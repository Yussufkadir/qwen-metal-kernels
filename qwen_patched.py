"""
qwen_custom.py — Custom Qwen forward: MLX attention + our Metal MLP
"""
import time, gc
import numpy as np, ctypes, mlx.core as mx
from mlx_lm import load
from metal_bridge import init, _lib

init()
print("Loading model...")
model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
print(f"{len(model.layers)} layers.\n")

B = len(model.layers)
MG, KG = 4864, 896
MD, KD = 896, 4864

# Extract weights
gate = np.zeros((B, MG, KG), dtype=np.float16)
up   = np.zeros((B, MG, KG), dtype=np.float16)
down = np.zeros((B, MD, KD), dtype=np.float16)
for i, l in enumerate(model.layers):
    gate[i] = np.array(l.mlp.gate_proj.weight.astype(mx.float32)).astype(np.float16)
    up[i]   = np.array(l.mlp.up_proj.weight.astype(mx.float32)).astype(np.float16)
    down[i] = np.array(l.mlp.down_proj.weight.astype(mx.float32)).astype(np.float16)

# Metal buffer setup
g_bits = np.ascontiguousarray(gate).view(np.uint16)
u_bits = np.ascontiguousarray(up).view(np.uint16)
d_bits = np.ascontiguousarray(down).view(np.uint16)
pg = g_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
pu = u_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
pd = d_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

g_out = np.zeros((B, MG), dtype=np.float32)
u_out = np.zeros((B, MG), dtype=np.float32)
d_out = np.zeros((B, MD), dtype=np.float32)
pgy = g_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
puy = u_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
pdy = d_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

x_batch = np.zeros((B, KG), dtype=np.float32)
px = x_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
h_batch = np.zeros((B, MG), dtype=np.float32)
ph = h_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

Bc = ctypes.c_uint32(B)
Mgc, Kgc = ctypes.c_uint32(MG), ctypes.c_uint32(KG)
Mdc, Kdc = ctypes.c_uint32(MD), ctypes.c_uint32(KD)

_lib.run_gate_up_batched(pg, pu, px, pgy, puy, Bc, Mgc, Kgc)
_lib.run_down_batched(pd, ph, pdy, Bc, Mdc, Kdc)
print("Metal ready.\n")

def metal_mlp_single_layer(layer_idx, hidden_np):
    """
    hidden_np: [1, seq, 896] -> returns this layer's MLP output [1, seq, 896]
    """
    seq = hidden_np.shape[1]
    res = np.zeros((1, seq, MD), dtype=np.float32)
    for pos in range(seq):
        x_batch[:] = hidden_np[0, pos, :]
        _lib.run_gate_up_batched(pg, pu, px, pgy, puy, Bc, Mgc, Kgc)
        np.multiply(g_out / (1.0 + np.exp(-g_out)), u_out, out=h_batch)
        _lib.run_down_batched(pd, ph, pdy, Bc, Mdc, Kdc)
        res[0, pos, :] = d_out[layer_idx]  # pick this layer's output
    return res

def custom_generate(prompt, max_new=20):
    tokens = mx.array([tokenizer.encode(prompt)])
    generated = []
    for step in range(max_new):
        # Embed
        h = model.model.embed_tokens(tokens)  # [1, seq, 896]
        mx.eval(h)
        seq_len = h.shape[1]
        
        # Run transformer layers
        for i, layer in enumerate(model.model.layers):
            # Attention
            residual = h
            h_norm = layer.input_layernorm(h)
            attn = layer.self_attn(h_norm)
            h = residual + attn
            mx.eval(h)
            
            # MLP (our Metal kernel)
            residual = h
            h_norm = layer.post_attention_layernorm(h)
            # Convert to numpy
            h_np = np.array(h_norm.astype(mx.float32)).astype(np.float32)
            mlp_np = metal_mlp_single_layer(i, h_np)  # [1, seq, 896]
            # Convert back to MLX
            mlp_out = mx.array(mlp_np.reshape(1, seq_len, MD))
            h = residual + mlp_out
            mx.eval(h)
            del h_np, mlp_np, mlp_out, residual, h_norm, attn
            gc.collect()
        
        # Final norm + lm_head
        h = model.model.norm(h)
        logits = model.lm_head(h[:, -1:, :])  # only last token
        mx.eval(logits)
        next_id = mx.argmax(logits[0, -1], axis=-1).item()
        del logits, h
        generated.append(next_id)
        new_token = mx.array([[next_id]])
        tokens = mx.concatenate([tokens, new_token], axis=1)
        mx.eval(tokens)
        del new_token
        if next_id in (tokenizer.eos_token_id, 151643, 151645):
            break
    return tokenizer.decode(generated)

# Test questions
prompts = [
    "What is the capital of France?",
    "What color is the sky?",
    "How many days in a week?",
    "What year did WWII end?",
]
print("="*60)
total_tok, total_t = 0, 0
for p in prompts:
    gc.collect()
    t0 = time.perf_counter()
    resp = custom_generate(p, max_new=20)
    t1 = time.perf_counter()
    n = len(tokenizer.encode(resp))
    total_tok += n
    total_t += (t1-t0)
    print(f"Q: {p}\nA: {resp}\n({n} tok in {t1-t0:.2f}s, {n/(t1-t0):.1f} tok/s)\n")
print("="*60)
print(f"PATCHED: {total_tok/total_t:.1f} tok/sec")
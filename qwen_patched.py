import time
import numpy as np
import ctypes
import mlx.core as mx
from mlx_lm import load
from metal_bridge import init, _lib

init()

print("Loading Qwen 0.5B Instruct...")
model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
print(f"Model loaded.{len(model.layers)} layers.\n")

print("Extracting MLP weights...")
all_gate_w = []
all_up_w = []
all_down_w = []

for layer in model.layers:
    mlp = layer.mlp
    gate_w_f32 = np.array(mlp.gate_proj.weight.astype(mx.float32))
    up_w_f32   = np.array(mlp.up_proj.weight.astype(mx.float32))
    down_w_f32 = np.array(mlp.down_proj.weight.astype(mx.float32))
    
    all_gate_w.append(gate_w_f32.astype(np.float16))
    all_up_w.append(up_w_f32.astype(np.float16))
    all_down_w.append(down_w_f32.astype(np.float16))

gate_w = np.stack(all_gate_w)   
up_w   = np.stack(all_up_w)     
down_w = np.stack(all_down_w)

BATCH = 24
M_GATE = 4864
K_GATE = 896
M_DOWN = 896
K_DOWN = 4864
print(f" gate_w: {gate_w.shape}, up_w: {up_w.shape}, down_w: {down_w.shape}")

gate_w_bits = np.ascontiguousarray(gate_w).view(np.uint16)
up_w_bits = np.ascontiguousarray(up_w).view(np.uint16)
down_w_bits = np.ascontiguousarray(down_w).view(np.uint16)

p_gate_w = gate_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
p_up_w = up_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
p_down_w = down_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

gate_out = np.zeros((BATCH, M_GATE), dtype=np.float32)
up_out = np.zeros((BATCH, M_GATE), dtype=np.float32)
down_out = np.zeros((BATCH, M_DOWN), dtype=np.float32)

p_gate_y = gate_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_up_y = up_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_down_y = down_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

B = ctypes.c_uint32(BATCH)
Mg = ctypes.c_uint32(M_GATE)
Kg = ctypes.c_uint32(K_GATE)
Md = ctypes.c_uint32(M_DOWN)
Kd = ctypes.c_uint32(K_DOWN)

x_batched = np.zeros((BATCH, K_GATE), dtype=np.float32)
p_x = x_batched.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
mlp_hidden = np.zeros((BATCH, M_GATE), dtype=np.float32)
p_mlp = mlp_hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

print("Pre-loading weights into Metal buffers...")
_lib.run_gate_up_batched(p_gate_w, p_up_w, p_x, p_gate_y, p_up_y, B, Mg, Kg)
_lib.run_down_batched(p_down_w, p_mlp, p_down_y, B, Md, Kd)
print("Weights loaded.\n")

def run_mlp_with_metal(hidden_states_np):
    seq_len = hidden_states_np.shape[1]

    result = np.zeros((seq_len, 24, 896), dtype=np.float32)
    
    for pos in range(seq_len):
        token_input = hidden_states_np[0, pos, :]  
        x_batched[:] = token_input
        
        _lib.run_gate_up_batched(p_gate_w, p_up_w, p_x, p_gate_y, p_up_y, B, Mg, Kg)
        np.multiply(
            gate_out / (1.0 + np.exp(-gate_out)),
            up_out,
            out=mlp_hidden
        )
        _lib.run_down_batched(p_down_w, p_mlp, p_down_y, B, Md, Kd)
        
        result[pos] = down_out  
    
    return result

class PatchedMLP:

    def __init__(self, original_mlp, layer_idx):
        self.original = original_mlp
        self.idx = layer_idx

    def __call__(self, x):
        x_np = np.array(x.astype(mx.float32)).astype(np.float32)
        result_np = run_mlp_with_metal(x_np) 

        layer_out = result_np[:, self.idx, :]  
        return mx.array(layer_out.reshape(1, -1, 896))
    
for i, layer in enumerate(model.layers):
    layer.mlp = PatchedMLP(layer.mlp, i)
print("MLP layers patched with custom Metal kernels.\n")

test_prompts = [
    "What is the capital of France?",
    "What color is the sky on a clear day?",
    "How many days are in a week?",
    "What year did World War II end?"
]

def generate_text(model, tokenizer, prompt, max_new=20):
    tokens = mx.array([tokenizer.encode(prompt)])

    generated_ids = []
    for _ in range(max_new):
        logits = model(tokens)
        next_id = mx.argmax(logits[:, -1, :], axis=-1).item()
        generated_ids.append(next_id)
        tokens = mx.concatenate([tokens, mx.array([[next_id]])], axis=1)

        if next_id in (tokenizer.eos_token_id, 151643, 151645):
            break

    return tokenizer.decode(generated_ids)

print("=" * 60)
print("PATCHED: Custom Metal kernels")
print("=" * 60)

total_tokens = 0
total_time = 0

for prompt in test_prompts:
    start = time.perf_counter()
    response = generate_text(model, tokenizer, prompt, max_new=20)
    elapsed = time.perf_counter() - start

    new_tokens = len(tokenizer.encode(response))
    total_tokens += new_tokens
    total_time += elapsed

    print(f"  Q: {prompt}")
    print(f"  A: {response}")
    print(f"     ({new_tokens} tokens in {elapsed:.2f}s, {new_tokens/elapsed:.1f} tok/s)\n")

patched_tok_sec = total_tokens / total_time

print("=" * 60)
print(f"  PATCHED: {patched_tok_sec:.1f} tok/sec")
print(f"  Total: {total_tokens} tokens in {total_time:.2f}s")
print("=" * 60)
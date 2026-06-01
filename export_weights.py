import numpy as np
import mlx.core as mx
import os
from mlx_lm import load

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
os.makedirs("qwen_weights", exist_ok=True)

def save_tensor(name, tensor):
    arr = np.array(tensor.astype(mx.float16))
    arr.tofile(f"qwen_weights/{name}.bin")

save_tensor("embed_tokens.weight", model.model.embed_tokens.weight)
save_tensor("norm.weight", model.model.norm.weight)
head_weight = model.lm_head.weight if hasattr(model, 'lm_head') else model.model.embed_tokens.weight
save_tensor("lm_head.weight", head_weight)


for i, layer in enumerate(model.layers):
    save_tensor(f"layer{i}.input_layernorm.weight", layer.input_layernorm.weight)
    save_tensor(f"layer{i}.post_attention_layernorm.weight", layer.post_attention_layernorm.weight)
    save_tensor(f"layer{i}.self_attn.q_proj.weight", layer.self_attn.q_proj.weight)
    save_tensor(f"layer{i}.self_attn.k_proj.weight", layer.self_attn.k_proj.weight)
    save_tensor(f"layer{i}.self_attn.v_proj.weight", layer.self_attn.v_proj.weight)
    save_tensor(f"layer{i}.self_attn.o_proj.weight", layer.self_attn.o_proj.weight)
    save_tensor(f"layer{i}.mlp.gate_proj.weight", layer.mlp.gate_proj.weight)
    save_tensor(f"layer{i}.mlp.up_proj.weight", layer.mlp.up_proj.weight)
    save_tensor(f"layer{i}.mlp.down_proj.weight", layer.mlp.down_proj.weight)

print("Weights exported to qwen_weights/")

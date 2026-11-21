import shutil
import numpy as np
import mlx.core as mx
from pathlib import Path
from huggingface_hub import snapshot_download
from mlx_lm import load

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
WEIGHTS_DIR = Path(__file__).resolve().parent / "kernel/qwen_kernel/qwen_kernel/qwen_weights"
TOKENIZER_DIR = WEIGHTS_DIR / "tokenizer"
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")

model, tokenizer = load(MODEL_NAME)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

snapshot = Path(snapshot_download(MODEL_NAME, allow_patterns=list(TOKENIZER_FILES)))
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
for name in TOKENIZER_FILES:
    shutil.copyfile(snapshot / name, TOKENIZER_DIR / name)
print(f"Tokenizer exported to {TOKENIZER_DIR}")

def save_tensor(name, tensor):
    arr = np.array(tensor.astype(mx.float16))
    arr.tofile(WEIGHTS_DIR / f"{name}.bin")

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

    if getattr(layer.self_attn.q_proj, "bias", None) is not None:
        save_tensor(f"layer{i}.self_attn.q_proj.bias", layer.self_attn.q_proj.bias)
    if getattr(layer.self_attn.k_proj, "bias", None) is not None:
        save_tensor(f"layer{i}.self_attn.k_proj.bias", layer.self_attn.k_proj.bias)
    if getattr(layer.self_attn.v_proj, "bias", None) is not None:
        save_tensor(f"layer{i}.self_attn.v_proj.bias", layer.self_attn.v_proj.bias)

    save_tensor(f"layer{i}.mlp.gate_proj.weight", layer.mlp.gate_proj.weight)
    save_tensor(f"layer{i}.mlp.up_proj.weight", layer.mlp.up_proj.weight)
    save_tensor(f"layer{i}.mlp.down_proj.weight", layer.mlp.down_proj.weight)
    print(f"Layer {i} exported")

print(f"Weights exported to {WEIGHTS_DIR}")

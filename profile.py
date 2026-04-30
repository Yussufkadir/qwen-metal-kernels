from mlx_lm import load
import mlx.core as mx

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")

print("=== Qwen 0.5B weight shapes ===")
for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        w = module.weight
        if hasattr(w, 'shape'):
            print(f"{name:50s} {str(w.shape):20s} {w.dtype}")

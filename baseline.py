from mlx_lm import load, generate
import time

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")

# warmup
generate(model, tokenizer, prompt="hi", max_tokens=5, verbose=False)

# baseline measurement
prompt = "explain how neural networks learn"
start = time.time()
response = generate(
    model,
    tokenizer, 
    prompt=prompt,
    max_tokens=100,
    verbose=False
)
elapsed = time.time() - start

print(f"tokens/sec: {100 / elapsed:.1f}")
print(f"response: {response}")

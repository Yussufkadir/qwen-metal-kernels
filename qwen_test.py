import time
import mlx.core as mx
from mlx_lm import load

print("Loading Qwen 0.5B Instruct...")
model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
print("Model loaded.\n")

test_prompts = [
    "What is the capital of France?",
    "What color is the sky on a clear day?",
    "How many days are in a week?",
    "What year did World War II end?",
]

def generate_text(model, tokenizer, prompt, max_new=20):
    tokens = mx.array([tokenizer.encode(prompt)])
    
    generated_ids = []
    for _ in range(max_new):
        logits = model(tokens)
        next_id = mx.argmax(logits[:, -1, :], axis=-1).item()
        generated_ids.append(next_id)

        tokens = mx.concatenate(
            [tokens, mx.array([[next_id]])], axis=1
        )

        if next_id in (tokenizer.eos_token_id, 151643, 151645):
            break
    
    return tokenizer.decode(generated_ids)

print("="*60)
print("BASELINE: MLX default kernels")
print("="*60)

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

baseline_tok_sec = total_tokens / total_time

print("="*60)
print(f"  BASELINE: {baseline_tok_sec:.1f} tok/sec")
print(f"  Total: {total_tokens} tokens in {total_time:.2f}s")
print("="*60)
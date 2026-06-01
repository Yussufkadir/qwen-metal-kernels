import sys 
from mlx_lm import load

_, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
text = sys.argv[1] if len(sys.argv) > 1 else "Hello"
ids = tokenizer.encode(text)
print(''.join(map(str, ids)))
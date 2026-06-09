import sys
from mlx_lm import load

_, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")

if sys.argv[1] == "encode":
    text = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    ids = tokenizer.encode(text)
    print(' '.join(map(str, ids)))
elif sys.argv[1] == "decode":
    ids = list(map(int, sys.argv[2:]))
    print(tokenizer.decode(ids))
else:
    print("Usage: token_util.py encode 'text' | decode id1 id2 ...")
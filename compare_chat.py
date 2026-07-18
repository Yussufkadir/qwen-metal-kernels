#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "kernel/qwen_kernel/qwen_kernel"
ENGINE = ENGINE_DIR / "inference_engine"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATED_RE = re.compile(r"Generated continuation:\s*([0-9 ]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="*", help="user prompt text")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--tokens", "-n", type=int, default=80)
    parser.add_argument("--backend", default="fp16", choices=["fp16", "int4", "int8", "int8-fp16", "mixed", "mps", "hybrid"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--engine-dir", type=Path, default=ENGINE_DIR)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--show-token-ids", action="store_true")
    return parser.parse_args()


def normalize_token_ids(value) -> list[int]:
    if isinstance(value, dict) and "input_ids" in value:
        return [int(token) for token in value["input_ids"]]
    if hasattr(value, "input_ids"):
        return [int(token) for token in value.input_ids]
    return [int(token) for token in value]


def load_transformers_tokenizer(model_name: str, allow_download: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=not allow_download,
    )


def make_prompt_tokens(tokenizer, system_prompt: str, user_prompt: str) -> list[int]:
    templated = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    return normalize_token_ids(templated)


def decode(tokenizer, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True)


def first_mismatch(left: list[int], right: list[int]) -> tuple[int, int | None]:
    matched = 0
    mismatch = None
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            mismatch = index
            break
        matched += 1
    if mismatch is None and len(left) != len(right):
        mismatch = matched
    return matched, mismatch


def generate_mlx(model_name: str, prompt_tokens: list[int], max_tokens: int) -> tuple[list[int], float]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    model, _ = load(model_name)
    output: list[int] = []
    start = time.perf_counter()
    for token, _ in generate_step(mx.array(prompt_tokens), model, max_tokens=max_tokens):
        output.append(int(token))
        if len(output) >= max_tokens:
            break
    elapsed = time.perf_counter() - start
    return output, elapsed


def generate_native(
    engine: Path,
    engine_dir: Path,
    backend: str,
    prompt_tokens: list[int],
    max_tokens: int,
) -> tuple[list[int], float, str]:
    command = [
        str(engine),
        "--backend",
        backend,
        "--max-tokens",
        str(max_tokens),
        "--quiet",
        *(str(token) for token in prompt_tokens),
    ]
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=engine_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(result.stdout)
    match = GENERATED_RE.search(result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse native output:\n{result.stdout}")
    return [int(token) for token in match.group(1).split()], elapsed, result.stdout


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = "Explain transformer KV cache in 3 bullet points. Be technically correct and concise."
    if not args.engine.exists():
        raise SystemExit(f"Native engine not found: {args.engine}\nRun: {args.engine_dir / 'build_engine.sh'}")

    tokenizer = load_transformers_tokenizer(args.model, args.allow_download)
    prompt_tokens = make_prompt_tokens(tokenizer, args.system, prompt)

    print(f"Prompt: {prompt!r}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    if args.show_token_ids:
        print("Prompt token IDs:")
        print(" ".join(str(token) for token in prompt_tokens))
    print()

    print("Generating MLX reference...")
    mlx_tokens, mlx_elapsed = generate_mlx(args.model, prompt_tokens, args.tokens)
    print("Generating native engine...")
    native_tokens, native_elapsed, native_raw = generate_native(
        args.engine, args.engine_dir, args.backend, prompt_tokens, args.tokens
    )

    matched, mismatch = first_mismatch(mlx_tokens, native_tokens)
    mismatch_text = "exact match" if mismatch is None else f"first mismatch at generated token {mismatch}"

    print("\n=== MLX decoded ===")
    print(decode(tokenizer, mlx_tokens))
    print("\n=== Native decoded ===")
    print(decode(tokenizer, native_tokens))

    print("\n=== Token comparison ===")
    print(f"Backend: {args.backend}")
    print(f"MLX tokens: {len(mlx_tokens)} in {mlx_elapsed:.3f}s ({len(mlx_tokens) / mlx_elapsed:.1f} tok/s)")
    print(
        f"Native tokens: {len(native_tokens)} in {native_elapsed:.3f}s "
        f"({len(native_tokens) / native_elapsed:.1f} tok/s, includes process/runtime overhead)"
    )
    print(f"Agreement: {matched}/{min(len(mlx_tokens), len(native_tokens))} matching, {mismatch_text}")
    if mismatch is not None and mismatch < len(mlx_tokens) and mismatch < len(native_tokens):
        print(f"MLX token at mismatch:    {mlx_tokens[mismatch]}")
        print(f"Native token at mismatch: {native_tokens[mismatch]}")
        start = max(0, mismatch - 8)
        end = mismatch + 8
        print(f"MLX window:    {mlx_tokens[start:end]}")
        print(f"Native window: {native_tokens[start:end]}")


if __name__ == "__main__":
    main()

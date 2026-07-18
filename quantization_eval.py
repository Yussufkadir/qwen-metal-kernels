#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "kernel/qwen_kernel/qwen_kernel"
ENGINE = ENGINE_DIR / "inference_engine"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATED_RE = re.compile(r"Generated continuation:\s*([0-9 ]+)")


DEFAULT_PROMPTS = [
    "Explain transformer KV cache in 3 technically correct bullet points.",
    "What is the difference between prefill and decode in LLM inference?",
    "Explain why quantization can make a model faster but less accurate.",
    "Write a Python function that returns the factorial of n.",
    "Summarize this in one sentence: Metal kernels reduce dispatch and memory overhead when operations are fused.",
]


@dataclass
class QuantEvalRow:
    prompt_index: int
    prompt: str
    backend: str
    reference_backend: str
    generated_tokens: int
    elapsed_ms: float
    tok_s: float
    matching_prefix: int
    agreement_ratio: float
    first_mismatch: int | None
    decoded: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare quantized native backends against FP16 output.")
    parser.add_argument("--backends", default="int8,int8-fp16,int4", help="comma-separated quantized backends")
    parser.add_argument("--reference", default="fp16", help="reference backend")
    parser.add_argument("--tokens", "-n", type=int, default=64)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--engine-dir", type=Path, default=ENGINE_DIR)
    parser.add_argument("--prompt-file", type=Path, help="optional newline-separated prompt file")
    parser.add_argument("--json", type=Path, help="optional raw result output")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def normalize_token_ids(value) -> list[int]:
    if isinstance(value, dict) and "input_ids" in value:
        return [int(token) for token in value["input_ids"]]
    if hasattr(value, "input_ids"):
        return [int(token) for token in value.input_ids]
    return [int(token) for token in value]


def load_tokenizer(model_name: str, allow_download: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=not allow_download,
    )


def make_prompt_tokens(tokenizer, system_prompt: str, prompt: str) -> list[int]:
    templated = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    return normalize_token_ids(templated)


def generate_native(
    engine: Path,
    engine_dir: Path,
    backend: str,
    prompt_tokens: list[int],
    max_tokens: int,
    timeout: float,
) -> tuple[list[int], float]:
    command = [
        str(engine),
        "--backend",
        backend,
        "--quiet",
        "--max-tokens",
        str(max_tokens),
        *(str(token) for token in prompt_tokens),
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=engine_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"{backend} failed:\n{result.stdout}")
    match = GENERATED_RE.search(result.stdout)
    if not match:
        raise RuntimeError(f"could not parse {backend} output:\n{result.stdout}")
    return [int(token) for token in match.group(1).split()], elapsed


def first_mismatch(reference: list[int], candidate: list[int]) -> tuple[int, int | None]:
    matched = 0
    mismatch = None
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            mismatch = index
            break
        matched += 1
    if mismatch is None and len(reference) != len(candidate):
        mismatch = matched
    return matched, mismatch


def read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [line.strip() for line in path.read_text().splitlines()]
    return [prompt for prompt in prompts if prompt and not prompt.startswith("#")]


def main() -> int:
    args = parse_args()
    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    valid = {"fp16", "int8", "int8-fp16", "int4", "mixed", "mps", "hybrid"}
    invalid = set(backends + [args.reference]) - valid
    if invalid:
        raise SystemExit(f"unknown backend(s): {', '.join(sorted(invalid))}")
    if not args.engine.exists():
        raise SystemExit(f"native engine not found: {args.engine}")

    tokenizer = load_tokenizer(args.model, args.allow_download)
    prompts = read_prompts(args.prompt_file)
    rows: list[QuantEvalRow] = []

    print(f"Quantization eval against {args.reference}")
    print(f"backends={','.join(backends)} prompts={len(prompts)} tokens={args.tokens}")
    print()

    for prompt_index, prompt in enumerate(prompts, start=1):
        prompt_tokens = make_prompt_tokens(tokenizer, args.system, prompt)
        ref_tokens, ref_elapsed = generate_native(
            args.engine, args.engine_dir, args.reference, prompt_tokens, args.tokens, args.timeout
        )
        ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
        print(f"[{prompt_index}/{len(prompts)}] {prompt}")
        print(f"  {args.reference:>10}: {len(ref_tokens):3d} tok, {len(ref_tokens) / ref_elapsed:6.1f} tok/s")

        for backend in backends:
            tokens, elapsed = generate_native(
                args.engine, args.engine_dir, backend, prompt_tokens, args.tokens, args.timeout
            )
            matched, mismatch = first_mismatch(ref_tokens, tokens)
            denominator = max(min(len(ref_tokens), len(tokens)), 1)
            ratio = matched / denominator
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            rows.append(
                QuantEvalRow(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    backend=backend,
                    reference_backend=args.reference,
                    generated_tokens=len(tokens),
                    elapsed_ms=elapsed * 1000.0,
                    tok_s=len(tokens) / elapsed if elapsed else 0.0,
                    matching_prefix=matched,
                    agreement_ratio=ratio,
                    first_mismatch=mismatch,
                    decoded=text,
                )
            )
            mismatch_text = "exact" if mismatch is None else f"mismatch@{mismatch}"
            print(f"  {backend:>10}: {len(tokens):3d} tok, {len(tokens) / elapsed:6.1f} tok/s, {ratio * 100:5.1f}% prefix, {mismatch_text}")

        if args.show_text:
            print()
            print(f"  {args.reference} text:")
            print("  " + ref_text.replace("\n", "\n  "))
            for backend in backends:
                text = next(row.decoded for row in rows if row.prompt_index == prompt_index and row.backend == backend)
                print()
                print(f"  {backend} text:")
                print("  " + text.replace("\n", "\n  "))
        print()

    print("Summary")
    print(f"{'backend':>10}  {'median prefix':>13}  {'mean prefix':>11}  {'median tok/s':>12}")
    print("-" * 54)
    for backend in backends:
        selected = [row for row in rows if row.backend == backend]
        if not selected:
            continue
        print(
            f"{backend:>10}  "
            f"{statistics.median(row.agreement_ratio for row in selected) * 100:12.1f}%  "
            f"{statistics.mean(row.agreement_ratio for row in selected) * 100:10.1f}%  "
            f"{statistics.median(row.tok_s for row in selected):12.1f}"
        )

    if args.json:
        args.json.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n")
        print(f"\nRaw results: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "kernel/qwen_kernel/qwen_kernel"
ENGINE = ENGINE_DIR / "inference_engine"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TIMING_RE = re.compile(
    r"\[timing\] prefill: ([0-9.]+) ms .*?\n"
    r"\[timing\] decode: ([0-9.]+) ms for (\d+) tokens \(([0-9.]+) tok/s\)"
)
TOKENS_RE = re.compile(r"Generated continuation:\s*([0-9 ]+)")


@dataclass
class Measurement:
    engine: str
    context: int
    repeat: int
    prefill_ms: float
    prefill_tps: float
    decode_ms: float
    decode_tps: float
    output_tokens: list[int]


def find_subsequence(sequence: list[int], needle: list[int]) -> int:
    for index in range(len(sequence) - len(needle) + 1):
        if sequence[index : index + len(needle)] == needle:
            return index
    raise ValueError("Could not locate marker inside the chat template")


def make_prompt_tokens(tokenizer, target_length: int) -> list[int]:
    marker = "ZXQMARKERZXQ"
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=True,
        add_generation_prompt=True,
    )
    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
    marker_at = find_subsequence(templated, marker_tokens)
    prefix = templated[:marker_at]
    suffix = templated[marker_at + len(marker_tokens) :]
    content_length = target_length - len(prefix) - len(suffix)
    if content_length < 1:
        minimum = len(prefix) + len(suffix) + 1
        raise ValueError(f"Context {target_length} is too short for the chat template; minimum is {minimum}")

    source = (
        "Explain how neural networks learn in simple terms, including examples, "
        "limitations, and practical applications. " * ((content_length // 12) + 4)
    )
    content = tokenizer.encode(source, add_special_tokens=False)
    if len(content) < content_length:
        raise RuntimeError("Failed to create enough deterministic prompt tokens")
    result = prefix + content[:content_length] + suffix
    assert len(result) == target_length
    return result


def benchmark_mlx(mx, generate_step, model, prompt: list[int], decode_tokens: int, repeat: int) -> Measurement:
    generator = generate_step(mx.array(prompt), model, max_tokens=decode_tokens)
    output: list[int] = []
    started = time.perf_counter()
    first_ready = None
    for index in range(decode_tokens):
        token, _ = next(generator)
        output.append(int(token))
        if index == 0:
            first_ready = time.perf_counter()
    finished = time.perf_counter()
    assert first_ready is not None
    prefill_seconds = first_ready - started
    decode_seconds = finished - first_ready
    measured_decode_tokens = max(decode_tokens - 1, 0)
    return Measurement(
        engine="mlx",
        context=len(prompt),
        repeat=repeat,
        prefill_ms=prefill_seconds * 1000.0,
        prefill_tps=len(prompt) / prefill_seconds,
        decode_ms=decode_seconds * 1000.0,
        decode_tps=(measured_decode_tokens / decode_seconds if decode_seconds else 0.0),
        output_tokens=output,
    )


def benchmark_native(backend: str, prompt: list[int], decode_tokens: int, repeat: int) -> Measurement:
    command = [
        str(ENGINE),
        "--backend",
        backend,
        "--max-tokens",
        str(decode_tokens),
        *(str(token) for token in prompt),
    ]
    process = subprocess.run(
        command,
        cwd=ENGINE_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"Native {backend} failed ({process.returncode}):\n{process.stdout}")
    timing = TIMING_RE.search(process.stdout)
    generated = TOKENS_RE.search(process.stdout)
    if not timing or not generated:
        raise RuntimeError(f"Could not parse native {backend} output:\n{process.stdout}")
    prefill_ms, decode_ms, measured_tokens, decode_tps = timing.groups()
    outputs = [int(token) for token in generated.group(1).split()]
    prefill_value = float(prefill_ms)
    return Measurement(
        engine=backend,
        context=len(prompt),
        repeat=repeat,
        prefill_ms=prefill_value,
        prefill_tps=len(prompt) / (prefill_value / 1000.0),
        decode_ms=float(decode_ms),
        decode_tps=float(decode_tps),
        output_tokens=outputs,
    )


def median(values: Iterable[float]) -> float:
    return statistics.median(values)


def agreement(reference: list[int], candidate: list[int]) -> tuple[int, int | None]:
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


def print_summary(measurements: list[Measurement], contexts: list[int], engines: list[str]) -> None:
    print("\nMedian results (model/process loading excluded)")
    print(f"{'context':>7}  {'engine':>6}  {'prefill ms':>10}  {'prefill t/s':>11}  {'decode t/s':>10}")
    print("-" * 55)
    for context in contexts:
        for engine in engines:
            rows = [m for m in measurements if m.context == context and m.engine == engine]
            if not rows:
                continue
            print(
                f"{context:7d}  {engine:>6}  {median(m.prefill_ms for m in rows):10.2f}  "
                f"{median(m.prefill_tps for m in rows):11.1f}  {median(m.decode_tps for m in rows):10.1f}"
            )

    if "mlx" in engines and "fp16" in engines:
        print("\nFP16 token agreement against MLX (first repeat)")
        for context in contexts:
            mlx_rows = [m for m in measurements if m.context == context and m.engine == "mlx"]
            native_rows = [m for m in measurements if m.context == context and m.engine == "fp16"]
            if not mlx_rows or not native_rows:
                continue
            matched, mismatch = agreement(mlx_rows[0].output_tokens, native_rows[0].output_tokens)
            status = "exact" if mismatch is None else f"first mismatch at generated token {mismatch}"
            print(f"  context {context:4d}: {matched:3d} matching tokens, {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="32,128,512,1024", help="comma-separated prompt lengths")
    parser.add_argument("--tokens", type=int, default=100, help="generated tokens per measurement")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", default="mlx,fp16,int4", help="subset of mlx,fp16,int4,int8,int8-fp16,mixed,mps,hybrid")
    parser.add_argument("--json", type=Path, help="optional raw measurement output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contexts = [int(value) for value in args.contexts.split(",")]
    engines = [value.strip() for value in args.backends.split(",")]
    invalid = set(engines) - {"mlx", "fp16", "int4", "int8", "int8-fp16", "mixed", "mps", "hybrid"}
    if invalid:
        raise SystemExit(f"Unknown backends: {', '.join(sorted(invalid))}")
    if args.tokens < 2:
        raise SystemExit("--tokens must be at least 2 so decode throughput is measurable")
    if not ENGINE.exists() and any(engine != "mlx" for engine in engines):
        raise SystemExit(f"Native engine not found: {ENGINE}; run build_engine.sh first")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    print(f"Loading {MODEL_NAME} for MLX/tokenizer...")
    model, tokenizer = load(MODEL_NAME)
    prompts = {length: make_prompt_tokens(tokenizer, length) for length in contexts}

    if "mlx" in engines:
        print("Warming MLX kernels...")
        list(generate_step(mx.array(prompts[contexts[0]]), model, max_tokens=4))

    measurements: list[Measurement] = []
    for context in contexts:
        prompt = prompts[context]
        print(f"\nContext {context}, decode {args.tokens}")
        for repeat in range(args.repeats):
            ordered = engines[repeat % len(engines) :] + engines[: repeat % len(engines)]
            for engine in ordered:
                if engine == "mlx":
                    result = benchmark_mlx(mx, generate_step, model, prompt, args.tokens, repeat)
                else:
                    result = benchmark_native(engine, prompt, args.tokens, repeat)
                measurements.append(result)
                print(
                    f"  run {repeat + 1}/{args.repeats} {engine:>6}: "
                    f"prefill {result.prefill_tps:8.1f} t/s, decode {result.decode_tps:8.1f} t/s"
                )

    print_summary(measurements, contexts, engines)
    if args.json:
        args.json.write_text(json.dumps([asdict(item) for item in measurements], indent=2) + "\n")
        print(f"\nRaw results: {args.json}")


if __name__ == "__main__":
    main()

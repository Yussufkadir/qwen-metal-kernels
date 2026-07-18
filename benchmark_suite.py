#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "kernel/qwen_kernel/qwen_kernel"
ENGINE = ENGINE_DIR / "inference_engine"
QWEN_LOCAL = ROOT / "qwen-local"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

NATIVE_TIMING_RE = re.compile(
    r"\[timing\] prefill: ([0-9.]+) ms for (\d+) tokens \(([0-9.]+) tok/s\).*?"
    r"\[timing\] decode: ([0-9.]+) ms for (\d+) tokens \(([0-9.]+) tok/s\)",
    re.S,
)
GENERATED_RE = re.compile(r"Generated continuation:\s*([0-9 ]+)")
STARTUP_RE = re.compile(r"\[startup\] ([a-zA-Z0-9_]+): ([0-9.]+) ms")
STARTUP_FLAG_RE = re.compile(r"\[startup\] ([a-zA-Z0-9_]+): (yes|no)")
RUNTIME_RE = re.compile(r"\[runtime\].*generated_tokens=(\d+).*elapsed=([0-9.]+)s")
CHAT_TIMING_RE = re.compile(
    r"\[timing\] prompt=([0-9.]+)ms native=([0-9.]+)ms "
    r"first_token=([0-9.]+|n/a)ms decode≈([0-9.]+) tok/s decode/print=([0-9.]+)ms"
)
CACHE_RE = re.compile(r"\[cache\] common=(\d+) delta=(\d+)( reset)?")


@dataclass
class BenchRow:
    kind: str
    backend: str
    repeat: int
    context_tokens: int | None = None
    generated_tokens: int | None = None
    elapsed_ms: float | None = None
    startup_ms: float | None = None
    prefill_ms: float | None = None
    prefill_tps: float | None = None
    first_token_ms: float | None = None
    decode_ms: float | None = None
    decode_tps: float | None = None
    prompt_ms: float | None = None
    native_ms: float | None = None
    decode_print_ms: float | None = None
    cache_common: int | None = None
    cache_delta: int | None = None
    cache_reset: bool | None = None
    gpu_warmup: bool | None = None
    packed_weights: bool | None = None
    mmap_weights: bool | None = None
    output_tokens: list[int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliable benchmark suite for the custom Qwen runtime.")
    parser.add_argument("--backends", default="fp16", help="comma-separated native backends")
    parser.add_argument("--contexts", default="32,128,512", help="comma-separated prompt lengths for core decode")
    parser.add_argument("--tokens", type=int, default=64, help="generated tokens per core/chat measurement")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--weight-warmup", action="store_true", help="pass --warmup to native/qwen-local runs")
    parser.add_argument(
        "--session-gpu-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warm the persistent GPU prefill path before measuring chat turns",
    )
    parser.add_argument("--session-turns", type=int, default=4)
    parser.add_argument("--session-tokens", type=int, default=32)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--engine-dir", type=Path, default=ENGINE_DIR)
    parser.add_argument("--qwen-local", type=Path, default=QWEN_LOCAL)
    parser.add_argument("--json", type=Path, help="write raw benchmark rows to JSON")
    parser.add_argument("--skip-core", action="store_true")
    parser.add_argument("--skip-startup", action="store_true")
    parser.add_argument("--skip-oneshot", action="store_true")
    parser.add_argument("--skip-session", action="store_true")
    parser.add_argument("--include-mlx", action="store_true", help="include MLX core decode reference")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--timeout", type=float, default=240.0)
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


def find_subsequence(sequence: list[int], needle: list[int]) -> int:
    for index in range(len(sequence) - len(needle) + 1):
        if sequence[index : index + len(needle)] == needle:
            return index
    raise ValueError("could not locate prompt marker inside chat template")


def make_prompt_tokens(tokenizer, target_length: int) -> list[int]:
    marker = "ZXQMARKERZXQ"
    templated = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
    marker_at = find_subsequence(templated, marker_tokens)
    prefix = templated[:marker_at]
    suffix = templated[marker_at + len(marker_tokens) :]
    content_length = target_length - len(prefix) - len(suffix)
    if content_length < 1:
        minimum = len(prefix) + len(suffix) + 1
        raise ValueError(f"context {target_length} is too short; minimum is {minimum}")

    source = (
        "Explain how neural networks learn in simple terms, including examples, "
        "limitations, and practical applications. "
    )
    encoded = tokenizer.encode(source * ((content_length // 12) + 8), add_special_tokens=False)
    if len(encoded) < content_length:
        raise RuntimeError("failed to create enough deterministic prompt tokens")
    result = prefix + encoded[:content_length] + suffix
    if len(result) != target_length:
        raise RuntimeError(f"prompt length mismatch: wanted {target_length}, got {len(result)}")
    return result


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    input_text: str | None = None,
) -> tuple[int, str, float]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return result.returncode, result.stdout, (time.perf_counter() - started) * 1000.0


def parse_generated(output: str) -> list[int]:
    match = GENERATED_RE.search(output)
    if not match:
        raise RuntimeError(f"could not parse generated tokens:\n{output}")
    return [int(token) for token in match.group(1).split()]


def benchmark_native_core(
    engine: Path,
    engine_dir: Path,
    backend: str,
    prompt_tokens: list[int],
    max_tokens: int,
    repeat: int,
    timeout: float,
    weight_warmup: bool,
) -> BenchRow:
    command = [
        str(engine),
        "--backend",
        backend,
        "--max-tokens",
        str(max_tokens),
    ]
    if weight_warmup:
        command.append("--warmup")
    command.extend(str(token) for token in prompt_tokens)
    code, output, elapsed_ms = run_command(command, cwd=engine_dir, timeout=timeout)
    if code:
        raise RuntimeError(f"native core {backend} failed:\n{output}")
    timing = NATIVE_TIMING_RE.search(output)
    if not timing:
        raise RuntimeError(f"could not parse native timing:\n{output}")
    prefill_ms, _, prefill_tps, decode_ms, _, decode_tps = timing.groups()
    output_tokens = parse_generated(output)
    return BenchRow(
        kind="native_core",
        backend=backend,
        repeat=repeat,
        context_tokens=len(prompt_tokens),
        generated_tokens=len(output_tokens),
        elapsed_ms=elapsed_ms,
        prefill_ms=float(prefill_ms),
        prefill_tps=float(prefill_tps),
        decode_ms=float(decode_ms),
        decode_tps=float(decode_tps),
        output_tokens=output_tokens,
    )


def benchmark_startup(
    engine: Path,
    engine_dir: Path,
    backend: str,
    repeat: int,
    timeout: float,
    weight_warmup: bool,
) -> BenchRow:
    command = [
        str(engine),
        "--backend",
        backend,
        "--startup-timing",
        "--quiet",
        "--max-tokens",
        "1",
        "1",
    ]
    if weight_warmup:
        command.append("--warmup")
    code, output, elapsed_ms = run_command(command, cwd=engine_dir, timeout=timeout)
    if code:
        raise RuntimeError(f"startup benchmark {backend} failed:\n{output}")
    timings = {name: float(value) for name, value in STARTUP_RE.findall(output)}
    flags = {name: value == "yes" for name, value in STARTUP_FLAG_RE.findall(output)}
    return BenchRow(
        kind="startup",
        backend=backend,
        repeat=repeat,
        elapsed_ms=elapsed_ms,
        startup_ms=timings.get("total"),
        packed_weights=flags.get("packed_weights"),
        mmap_weights=flags.get("mmap_weights"),
    )


def benchmark_oneshot(
    qwen_local: Path,
    backend: str,
    max_tokens: int,
    repeat: int,
    timeout: float,
    weight_warmup: bool,
) -> BenchRow:
    prompt = "Say hello in one short sentence."
    command = [
        str(qwen_local),
        "--backend",
        backend,
        "--stats",
        *(["--warmup"] if weight_warmup else []),
        "-n",
        str(max_tokens),
        prompt,
    ]
    code, output, elapsed_ms = run_command(command, cwd=ROOT, timeout=timeout)
    if code:
        raise RuntimeError(f"one-shot benchmark {backend} failed:\n{output}")
    runtime = RUNTIME_RE.search(output)
    timing = CHAT_TIMING_RE.search(output)
    if not runtime or not timing:
        raise RuntimeError(f"could not parse one-shot output:\n{output}")
    generated_tokens, runtime_elapsed = runtime.groups()
    prompt_ms, native_ms, first_token_ms, decode_tps, decode_print_ms = timing.groups()
    first = None if first_token_ms == "n/a" else float(first_token_ms)
    return BenchRow(
        kind="oneshot",
        backend=backend,
        repeat=repeat,
        generated_tokens=int(generated_tokens),
        elapsed_ms=elapsed_ms,
        prompt_ms=float(prompt_ms),
        native_ms=float(native_ms),
        first_token_ms=first,
        decode_tps=float(decode_tps),
        decode_print_ms=float(decode_print_ms),
    )


def benchmark_session(
    qwen_local: Path,
    backend: str,
    turns: int,
    max_tokens: int,
    repeat: int,
    timeout: float,
    weight_warmup: bool,
    gpu_warmup: bool,
) -> list[BenchRow]:
    prompts = [
        "Say hello in one short sentence.",
        "Say it shorter.",
        "Now answer with only three words.",
        "What did I ask first?",
        "Give one tiny summary.",
        "End with a friendly goodbye.",
    ]
    selected = prompts[:turns]
    input_text = "\n".join(selected + ["/exit"]) + "\n"
    command = [
        str(qwen_local),
        "--chat",
        "--persistent",
        "--backend",
        backend,
        "--stats",
        "--no-stream",
        "--gpu-warmup" if gpu_warmup else "--no-gpu-warmup",
        *(["--warmup"] if weight_warmup else []),
        "-n",
        str(max_tokens),
    ]
    code, output, elapsed_ms = run_command(command, cwd=ROOT, timeout=timeout, input_text=input_text)
    if code:
        raise RuntimeError(f"session benchmark {backend} failed:\n{output}")

    timing_rows = CHAT_TIMING_RE.findall(output)
    cache_rows = CACHE_RE.findall(output)
    if len(timing_rows) < turns or len(cache_rows) < turns:
        raise RuntimeError(f"could not parse session output:\n{output}")

    rows: list[BenchRow] = []
    for turn_index, timing in enumerate(timing_rows[:turns]):
        prompt_ms, native_ms, first_token_ms, decode_tps, decode_print_ms = timing
        common, delta, reset = cache_rows[turn_index]
        first = None if first_token_ms == "n/a" else float(first_token_ms)
        rows.append(
            BenchRow(
                kind="session_turn",
                backend=backend,
                repeat=repeat,
                context_tokens=turn_index + 1,
                generated_tokens=max_tokens,
                elapsed_ms=elapsed_ms if turn_index == 0 else None,
                prompt_ms=float(prompt_ms),
                native_ms=float(native_ms),
                first_token_ms=first,
                decode_tps=float(decode_tps),
                decode_print_ms=float(decode_print_ms),
                cache_common=int(common),
                cache_delta=int(delta),
                cache_reset=bool(reset),
                gpu_warmup=gpu_warmup,
            )
        )
    return rows


def benchmark_mlx_core(mx, generate_step, model, prompt_tokens: list[int], max_tokens: int, repeat: int) -> BenchRow:
    output: list[int] = []
    generator = generate_step(mx.array(prompt_tokens), model, max_tokens=max_tokens)
    started = time.perf_counter()
    first_ready = None
    for index in range(max_tokens):
        token, _ = next(generator)
        output.append(int(token))
        if index == 0:
            first_ready = time.perf_counter()
    finished = time.perf_counter()
    if first_ready is None:
        raise RuntimeError("MLX produced no tokens")
    prefill_ms = (first_ready - started) * 1000.0
    decode_ms = (finished - first_ready) * 1000.0
    decode_tokens = max(max_tokens - 1, 0)
    return BenchRow(
        kind="mlx_core",
        backend="mlx",
        repeat=repeat,
        context_tokens=len(prompt_tokens),
        generated_tokens=max_tokens,
        elapsed_ms=(finished - started) * 1000.0,
        prefill_ms=prefill_ms,
        prefill_tps=len(prompt_tokens) / (prefill_ms / 1000.0),
        decode_ms=decode_ms,
        decode_tps=decode_tokens / (decode_ms / 1000.0) if decode_ms else 0.0,
        output_tokens=output,
    )


def median(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return statistics.median(clean)


def print_core_summary(rows: list[BenchRow], contexts: list[int], backends: list[str]) -> None:
    core = [row for row in rows if row.kind in {"native_core", "mlx_core"}]
    if not core:
        return
    print("\nCore generation median")
    print(f"{'context':>7}  {'backend':>7}  {'prefill ms':>10}  {'prefill t/s':>11}  {'decode t/s':>10}")
    print("-" * 60)
    engines = ["mlx"] + backends if any(row.backend == "mlx" for row in core) else backends
    for context in contexts:
        for backend in engines:
            selected = [row for row in core if row.context_tokens == context and row.backend == backend]
            if not selected:
                continue
            print(
                f"{context:7d}  {backend:>7}  "
                f"{median(row.prefill_ms for row in selected):10.2f}  "
                f"{median(row.prefill_tps for row in selected):11.1f}  "
                f"{median(row.decode_tps for row in selected):10.1f}"
            )


def print_runtime_summary(rows: list[BenchRow], backends: list[str]) -> None:
    startup = [row for row in rows if row.kind == "startup"]
    if startup:
        print("\nStartup median")
        print(f"{'backend':>7}  {'startup ms':>10}  {'wall ms':>10}  {'packed':>7}  {'mmap':>5}")
        print("-" * 50)
        for backend in backends:
            selected = [row for row in startup if row.backend == backend]
            if selected:
                print(
                    f"{backend:>7}  {median(row.startup_ms for row in selected):10.1f}  "
                    f"{median(row.elapsed_ms for row in selected):10.1f}  "
                    f"{str(selected[-1].packed_weights):>7}  {str(selected[-1].mmap_weights):>5}"
                )

    oneshot = [row for row in rows if row.kind == "oneshot"]
    if oneshot:
        print("\nOne-shot wrapper median")
        print(f"{'backend':>7}  {'wall ms':>10}  {'native ms':>10}  {'first ms':>10}  {'decode t/s':>10}")
        print("-" * 58)
        for backend in backends:
            selected = [row for row in oneshot if row.backend == backend]
            if selected:
                print(
                    f"{backend:>7}  {median(row.elapsed_ms for row in selected):10.1f}  "
                    f"{median(row.native_ms for row in selected):10.1f}  "
                    f"{median(row.first_token_ms for row in selected):10.1f}  "
                    f"{median(row.decode_tps for row in selected):10.1f}"
                )

    session = [row for row in rows if row.kind == "session_turn"]
    if session:
        print("\nPersistent session median")
        print(f"{'backend':>7}  {'turn':>4}  {'first ms':>10}  {'native ms':>10}  {'decode t/s':>10}  {'cache':>14}")
        print("-" * 72)
        for backend in backends:
            turns = sorted({row.context_tokens for row in session if row.backend == backend})
            for turn in turns:
                selected = [row for row in session if row.backend == backend and row.context_tokens == turn]
                cache = selected[-1]
                cache_text = f"{cache.cache_common}/{cache.cache_delta}"
                if cache.cache_reset:
                    cache_text += " reset"
                print(
                    f"{backend:>7}  {turn:4d}  {median(row.first_token_ms for row in selected):10.1f}  "
                    f"{median(row.native_ms for row in selected):10.1f}  "
                    f"{median(row.decode_tps for row in selected):10.1f}  {cache_text:>14}"
                )


def default_json_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / f"benchmark_{stamp}.json"


def main() -> int:
    args = parse_args()
    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    contexts = [int(value) for value in args.contexts.split(",") if value.strip()]
    valid_backends = {"fp16", "int8", "int8-fp16", "int4", "mixed", "mps", "hybrid"}
    invalid = set(backends) - valid_backends
    if invalid:
        raise SystemExit(f"unknown backend(s): {', '.join(sorted(invalid))}")
    if args.tokens < 2:
        raise SystemExit("--tokens must be at least 2")
    if args.session_turns < 2:
        raise SystemExit("--session-turns must be at least 2")
    if not args.engine.exists():
        raise SystemExit(f"native engine not found: {args.engine}")
    if not args.qwen_local.exists():
        raise SystemExit(f"qwen-local wrapper not found: {args.qwen_local}")

    tokenizer = load_tokenizer(args.model, args.allow_download)
    prompts = {context: make_prompt_tokens(tokenizer, context) for context in contexts}
    mlx_bundle = None
    if args.include_mlx:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step

        model, _ = load(args.model)
        list(generate_step(mx.array(prompts[contexts[0]]), model, max_tokens=4))
        mlx_bundle = (mx, generate_step, model)

    rows: list[BenchRow] = []
    print(f"Benchmarking {args.model}")
    print(
        f"backends={','.join(backends)} contexts={','.join(map(str, contexts))} "
        f"tokens={args.tokens} session_gpu_warmup={args.session_gpu_warmup}"
    )

    if not args.skip_core:
        for context in contexts:
            prompt = prompts[context]
            print(f"\nCore context={context}")
            for warmup in range(args.warmups):
                for backend in backends:
                    benchmark_native_core(
                        args.engine, args.engine_dir, backend, prompt,
                        min(args.tokens, 8), -warmup - 1, args.timeout, args.weight_warmup
                    )
            for repeat in range(args.repeats):
                if args.include_mlx:
                    assert mlx_bundle is not None
                    row = benchmark_mlx_core(*mlx_bundle, prompt, args.tokens, repeat)
                    rows.append(row)
                    print(f"  run {repeat + 1}/{args.repeats} {'mlx':>7}: decode {row.decode_tps:8.1f} t/s")
                for backend in backends:
                    row = benchmark_native_core(
                        args.engine, args.engine_dir, backend, prompt,
                        args.tokens, repeat, args.timeout, args.weight_warmup
                    )
                    rows.append(row)
                    print(f"  run {repeat + 1}/{args.repeats} {backend:>7}: decode {row.decode_tps:8.1f} t/s")

    for backend in backends:
        if not args.skip_startup:
            print(f"\nStartup backend={backend}")
            for repeat in range(args.repeats):
                row = benchmark_startup(
                    args.engine, args.engine_dir, backend, repeat, args.timeout, args.weight_warmup
                )
                rows.append(row)
                print(f"  run {repeat + 1}/{args.repeats}: startup {row.startup_ms:.1f} ms")

        if not args.skip_oneshot:
            print(f"\nOne-shot backend={backend}")
            for repeat in range(args.repeats):
                row = benchmark_oneshot(
                    args.qwen_local, backend, args.tokens, repeat, args.timeout, args.weight_warmup
                )
                rows.append(row)
                print(f"  run {repeat + 1}/{args.repeats}: first {row.first_token_ms:.1f} ms, decode {row.decode_tps:.1f} t/s")

        if not args.skip_session:
            print(f"\nSession backend={backend}")
            for repeat in range(args.repeats):
                session_rows = benchmark_session(
                    args.qwen_local,
                    backend,
                    args.session_turns,
                    args.session_tokens,
                    repeat,
                    args.timeout,
                    args.weight_warmup,
                    args.session_gpu_warmup,
                )
                rows.extend(session_rows)
                cached = [row for row in session_rows[1:] if row.first_token_ms is not None]
                cached_first = statistics.mean(row.first_token_ms for row in cached) if cached else 0.0
                print(f"  run {repeat + 1}/{args.repeats}: cached first avg {cached_first:.1f} ms")

    print_core_summary(rows, contexts, backends)
    print_runtime_summary(rows, backends)

    json_path = args.json
    if json_path is None:
        json_path = default_json_path()
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n")
    print(f"\nRaw results: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

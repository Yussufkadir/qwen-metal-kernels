#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "kernel/qwen_kernel/qwen_kernel"
ENGINE = ENGINE_DIR / "inference_engine"
BUILD_SCRIPT = ENGINE_DIR / "build_engine.sh"

RUNTIME_RE = re.compile(r"\[runtime\].*generated_tokens=(\d+).*elapsed=([0-9.]+)s")
CACHE_RE = re.compile(r"\[cache\] common=(\d+) delta=(\d+)( reset)?")
TIMING_RE = re.compile(
    r"\[timing\] prompt=([0-9.]+)ms native=([0-9.]+)ms "
    r"first_token=([0-9.]+|n/a)ms decode≈([0-9.]+) tok/s decode/print=([0-9.]+)ms"
)
AGREEMENT_RE = re.compile(r"Agreement:\s+(\d+)/(\d+)\s+matching,\s+(.*)")
BENCH_RE = re.compile(r"^\s*(\d+)\s+([\w-]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", re.MULTILINE)


@dataclass
class CheckResult:
    name: str
    passed: bool
    elapsed: float
    detail: str
    output: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="fp16", choices=["fp16", "int8", "int8-fp16", "int4", "mixed", "mps", "hybrid"])
    parser.add_argument("--benchmark-backends", default="fp16,int8")
    parser.add_argument("--benchmark-contexts", default="32,128")
    parser.add_argument("--benchmark-tokens", type=int, default=32)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--session-turns", type=int, default=4)
    parser.add_argument("--session-tokens", type=int, default=32)
    parser.add_argument("--compare-tokens", type=int, default=32)
    parser.add_argument("--chat-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-session-benchmark", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: float = 180.0,
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
    return result.returncode, result.stdout, time.perf_counter() - started


def finish(name: str, passed: bool, elapsed: float, detail: str, output: str = "") -> CheckResult:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail} ({elapsed:.2f}s)")
    return CheckResult(name, passed, elapsed, detail, output)


def check_build(args: argparse.Namespace) -> CheckResult:
    if args.no_build:
        return finish("build", True, 0.0, "skipped")
    if not BUILD_SCRIPT.exists():
        return finish("build", False, 0.0, f"missing {BUILD_SCRIPT}")
    code, output, elapsed = run_command([str(BUILD_SCRIPT)], cwd=ENGINE_DIR, timeout=args.timeout)
    passed = code == 0 and ENGINE.exists()
    detail = "engine built" if passed else "build failed"
    return finish("build", passed, elapsed, detail, output)


def check_native_smoke(args: argparse.Namespace) -> CheckResult:
    if not ENGINE.exists():
        return finish("native smoke", False, 0.0, "engine binary missing")
    command = [str(ENGINE), "--backend", args.backend, "--quiet", "--max-tokens", "1", "1"]
    code, output, elapsed = run_command(command, cwd=ENGINE_DIR, timeout=args.timeout)
    passed = code == 0 and "Generated continuation:" in output
    detail = "native engine generated one token" if passed else "native engine failed"
    return finish("native smoke", passed, elapsed, detail, output)


def check_chat_smoke(args: argparse.Namespace) -> CheckResult:
    command = [
        sys.executable,
        str(ROOT / "qwen_chat.py"),
        "--backend",
        args.backend,
        "--no-stream",
        "--stats",
        "-n",
        str(args.chat_tokens),
        "Say hello in one short sentence.",
    ]
    code, output, elapsed = run_command(command, cwd=ROOT, timeout=args.timeout)
    match = RUNTIME_RE.search(output)
    generated = int(match.group(1)) if match else 0
    passed = code == 0 and generated > 0 and "Assistant:" in output
    detail = f"generated_tokens={generated}" if passed else "chat wrapper failed"
    return finish("chat smoke", passed, elapsed, detail, output)


def check_persistent_cache(args: argparse.Namespace) -> CheckResult:
    command = [
        sys.executable,
        str(ROOT / "qwen_chat.py"),
        "--chat",
        "--persistent",
        "--backend",
        args.backend,
        "--no-stream",
        "--stats",
        "-n",
        str(args.chat_tokens),
    ]
    input_text = "Say hello in one short sentence.\nSay it shorter.\n/exit\n"
    code, output, elapsed = run_command(command, cwd=ROOT, timeout=args.timeout, input_text=input_text)
    cache_rows = [(int(a), int(b), bool(c)) for a, b, c in CACHE_RE.findall(output)]
    reused = any(common > 0 and not reset for common, _, reset in cache_rows[1:])
    passed = code == 0 and reused
    if cache_rows:
        detail = "cache " + ", ".join(f"common={a} delta={b}{' reset' if c else ''}" for a, b, c in cache_rows)
    else:
        detail = "no cache rows found"
    return finish("persistent cache", passed, elapsed, detail, output)


def check_compare(args: argparse.Namespace) -> CheckResult:
    if args.skip_compare:
        return finish("mlx compare", True, 0.0, "skipped")
    command = [
        sys.executable,
        str(ROOT / "compare_chat.py"),
        "--backend",
        args.backend,
        "-n",
        str(args.compare_tokens),
    ]
    if args.allow_download:
        command.append("--allow-download")
    code, output, elapsed = run_command(command, cwd=ROOT, timeout=args.timeout)
    match = AGREEMENT_RE.search(output)
    matched = int(match.group(1)) if match else 0
    total = int(match.group(2)) if match else 0
    passed = code == 0 and matched > 0
    detail = f"agreement={matched}/{total}" if match else "agreement line missing"
    return finish("mlx compare", passed, elapsed, detail, output)


def check_benchmark(args: argparse.Namespace) -> CheckResult:
    if args.skip_benchmark:
        return finish("benchmark", True, 0.0, "skipped")
    command = [
        sys.executable,
        str(ROOT / "benchmark_real.py"),
        "--backends",
        args.benchmark_backends,
        "--contexts",
        args.benchmark_contexts,
        "--tokens",
        str(args.benchmark_tokens),
        "--repeats",
        str(args.benchmark_repeats),
    ]
    code, output, elapsed = run_command(command, cwd=ROOT, timeout=args.timeout)
    rows = BENCH_RE.findall(output)
    passed = code == 0 and bool(rows)
    if rows:
        parts = [f"{context}/{engine}: decode {decode_tps} t/s" for context, engine, _, _, decode_tps in rows]
        detail = "; ".join(parts)
    else:
        detail = "benchmark summary missing"
    return finish("benchmark", passed, elapsed, detail, output)


def check_session_latency(args: argparse.Namespace) -> CheckResult:
    if args.skip_session_benchmark:
        return finish("session latency", True, 0.0, "skipped")
    if args.session_turns < 2:
        return finish("session latency", False, 0.0, "--session-turns must be at least 2")

    prompts = [
        "Say hello in one short sentence.",
        "Say it shorter.",
        "Now answer with only three words.",
        "What did I ask first?",
        "Give one tiny summary.",
        "End with a friendly goodbye.",
    ]
    selected = prompts[: args.session_turns]
    input_text = "\n".join(selected + ["/exit"]) + "\n"
    command = [
        sys.executable,
        str(ROOT / "qwen_chat.py"),
        "--chat",
        "--persistent",
        "--backend",
        args.backend,
        "--no-stream",
        "--stats",
        "-n",
        str(args.session_tokens),
    ]
    code, output, elapsed = run_command(command, cwd=ROOT, timeout=args.timeout, input_text=input_text)

    timings = []
    for prompt_ms, native_ms, first_token_ms, decode_tps, decode_print_ms in TIMING_RE.findall(output):
        timings.append(
            {
                "prompt_ms": float(prompt_ms),
                "native_ms": float(native_ms),
                "first_token_ms": None if first_token_ms == "n/a" else float(first_token_ms),
                "decode_tps": float(decode_tps),
                "decode_print_ms": float(decode_print_ms),
            }
        )
    cache_rows = [(int(a), int(b), bool(c)) for a, b, c in CACHE_RE.findall(output)]
    reused_rows = [(common, delta, reset) for common, delta, reset in cache_rows[1:] if common > 0 and not reset]

    first_tokens = [row["first_token_ms"] for row in timings if row["first_token_ms"] is not None]
    cached_first_tokens = first_tokens[1:]
    passed = (
        code == 0
        and len(timings) >= args.session_turns
        and len(cache_rows) >= args.session_turns
        and len(reused_rows) >= args.session_turns - 1
        and bool(cached_first_tokens)
    )

    if passed:
        avg_cached_first = sum(cached_first_tokens) / len(cached_first_tokens)
        avg_cached_native = sum(row["native_ms"] for row in timings[1:]) / max(len(timings) - 1, 1)
        detail = (
            f"turns={len(timings)}, first={first_tokens[0]:.1f}ms, "
            f"cached_first_avg={avg_cached_first:.1f}ms, cached_native_avg={avg_cached_native:.1f}ms"
        )
    else:
        detail = f"timings={len(timings)}, cache_rows={len(cache_rows)}, reused={len(reused_rows)}"
    return finish("session latency", passed, elapsed, detail, output)


def print_failure_logs(results: list[CheckResult], verbose: bool) -> None:
    for result in results:
        if result.passed and not verbose:
            continue
        if not result.output.strip():
            continue
        print()
        print(f"===== {result.name} output =====")
        print(result.output.rstrip())


def main() -> int:
    args = parse_args()
    checks = [
        check_build,
        check_native_smoke,
        check_chat_smoke,
        check_persistent_cache,
        check_session_latency,
        check_compare,
        check_benchmark,
    ]
    results: list[CheckResult] = []
    print(f"Verifying engine in {ROOT}")
    print(f"backend={args.backend}")
    print()
    for check in checks:
        try:
            result = check(args)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            result = finish(check.__name__, False, args.timeout, "timed out", output)
        except Exception as exc:
            result = finish(check.__name__, False, 0.0, str(exc))
        results.append(result)

    print()
    passed = sum(1 for result in results if result.passed)
    print(f"Summary: {passed}/{len(results)} checks passed")
    print_failure_logs(results, args.verbose)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

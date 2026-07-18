#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "kernel/qwen_kernel/qwen_kernel"
ENGINE = ENGINE_DIR / "inference_engine"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
HISTORY_FILE = ROOT / ".qwen_chat_history"
TOKEN_RE = re.compile(r"^TOKEN\s+(-?\d+)\s*$")
GENERATED_RE = re.compile(r"Generated continuation:\s*([0-9 ]+)")
VALID_BACKENDS = ("fp16", "int8", "int8-fp16", "int4", "mixed", "mps", "hybrid")
BACKEND_NOTES = {
    "fp16": "stable/default quality path",
    "int8": "experimental: slightly faster decode, worse session latency today",
    "int8-fp16": "experimental: INT8 body with FP16 LM head",
    "int4": "experimental: aggressive compression, quality not reliable yet",
    "mixed": "experimental: INT4 body with FP16 LM head",
    "mps": "comparison path through Apple MPS",
    "hybrid": "experimental mixed execution path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="*", help="user prompt text")
    parser.add_argument("--backend", default="fp16", choices=VALID_BACKENDS)
    parser.add_argument("--max-tokens", "-n", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, >0 enables sampling")
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k filtering")
    parser.add_argument("--top-p", type=float, default=1.0, help="1 disables nucleus filtering")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="1 disables repetition penalty")
    parser.add_argument("--seed", type=int, default=0, help="0 uses a time-based seed")
    parser.add_argument("--system", default="You are a helpful assistant.", help="system prompt")
    parser.add_argument("--model", default=MODEL_NAME, help="tokenizer/model name")
    parser.add_argument(
        "--tokenizer",
        choices=["auto", "fast", "transformers"],
        default="auto",
        help="tokenizer implementation; auto uses the fast local Qwen path when available",
    )
    parser.add_argument("--engine", type=Path, default=ENGINE, help="path to native inference_engine binary")
    parser.add_argument("--engine-dir", type=Path, default=ENGINE_DIR, help="working directory containing qwen_weights")
    parser.add_argument("--build", action="store_true", help="run build_engine.sh before generation")
    parser.add_argument("--warmup", action="store_true", help="pre-touch loaded mmap weights during native startup")
    parser.add_argument(
        "--gpu-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warm the persistent GPU prefill path before the first chat turn",
    )
    parser.add_argument("--chat", action="store_true", help="start an interactive terminal chat loop")
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep the native engine process loaded between turns; default: on in chat, off for one-shot prompts",
    )
    parser.add_argument("--no-stream", action="store_true", help="wait for all tokens before printing decoded text")
    parser.add_argument("--stats", action="store_true", help="print runtime timing and cache stats")
    parser.add_argument("--show-token-ids", action="store_true", help="print prompt and generated token IDs")
    parser.add_argument("--save-chat", type=Path, help="autosave chat transcript to this JSON file")
    parser.add_argument("--no-history", action="store_true", help="disable terminal input history file")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow the tokenizer to download if it is not already cached",
    )
    return parser.parse_args()


def normalize_token_ids(value) -> list[int]:
    """Handle tokenizer return shapes across Transformers versions."""
    if isinstance(value, dict) and "input_ids" in value:
        return [int(token) for token in value["input_ids"]]
    if hasattr(value, "input_ids"):
        return [int(token) for token in value.input_ids]
    return [int(token) for token in value]


def find_hf_snapshot(model_name: str) -> Path | None:
    cache_home = os.environ.get("HF_HOME")
    if cache_home:
        hub = Path(cache_home) / "hub"
    else:
        hub = Path.home() / ".cache/huggingface/hub"
    repo_dir = hub / ("models--" + model_name.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [
        path for path in snapshots.iterdir()
        if (path / "vocab.json").exists() and (path / "merges.txt").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


class FastQwenTokenizer:
    eos_token_id = 151645

    def __init__(self, snapshot: Path):
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel

        self.snapshot = snapshot
        model = BPE.from_file(str(snapshot / "vocab.json"), str(snapshot / "merges.txt"), unk_token=None)
        self.tokenizer = Tokenizer(model)
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        config_path = snapshot / "tokenizer_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            decoder = config.get("added_tokens_decoder", {})
            for _, token_config in sorted(decoder.items(), key=lambda item: int(item[0])):
                content = token_config["content"]
                token = AddedToken(
                    content,
                    single_word=bool(token_config.get("single_word", False)),
                    lstrip=bool(token_config.get("lstrip", False)),
                    rstrip=bool(token_config.get("rstrip", False)),
                    normalized=bool(token_config.get("normalized", False)),
                    special=bool(token_config.get("special", False)),
                )
                if token.special:
                    self.tokenizer.add_special_tokens([token])
                else:
                    self.tokenizer.add_tokens([token])
        else:
            self.tokenizer.add_special_tokens([
                AddedToken("<|endoftext|>", special=True),
                AddedToken("<|im_start|>", special=True),
                AddedToken("<|im_end|>", special=True),
            ])

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        if not tokenize:
            return self.render_chat(messages, add_generation_prompt)
        return self.encode(self.render_chat(messages, add_generation_prompt))

    def render_chat(self, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
        parts: list[str] = []
        start = 0
        if messages and messages[0]["role"] == "system":
            parts.append(f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n")
            start = 1
        else:
            parts.append(
                "<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                "<|im_end|>\n"
            )
        for message in messages[start:]:
            role = message["role"]
            content = message["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return int(token_id) if token_id is not None else -1


def load_fast_qwen_tokenizer(model_name: str) -> FastQwenTokenizer:
    if model_name != MODEL_NAME:
        raise RuntimeError("fast tokenizer is only wired for the default Qwen model")
    snapshot = find_hf_snapshot(model_name)
    if snapshot is None:
        raise RuntimeError(f"could not find local Hugging Face snapshot for {model_name}")
    return FastQwenTokenizer(snapshot)


def load_tokenizer(model_name: str, allow_download: bool, implementation: str):
    if implementation in {"auto", "fast"}:
        try:
            return load_fast_qwen_tokenizer(model_name)
        except Exception as exc:
            if implementation == "fast":
                raise SystemExit(f"Could not load fast tokenizer: {exc}") from exc

    try:
        from transformers import AutoTokenizer
    except Exception as exc: 
        raise SystemExit(
            "Could not import transformers. Install it in the venv or run from the project venv."
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=not allow_download,
        )
    except Exception as exc:
        if not allow_download:
            raise SystemExit(
                f"Could not load cached tokenizer for {model_name!r}.\n"
                "If you want to download it, rerun with --allow-download."
            ) from exc
        raise


@dataclass
class Timing:
    prompt_ms: float = 0.0
    native_load_ms: float = 0.0
    first_token_ms: float | None = None
    native_ms: float = 0.0
    decode_print_ms: float = 0.0
    total_ms: float = 0.0
    cache_common: int | None = None
    cache_delta: int | None = None
    cache_reset: bool = False


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: list[int]
    generated_tokens: list[int]
    visible_tokens: list[int]
    elapsed: float
    timing: Timing


def make_chat_tokens(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return normalize_token_ids(templated)


def make_prompt_tokens(tokenizer, system_prompt: str, user_prompt: str) -> list[int]:
    return make_chat_tokens(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )


def special_stop_ids(tokenizer) -> set[int]:
    stop_ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stop_ids.add(int(eos))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for token in ("<|im_end|>", "<|endoftext|>"):
            try:
                token_id = convert(token)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0:
                stop_ids.add(token_id)
    return stop_ids


def decode(tokenizer, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)


def build_engine(engine_dir: Path) -> None:
    script = engine_dir / "build_engine.sh"
    if not script.exists():
        raise SystemExit(f"Build script not found: {script}")
    subprocess.run([str(script)], cwd=engine_dir, check=True)


def setup_readline_history(enabled: bool) -> None:
    if not enabled or not sys.stdin.isatty():
        return
    try:
        import atexit
        import readline
    except Exception:
        return
    try:
        if HISTORY_FILE.exists():
            readline.read_history_file(str(HISTORY_FILE))
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, str(HISTORY_FILE))
    except Exception:
        pass


def run_native_stream(
    engine: Path,
    engine_dir: Path,
    backend: str,
    prompt_tokens: list[int],
    max_tokens: int,
    stop_ids: set[int],
    sampling: dict[str, float | int],
    warmup: bool = False,
):
    command = [
        str(engine),
        "--backend",
        backend,
        "--max-tokens",
        str(max_tokens),
        "--stream-tokens",
        "--quiet",
    ]
    if warmup:
        command.append("--warmup")
    command.extend([
        "--temperature", str(sampling["temperature"]),
        "--top-k", str(sampling["top_k"]),
        "--top-p", str(sampling["top_p"]),
        "--repetition-penalty", str(sampling["repetition_penalty"]),
        "--seed", str(sampling["seed"]),
    ])
    for token in sorted(stop_ids):
        command.extend(["--stop-token", str(token)])
    command.extend(str(token) for token in prompt_tokens)
    return subprocess.Popen(
        command,
        cwd=engine_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )


def run_native_blocking(
    engine: Path,
    engine_dir: Path,
    backend: str,
    prompt_tokens: list[int],
    max_tokens: int,
    stop_ids: set[int],
    sampling: dict[str, float | int],
    warmup: bool = False,
) -> tuple[list[int], str]:
    command = [
        str(engine),
        "--backend",
        backend,
        "--max-tokens",
        str(max_tokens),
        "--quiet",
    ]
    if warmup:
        command.append("--warmup")
    command.extend([
        "--temperature", str(sampling["temperature"]),
        "--top-k", str(sampling["top_k"]),
        "--top-p", str(sampling["top_p"]),
        "--repetition-penalty", str(sampling["repetition_penalty"]),
        "--seed", str(sampling["seed"]),
    ])
    for token in sorted(stop_ids):
        command.extend(["--stop-token", str(token)])
    command.extend(str(token) for token in prompt_tokens)
    result = subprocess.run(
        command,
        cwd=engine_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout or "native engine exited without diagnostic output")
    match = GENERATED_RE.search(result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse engine output:\n{result.stdout}")
    return [int(token) for token in match.group(1).split()], result.stdout


class PersistentNativeEngine:
    def __init__(
        self,
        engine: Path,
        engine_dir: Path,
        backend: str,
        warmup: bool,
        gpu_warmup: bool,
    ):
        self.engine = engine
        self.engine_dir = engine_dir
        self.backend = backend
        self.warmup = warmup
        command = [str(engine), "--backend", backend, "--server", "--quiet"]
        if warmup:
            command.append("--warmup")
        if gpu_warmup:
            command.append("--gpu-warmup")
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            cwd=engine_dir,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert self.process.stdout is not None
        assert self.process.stdin is not None
        ready = self.process.stdout.readline().strip()
        if ready != f"READY {backend}":
            self.close()
            raise RuntimeError(f"native server failed to start, got: {ready!r}")
        self.load_elapsed = time.perf_counter() - started

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write("QUIT\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()

    def reset_session(self) -> None:
        if self.process.poll() is not None:
            return
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write("RESET\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip()
        if response != "RESET_OK":
            raise RuntimeError(f"native server reset failed: {response}")

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        stop_ids: set[int],
        sampling: dict[str, float | int],
        on_token=None,
    ) -> tuple[list[int], float | None, float, tuple[int, int, bool] | None]:
        if self.process.poll() is not None:
            raise RuntimeError("native server is not running")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        stop_tokens = sorted(stop_ids)
        line = (
            "GENERATE_CACHE_SAMPLE "
            + str(max_tokens)
            + " "
            + str(sampling["temperature"])
            + " "
            + str(sampling["top_k"])
            + " "
            + str(sampling["top_p"])
            + " "
            + str(sampling["repetition_penalty"])
            + " "
            + str(sampling["seed"])
            + " "
            + str(len(stop_tokens))
            + " "
            + " ".join(str(token) for token in stop_tokens)
            + " "
            + " ".join(str(token) for token in prompt_tokens)
            + "\n"
        )
        self.process.stdin.write(line)
        self.process.stdin.flush()

        generated: list[int] = []
        saw_begin = False
        cache_info: tuple[int, int, bool] | None = None
        started = time.perf_counter()
        first_token_at: float | None = None
        while True:
            output = self.process.stdout.readline()
            if output == "":
                raise RuntimeError("native server exited while generating")
            output = output.strip()
            if output == "BEGIN":
                saw_begin = True
                continue
            if output.startswith("CACHE "):
                _, common, delta, reset = output.split()
                cache_info = (int(common), int(delta), bool(int(reset)))
                continue
            if output.startswith("TOKEN "):
                token_id = int(output.split()[1])
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                generated.append(token_id)
                if on_token is not None:
                    on_token(token_id)
                continue
            if output.startswith("END "):
                ended = time.perf_counter()
                return (
                    generated,
                    (first_token_at - started if first_token_at is not None else None),
                    ended - started,
                    cache_info,
                )
            if output.startswith("ERROR "):
                raise RuntimeError(output)
            if not saw_begin:
                raise RuntimeError(f"unexpected native server output: {output}")


def visible_prefix(tokens: list[int], stop_ids: set[int]) -> list[int]:
    result: list[int] = []
    for token in tokens:
        if token in stop_ids:
            break
        result.append(token)
    return result


def generate_turn(
    *,
    tokenizer,
    messages: list[dict[str, str]],
    stop_ids: set[int],
    engine: Path,
    engine_dir: Path,
    backend: str,
    max_tokens: int,
    no_stream: bool,
    show_token_ids: bool,
    print_runtime: bool,
    sampling: dict[str, float | int],
    warmup: bool = False,
    native_server: PersistentNativeEngine | None = None,
) -> GenerationResult:
    prompt_started = time.perf_counter()
    prompt_tokens = make_chat_tokens(tokenizer, messages)
    prompt_elapsed = time.perf_counter() - prompt_started

    if show_token_ids:
        print("Prompt token IDs:")
        print(" ".join(str(token) for token in prompt_tokens))
        print()

    started = time.perf_counter()
    first_token_at: float | None = None
    native_elapsed = 0.0
    decode_print_elapsed = 0.0
    cache_info: tuple[int, int, bool] | None = None
    generated: list[int] = []
    visible: list[int] = []
    assistant_text = ""

    print("Assistant: ", end="", flush=True)
    if no_stream:
        if native_server is not None:
            generated, first_token_elapsed, native_elapsed, cache_info = native_server.generate(
                prompt_tokens, max_tokens, stop_ids, sampling
            )
            first_token_at = started + first_token_elapsed if first_token_elapsed is not None else None
        else:
            native_started = time.perf_counter()
            generated, raw_output = run_native_blocking(
                engine, engine_dir, backend, prompt_tokens, max_tokens, stop_ids, sampling, warmup
            )
            native_elapsed = time.perf_counter() - native_started
        visible = visible_prefix(generated, stop_ids)
        decode_started = time.perf_counter()
        assistant_text = decode(tokenizer, visible)
        print(assistant_text, flush=True)
        decode_print_elapsed += time.perf_counter() - decode_started
    elif native_server is not None:
        printed = ""
        stopped_printing = False

        def on_token(token_id: int) -> None:
            nonlocal printed, stopped_printing, assistant_text
            generated.append(token_id)
            if token_id in stop_ids:
                stopped_printing = True
            if stopped_printing:
                return
            visible.append(token_id)
            decode_started = time.perf_counter()
            current = decode(tokenizer, visible)
            if current.startswith(printed):
                delta = current[len(printed) :]
            else:
                delta = decode(tokenizer, [token_id])
            if delta:
                print(delta, end="", flush=True)
                printed = current
                assistant_text = current
            nonlocal decode_print_elapsed
            decode_print_elapsed += time.perf_counter() - decode_started

        generated_result, first_token_elapsed, native_elapsed, cache_info = native_server.generate(
            prompt_tokens, max_tokens, stop_ids, sampling, on_token=on_token
        )
        if len(generated) != len(generated_result):
            generated = generated_result
        first_token_at = started + first_token_elapsed if first_token_elapsed is not None else None
        print(flush=True)
        decode_started = time.perf_counter()
        assistant_text = decode(tokenizer, visible)
        decode_print_elapsed += time.perf_counter() - decode_started
    else:
        native_started = time.perf_counter()
        process = run_native_stream(engine, engine_dir, backend, prompt_tokens, max_tokens, stop_ids, sampling, warmup)
        assert process.stdout is not None
        printed = ""
        stopped_printing = False
        engine_output: list[str] = []
        for line in process.stdout:
            engine_output.append(line)
            match = TOKEN_RE.match(line)
            if not match:
                continue
            token_id = int(match.group(1))
            if first_token_at is None:
                first_token_at = time.perf_counter()
            generated.append(token_id)
            if token_id in stop_ids:
                stopped_printing = True
            if stopped_printing:
                continue

            visible.append(token_id)
            decode_started = time.perf_counter()
            current = decode(tokenizer, visible)
            if current.startswith(printed):
                delta = current[len(printed) :]
            else:
                delta = decode(tokenizer, [token_id])
            if delta:
                print(delta, end="", flush=True)
                printed = current
                assistant_text = current
            decode_print_elapsed += time.perf_counter() - decode_started
        return_code = process.wait()
        native_elapsed = time.perf_counter() - native_started
        if return_code:
            print(file=sys.stderr)
            output = "".join(engine_output).strip()
            if not output:
                output = "native engine exited without diagnostic output"
            raise RuntimeError("Native engine failed:\n" + output)
        print(flush=True)
        decode_started = time.perf_counter()
        assistant_text = decode(tokenizer, visible)
        decode_print_elapsed += time.perf_counter() - decode_started

    elapsed = time.perf_counter() - started
    timing = Timing(
        prompt_ms=prompt_elapsed * 1000.0,
        native_load_ms=native_server.load_elapsed * 1000.0 if native_server is not None else 0.0,
        first_token_ms=(first_token_at - started) * 1000.0 if first_token_at is not None else None,
        native_ms=native_elapsed * 1000.0,
        decode_print_ms=decode_print_elapsed * 1000.0,
        total_ms=elapsed * 1000.0,
        cache_common=cache_info[0] if cache_info is not None else None,
        cache_delta=cache_info[1] if cache_info is not None else None,
        cache_reset=cache_info[2] if cache_info is not None else False,
    )
    if show_token_ids:
        print()
        print("Generated token IDs:")
        print(" ".join(str(token) for token in generated))
    if print_runtime:
        print(
            f"\n[runtime] backend={backend} prompt_tokens={len(prompt_tokens)} "
            f"generated_tokens={len(generated)} visible_tokens≈{len(visible)} elapsed={elapsed:.3f}s"
        )
        first = "n/a" if timing.first_token_ms is None else f"{timing.first_token_ms:.1f}"
        decode_tokens = max(len(generated) - 1, 0) if timing.first_token_ms is not None else len(generated)
        decode_ms = max(timing.native_ms - (timing.first_token_ms or 0.0), 0.0)
        decode_tps = (decode_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        print(
            f"[timing] prompt={timing.prompt_ms:.1f}ms native={timing.native_ms:.1f}ms "
            f"first_token={first}ms decode≈{decode_tps:.1f} tok/s "
            f"decode/print={timing.decode_print_ms:.1f}ms"
        )
        if timing.cache_common is not None:
            reset = " reset" if timing.cache_reset else ""
            print(f"[cache] common={timing.cache_common} delta={timing.cache_delta}{reset}")
    return GenerationResult(
        text=assistant_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated,
        visible_tokens=visible,
        elapsed=elapsed,
        timing=timing,
    )


def print_chat_help() -> None:
    print(
        "Commands:\n"
        "  /help              show this help\n"
        "  /exit, /quit       leave chat\n"
        "  /status            show current backend, sampling, history, and cache mode\n"
        "  /backends          list available backends and current recommendations\n"
        "  /history [N]       show the last N conversation messages, default 8\n"
        "  /save [PATH]       save transcript JSON, default from --save-chat or chat_TIMESTAMP.json\n"
        "  /clear             clear the terminal screen\n"
        "  /multi             enter a multiline prompt; finish with /end\n"
        "  /reset             clear conversation history\n"
        "  /tokens N          set max generated tokens per reply\n"
        "  /backend NAME      switch backend: fp16, int8, int8-fp16, int4, mixed, mps, hybrid\n"
        "  /temperature T     0 = greedy, try 0.7 for sampling\n"
        "  /top-p P           nucleus sampling cutoff, e.g. 0.9\n"
        "  /top-k K           0 disables top-k, try 40\n"
        "  /repeat R          repetition penalty, e.g. 1.1\n"
        "  /seed S            0 = time-based seed\n"
        "  /stats on|off      show or hide runtime timing/cache stats\n"
        "  /system TEXT       replace the system prompt and reset history\n"
    )


def print_backends(current: str) -> None:
    for name in VALID_BACKENDS:
        marker = "*" if name == current else " "
        print(f"{marker} {name:10s} {BACKEND_NOTES[name]}")


def print_status(
    *,
    backend: str,
    max_tokens: int,
    sampling: dict[str, float | int],
    persistent: bool,
    warmup: bool,
    gpu_warmup: bool,
    stats_enabled: bool,
    messages: list[dict[str, str]],
    save_chat: Path | None,
) -> None:
    turns = sum(1 for message in messages if message["role"] == "user")
    assistant_turns = sum(1 for message in messages if message["role"] == "assistant")
    print(f"backend:     {backend} ({BACKEND_NOTES[backend]})")
    print(f"max_tokens:  {max_tokens}")
    print(f"persistent:  {persistent}")
    print(f"warmup:      {warmup}")
    print(f"gpu_warmup:  {gpu_warmup}")
    print(f"stats:       {stats_enabled}")
    print(f"temperature: {sampling['temperature']}")
    print(f"top_p:       {sampling['top_p']}")
    print(f"top_k:       {sampling['top_k']}")
    print(f"repeat:      {sampling['repetition_penalty']}")
    print(f"seed:        {sampling['seed']}")
    print(f"history:     {turns} user turns, {assistant_turns} assistant turns")
    print(f"autosave:    {save_chat if save_chat else 'off'}")


def print_history(messages: list[dict[str, str]], count: int) -> None:
    visible = [message for message in messages if message["role"] != "system"]
    if not visible:
        print("history is empty")
        return
    for message in visible[-count:]:
        content = message["content"].replace("\n", "\n    ")
        print(f"{message['role']}: {content}")


def save_transcript(
    path: Path,
    *,
    model: str,
    backend: str,
    max_tokens: int,
    sampling: dict[str, float | int],
    messages: list[dict[str, str]],
) -> Path:
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "backend": backend,
        "max_tokens": max_tokens,
        "sampling": sampling,
        "messages": messages,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def default_transcript_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / f"chat_{stamp}.json"


def read_multiline_prompt() -> str:
    print("Enter multiline prompt. Finish with /end on its own line.")
    lines: list[str] = []
    while True:
        try:
            line = input("... ")
        except (EOFError, KeyboardInterrupt):
            print()
            return ""
        if line.strip() == "/end":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_chat(args: argparse.Namespace, tokenizer, stop_ids: set[int]) -> None:
    backend = args.backend
    max_tokens = args.max_tokens
    sampling: dict[str, float | int] = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
    }
    system_prompt = args.system
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    native_server: PersistentNativeEngine | None = None
    stats_enabled = args.stats
    save_chat: Path | None = args.save_chat
    setup_readline_history(not args.no_history)

    print("Custom Qwen terminal chat")
    print(
        f"backend={backend}, max_tokens={max_tokens}, persistent={args.persistent}, "
        f"temperature={sampling['temperature']}, stats={stats_enabled}, "
        f"warmup={args.warmup}, gpu_warmup={args.gpu_warmup}"
    )
    print("Type /help for commands, /status for settings, /exit to quit.\n")

    def ensure_server() -> PersistentNativeEngine | None:
        nonlocal native_server
        if not args.persistent:
            return None
        if native_server is None or native_server.backend != backend:
            if native_server is not None:
                native_server.close()
            warmups = []
            if args.warmup:
                warmups.append("weight")
            if args.gpu_warmup:
                warmups.append("GPU")
            suffix = f" with {' + '.join(warmups)} warmup" if warmups else ""
            print(f"[runtime] loading persistent native engine ({backend}){suffix}...")
            native_server = PersistentNativeEngine(
                args.engine,
                args.engine_dir,
                backend,
                args.warmup,
                args.gpu_warmup,
            )
            print(f"[runtime] native engine ready in {native_server.load_elapsed * 1000.0:.1f}ms")
        return native_server

    try:
        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return
            if not user_text:
                continue

            if user_text == "/multi":
                user_text = read_multiline_prompt()
                if not user_text:
                    continue

            if user_text.startswith("/"):
                parts = user_text.split(maxsplit=1)
                command = parts[0].lower()
                value = parts[1].strip() if len(parts) > 1 else ""
                if command in {"/exit", "/quit"}:
                    print("bye")
                    return
                if command == "/help":
                    print_chat_help()
                    continue
                if command == "/status":
                    print_status(
                        backend=backend,
                        max_tokens=max_tokens,
                        sampling=sampling,
                        persistent=bool(args.persistent),
                        warmup=bool(args.warmup),
                        gpu_warmup=bool(args.gpu_warmup),
                        stats_enabled=stats_enabled,
                        messages=messages,
                        save_chat=save_chat,
                    )
                    continue
                if command == "/backends":
                    print_backends(backend)
                    continue
                if command == "/history":
                    if value:
                        try:
                            count = int(value)
                        except ValueError:
                            print("usage: /history 8")
                            continue
                    else:
                        count = 8
                    if count <= 0:
                        print("usage: /history 8")
                        continue
                    print_history(messages, count)
                    continue
                if command == "/save":
                    target = Path(value) if value else save_chat or default_transcript_path()
                    saved = save_transcript(
                        target,
                        model=args.model,
                        backend=backend,
                        max_tokens=max_tokens,
                        sampling=sampling,
                        messages=messages,
                    )
                    save_chat = saved
                    print(f"saved transcript: {saved}")
                    continue
                if command == "/clear":
                    if sys.stdout.isatty():
                        print("\033c", end="")
                    continue
                if command == "/reset":
                    messages = [{"role": "system", "content": system_prompt}]
                    if native_server is not None:
                        native_server.reset_session()
                    print("history reset")
                    continue
                if command == "/tokens":
                    if not value.isdigit() or int(value) <= 0:
                        print("usage: /tokens 128")
                        continue
                    max_tokens = int(value)
                    print(f"max_tokens={max_tokens}")
                    continue
                if command == "/temperature":
                    try:
                        sampling["temperature"] = float(value)
                    except ValueError:
                        print("usage: /temperature 0.7")
                        continue
                    print(f"temperature={sampling['temperature']}")
                    continue
                if command == "/top-p":
                    try:
                        sampling["top_p"] = float(value)
                    except ValueError:
                        print("usage: /top-p 0.9")
                        continue
                    print(f"top_p={sampling['top_p']}")
                    continue
                if command == "/top-k":
                    if not value.isdigit() or int(value) < 0:
                        print("usage: /top-k 40")
                        continue
                    sampling["top_k"] = int(value)
                    print(f"top_k={sampling['top_k']}")
                    continue
                if command == "/repeat":
                    try:
                        sampling["repetition_penalty"] = float(value)
                    except ValueError:
                        print("usage: /repeat 1.1")
                        continue
                    print(f"repetition_penalty={sampling['repetition_penalty']}")
                    continue
                if command == "/seed":
                    if not value.isdigit() or int(value) < 0:
                        print("usage: /seed 1234")
                        continue
                    sampling["seed"] = int(value)
                    print(f"seed={sampling['seed']}")
                    continue
                if command == "/stats":
                    normalized = value.lower()
                    if normalized not in {"on", "off"}:
                        print("usage: /stats on")
                        continue
                    stats_enabled = normalized == "on"
                    print(f"stats={stats_enabled}")
                    continue
                if command == "/backend":
                    if value not in VALID_BACKENDS:
                        print("backend must be one of: " + ", ".join(VALID_BACKENDS))
                        continue
                    backend = value
                    if native_server is not None:
                        native_server.close()
                        native_server = None
                    print(f"backend={backend}")
                    continue
                if command == "/system":
                    if not value:
                        print("usage: /system You are a concise assistant.")
                        continue
                    system_prompt = value
                    messages = [{"role": "system", "content": system_prompt}]
                    if native_server is not None:
                        native_server.reset_session()
                    print("system prompt updated and history reset")
                    continue
                print(f"unknown command: {command}")
                print_chat_help()
                continue

            messages.append({"role": "user", "content": user_text})
            try:
                result = generate_turn(
                    tokenizer=tokenizer,
                    messages=messages,
                    stop_ids=stop_ids,
                    engine=args.engine,
                    engine_dir=args.engine_dir,
                    backend=backend,
                    max_tokens=max_tokens,
                    no_stream=args.no_stream,
                    show_token_ids=args.show_token_ids,
                    print_runtime=stats_enabled,
                    sampling=sampling,
                    warmup=args.warmup,
                    native_server=ensure_server(),
                )
            except RuntimeError as exc:
                messages.pop()
                print(f"\nerror: {exc}", file=sys.stderr)
                continue
            messages.append({"role": "assistant", "content": result.text})
            if save_chat is not None:
                try:
                    save_transcript(
                        save_chat,
                        model=args.model,
                        backend=backend,
                        max_tokens=max_tokens,
                        sampling=sampling,
                        messages=messages,
                    )
                except OSError as exc:
                    print(f"warning: could not save transcript: {exc}", file=sys.stderr)
            print()
    finally:
        if native_server is not None:
            native_server.close()


def main() -> None:
    args = parse_args()
    if args.persistent is None:
        args.persistent = bool(args.chat)

    if args.build:
        build_engine(args.engine_dir)
    if not args.engine.exists():
        raise SystemExit(f"Native engine not found: {args.engine}\nRun: {args.engine_dir / 'build_engine.sh'}")

    tokenizer = load_tokenizer(args.model, args.allow_download, args.tokenizer)
    stop_ids = special_stop_ids(tokenizer)

    if args.chat:
        run_chat(args, tokenizer, stop_ids)
        return

    user_prompt = " ".join(args.prompt).strip()
    if not user_prompt:
        if sys.stdin.isatty():
            args.chat = True
            args.persistent = True
            run_chat(args, tokenizer, stop_ids)
            return
        raise SystemExit(
            "Provide a prompt, for example: ./qwen_chat.py 'Explain KV cache simply'\n"
            "Or start interactive mode with: ./qwen_chat.py --chat"
        )

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": user_prompt},
    ]
    result = generate_turn(
        tokenizer=tokenizer,
        messages=messages,
        stop_ids=stop_ids,
        engine=args.engine,
        engine_dir=args.engine_dir,
        backend=args.backend,
        max_tokens=args.max_tokens,
        no_stream=args.no_stream,
        show_token_ids=args.show_token_ids,
        print_runtime=args.stats,
        sampling={
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        warmup=args.warmup,
    )
    if args.save_chat is not None:
        messages.append({"role": "assistant", "content": result.text})
        saved = save_transcript(
            args.save_chat,
            model=args.model,
            backend=args.backend,
            max_tokens=args.max_tokens,
            sampling={
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "seed": args.seed,
            },
            messages=messages,
        )
        print(f"\n[chat] saved transcript: {saved}")


if __name__ == "__main__":
    main()

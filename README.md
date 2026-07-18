# Qwen Metal Local Inference Runtime

Custom local Qwen inference runtime for Apple Silicon, written around Objective-C++ and Metal kernels. The project started as a kernel benchmark and grew into a usable terminal LLM runtime with KV cache reuse, persistent chat, startup optimization, benchmark tooling, and experimental quantized backends.

The stable path is currently `fp16`. Quantized backends are kept for experimentation and measurement, but `fp16` is the recommended backend for normal local chat quality.

## Current status

- Native Qwen2.5-0.5B-Instruct inference engine on Apple Silicon
- Terminal one-shot generation and interactive chat wrapper
- Persistent server mode for low-latency multi-turn chat
- KV cache reuse across chat turns
- Metal kernels for projection, attention, RMS norm, RoPE, MLP, sampling support paths, and fused hot-path operations
- Packed + mmap-backed weight loading for much faster startup
- Verification and benchmark scripts for comparing correctness, latency, cache behavior, and MLX reference behavior
- FP16 stable backend plus experimental `int8`, `int8-fp16`, `int4`, `mixed`, `mps`, and `hybrid` backends

## Quick start

Build the native engine:

```bash
cd kernel/qwen_kernel/qwen_kernel
./build_engine.sh
cd ../../..
```

Run a one-shot prompt:

```bash
./qwen-local "Explain transformer KV cache in simple terms."
```

By default, `qwen-local` uses a lightweight local Qwen tokenizer path for the default model. This avoids importing the full Transformers stack on every one-shot command. If you need the original Hugging Face tokenizer path for debugging, use:

```bash
./qwen-local --tokenizer transformers "Explain transformer KV cache in simple terms."
```

Start interactive chat:

```bash
./qwen-local --chat
```

Or just run the local command with no prompt from an interactive terminal:

```bash
./qwen-local
```

Useful chat commands:

```text
/help
/status
/backends
/history
/multi
/save chat.json
/reset
/tokens 128
/temperature 0.7
/top-p 0.9
/stats on
/stats off
/backend fp16
/exit
```

## Stable local LLM path

The intended daily-use path is:

```bash
./qwen-local --chat --backend fp16
```

In chat mode the wrapper keeps the native engine loaded by default. That avoids paying engine startup on every turn and lets the runtime reuse the KV cache when the conversation grows normally. The persistent runtime also performs a 32-token GPU prefill warmup before reporting that it is ready. This moves MPS pipeline initialization out of the first user turn.

To measure the un-warmed first request:

```bash
./qwen-local --chat --no-gpu-warmup --stats
```

For timing output:

```bash
./qwen-local --chat --stats
```

For a single prompt with timing:

```bash
./qwen-local --stats -n 128 "Explain KV cache in three bullet points."
```

Optional warm-start mode:

```bash
./qwen-local --warmup --stats "Explain KV cache simply."
```

`--warmup` pre-touches the mmap-backed weights during native startup. This intentionally moves some page-fault cost into startup. It can be useful when testing cold-start behavior, but it is not the default because on an already warm OS file cache it can make one-shot latency worse.

Save a one-shot transcript:

```bash
./qwen-local --save-chat chat.json "Explain KV cache simply."
```

Autosave an interactive chat transcript:

```bash
./qwen-local --chat --save-chat chat.json
```

## Weight loading

The fastest startup path uses a packed weight archive:

```bash
python3 pack_weights.py --profile fp16
```

This creates ignored local runtime files under:

```text
kernel/qwen_kernel/qwen_kernel/qwen_weights/weights.pack
kernel/qwen_kernel/qwen_kernel/qwen_weights/weights.index
```

The engine memory-maps the pack file and creates Metal buffers from the mapped weight ranges when possible. This reduces cold engine startup substantially compared with opening hundreds of loose weight files.

## Verification

Run the main verification suite:

```bash
venv/bin/python verify_engine.py \
  --backend fp16 \
  --compare-tokens 24 \
  --benchmark-tokens 24 \
  --benchmark-contexts 32,128 \
  --benchmark-repeats 1 \
  --session-turns 4 \
  --session-tokens 32 \
  --timeout 240
```

The verifier checks:

- native engine smoke generation
- Python chat wrapper
- persistent KV cache reuse
- session latency
- MLX token comparison
- benchmark output

## Benchmarking

Run the reliable benchmark suite:

```bash
venv/bin/python benchmark_suite.py \
  --backends fp16,int8 \
  --contexts 32,128,512 \
  --tokens 64 \
  --repeats 3 \
  --warmups 1 \
  --session-turns 4 \
  --session-tokens 32
```

Compare the same suite with weight warmup enabled:

```bash
venv/bin/python benchmark_suite.py \
  --backends fp16 \
  --contexts 32,128 \
  --tokens 32 \
  --repeats 1 \
  --weight-warmup
```

This reports separate medians for:

- core native prefill/decode speed
- native engine startup
- one-shot wrapper latency
- persistent chat/session latency
- KV cache reuse across turns

The suite also writes raw rows to an ignored `benchmark_*.json` file for later comparison or plotting.

Persistent session measurements use GPU prefill warmup by default. Use `--no-session-gpu-warmup` to expose the raw first-use cost. Core and one-shot measurements are never GPU-warmed unless explicitly launched through the native `--gpu-warmup` option.

To compare the optimized decode-attention path with the original three-pass implementation:

```bash
QWEN_FUSED_ATTENTION=0 venv/bin/python benchmark_suite.py \
  --backends fp16 \
  --contexts 512,1024,2048 \
  --tokens 32 \
  --repeats 3 \
  --skip-startup \
  --skip-oneshot \
  --skip-session
```

Long-context decode switches from independent query-head blocks to paired GQA blocks once attention reaches 14 blocks. To disable that adaptive path for an A/B run:

```bash
QWEN_GROUPED_GQA=0 venv/bin/python benchmark_suite.py \
  --backends fp16 \
  --contexts 3200,3456,3968 \
  --tokens 32 \
  --repeats 3 \
  --skip-startup \
  --skip-oneshot \
  --skip-session
```

Prefill attention switches to an 8-query by 32-key SIMD-matrix tiled kernel for prompt deltas of at least 512 tokens. Both QK scores and probability-by-value accumulation use 8-by-8 SIMD-group matrix operations, with online softmax between them. To compare it with the score-matrix three-pass path:

```bash
QWEN_TILED_PREFILL=0 venv/bin/python benchmark_suite.py \
  --backends fp16 \
  --contexts 512,1024,2048,3072 \
  --tokens 32 \
  --repeats 3 \
  --skip-startup \
  --skip-oneshot \
  --skip-session
```

The equivalent native engine switch is `--legacy-prefill-attention`.

To compare combined prefill matrices with the original separate MPS launches:

```bash
QWEN_COMBINED_PREFILL=0 venv/bin/python benchmark_suite.py \
  --backends fp16 \
  --contexts 128,512,1024 \
  --tokens 32 \
  --repeats 3 \
  --skip-startup \
  --skip-oneshot \
  --skip-session
```

To include an MLX core decode reference:

```bash
venv/bin/python benchmark_suite.py \
  --include-mlx \
  --backends fp16 \
  --contexts 32,128,512 \
  --tokens 64 \
  --repeats 3
```

Run decode benchmarks:

```bash
venv/bin/python benchmark_real.py \
  --backends fp16,int8 \
  --contexts 32,128 \
  --tokens 64 \
  --repeats 3
```

Compare native output against MLX:

```bash
venv/bin/python compare_chat.py --backend fp16 -n 80
```

Evaluate quantized backends against native FP16 behavior:

```bash
venv/bin/python quantization_eval.py \
  --backends int8,int8-fp16,int4 \
  --tokens 64
```

This measures generated-token prefix agreement against the FP16 backend. It is a quantization-health check: high agreement means the quantized backend is preserving FP16 behavior; low agreement means quantization is changing token choices early.

## Backend notes

| Backend | Intended use | Current recommendation |
| --- | --- | --- |
| `fp16` | Stable local chat and correctness testing | Use by default |
| `int8` | Speed/memory experiments | Promising, keep testing quality |
| `int8-fp16` | INT8 transformer body with FP16 LM head | Experimental stable-lite candidate |
| `int4` | Aggressive compression experiments | Fast but quality is not reliable enough yet |
| `mixed` | INT4 body with FP16 LM head experiment | Experimental |
| `mps` | Apple MPS comparison path | Useful for comparison, not the main optimized path |
| `hybrid` | Experimental mixed execution path | Experimental |

## What was optimized

The main engineering wins so far:

- KV cache reuse for decode and persistent chat
- batched prompt delta prefill for cached sessions
- zero-copy combined QKV and gate/up prefill matrices
- fused QKV split/bias and gate/up activation kernels
- SIMD-group matrix QK and probability-by-value tiled prefill for prompt deltas of at least 512 tokens
- fused FP16 QKV projection
- fused RoPE plus KV cache append for single-token decode
- context-adaptive fused and blockwise decode attention
- paired grouped-query attention near the 4096-token cache limit
- GPU greedy argmax path
- backend-specific pipeline loading
- direct Metal buffer weight loading
- packed + mmap-backed weights
- fast local tokenizer path for terminal one-shot usage
- persistent GPU prefill warmup for low first-turn latency
- startup timing instrumentation
- session latency verification

## Honest limitations

This is a custom learning/runtime project, not a replacement for mature engines like MLX or llama.cpp across every workload.

The native FP16 path is in the same rough short-context decode-speed class as MLX on the tested small model, but mature frameworks still have advantages in generality, long-context behavior, quality-preserving quantization, model coverage, and production hardening.

The project is strongest as a systems portfolio piece because it exposes the inference stack directly: kernels, memory layout, dispatch overhead, KV cache behavior, startup cost, backend tradeoffs, and measurement discipline.

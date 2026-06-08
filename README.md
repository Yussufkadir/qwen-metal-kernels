# Qwen Metal Kernels

Custom Metal GPU kernels for Qwen-0.5B-Instruct inference on Apple Silicon built from scratch to understand and push the limits of local LLM performance without cloud dependency or framework abstraction.

## Why

Running LLMs locally on Apple Silicon is slow compared to CUDA equivalents. Existing frameworks (MLX, llama.cpp) abstract away the kernel layer entirely, which means you can't go below them to recover performance. This project is an attempt to do exactly that write the compute kernels directly in Metal Shading Language and build upward to understand where performance is lost and what can be recovered.

Secondary motivation: privacy and cost. No API keys, no data leaving the machine, no per-token billing.

## Results

| Phase | Approach | Throughput | Status |
|-------|----------|------------|--------|
| Baseline | MLX default | ~120 tok/s | ✅ measured |
| Phase 1 | Custom batched Metal kernels | ~324 tok/s (**2.7×**) | ✅ working |
| Phase 2 | Kernels called via Python/ctypes bridge | ~120 tok/s | ✅ built, overhead documented |
| Phase 3 | Native Obj-C++ inference engine | — | 🔧 in progress |

## Architecture

### Phase 1 — Kernel optimization (`kernels.metal`, `bridge.mm`)

Wrote batched matrix-vector multiply kernels in Metal Shading Language targeting Qwen 0.5B's exact weight dimensions on M1 Pro:

- **Fused gate+up projection**: single Metal compute pass handles both projections simultaneously, eliminating a redundant GPU memory round-trip
- **SIMD reduction**: uses `simd_sum` within each warp, then threadgroup reduction — avoids global atomics
- **Half-precision weights, float accumulation**: weights stored as `half` (fp16), dot products accumulated in `float` to avoid precision loss
- **Buffer caching**: `bridge.mm` reuses MTLBuffers across calls (size-matched) to avoid Metal heap allocation overhead on the hot path
- **Benchmarking**: 100-rep timing with warmup in both `main.mm` (native) and `benchmark_qwen.py` (Python)

This gave the **2.7× throughput improvement** over MLX baseline (120 → 324 tok/s) on MLP layers alone.

### Phase 2 — Python integration (`qwen_patched.py`, `metal_bridge.py`)

Built a Python-to-Objective-C++ interop layer via `ctypes` to call the kernels from inside MLX's forward pass (`qwen_patched.py` monkey-patches each layer's MLP). The overhead of:

- copying tensors from MLX's memory space to numpy
- crossing the Python/native boundary per forward pass
- `mx.eval()` synchronisation barriers

...negated the kernel-level gains entirely. This is a structural issue with how MLX exposes its Python API — unlike PyTorch where the C++ backend is tightly coupled to the Python runtime, MLX's design makes low-level kernel injection costly without modifying the framework itself.

### Phase 3 — Native inference engine (`inference.mm`)

To bypass Python overhead entirely, began writing the full transformer loop natively in Objective-C++:

- Weight loading from binary files (`export_weights.py` dumps MLX weights to raw fp16 `.bin` files)
- RMS norm, residual add via Metal kernels
- Multi-head attention with GQA (14 query heads, 2 KV heads) computed in CPU float for now
- RoPE position embeddings applied in float
- KV cache (growing vectors per layer)
- SiLU activation in the MLP
- Greedy decoding over 151936-token vocabulary

Currently blocked on a memory management bug in the MLP section of the layer loop — likely a mismatched buffer access when batching all 24 layers' weights against a single token's hidden state.

## What this revealed

- **Why there is no Unsloth for Mac**: customising kernels below the MLX abstraction requires eventually writing your own runtime. The Python integration overhead problem means you can't get kernel-level gains without bypassing Python entirely — which means reimplementing the inference loop in C++/Obj-C++. That is a significant engineering lift.
- **Metal buffer lifetime is strict**: `newBufferWithBytesNoCopy` requires page-aligned memory and the backing pointer to outlive the GPU command. Getting this wrong silently corrupts results or bus-errors rather than producing a clean error.
- **ARC and manual memory interact badly**: mixing `new`/`delete` for float arrays with ARC-managed MTLBuffers in the same scope creates subtle lifetime issues that are hard to debug.
- **MLX's integration model**: designed for ease of use in Python, not for kernel injection. The abstraction is the feature for most users; it's the obstacle here.

## File structure

```
├── kernels.metal          # Metal compute kernels (matmul, RMS norm, RoPE, attention, softmax)
├── bridge.mm              # Obj-C++ bridge: buffer caching, kernel dispatch, C extern interface
├── inference.mm           # Phase 3: full transformer inference loop in Obj-C++ (WIP)
├── main.mm                # Native benchmark harness (100-rep timing)
├── build.sh               # Compiles .metal → .metallib, bridge.mm → .dylib
├── metal_bridge.py        # Python ctypes bindings for the dylib
├── benchmark_qwen.py      # Python benchmark (Phase 2 overhead measurement)
├── qwen_patched.py        # MLX monkey-patch: swaps Metal MLP into MLX forward pass
├── qwen_test.py           # Baseline MLX measurement
├── export_weights.py      # Dumps Qwen weights from MLX to raw fp16 .bin files
└── tokenize.py            # Quick tokenizer utility
```

## Requirements

- Apple Silicon Mac (M1 or later)
- Xcode command line tools
- Python 3.10+, `mlx-lm`, `numpy`
- Qwen2.5-0.5B-Instruct (downloaded via `mlx_lm.load`)

## Build

```bash
# Export weights first
python export_weights.py

# Build Metal library and dylib
cd kernel/qwen_kernel/qwen_kernel
./build.sh

# Run benchmark
cd ../../..
python benchmark_qwen.py
```

## Status

- [x] Phase 1: Custom matmul kernels — **2.7× throughput gain**
- [x] Phase 2: Python ctypes bridge — built, overhead issue identified and documented  
- [ ] Phase 3: Native Obj-C++ inference engine — blocked on memory bug in MLP layer loop
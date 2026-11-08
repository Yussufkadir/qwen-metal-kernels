# Kernel microbenchmarks

Standalone harnesses for measuring single decode kernels outside the engine.
Build any of them with:

```bash
xcrun clang++ -std=c++20 -O2 -fobjc-arc prof.mm -framework Foundation -framework Metal -o prof
```

- `prof.mm <default.metallib> <context>`: per-op decode cost with cold weights
  (24 distinct weight copies, one dispatch per layer), the closest thing to a
  per-kernel profile of `forward_step_gpu`.
- `attnbench.mm <default.metallib> attn_prototype.metal`: old vs new decode
  attention, timed and checked against a double-precision CPU reference.
- `bench.mm <kernels.metal> <optimized_kernels.metal> gemv.metal`: matvec
  variants. Note that shapes under ~24 MB stay in the system-level cache across
  repeats and report bandwidth above DRAM peak; trust `prof` for those.

Prototype kernels are compiled at runtime with `newLibraryWithSource`, so no
offline Metal toolchain is needed to iterate.

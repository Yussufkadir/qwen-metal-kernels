import time
import numpy as np
from metal_bridge import init, gate_up_batched_reuse, down_batched_reuse
init()


BATCH   = 24
M_GATE  = 4864
K_GATE  = 896
M_DOWN  = 896
K_DOWN  = 4864
WARMUP  = 5
REPS    = 100


print("\nCreating test data...")
np.random.seed(42)

gate_w = np.random.randn(BATCH, M_GATE, K_GATE).astype(np.float16)
up_w   = np.random.randn(BATCH, M_GATE, K_GATE).astype(np.float16)
x_gate = np.random.randn(BATCH, K_GATE).astype(np.float32)

down_w = np.random.randn(BATCH, M_DOWN, K_DOWN).astype(np.float16)
x_down = np.random.randn(BATCH, K_DOWN).astype(np.float32)

print(f"  gate_w: {gate_w.shape} ({gate_w.nbytes / 1e6:.1f} MB)")
print(f"  up_w:   {up_w.shape} ({up_w.nbytes / 1e6:.1f} MB)")
print(f"  down_w: {down_w.shape} ({down_w.nbytes / 1e6:.1f} MB)")


print("\nBenchmarking gate+up (48 matvecs)...")
for _ in range(WARMUP):
    gate_up_batched_reuse(gate_w, up_w, x_gate)

t0 = time.perf_counter()
for _ in range(REPS):
    gate_out, up_out = gate_up_batched_reuse(gate_w, up_w, x_gate)
t1 = time.perf_counter()

ms_gate_up = (t1 - t0) / REPS * 1000
print(f"  Time:       {ms_gate_up:.3f} ms")
print(f"  Per-matvec: {ms_gate_up / 48:.4f} ms")


print("\nBenchmarking down (24 matvecs)...")
for _ in range(WARMUP):
    down_batched_reuse(down_w, x_down)

t0 = time.perf_counter()
for _ in range(REPS):
    out = down_batched_reuse(down_w, x_down)
t1 = time.perf_counter()

ms_down = (t1 - t0) / REPS * 1000
print(f"  Time:       {ms_down:.3f} ms")
print(f"  Per-matvec: {ms_down / 24:.4f} ms")


mlp_total = ms_gate_up + ms_down
est_tok_sec = 1000.0 / (mlp_total * 1.3)

print(f"\n{'='*55}")
print(f"  MLP gate+up:  {ms_gate_up:.3f} ms  (48 matvecs in 1 dispatch)")
print(f"  MLP down:     {ms_down:.3f} ms  (24 matvecs in 1 dispatch)")
print(f"  MLP total:    {mlp_total:.3f} ms")
print(f"  Est tok/sec:  {est_tok_sec:.0f}")
print(f"  MLX baseline: 106 tok/sec")
print(f"  Status:       {'🔥 FASTER' if est_tok_sec > 106 else '🐢 slower'}")
print(f"{'='*55}")
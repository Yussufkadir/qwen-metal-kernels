import time
import numpy as np
import ctypes
from metal_bridge import init, _lib

init()

BATCH   = 24
M_GATE  = 4864
K_GATE  = 896
M_DOWN  = 896
K_DOWN  = 4864
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

gate_out = np.zeros((BATCH, M_GATE), dtype=np.float32)
up_out   = np.zeros((BATCH, M_GATE), dtype=np.float32)
down_out = np.zeros((BATCH, M_DOWN), dtype=np.float32)

gate_w_bits = gate_w.view(np.uint16)
up_w_bits   = up_w.view(np.uint16)
down_w_bits = down_w.view(np.uint16)
x_gate_f32  = np.ascontiguousarray(x_gate, dtype=np.float32)
x_down_f32  = np.ascontiguousarray(x_down, dtype=np.float32)

p_gate_w = gate_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
p_up_w   = up_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
p_x_gate = x_gate_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_gate_y = gate_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_up_y   = up_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

p_down_w = down_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
p_x_down = x_down_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_down_y = down_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

B  = ctypes.c_uint32(BATCH)
Mg = ctypes.c_uint32(M_GATE)
Kg = ctypes.c_uint32(K_GATE)
Md = ctypes.c_uint32(M_DOWN)
Kd = ctypes.c_uint32(K_DOWN)

print("\nBenchmarking gate+up (48 matvecs)...")

_lib.run_gate_up_batched(p_gate_w, p_up_w, p_x_gate, p_gate_y, p_up_y, B, Mg, Kg)

for _ in range(5):
    _lib.run_gate_up_batched(p_gate_w, p_up_w, p_x_gate, p_gate_y, p_up_y, B, Mg, Kg)

t0 = time.perf_counter()
for _ in range(REPS):
    _lib.run_gate_up_batched(p_gate_w, p_up_w, p_x_gate, p_gate_y, p_up_y, B, Mg, Kg)
t1 = time.perf_counter()

ms_gate = (t1 - t0) / REPS * 1000
print(f"  Time:       {ms_gate:.3f} ms")
print(f"  Per-matvec: {ms_gate / 48:.4f} ms")

print("\nBenchmarking down (24 matvecs)...")

_lib.run_down_batched(p_down_w, p_x_down, p_down_y, B, Md, Kd)

for _ in range(5):
    _lib.run_down_batched(p_down_w, p_x_down, p_down_y, B, Md, Kd)

t0 = time.perf_counter()
for _ in range(REPS):
    _lib.run_down_batched(p_down_w, p_x_down, p_down_y, B, Md, Kd)
t1 = time.perf_counter()

ms_down = (t1 - t0) / REPS * 1000
print(f"  Time:       {ms_down:.3f} ms")
print(f"  Per-matvec: {ms_down / 24:.4f} ms")

ms_gate_per_call = ms_gate / REPS
ms_down_per_call = ms_down / REPS

mlp_total = ms_gate_per_call + ms_down_per_call
est_tok_sec = 1000.0 / (mlp_total * 1.3)

print(f"\n{'='*55}")
print(f"  gate+up per call: {ms_gate_per_call:.3f} ms  (48 matvecs)")
print(f"  down per call:    {ms_down_per_call:.3f} ms  (24 matvecs)")
print(f"  MLP total:        {mlp_total:.3f} ms")
print(f"  Est tok/sec:      {est_tok_sec:.0f}")
print(f"  MLX baseline:     106 tok/sec")
print(f"  Status:           {'🔥 FASTER' if est_tok_sec > 106 else '🐢 slower'}")
print(f"{'='*55}")
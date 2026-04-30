import ctypes
import numpy as np
import os


_dylib_path = os.path.join(os.path.dirname(__file__), "libmetal_kernels.dylib")

if not os.path.exists(_dylib_path):
    raise FileNotFoundError(
        f"libmetal_kernels.dylib not found at {_dylib_path}\n"
        "Run: cd kernel/qwen_kernel && ./build.sh"
    )

_buffer_cache = {}

def _get_or_create_buffer(name, shape, dtype):
    key = (name, shape, dtype)
    if key not in _buffer_cache:
        _buffer_cache[key] = np.zeros(shape, dtype=dtype)
    return _buffer_cache[key]


def gate_up_batched_reuse(gate_w, up_w, x):
    B, M, K = gate_w.shape
    
    gate_w_bits = gate_w.view(np.uint16)
    up_w_bits   = up_w.view(np.uint16)
    x_f32       = np.ascontiguousarray(x, dtype=np.float32)
    
    gate_out = _get_or_create_buffer("gate_out", (B, M), np.float32)
    up_out   = _get_or_create_buffer("up_out", (B, M), np.float32)
    
    _lib.run_gate_up_batched(
        gate_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        up_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        up_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(B), ctypes.c_uint32(M), ctypes.c_uint32(K),
    )
    return gate_out, up_out


def down_batched_reuse(down_w, x):
    B, M, K = down_w.shape
    
    down_w_bits = down_w.view(np.uint16)
    x_f32       = np.ascontiguousarray(x, dtype=np.float32)
    
    out = _get_or_create_buffer("down_out", (B, M), np.float32)
    
    _lib.run_down_batched(
        down_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(B), ctypes.c_uint32(M), ctypes.c_uint32(K),
    )
    return out

_lib = ctypes.CDLL(_dylib_path)


_lib.metal_init.restype = ctypes.c_int

_lib.run_gate_up_batched.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
]
_lib.run_gate_up_batched.restype = ctypes.c_int

_lib.run_down_batched.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
]
_lib.run_down_batched.restype = ctypes.c_int

def init():
    result = _lib.metal_init()
    if result != 0:
        raise RuntimeError("Metal kernel initialization failed")
    print("✅ Metal kernels initialized")


def gate_up_batched(gate_w: np.ndarray, up_w: np.ndarray, x: np.ndarray):
    B, M, K = gate_w.shape

    gate_w_bits = gate_w.view(np.uint16)
    up_w_bits   = up_w.view(np.uint16)
    x_f32       = np.ascontiguousarray(x, dtype=np.float32)
    
    gate_out = np.zeros((B, M), dtype=np.float32)
    up_out   = np.zeros((B, M), dtype=np.float32)
    
    _lib.run_gate_up_batched(
        gate_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        up_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        up_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(B),
        ctypes.c_uint32(M),
        ctypes.c_uint32(K),
    )
    
    return gate_out, up_out


def down_batched(down_w: np.ndarray, x: np.ndarray):
    B, M, K = down_w.shape
    
    down_w_bits = down_w.view(np.uint16)
    x_f32       = np.ascontiguousarray(x, dtype=np.float32)
    
    out = np.zeros((B, M), dtype=np.float32)
    
    _lib.run_down_batched(
        down_w_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(B),
        ctypes.c_uint32(M),
        ctypes.c_uint32(K),
    )
    
    return out
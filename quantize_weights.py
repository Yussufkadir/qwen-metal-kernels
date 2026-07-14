import argparse
from pathlib import Path
import numpy as np

NUM_LAYERS = 24
HIDDEN = 896
INTERMEDIATE = 4864
KV_DIM = 128
VOCAB = 151936
GROUP_SIZE = 64
ROWS_PER_CHUNK = 256


def projection_shapes():
    yield "lm_head.weight", VOCAB, HIDDEN
    for layer in range(NUM_LAYERS):
        p = f"layer{layer}"
        yield f"{p}.self_attn.q_proj.weight", HIDDEN, HIDDEN
        yield f"{p}.self_attn.k_proj.weight", KV_DIM, HIDDEN
        yield f"{p}.self_attn.v_proj.weight", KV_DIM, HIDDEN
        yield f"{p}.self_attn.o_proj.weight", HIDDEN, HIDDEN
        yield f"{p}.mlp.gate_proj.weight", INTERMEDIATE, HIDDEN
        yield f"{p}.mlp.up_proj.weight", INTERMEDIATE, HIDDEN
        yield f"{p}.mlp.down_proj.weight", HIDDEN, INTERMEDIATE


def quantize_q4_file(weights_dir: Path, name: str, rows: int, columns: int) -> tuple[int, int]:
    source = weights_dir / f"{name}.bin"
    packed_path = weights_dir / f"{name}.q4.bin"
    scales_path = weights_dir / f"{name}.scales.bin"
    expected = rows * columns * np.dtype(np.float16).itemsize
    if source.stat().st_size != expected:
        raise ValueError(f"{source}: expected {expected} bytes, found {source.stat().st_size}")
    if columns % GROUP_SIZE or columns % 2:
        raise ValueError(f"{name}: input dimension {columns} is not INT4-group aligned")

    source_map = np.memmap(source, dtype=np.float16, mode="r", shape=(rows, columns))
    with packed_path.open("wb") as packed_file, scales_path.open("wb") as scales_file:
        for start in range(0, rows, ROWS_PER_CHUNK):
            chunk = np.asarray(source_map[start : start + ROWS_PER_CHUNK], dtype=np.float32)
            grouped = chunk.reshape(chunk.shape[0], columns // GROUP_SIZE, GROUP_SIZE)
            scales = np.max(np.abs(grouped), axis=2) / 7.0
            scales = np.where(scales == 0.0, 1.0, scales)
            quantized = np.clip(np.rint(grouped / scales[..., None]), -7, 7).astype(np.int8)
            encoded = (quantized.reshape(chunk.shape[0], columns).astype(np.int16) + 8).astype(np.uint8)
            packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)
            packed.tofile(packed_file)
            scales.astype(np.float16).tofile(scales_file)

    return packed_path.stat().st_size, scales_path.stat().st_size


def quantize_q8_file(weights_dir: Path, name: str, rows: int, columns: int) -> tuple[int, int]:
    source = weights_dir / f"{name}.bin"
    quant_path = weights_dir / f"{name}.q8.bin"
    scales_path = weights_dir / f"{name}.q8.scales.bin"
    expected = rows * columns * np.dtype(np.float16).itemsize
    if source.stat().st_size != expected:
        raise ValueError(f"{source}: expected {expected} bytes, found {source.stat().st_size}")
    if columns % GROUP_SIZE:
        raise ValueError(f"{name}: input dimension {columns} is not INT8-group aligned")

    source_map = np.memmap(source, dtype=np.float16, mode="r", shape=(rows, columns))
    with quant_path.open("wb") as quant_file, scales_path.open("wb") as scales_file:
        for start in range(0, rows, ROWS_PER_CHUNK):
            chunk = np.asarray(source_map[start : start + ROWS_PER_CHUNK], dtype=np.float32)
            grouped = chunk.reshape(chunk.shape[0], columns // GROUP_SIZE, GROUP_SIZE)
            scales = np.max(np.abs(grouped), axis=2) / 127.0
            scales = np.where(scales == 0.0, 1.0, scales)
            quantized = np.clip(np.rint(grouped / scales[..., None]), -127, 127).astype(np.int8)
            quantized.reshape(chunk.shape[0], columns).tofile(quant_file)
            scales.astype(np.float16).tofile(scales_file)

    return quant_path.stat().st_size, scales_path.stat().st_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_dir", nargs="?", default="kernel/qwen_kernel/qwen_kernel/qwen_weights")
    parser.add_argument("--bits", choices=["4", "8", "all"], default="4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir)
    if not weights_dir.is_dir():
        raise SystemExit(f"Weights directory not found: {weights_dir}")

    modes = ["4", "8"] if args.bits == "all" else [args.bits]
    for bits in modes:
        fp16_bytes = quantized_bytes = scale_bytes = 0
        quantize = quantize_q4_file if bits == "4" else quantize_q8_file
        label = f"INT{bits}"
        print(f"\nWriting {label} projection weights")
        for index, (name, rows, columns) in enumerate(projection_shapes(), start=1):
            quantized, scales = quantize(weights_dir, name, rows, columns)
            fp16_bytes += rows * columns * 2
            quantized_bytes += quantized
            scale_bytes += scales
            print(f"[{index:3d}/169] {name}: {(quantized + scales) / 1e6:7.2f} MB")

        total_bytes = quantized_bytes + scale_bytes
        print(f"FP16 projections: {fp16_bytes / 1e6:.1f} MB")
        print(f"{label} projections: {total_bytes / 1e6:.1f} MB")
        print(f"Compression:       {fp16_bytes / total_bytes:.2f}x")


if __name__ == "__main__":
    main()

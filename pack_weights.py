#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ALIGNMENT = 4096


def align(offset: int, alignment: int = ALIGNMENT) -> int:
    remainder = offset % alignment
    return offset if remainder == 0 else offset + alignment - remainder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weights_dir",
        nargs="?",
        default="kernel/qwen_kernel/qwen_kernel/qwen_weights",
    )
    parser.add_argument("--pack-name", default="weights.pack")
    parser.add_argument("--index-name", default="weights.index")
    parser.add_argument("--profile", default="fp16", choices=["fp16", "all"])
    return parser.parse_args()


def fp16_load_order() -> list[str]:
    names = [
        "embed_tokens.weight.bin",
        "norm.weight.bin",
        "lm_head.weight.bin",
    ]
    for layer in range(24):
        prefix = f"layer{layer}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight.bin",
                f"{prefix}.post_attention_layernorm.weight.bin",
                f"{prefix}.self_attn.q_proj.weight.bin",
                f"{prefix}.self_attn.k_proj.weight.bin",
                f"{prefix}.self_attn.v_proj.weight.bin",
                f"{prefix}.self_attn.o_proj.weight.bin",
                f"{prefix}.self_attn.q_proj.bias.bin",
                f"{prefix}.self_attn.k_proj.bias.bin",
                f"{prefix}.self_attn.v_proj.bias.bin",
                f"{prefix}.mlp.gate_proj.weight.bin",
                f"{prefix}.mlp.up_proj.weight.bin",
                f"{prefix}.mlp.down_proj.weight.bin",
            ]
        )
    return names


def ordered_files(weights_dir: Path, profile: str) -> list[Path]:
    files = {path.name: path for path in weights_dir.glob("*.bin") if path.is_file()}
    if profile == "all":
        return [files[name] for name in sorted(files)]

    result: list[Path] = []
    used: set[str] = set()
    for name in fp16_load_order():
        path = files.get(name)
        if path is not None:
            result.append(path)
            used.add(name)
    for name in sorted(files):
        if name not in used:
            result.append(files[name])
    return result


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        raise SystemExit(f"missing weights directory: {weights_dir}")

    pack_path = weights_dir / args.pack_name
    index_path = weights_dir / args.index_name
    files = ordered_files(weights_dir, args.profile)
    if not files:
        raise SystemExit(f"no .bin files found in {weights_dir}")

    entries: list[tuple[str, int, int]] = []
    offset = 0
    with pack_path.open("wb") as pack:
        for path in files:
            aligned = align(offset)
            if aligned > offset:
                pack.write(b"\0" * (aligned - offset))
                offset = aligned

            data = path.read_bytes()
            name = path.name
            entries.append((name, offset, len(data)))
            pack.write(data)
            offset += len(data)

    with index_path.open("w", encoding="utf-8") as index:
        for name, entry_offset, size in entries:
            index.write(f"{name}\t{entry_offset}\t{size}\n")

    total_size = sum(size for _, _, size in entries)
    print(f"Packed {len(entries)} files")
    print(f"Payload: {total_size / (1024 * 1024):.1f} MB")
    print(f"Archive: {pack_path}")
    print(f"Index:   {index_path}")


if __name__ == "__main__":
    main()

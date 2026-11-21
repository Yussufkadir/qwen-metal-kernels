import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import support
import pack_weights
import quantize_weights

ROWS = 6
COLUMNS = 128
GROUPS = COLUMNS // quantize_weights.GROUP_SIZE

def write_weights(directory: Path, name: str = "w") -> np.ndarray:
    rng = np.random.default_rng(7)
    weights = (rng.standard_normal((ROWS, COLUMNS)) * 0.05).astype(np.float16)
    weights[0, : quantize_weights.GROUP_SIZE] = 0
    weights.tofile(directory / f"{name}.bin")
    return weights.astype(np.float32)

class QuantizeInt8Test(unittest.TestCase):
    def test_round_trip_error_is_within_half_a_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            weights = write_weights(directory)
            sizes = quantize_weights.quantize_q8_file(directory, "w", ROWS, COLUMNS)
            self.assertEqual(sizes, (ROWS * COLUMNS, ROWS * GROUPS * 2))
            quantized = np.fromfile(directory / "w.q8.bin", dtype=np.int8).reshape(ROWS, GROUPS, -1)
            scales = np.fromfile(directory / "w.q8.scales.bin", dtype=np.float16).reshape(ROWS, GROUPS)
            scales = scales.astype(np.float32)
            self.assertLessEqual(int(np.abs(quantized).max()), 127)
            restored = (quantized * scales[..., None]).reshape(ROWS, COLUMNS)
            tolerance = np.repeat(scales, quantize_weights.GROUP_SIZE, axis=1) * 0.51 + 1e-4
            self.assertTrue(np.all(np.abs(restored - weights) <= tolerance))

    def test_zero_group_stays_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_weights(directory)
            quantize_weights.quantize_q8_file(directory, "w", ROWS, COLUMNS)
            quantized = np.fromfile(directory / "w.q8.bin", dtype=np.int8).reshape(ROWS, COLUMNS)
            self.assertFalse(quantized[0, : quantize_weights.GROUP_SIZE].any())

    def test_wrong_file_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_weights(directory)
            with self.assertRaises(ValueError):
                quantize_weights.quantize_q8_file(directory, "w", ROWS + 1, COLUMNS)

class QuantizeInt4Test(unittest.TestCase):
    def test_round_trip_error_is_within_half_a_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            weights = write_weights(directory)
            sizes = quantize_weights.quantize_q4_file(directory, "w", ROWS, COLUMNS)
            self.assertEqual(sizes, (ROWS * COLUMNS // 2, ROWS * GROUPS * 2))
            packed = np.fromfile(directory / "w.q4.bin", dtype=np.uint8).reshape(ROWS, COLUMNS // 2)
            values = np.empty((ROWS, COLUMNS), dtype=np.int16)
            values[:, 0::2] = packed & 0x0F
            values[:, 1::2] = packed >> 4
            values -= 8
            self.assertGreaterEqual(int(values.min()), -7)
            self.assertLessEqual(int(values.max()), 7)
            scales = np.fromfile(directory / "w.scales.bin", dtype=np.float16).reshape(ROWS, GROUPS)
            scales = scales.astype(np.float32)
            expanded = np.repeat(scales, quantize_weights.GROUP_SIZE, axis=1)
            restored = values.astype(np.float32) * expanded
            self.assertTrue(np.all(np.abs(restored - weights) <= expanded * 0.51 + 1e-4))

    def test_unaligned_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            np.zeros((2, 100), dtype=np.float16).tofile(directory / "w.bin")
            with self.assertRaises(ValueError):
                quantize_weights.quantize_q4_file(directory, "w", 2, 100)

class ProjectionShapesTest(unittest.TestCase):
    def test_covers_every_projection(self):
        shapes = list(quantize_weights.projection_shapes())
        self.assertEqual(len(shapes), 1 + 7 * quantize_weights.NUM_LAYERS)
        self.assertEqual(shapes[0], ("lm_head.weight", quantize_weights.VOCAB, quantize_weights.HIDDEN))
        self.assertEqual(len({name for name, _, _ in shapes}), len(shapes))
        for _, _, columns in shapes:
            self.assertEqual(columns % quantize_weights.GROUP_SIZE, 0)

class PackWeightsTest(unittest.TestCase):
    def test_align(self):
        self.assertEqual(pack_weights.align(0), 0)
        self.assertEqual(pack_weights.align(1), 4096)
        self.assertEqual(pack_weights.align(4096), 4096)
        self.assertEqual(pack_weights.align(4097), 8192)

    def test_fp16_profile_orders_hot_files_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("zzz.q8.bin", "norm.weight.bin", "embed_tokens.weight.bin", "aaa.q4.bin"):
                (directory / name).write_bytes(b"x")
            names = [path.name for path in pack_weights.ordered_files(directory, "fp16")]
            self.assertEqual(names, ["embed_tokens.weight.bin", "norm.weight.bin", "aaa.q4.bin", "zzz.q8.bin"])
            names = [path.name for path in pack_weights.ordered_files(directory, "all")]
            self.assertEqual(names, sorted(names))

    def test_pack_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            contents = {
                "embed_tokens.weight.bin": bytes(range(256)) * 20,
                "norm.weight.bin": b"\x01\x02\x03",
                "layer0.mlp.down_proj.weight.bin": b"\xff" * 5000,
            }
            for name, data in contents.items():
                (directory / name).write_bytes(data)
            with mock.patch.object(sys, "argv", ["pack_weights.py", str(directory)]):
                with mock.patch("builtins.print"):
                    pack_weights.main()
            pack = (directory / "weights.pack").read_bytes()
            seen = {}
            for line in (directory / "weights.index").read_text().splitlines():
                name, offset, size = line.split("\t")
                offset, size = int(offset), int(size)
                self.assertEqual(offset % pack_weights.ALIGNMENT, 0)
                seen[name] = pack[offset : offset + size]
            self.assertEqual(seen, contents)

if __name__ == "__main__":
    unittest.main()

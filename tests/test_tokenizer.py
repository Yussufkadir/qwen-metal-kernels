import tempfile
import unittest
from pathlib import Path

import support
import qwen_chat

TOKENIZER = support.load_tokenizer_or_none()

SAMPLES = [
    "What is the capital of France?",
    "def add(a, b):\n    return a + b\n",
    "Türkçe karakterler: ğüşiöç ĞÜŞİÖÇ",
    "日本語のテキスト and emoji 🚀",
    "  leading and trailing spaces  ",
    "Explain KV caching.\nUse two sentences.\n\n- one\n- two",
    "caf\u00e9 r\u00e9sum\u00e9 12345 3.14 it's don't",
    "",
]

class ChatTemplateTest(unittest.TestCase):
    def render(self, messages, add_generation_prompt=True):
        return qwen_chat.FastQwenTokenizer.render_chat(None, messages, add_generation_prompt)

    def test_custom_system_prompt(self):
        text = self.render([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ])
        self.assertEqual(
            text,
            "<|im_start|>system\nBe brief.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_default_system_prompt_is_added(self):
        text = self.render([{"role": "user", "content": "Hi"}])
        self.assertTrue(text.startswith("<|im_start|>system\nYou are Qwen"))
        self.assertIn("<|im_start|>user\nHi<|im_end|>\n", text)

    def test_generation_prompt_is_optional(self):
        text = self.render([{"role": "user", "content": "Hi"}], add_generation_prompt=False)
        self.assertTrue(text.endswith("<|im_start|>user\nHi<|im_end|>\n"))

    def test_multi_turn_order(self):
        text = self.render([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ])
        self.assertLess(text.index("one"), text.index("two"))
        self.assertLess(text.index("two"), text.index("three"))

class TokenizerFilesTest(unittest.TestCase):
    def test_empty_directory_has_no_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(qwen_chat.has_tokenizer_files(Path(directory)))

    def test_copy_makes_a_directory_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            target = Path(directory) / "nested" / "target"
            source.mkdir()
            for name in qwen_chat.TOKENIZER_REQUIRED_FILES:
                (source / name).write_text("x")
            self.assertTrue(qwen_chat.copy_tokenizer_files(source, target))
            self.assertTrue(qwen_chat.has_tokenizer_files(target))

    def test_copy_fails_when_source_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "vocab.json").write_text("{}")
            self.assertFalse(qwen_chat.copy_tokenizer_files(source, Path(directory) / "target"))

    def test_other_models_are_rejected(self):
        with self.assertRaises(RuntimeError):
            qwen_chat.load_fast_qwen_tokenizer("some/other-model")

@unittest.skipIf(TOKENIZER is None, "tokenizer files are not available")
class FastTokenizerTest(unittest.TestCase):
    def test_special_token_ids(self):
        self.assertEqual(TOKENIZER.convert_tokens_to_ids("<|endoftext|>"), 151643)
        self.assertEqual(TOKENIZER.convert_tokens_to_ids("<|im_start|>"), 151644)
        self.assertEqual(TOKENIZER.convert_tokens_to_ids("<|im_end|>"), 151645)
        self.assertEqual(TOKENIZER.convert_tokens_to_ids("definitely-not-a-token-xyz"), -1)

    def test_stop_ids(self):
        self.assertEqual(qwen_chat.special_stop_ids(TOKENIZER), {151643, 151645})

    def test_round_trip(self):
        for sample in SAMPLES:
            with self.subTest(sample=sample):
                self.assertEqual(TOKENIZER.decode(TOKENIZER.encode(sample)), sample)

    def test_ids_are_in_vocabulary(self):
        for sample in SAMPLES:
            for token in TOKENIZER.encode(sample):
                self.assertTrue(0 <= token < 151936)

    def test_special_tokens_are_single_ids(self):
        tokens = TOKENIZER.encode("<|im_start|>user\nHi<|im_end|>\n")
        self.assertEqual(tokens[0], 151644)
        self.assertEqual(tokens.count(151645), 1)

    def test_decode_can_skip_special_tokens(self):
        tokens = TOKENIZER.encode("<|im_start|>Hi<|im_end|>")
        self.assertEqual(TOKENIZER.decode(tokens, skip_special_tokens=True), "Hi")
        self.assertEqual(TOKENIZER.decode(tokens, skip_special_tokens=False), "<|im_start|>Hi<|im_end|>")

    def test_chat_tokens_start_and_end(self):
        tokens = qwen_chat.make_prompt_tokens(TOKENIZER, "You are a helpful assistant.", "Hi")
        self.assertEqual(tokens[0], 151644)
        self.assertEqual(tokens[-3:], TOKENIZER.encode("<|im_start|>assistant\n"))

    def test_loads_from_a_plain_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "tokenizer"
            self.assertTrue(qwen_chat.copy_tokenizer_files(TOKENIZER.snapshot, target))
            clone = qwen_chat.load_fast_qwen_tokenizer(qwen_chat.MODEL_NAME, target)
            for sample in SAMPLES:
                self.assertEqual(clone.encode(sample), TOKENIZER.encode(sample))

    def test_matches_transformers_reference(self):
        try:
            from transformers import AutoTokenizer
            reference = AutoTokenizer.from_pretrained(qwen_chat.MODEL_NAME, local_files_only=True)
        except Exception as exc:
            self.skipTest(f"transformers reference tokenizer not available: {exc}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain KV caching.\nUse two sentences."},
            {"role": "assistant", "content": "It stores keys and values."},
            {"role": "user", "content": "Türkçe söyle 🚀"},
        ]
        expected = qwen_chat.make_chat_tokens(reference, messages)
        self.assertEqual(qwen_chat.make_chat_tokens(TOKENIZER, messages), expected)
        for sample in SAMPLES:
            with self.subTest(sample=sample):
                self.assertEqual(
                    TOKENIZER.encode(sample),
                    reference.encode(sample, add_special_tokens=False),
                )

    def test_normalizes_unicode_like_the_reference(self):
        self.assertEqual(TOKENIZER.encode("e\u0301"), TOKENIZER.encode("\u00e9"))

if __name__ == "__main__":
    unittest.main()

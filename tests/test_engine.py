import subprocess
import unittest

import support
import qwen_chat

TOKENIZER = support.load_tokenizer_or_none() if support.engine_available() else None
READY = support.engine_available() and TOKENIZER is not None
GREEDY = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "seed": 0}
STOP = {151643, 151645}

def chat_tokens(*turns):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for index, text in enumerate(turns):
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": text})
    return qwen_chat.make_chat_tokens(TOKENIZER, messages)

def run_engine(*args, timeout=120):
    return subprocess.run(
        [str(support.ENGINE), *[str(arg) for arg in args]],
        cwd=support.ENGINE_DIR,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

def generate(prompt, *options, max_tokens=12):
    result = run_engine("--quiet", "-n", max_tokens, *options, *prompt)
    if result.returncode != 0:
        raise AssertionError(f"engine failed ({result.returncode}): {result.stderr}")
    match = qwen_chat.GENERATED_RE.search(result.stdout)
    if match is None:
        raise AssertionError(f"no generated tokens in output: {result.stdout!r}")
    return [int(token) for token in match.group(1).split()]

def start_server(backend="fp16"):
    return qwen_chat.PersistentNativeEngine(
        engine=support.ENGINE,
        engine_dir=support.ENGINE_DIR,
        backend=backend,
        warmup=False,
        gpu_warmup=False,
    )

@unittest.skipUnless(READY, "native engine, weights or tokenizer not available")
class CommandLineTest(unittest.TestCase):
    def assert_usage_error(self, *args):
        result = run_engine(*args)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue(result.stderr.strip())

    def test_help(self):
        result = run_engine("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--backend", result.stdout)

    def test_unknown_option(self):
        self.assert_usage_error("--definitely-not-an-option", "1")

    def test_non_numeric_token(self):
        self.assert_usage_error("--quiet", "12", "abc")

    def test_non_numeric_option_value(self):
        self.assert_usage_error("--quiet", "-n", "abc", "12")

    def test_non_positive_max_tokens(self):
        self.assert_usage_error("--quiet", "-n", "0", "12")
        self.assert_usage_error("--quiet", "-n", "-3", "12")

    def test_unknown_backend(self):
        self.assert_usage_error("--quiet", "--backend", "float128", "12")

    def test_out_of_vocabulary_token_fails_cleanly(self):
        for token in (151936, 99999999, -1):
            with self.subTest(token=token):
                result = run_engine("--quiet", "-n", "2", "--", token) if token < 0 else run_engine("--quiet", "-n", "2", token)
                self.assertGreater(result.returncode, 0)

    def test_prompt_longer_than_context_fails_cleanly(self):
        result = run_engine("--quiet", "-n", "8", *([198] * 4100))
        self.assertGreater(result.returncode, 0)

@unittest.skipUnless(READY, "native engine, weights or tokenizer not available")
class GenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = chat_tokens("What is the capital of France?")
        cls.reference = generate(cls.prompt)

    def test_greedy_answer_is_correct(self):
        text = TOKENIZER.decode(qwen_chat.visible_prefix(self.reference, STOP))
        self.assertIn("Paris", text)

    def test_greedy_is_deterministic(self):
        self.assertEqual(generate(self.prompt), self.reference)

    def test_stop_token_ends_generation(self):
        tokens = generate(self.prompt, "--stop-token", 151645, max_tokens=64)
        self.assertEqual(tokens[-1], 151645)
        self.assertEqual(tokens.count(151645), 1)
        self.assertLess(len(tokens), 64)

    def test_max_tokens_is_respected(self):
        self.assertEqual(len(generate(self.prompt, max_tokens=3)), 3)
        self.assertEqual(generate(self.prompt, max_tokens=3), self.reference[:3])

    def test_stream_tokens_match_blocking_output(self):
        result = run_engine("--quiet", "-n", 12, "--stream-tokens", *self.prompt)
        self.assertEqual(result.returncode, 0, result.stderr)
        streamed = [int(line.split()[1]) for line in result.stdout.splitlines() if line.startswith("TOKEN ")]
        self.assertEqual(streamed, self.reference)

    def test_legacy_kernels_match_optimized_kernels(self):
        for flag in ("--legacy-attention", "--legacy-prefill", "--legacy-prefill-attention"):
            with self.subTest(flag=flag):
                self.assertEqual(generate(self.prompt, flag), self.reference)

    def test_long_prompt_uses_tiled_prefill_consistently(self):
        prompt = chat_tokens("Summarize this text. " + "The quick brown fox jumps over the lazy dog. " * 60)
        self.assertGreater(len(prompt), 512)
        tiled = generate(prompt, max_tokens=8)
        legacy = generate(prompt, "--legacy-prefill-attention", max_tokens=8)
        self.assertEqual(tiled[:4], legacy[:4])

    def test_backends_agree_with_fp16(self):
        expected = qwen_chat.visible_prefix(self.reference, STOP)
        self.assertTrue(expected)
        for backend in ("mps", "int8", "int8-fp16"):
            with self.subTest(backend=backend):
                if not support.backend_available(backend):
                    self.skipTest(f"weights for {backend} are not available")
                tokens = generate(self.prompt, "--backend", backend)
                self.assertEqual(qwen_chat.visible_prefix(tokens, STOP), expected)

    def test_hybrid_prefill_matches_fp16(self):
        if not support.backend_available("hybrid"):
            self.skipTest("weights for hybrid are not available")
        tokens = generate(self.prompt, "--backend", "hybrid", max_tokens=2)
        self.assertEqual(tokens[0], self.reference[0])

    def test_int4_backends_run(self):
        for backend in ("int4", "mixed"):
            with self.subTest(backend=backend):
                if not support.backend_available(backend):
                    self.skipTest(f"weights for {backend} are not available")
                tokens = generate(self.prompt, "--backend", backend, max_tokens=4)
                self.assertEqual(len(tokens), 4)
                self.assertTrue(all(0 <= token < 151936 for token in tokens))

@unittest.skipUnless(READY, "native engine, weights or tokenizer not available")
class SamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = chat_tokens("Write one sentence about the sea.")

    def sample(self, seed, backend="fp16"):
        return generate(
            self.prompt, "--backend", backend, "--temperature", 0.9, "--top-k", 40,
            "--top-p", 0.95, "--seed", seed, max_tokens=16,
        )

    def test_same_seed_gives_same_tokens(self):
        self.assertEqual(self.sample(1234), self.sample(1234))

    def test_different_seeds_give_different_tokens(self):
        outputs = {tuple(self.sample(seed)) for seed in (1, 2, 3, 4)}
        self.assertGreater(len(outputs), 1)

    def test_seeded_sampling_matches_across_prefill_paths(self):
        if not support.backend_available("mps"):
            self.skipTest("mps backend not available")
        self.assertEqual(self.sample(77, "mps")[0], self.sample(77, "fp16")[0])

    def test_top_k_one_is_greedy(self):
        greedy = generate(self.prompt, max_tokens=8)
        sampled = generate(self.prompt, "--temperature", 1.0, "--top-k", 1, "--seed", 5, max_tokens=8)
        self.assertEqual(sampled, greedy)

@unittest.skipUnless(READY, "native engine, weights or tokenizer not available")
class SessionTest(unittest.TestCase):
    def setUp(self):
        self.server = start_server()
        self.addCleanup(self.server.close)

    def ask(self, prompt, max_tokens=12):
        tokens, _, _, cache = self.server.generate(prompt, max_tokens, STOP, GREEDY)
        return tokens, cache

    def test_first_request_matches_one_shot(self):
        prompt = chat_tokens("What is the capital of France?")
        tokens, cache = self.ask(prompt)
        self.assertEqual(cache, (0, len(prompt), False))
        self.assertEqual(tokens, generate(prompt, "--stop-token", 151643, "--stop-token", 151645))

    def test_follow_up_reuses_the_cache_and_matches_a_fresh_run(self):
        first = chat_tokens("What is the capital of France?")
        answer, _ = self.ask(first)
        reply = TOKENIZER.decode(qwen_chat.visible_prefix(answer, STOP))
        second = chat_tokens("What is the capital of France?", reply, "And of Italy?")
        tokens, cache = self.ask(second)
        common, delta, reset = cache
        self.assertGreater(common, len(first))
        self.assertEqual(common + delta, len(second))
        self.assertFalse(reset)
        self.assertEqual(tokens, generate(second, "--stop-token", 151643, "--stop-token", 151645))

    def test_diverging_prompt_keeps_the_shared_prefix(self):
        first = chat_tokens("Tell me about rivers in Europe.")
        self.ask(first)
        second = chat_tokens("Tell me about mountains in Asia.")
        shared = 0
        while first[shared] == second[shared]:
            shared += 1
        self.assertGreater(shared, 0)
        tokens, cache = self.ask(second)
        self.assertEqual(cache, (shared, len(second) - shared, False))
        self.assertEqual(tokens, generate(second, "--stop-token", 151643, "--stop-token", 151645))

    def test_repeating_the_same_prompt_gives_the_same_answer(self):
        prompt = chat_tokens("Name a primary color.")
        first, _ = self.ask(prompt)
        second, cache = self.ask(prompt)
        self.assertEqual(second, first)
        self.assertEqual(cache, (len(prompt) - 1, 1, False))

    def test_reset_clears_the_cache(self):
        prompt = chat_tokens("Name a primary color.")
        first, _ = self.ask(prompt)
        self.server.reset_session()
        second, cache = self.ask(prompt)
        self.assertEqual(cache, (0, len(prompt), False))
        self.assertEqual(second, first)

    def test_unrelated_prompt_resets_the_cache(self):
        self.ask(chat_tokens("Name a primary color."))
        prompt = [198, 9707, 11, 1879]
        _, cache = self.ask(prompt, max_tokens=2)
        self.assertEqual(cache, (0, len(prompt), True))

    def test_context_overflow_is_an_error_and_the_server_survives(self):
        with self.assertRaises(RuntimeError) as raised:
            self.ask([198] * 4090, max_tokens=64)
        self.assertIn("context window exceeded", str(raised.exception))
        self.assertTrue(self.server.is_alive())
        tokens, _ = self.ask(chat_tokens("What is the capital of France?"))
        self.assertIn("Paris", TOKENIZER.decode(tokens))

    def test_invalid_token_is_an_error_and_the_server_survives(self):
        with self.assertRaises(RuntimeError):
            self.ask([151936, 5])
        self.assertTrue(self.server.is_alive())
        tokens, cache = self.ask(chat_tokens("What is the capital of France?"))
        self.assertIn("Paris", TOKENIZER.decode(tokens))
        self.assertEqual(cache[0], 0)

    def test_full_context_window_can_be_used(self):
        prompt = chat_tokens("Repeat after me. " + "word " * 4200)
        self.assertGreater(len(prompt), qwen_chat.prompt_budget(4))
        prompt = prompt[: qwen_chat.prompt_budget(4)]
        tokens, cache = self.ask(prompt, max_tokens=4)
        self.assertLessEqual(len(tokens), 4)
        self.assertEqual(cache[1], len(prompt))
        with self.assertRaises(RuntimeError):
            self.ask(prompt + [198], max_tokens=4)

    def test_close_stops_the_process(self):
        self.server.close()
        self.assertFalse(self.server.is_alive())
        with self.assertRaises(RuntimeError):
            self.ask([198, 9707])

@unittest.skipUnless(READY and support.backend_available("mps"), "mps backend not available")
class MpsSessionTest(unittest.TestCase):
    def test_mps_chat_works_without_a_cache(self):
        server = start_server("mps")
        self.addCleanup(server.close)
        prompt = chat_tokens("What is the capital of France?")
        tokens, _, _, cache = server.generate(prompt, 12, STOP, GREEDY)
        self.assertEqual(cache, (0, len(prompt), False))
        self.assertIn("Paris", TOKENIZER.decode(tokens))
        tokens, _, _, cache = server.generate(prompt, 12, STOP, GREEDY)
        self.assertEqual(cache, (0, len(prompt), False))

if __name__ == "__main__":
    unittest.main()

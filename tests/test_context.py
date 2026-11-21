import unittest

import support
import qwen_chat

class WordTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        tokens = []
        for message in messages:
            tokens.extend([0] * (4 + len(message["content"].split())))
        if add_generation_prompt:
            tokens.extend([0] * 3)
        return tokens

def words(count):
    return " ".join(["w"] * count)

def conversation(turns, words_per_message):
    messages = [{"role": "system", "content": "sys"}]
    for index in range(turns):
        messages.append({"role": "user", "content": words(words_per_message) + f" u{index}"})
        messages.append({"role": "assistant", "content": words(words_per_message) + f" a{index}"})
    return messages

def count(messages):
    return len(qwen_chat.make_chat_tokens(WordTokenizer(), messages))

class FitsContextTest(unittest.TestCase):
    def test_boundary(self):
        limit = qwen_chat.MAX_CONTEXT_TOKENS
        self.assertTrue(qwen_chat.fits_context(limit, 1))
        self.assertFalse(qwen_chat.fits_context(limit + 1, 1))
        self.assertTrue(qwen_chat.fits_context(limit - 127, 128))
        self.assertFalse(qwen_chat.fits_context(limit - 126, 128))

    def test_prompt_budget_matches_fits_context(self):
        for max_tokens in (1, 64, 128, 1024):
            budget = qwen_chat.prompt_budget(max_tokens)
            self.assertTrue(qwen_chat.fits_context(budget, max_tokens))
            self.assertFalse(qwen_chat.fits_context(budget + 1, max_tokens))

class TrimToContextTest(unittest.TestCase):
    def test_no_trim_when_it_fits(self):
        messages = conversation(3, 20)
        messages.append({"role": "user", "content": "next"})
        before = [dict(message) for message in messages]
        self.assertEqual(qwen_chat.trim_to_context(WordTokenizer(), messages, 128), [])
        self.assertEqual(messages, before)

    def test_trim_keeps_system_and_newest_message(self):
        messages = conversation(40, 100)
        messages.append({"role": "user", "content": "newest question"})
        self.assertGreater(count(messages), qwen_chat.prompt_budget(128))
        removed = qwen_chat.trim_to_context(WordTokenizer(), messages, 128)
        self.assertTrue(removed)
        self.assertEqual(messages[0], {"role": "system", "content": "sys"})
        self.assertEqual(messages[-1], {"role": "user", "content": "newest question"})
        self.assertTrue(qwen_chat.fits_context(count(messages), 128))

    def test_trim_drops_whole_turns_oldest_first(self):
        messages = conversation(40, 100)
        messages.append({"role": "user", "content": "newest question"})
        original = [dict(message) for message in messages]
        removed = qwen_chat.trim_to_context(WordTokenizer(), messages, 128)
        self.assertEqual(len(removed) % 2, 0)
        self.assertEqual(removed, original[1 : 1 + len(removed)])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages, original[:1] + original[1 + len(removed) :])

    def test_trim_leaves_headroom_for_later_turns(self):
        messages = conversation(40, 100)
        messages.append({"role": "user", "content": "newest question"})
        qwen_chat.trim_to_context(WordTokenizer(), messages, 128)
        goal = int(qwen_chat.prompt_budget(128) * qwen_chat.CONTEXT_TRIM_TARGET)
        self.assertLessEqual(count(messages), goal)
        messages.append({"role": "assistant", "content": words(100)})
        messages.append({"role": "user", "content": words(100)})
        self.assertEqual(qwen_chat.trim_to_context(WordTokenizer(), messages, 128), [])

    def test_trim_cannot_fix_a_single_oversized_message(self):
        messages = conversation(2, 10)
        messages.append({"role": "user", "content": words(5000)})
        removed = qwen_chat.trim_to_context(WordTokenizer(), messages, 128)
        self.assertEqual(len(removed), 4)
        self.assertEqual(len(messages), 2)
        self.assertFalse(qwen_chat.fits_context(count(messages), 128))

    def test_trim_without_system_prompt(self):
        messages = conversation(40, 100)[1:]
        messages.append({"role": "user", "content": "newest question"})
        removed = qwen_chat.trim_to_context(WordTokenizer(), messages, 128)
        self.assertTrue(removed)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[-1]["content"], "newest question")

class FullTranscriptTest(unittest.TestCase):
    def test_archived_messages_are_restored_in_order(self):
        messages = conversation(40, 100)
        messages.append({"role": "user", "content": "newest question"})
        original = [dict(message) for message in messages]
        removed = qwen_chat.trim_to_context(WordTokenizer(), messages, 128)
        self.assertEqual(qwen_chat.full_transcript(messages, removed), original)

    def test_no_archive(self):
        messages = conversation(2, 5)
        self.assertEqual(qwen_chat.full_transcript(messages, []), messages)

class HelpersTest(unittest.TestCase):
    def test_visible_prefix_stops_at_stop_token(self):
        self.assertEqual(qwen_chat.visible_prefix([1, 2, 9, 3], {9}), [1, 2])
        self.assertEqual(qwen_chat.visible_prefix([1, 2, 3], {9}), [1, 2, 3])
        self.assertEqual(qwen_chat.visible_prefix([9], {9}), [])

    def test_normalize_token_ids(self):
        class Encoding:
            input_ids = [1, 2]
        self.assertEqual(qwen_chat.normalize_token_ids([1, 2]), [1, 2])
        self.assertEqual(qwen_chat.normalize_token_ids({"input_ids": [3]}), [3])
        self.assertEqual(qwen_chat.normalize_token_ids(Encoding()), [1, 2])

if __name__ == "__main__":
    unittest.main()

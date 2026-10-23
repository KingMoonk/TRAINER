"""The expectation line: tiers, word-list effects and the pages' use of it. No Tk window is created."""
import unittest

from studio_ia.expectation import T
from studio_ia.expectation import expectation as both, table_params
from studio_ia import design as D
from studio_ia.pretraining_plan import version_expectation
from studio_spec import default_spec, validate_spec
from studio_estimate import estimate


def expectation(*args, **kw):
    """The details (hover) text of a design's expectation."""
    return both(*args, **kw)[1]


def design(s):
    """The Builder's expectation for the choices `s` made on the starting design."""
    return D.design_expectation(s, D.write_spec(default_spec(), s))


class Tiers(unittest.TestCase):
    def test_default_design_is_letter_patterns(self):
        c = D.sizes(default_spec(), 0)
        text = expectation(c["params"], "byte", embedding_params=c["table"])
        self.assertIn(T["letters"], text)
        self.assertIn(T["guide"], text)
        self.assertIn(T["wider_deeper"], text)
        self.assertNotIn(T["table_heavy"], text)

    def test_little_text_caps_a_bigger_model(self):
        text = expectation(5_000_000, "bpe_custom", tokens_seen=1_000_000)
        self.assertIn(T["letters"], text)
        self.assertIn(T["more_text"], text)
        self.assertNotIn(T["wider_deeper"], text)

    def test_millions_of_parameters_get_real_words(self):
        text = expectation(5_000_000, "bpe_custom", tokens_seen=10_000_000)
        self.assertIn(T["words"], text)
        self.assertIn(T["wider_deeper"], text)

    def test_byte_at_twenty_million_is_asked_for_word_pieces(self):
        text = expectation(20_000_000, "byte", tokens_seen=500_000_000)
        self.assertIn(T["words"], text)
        self.assertIn(T["custom_pieces"], text)
        self.assertNotIn(T["sentences"], text)

    def test_twenty_million_custom_pieces_with_lots_of_text_write_simple_sentences(self):
        text = expectation(20_000_000, "bpe_custom", tokens_seen=500_000_000, data_kind="text", embedding_params=8192 * 512)
        self.assertIn(T["sentences"], text)
        self.assertIn(T["chat_scale"], text)
        self.assertNotIn(T["move_up"].split("{")[0], text)
        self.assertNotIn(T["table_heavy"], text)

    def test_text_that_is_not_yet_known_is_named_as_a_condition(self):
        self.assertIn(T["sentences_text"], expectation(20_000_000, "bpe_custom"))
        self.assertIn(T["words_text"], expectation(5_000_000, "byte"))


class WordLists(unittest.TestCase):
    def test_each_word_list_is_described(self):
        self.assertIn(T["byte"], expectation(100_000, "byte"))
        self.assertIn(T["bpe_custom"], expectation(100_000, "bpe_custom"))
        self.assertIn(T["bpe_gpt2"], expectation(100_000, "bpe_gpt2"))
        self.assertIn("letter by letter", T["byte"])
        self.assertIn("3 to 4 times", T["bpe_custom"])

    def test_gpt2_at_a_small_width_is_all_word_table(self):
        s = dict(D.read_spec(default_spec()), V=2)
        self.assertEqual(s["d"], 64)
        text = design(s)[1]
        self.assertIn(T["bpe_gpt2"], text)
        self.assertIn(T["table_heavy"], text)
        self.assertIn(T["letters"], text)  # the table is not thinking capacity, so 3 million parameters still read as small

    def test_a_wide_gpt2_design_is_not_table_heavy(self):
        total = 120_000_000
        self.assertNotIn(T["table_heavy"], expectation(total, "bpe_gpt2", embedding_params=table_params("bpe_gpt2", 768)))

    def test_whole_words_are_described_and_lose_rare_words(self):
        text = expectation(100_000, "word")
        self.assertIn(T["word"], text)
        self.assertIn("lost", T["word"])
        self.assertIn("unknown", T["word"])

    def test_whole_words_reach_sentences_like_other_word_lists_and_are_sized_by_the_data(self):
        self.assertIn(T["sentences"], expectation(20_000_000, "word", tokens_seen=500_000_000))
        self.assertNotIn(T["custom_pieces"], expectation(5_000_000, "word", tokens_seen=10_000_000))
        self.assertIsNone(table_params("word", 64))
        self.assertEqual(table_params("word", 64, vocab=3000), 3000 * 64)
        self.assertEqual(design(dict(D.read_spec(default_spec()), V=3))[0].split(":")[0], "Expect")

    def test_table_size(self):
        self.assertEqual(table_params("bpe_gpt2", 64), 50257 * 64)
        self.assertEqual(table_params("byte", 64), 256 * 64)
        self.assertIsNone(table_params("bpe_custom", 64))
        self.assertEqual(table_params("bpe_custom", 64, vocab=700), 700 * 64)


class Headline(unittest.TestCase):
    def test_one_short_line_and_the_long_text_behind_it(self):
        for tokenizer in ("byte", "bpe_custom", "bpe_gpt2", "word"):
            line, details = both(5_000_000, tokenizer, tokens_seen=10_000_000)
            self.assertTrue(line.startswith("Expect: "))
            self.assertLess(len(line), 50)
            self.assertGreater(len(details), 150)
            self.assertNotIn(line, details)

    def test_each_tier_has_its_headline(self):
        self.assertEqual(both(100_000, "byte")[0], T["short_letters"])
        self.assertEqual(both(5_000_000, "bpe_custom", tokens_seen=10_000_000)[0], T["short_words"])
        self.assertEqual(both(20_000_000, "bpe_custom", tokens_seen=500_000_000)[0], T["short_sentences"])
        self.assertEqual(both(20_000_000, "bpe_custom")[0], T["short_sentences_text"])
        self.assertEqual(both(5_000_000, "byte")[0], T["short_words_text"])


class Pages(unittest.TestCase):
    def test_builder_line_follows_the_draft_word_list(self):
        s = D.read_spec(default_spec())
        lines = [design(dict(s, V=v))[1] for v in range(4)]
        self.assertEqual(len(set(lines)), 4)

    def test_pretraining_line_names_plain_text_and_waits_for_the_preview(self):
        spec = validate_spec(default_spec())
        self.assertEqual(version_expectation(spec, {}, 2), ("", ""))
        self.assertEqual(version_expectation(None, {"params": 1}, 2), ("", ""))
        preview = {"pieces": 2_000_000, "params": estimate(spec, 256)["total"]}
        line, text = version_expectation(spec, preview, 2)
        self.assertEqual(line, T["short_letters"])
        self.assertIn(T["letters"], text)
        self.assertIn(T["continues"], text)
        self.assertIn("Post-training round 1", text)

    def test_pretraining_counts_the_passes_over_the_text(self):
        spec = validate_spec(default_spec())
        spec["tokenizer"] = {"id": "bpe_custom", "config": {}}
        spec["dims"]["d_model"] = 512
        preview = {"pieces": 10_000_000, "params": 20_000_000, "vocab": 8192}
        self.assertIn(T["letters"], version_expectation(spec, preview, 0)[1])  # no passes: nothing seen
        self.assertIn(T["sentences"], version_expectation(spec, preview, 4)[1])


if __name__ == "__main__":
    unittest.main()

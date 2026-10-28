"""The Builder's pieces card: real word lists fitted on your text, the example they cut, and the counts. No Tk window."""
import unittest
from unittest import mock

from studio_ia import pieces as cut
from studio_ia.services import FakeServices
from studio_modules.tokenizer.bpe_fast import FastBPETokenizer
from studio_modules.tokenizer.bpe_gpt2 import GPT2Tokenizer, bytes_to_unicode

TEXT = ["The heron waited by the river. The heron waited by the river. A kingfisher flashed past the old mill. Version 3.12 is out."]


def tiny_gpt2():
    """GPT2Tokenizer with every byte plus two merges: ' t' and 'he', so ' the' is two pieces with the space on the first."""
    symbols = list(bytes_to_unicode().values())
    merges = [("Ġ", "t"), ("h", "e")]
    return GPT2Tokenizer({s: i for i, s in enumerate(symbols + ["Ġt", "he"])}, merges)


def joined(chips):
    return "".join(label for label, _ in chips).replace("␣", " ")


class Pieces(unittest.TestCase):
    def test_no_text_keeps_letters_and_says_nothing_else(self):
        lists = cut.word_lists([])
        letters, small, large = lists["tokenizers"]
        self.assertEqual((small, large), (None, None))
        self.assertEqual(lists["sentence"], cut.EXAMPLE)
        self.assertEqual(cut.counts(lists), [len(cut.EXAMPLE), None, None])

    def test_lists_are_the_real_tokenizers_and_cut_the_sentence_back_whole(self):
        lists = cut.word_lists(TEXT)
        letters, small, _ = lists["tokenizers"]
        self.assertIsInstance(small, FastBPETokenizer)
        sentence = lists["sentence"]
        for tok in (letters, small):
            chips = cut.chips(tok, sentence)
            self.assertEqual(joined(chips), sentence)
            self.assertEqual(len(chips), len(tok.encode(sentence)))
        self.assertGreater(len(cut.chips(letters, sentence)), len(cut.chips(small, sentence)))

    def test_sample_is_capped(self):
        with mock.patch.object(cut, "SAMPLE_CHARS", 40), mock.patch.object(cut.FastBPETokenizer, "fit", wraps=FastBPETokenizer.fit) as fit:
            cut.word_lists(["a" * 30, "b" * 30, "c" * 30])
        self.assertEqual([e["text"] for e in fit.call_args.args[0]], ["a" * 30, "b" * 10])

    def test_example_prefers_a_split_word(self):
        texts = ["".join([" The cat sat on the mat."] * 5 + [" The cat met an ocelot today."])]
        small = FastBPETokenizer.fit([{"text": texts[0]}], 7 + 256 + 10)
        self.assertEqual(cut.example_sentence(texts, small), " The cat sat on the mat.")
        self.assertEqual(cut.example_sentence(texts), " The cat sat on the mat.")

    def test_example_is_plain_prose_from_the_text(self):
        self.assertEqual(cut.example_sentence(["Version 3.12 is here. The heron waited by the river."]), " The heron waited by the river.")
        self.assertEqual(cut.example_sentence(["Run `tasks.py test` for it all.", "too short."]), cut.EXAMPLE)

    def test_gpt2_leading_space_shows(self):
        self.assertEqual(cut.chips(tiny_gpt2(), "the"), [("t", True), ("he", True)])
        self.assertEqual(cut.chips(tiny_gpt2(), " the"), [("␣t", True), ("he", True)])

    def test_part_of_a_character_shows_as_hex(self):
        letters = cut.word_lists([])["tokenizers"][0]
        self.assertEqual(cut.chips(letters, "é"), [("c3", True), ("a9", True)])

    def test_fake_services_fit_on_their_rows_and_download_nothing(self):
        lists = FakeServices().word_lists()
        self.assertIsNone(lists["tokenizers"][2])
        self.assertIsInstance(lists["tokenizers"][1], FastBPETokenizer)


if __name__ == "__main__":
    unittest.main()

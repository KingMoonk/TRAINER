"""The whole-word tokenizer: round trips on known words, unknown words, the vocabulary cap and checkpoint JSON."""
import json
import unittest

from studio_model import vocabulary_guess
from studio_modules.tokenizer.word import PIECE, WordTokenizer
from studio_registry import get
from studio_slots import OFFSET, UNK
from studio_spec import default_spec

TEXT = "The cat sat on the mat.\nThe dog sat on the log, and the cat ran!\n"
EXAMPLES = [{"text": TEXT}, {"user": "Where is the cat?", "assistant": "The cat sat on the mat."}]


class WordTokenizerTest(unittest.TestCase):
    def test_known_text_round_trips_exactly(self):
        tokenizer = WordTokenizer.fit(EXAMPLES, 8192)
        for text in (TEXT, "The cat sat on the mat.", "The dog sat on the log, and the cat ran!", ""):
            ids = tokenizer.encode(text)
            self.assertTrue(all(OFFSET <= i < tokenizer.vocab_size for i in ids))
            self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.vocab_size, len(tokenizer.words) + OFFSET)

    def test_a_word_is_one_token_with_its_space(self):
        tokenizer = WordTokenizer.fit(EXAMPLES, 8192)
        ids = tokenizer.encode("The cat sat")
        self.assertEqual([tokenizer.words[i - OFFSET] for i in ids], ["The", " cat", " sat"])

    def test_split_keeps_every_character(self):
        for text in ("héllo wörld 🍂!", "a  b\n\nc", "x_1 + y2 = 3.5", "'quoted' (text)"):
            self.assertEqual("".join(PIECE.findall(text)), text)

    def test_unseen_words_become_unknown_and_drop_out_of_the_decoded_text(self):
        tokenizer = WordTokenizer.fit(EXAMPLES, 8192)
        ids = tokenizer.encode("The elephant sat")
        self.assertEqual(ids[1], UNK)
        self.assertNotEqual(ids[0], UNK)
        self.assertEqual(tokenizer.decode(ids), "The sat")
        self.assertEqual(tokenizer.encode("zzz"), [UNK])

    def test_vocabulary_is_capped_and_keeps_the_most_frequent(self):
        tokenizer = WordTokenizer.fit([{"text": "a a a a b b b c c d e"}], OFFSET + 3)
        self.assertEqual(tokenizer.vocab_size, OFFSET + 3)
        counts = {}
        for piece in PIECE.findall("a a a a b b b c c d e"):
            counts[piece] = counts.get(piece, 0) + 1
        best = sorted(counts, key=lambda piece: (-counts[piece], piece))[:3]
        self.assertEqual(tokenizer.words, best)
        self.assertEqual(tokenizer.encode(" d"), [UNK])

    def test_a_small_corpus_makes_a_small_list(self):
        tokenizer = WordTokenizer.fit([{"text": "one two one"}], 8192)
        self.assertEqual(sorted(tokenizer.words), [" one", " two", "one"])
        self.assertEqual(tokenizer.vocab_size, OFFSET + 3)

    def test_fit_is_deterministic_whatever_the_order(self):
        self.assertEqual(WordTokenizer.fit(EXAMPLES, 20).serialize(), WordTokenizer.fit(EXAMPLES[::-1], 20).serialize())

    def test_save_and_load_round_trip(self):
        module = get("tokenizer", "word")
        fitted = module.fit(module.parse({"vocab_size": 263}), EXAMPLES)
        restored = module.restore(json.loads(json.dumps(fitted.serialize())))
        self.assertEqual(restored.words, fitted.words)
        self.assertEqual(restored.vocab_size, fitted.vocab_size)
        self.assertEqual(restored.encode(TEXT), fitted.encode(TEXT))
        self.assertEqual(restored.decode(fitted.encode(TEXT)), fitted.decode(fitted.encode(TEXT)))

    def test_restore_rejects_a_bad_checkpoint(self):
        for data in ({"kind": "word"}, {"kind": "word", "words": "abc"}, {"kind": "word", "words": ["a", "a"]},
                     {"kind": "word", "words": ["a", ""]}, {"kind": "word", "words": [1]}, {"kind": "bpe_custom", "words": ["a"]}):
            with self.assertRaises(ValueError):
                WordTokenizer.restore(data)

    def test_registered_with_a_data_sized_vocabulary(self):
        module = get("tokenizer", "word")
        self.assertIsNone(module.vocab(module.parse()))
        self.assertEqual(module.parse().vocab_size, 8192)
        spec = default_spec()
        self.assertEqual(vocabulary_guess({"architecture": spec}), 263)   # bytes: fixed
        spec["tokenizer"] = {"id": "word", "config": {"vocab_size": 1000}}
        self.assertEqual(vocabulary_guess({"architecture": spec}), 1000)
        spec["tokenizer"] = {"id": "word", "config": {}}
        self.assertEqual(vocabulary_guess({"architecture": spec}), 8192)

    def test_pieces_are_bytes_for_matching_rows_across_tokenizers(self):
        tokenizer = WordTokenizer.fit(EXAMPLES, 8192)
        self.assertEqual(tokenizer.pieces[tokenizer.encode(" cat")[0] - OFFSET], b" cat")


if __name__ == "__main__":
    unittest.main()

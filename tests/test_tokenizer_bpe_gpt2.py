"""Regression tests for GPT-2 and modern Byte-Level BPE tokenizers."""

import unittest
from studio_modules.tokenizer.bpe_fast import FastBPETokenizer
from studio_modules.tokenizer.bpe_gpt2 import GPT2Tokenizer, bytes_to_unicode


def tiny_gpt2():
    symbols = list(bytes_to_unicode().values())
    merges = [("Ġ", "t"), ("h", "e"), ("'", "s")]
    return GPT2Tokenizer({s: i for i, s in enumerate(symbols + ["Ġt", "he", "'s"])}, merges)


class TestGPT2AndFastBPE(unittest.TestCase):
    def test_gpt2_roundtrip_unicode_and_whitespace(self):
        tok = tiny_gpt2()
        for text in ("hello world", "it's", "a\x1cb", "a\x85b", "a  b", "abcABC漢字мир 0123456789 !@#$%"):
            with self.subTest(text=text):
                ids = tok.encode(text)
                self.assertEqual(tok.decode(ids), text)

    def test_fast_bpe_cl100k_pretokenization_splits_digits_and_contractions(self):
        corpus = [{"text": "It's 1234567890 and IT'S 9999! The price is $1000.\nNew line here."}] * 20
        tok = FastBPETokenizer.fit(corpus, vocab_size=300)
        pieces = [tok.decode_bytes([i]).decode("utf-8", errors="replace") for i in tok.encode("It's 1234567")]
        # Digits are split into at most 3-digit chunks by the pre-tokenizer regex
        for piece in pieces:
            digits = "".join(ch for ch in piece if ch.isdigit())
            self.assertLessEqual(len(digits), 3, f"Piece {piece!r} has more than 3 digits")
        self.assertEqual(tok.decode(tok.encode("It's 1234567")), "It's 1234567")


if __name__ == "__main__":
    unittest.main()

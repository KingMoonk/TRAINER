"""Regression tests for GPT-2 BPE tokenizer Unicode handling."""

import unittest
from studio_modules.tokenizer.bpe_gpt2 import category, pretokenize


class TestCategoryFunction(unittest.TestCase):
    """Test the category() function for correct Unicode classification."""

    def test_ascii_space_characters(self):
        """Test that ASCII whitespace is correctly identified."""
        for char in (" ", "\t", "\n", "\r", "\f", "\v"):
            with self.subTest(char=repr(char)):
                self.assertEqual(category(char), "space")

    def test_nel_character(self):
        """Test that NEL (Next Line, U+0085) is classified as space."""
        self.assertEqual(category("\x85"), "space")

    def test_control_characters_not_space(self):
        """Test that control chars like U+001C-U+001F are NOT classified as space."""
        for cp in range(0x001C, 0x0020):  # U+001C, U+001D, U+001E, U+001F
            char = chr(cp)
            with self.subTest(codepoint=f"U+{cp:04X}"):
                self.assertNotEqual(category(char), "space", f"{repr(char)} should not be space")
                self.assertEqual(category(char), "other")

    def test_unicode_space_characters(self):
        """Test that Unicode space separators (Zs) are correctly identified."""
        space_chars = [
            " ",  # EN QUAD (Zs)
            " ",  # EM QUAD (Zs)
            " ",  # EN SPACE (Zs)
            " ",  # EM SPACE (Zs)
            "　",  # IDEOGRAPHIC SPACE (Zs)
        ]
        for char in space_chars:
            with self.subTest(char=repr(char)):
                self.assertEqual(category(char), "space")

    def test_letters(self):
        """Test that letters are correctly identified."""
        letters = "abcABC漢字мир"
        for char in letters:
            with self.subTest(char=repr(char)):
                self.assertEqual(category(char), "letter")

    def test_numbers(self):
        """Test that digits are correctly identified."""
        digits = "0123456789"
        for char in digits:
            with self.subTest(char=repr(char)):
                self.assertEqual(category(char), "number")

    def test_other_characters(self):
        """Test that punctuation and other chars are classified as 'other'."""
        others = "!@#$%^&*()[]{}.,;:'\"-"
        for char in others:
            with self.subTest(char=repr(char)):
                self.assertEqual(category(char), "other")


class TestPretokenizeFunction(unittest.TestCase):
    """Test the pretokenize() function."""

    def test_simple_text(self):
        """Test basic tokenization."""
        result = pretokenize("hello world")
        # Consecutive letters are grouped, space + following letters are grouped
        self.assertEqual(result, ["hello", " world"])

    def test_contractions(self):
        """Test that common contractions are preserved."""
        result = pretokenize("it's")
        self.assertIn("'s", result)

    def test_control_char_separation(self):
        """Test that control characters are treated as separate tokens."""
        # U+001C should NOT be grouped with surrounding letters
        result = pretokenize("a\x1cb")
        self.assertIn("\x1c", result)
        self.assertNotEqual(result, ["a\x1cb"])

    def test_nel_space_handling(self):
        """Test that NEL (U+0085) is treated as space."""
        result = pretokenize("a\x85b")
        # NEL is space, so should be separate
        self.assertEqual(result, ["a", "\x85", "b"])

    def test_multiple_spaces(self):
        """Test handling of multiple consecutive spaces."""
        result = pretokenize("a  b")
        # Multiple spaces should be mostly preserved
        self.assertIn(" ", result)

    def test_multiple_control_chars(self):
        """Test that multiple control chars are grouped together."""
        result = pretokenize("a\x1c\x1db")
        # Control chars are "other" type, so they should group together
        self.assertIn("\x1c\x1d", result)

    def test_mixed_unicode(self):
        """Test with mixed ASCII and Unicode."""
        result = pretokenize("café")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_empty_string(self):
        """Test tokenization of empty string."""
        result = pretokenize("")
        self.assertEqual(result, [])

    def test_space_only(self):
        """Test tokenization of space-only strings."""
        result = pretokenize("   ")
        # All consecutive spaces are kept together with the logic in pretokenize
        self.assertEqual(result, ["   "])

    def test_spaces_at_word_boundaries(self):
        """Test optional leading spaces in pattern."""
        result = pretokenize(" hello world ")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_numbers_with_spaces(self):
        """Test digit tokenization."""
        result = pretokenize("123 456")
        self.assertIsInstance(result, list)
        self.assertTrue(all(len(p) > 0 for p in result))


if __name__ == "__main__":
    unittest.main()

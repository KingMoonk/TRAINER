"""Theme floor: text tokens stay readable on every surface and fonts never go below 12px. Window-free."""

import unittest
from unittest import mock

from studio_ia import theme
from studio_ia.theme import C

TEXT = ("text", "text2", "muted", "dim")
SURFACES = ("bg", "panel", "panel2")
WCAG_AA = 4.5


def _linear(channel):
    c = channel / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class TestTheme(unittest.TestCase):
    def test_contrast_helper_matches_wcag_extremes(self):
        self.assertAlmostEqual(contrast("#000000", "#ffffff"), 21.0, places=1)
        self.assertAlmostEqual(contrast("#777777", "#777777"), 1.0, places=2)

    def test_text_tokens_pass_wcag_aa_on_every_surface(self):
        for fg in TEXT:
            for bg in SURFACES:
                with self.subTest(fg=fg, bg=bg):
                    self.assertGreaterEqual(contrast(C[fg], C[bg]), WCAG_AA, f"{fg} {C[fg]} on {bg} {C[bg]}")

    def test_dim_stays_a_lower_tier_than_muted(self):
        for bg in SURFACES:
            self.assertLess(contrast(C["dim"], C[bg]), contrast(C["muted"], C[bg]))

    def test_font_never_below_the_minimum(self):
        fake = mock.Mock()
        fake.actual.return_value = "Test"
        with mock.patch.object(theme.tkfont, "nametofont", return_value=fake):
            for size in (8, 11, 12):
                self.assertEqual(theme.font(size), ("Test", -12))
            self.assertEqual(theme.font(11, bold=True), ("Test", -12, "bold"))
            self.assertEqual(theme.font(11, mono=True), ("Test", -12))
            self.assertEqual(theme.font(15), ("Test", -15))

    def test_text_boxes_share_the_padded_ui_font_style(self):
        fake = mock.Mock()
        fake.actual.return_value = "Test"
        with mock.patch.object(theme.tkfont, "nametofont", return_value=fake), \
                mock.patch.object(theme.tk, "Text") as text, mock.patch.object(theme.ttk, "Entry") as entry:
            theme.text_box("parent", height=3, state="disabled")
            theme.entry("parent", 14, width=8)
        kw = text.call_args.kwargs
        self.assertEqual((kw["padx"], kw["pady"]), theme.TEXT_PAD)
        self.assertEqual(kw["font"], ("Test", -15))
        self.assertEqual((kw["highlightthickness"], kw["highlightbackground"], kw["highlightcolor"]), (1, C["border"], C["violet"]))
        self.assertEqual((kw["height"], kw["state"]), (3, "disabled"))
        self.assertEqual(entry.call_args.kwargs, dict(font=("Test", -14), width=8))


if __name__ == "__main__":
    unittest.main()

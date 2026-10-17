"""Theme floor: the viewer's text tokens stay readable on every surface. Window-free."""

import unittest

from studio_viewer_ui import C

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


if __name__ == "__main__":
    unittest.main()

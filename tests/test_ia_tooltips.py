"""The hover-help helper's window-free logic: what text shows, where the box goes, which widgets carry help."""
import unittest

from studio_ia.widgets import TIP_DELAY, TIP_GAP, TIP_WRAP, tip, tip_place, tip_text, tips_in


class Fake:
    """Just enough of a Tk widget for tip(): bind() and winfo_children()."""

    def __init__(self, *children):
        self.children, self.bound = list(children), []

    def bind(self, seq, _fn, add=None):
        self.bound.append(seq)

    def winfo_children(self):
        return self.children


class TipTextTest(unittest.TestCase):
    def test_text_is_stripped(self):
        self.assertEqual(tip_text("  Why this matters.\n"), "Why this matters.")

    def test_blank_and_none_show_nothing(self):
        for empty in (None, "", "   \n"):
            self.assertEqual(tip_text(empty), "")

    def test_a_function_is_asked_each_time(self):
        state = {"n": 1}
        text = lambda: f"{state['n']} step"  # noqa: E731
        self.assertEqual(tip_text(text), "1 step")
        state["n"] = 2
        self.assertEqual(tip_text(text), "2 step")
        self.assertEqual(tip_text(lambda: None), "")

    def test_defaults_are_unobtrusive(self):
        self.assertEqual((TIP_DELAY, TIP_WRAP), (400, 360))


class TipPlaceTest(unittest.TestCase):
    screen = (1000, 800)

    def test_below_and_right_of_the_pointer(self):
        x, y = tip_place((100, 100), (200, 50), self.screen)
        self.assertEqual((x, y), (100 + TIP_GAP, 100 + TIP_GAP + 8))

    def test_flips_left_at_the_right_edge(self):
        x, _ = tip_place((900, 100), (200, 50), self.screen)
        self.assertEqual(x, 900 - TIP_GAP - 200)

    def test_flips_above_at_the_bottom_edge(self):
        _, y = tip_place((100, 780), (200, 50), self.screen)
        self.assertEqual(y, 780 - TIP_GAP - 50)

    def test_stays_on_screen_whatever_the_pointer(self):
        for pointer in ((0, 0), (999, 799), (500, 400), (999, 0), (0, 799)):
            for size in ((360, 200), (120, 30), (990, 790)):
                x, y = tip_place(pointer, size, self.screen)
                self.assertGreaterEqual(x, 8)
                self.assertGreaterEqual(y, 8)
                if size[0] < self.screen[0] - 16:
                    self.assertLessEqual(x + size[0], self.screen[0] - 8, (pointer, size))
                if size[1] < self.screen[1] - 16:
                    self.assertLessEqual(y + size[1], self.screen[1] - 8, (pointer, size))

    def test_a_box_wider_than_the_screen_pins_to_the_left_margin(self):
        x, _ = tip_place((500, 100), (2000, 50), self.screen)
        self.assertEqual(x, 8)


class TipsInTest(unittest.TestCase):
    def test_collects_every_tip_under_a_widget(self):
        leaf, other = Fake(), Fake()
        card = Fake(tip(leaf, "first"), Fake(tip(other, lambda: "second  ")))
        tip(card, "whole card")
        self.assertEqual(tips_in(card), ["whole card", "first", "second"])

    def test_tip_covers_the_children_present_now(self):
        part = Fake()
        whole = tip(Fake(part), "x")
        for widget in (whole, part):
            self.assertIn("<Enter>", widget.bound)
            self.assertIn("<Leave>", widget.bound)
            self.assertIn("<ButtonPress>", widget.bound)

    def test_tip_returns_its_widget_for_chaining(self):
        w = Fake()
        self.assertIs(tip(w, "t"), w)

    def test_a_widget_without_help_has_none(self):
        self.assertEqual(tips_in(Fake(Fake())), [])


if __name__ == "__main__":
    unittest.main()

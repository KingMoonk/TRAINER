"""Chat page on FakeServices: renders, compare toggles two panes, picking a version updates the header."""
import tkinter as tk
import unittest
from tkinter import ttk

from studio_spec import default_spec
from studio_ia.pages.chat import T
from studio_ia.services import FakeServices
from studio_ia.shell import Studio


def texts(widget):
    own = [str(widget.cget("text"))] if isinstance(widget, (tk.Label, ttk.Label, ttk.Button)) else []
    return own + [t for child in widget.winfo_children() for t in texts(child)]


class Fake(FakeServices):
    def __init__(self):
        super().__init__()
        self.rated, self.taught = [], []

    def best_version(self):
        return min(self.versions(), key=lambda v: v.scorecard["bpc"])

    def chat(self, version, messages, on_text, stop):
        for word in (f"v{version} ", "says ", "hi"):
            on_text(word)
        return ""

    def rate_pair(self, messages, chosen, rejected):
        self.rated.append((chosen, rejected))

    def teach_correction(self, version, messages, correction):
        self.taught.append((version, correction))


class ChatTest(unittest.TestCase):
    def setUp(self):
        self.fake = Fake()
        for _ in range(3):
            self.fake.save_version(default_spec(), "change")
        for v, bpc in zip(self.fake.versions(), (2.0, 1.5, 1.8)):
            v.best_checkpoint, v.scorecard, v.status = f"ck{v.number}", {"bpc": bpc}, "trained"
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)
        self.page = self.app.show("chat")
        self.app.update()

    def test_renders_header_follows_best(self):
        self.assertIn("Chat", texts(self.page))
        self.assertIn(T["head_using"].format(version="Version 2"), texts(self.page))
        self.assertIn(T["head_mode_follows"], texts(self.page))
        self.assertEqual(len(self.page.panes()), 1)

    def test_picking_version_updates_header(self):
        self.page.pick(3)
        self.app.update()
        self.assertIn(T["head_using"].format(version="Version 3"), texts(self.page))
        self.assertIn(T["head_mode_picked"].format(best_version="Version 2"), texts(self.page))

    def test_compare_toggles_two_panes_and_rates(self):
        self.page.toggle_compare()
        self.app.update()
        self.assertEqual(len(self.page.panes()), 2)
        self.assertIn(T["back_to_one"], texts(self.page))
        self.assertTrue(self.page.verdict_line.startswith("No verdict yet"))
        self.page.entry.insert(0, "hello")
        self.page.send()
        for _ in range(100):
            self.app.update()
            if not self.page.pending:
                break
        self.assertEqual(self.page.replies, {0: "v2 says hi", 1: "v1 says hi"})
        self.page.better(1)
        self.assertEqual(self.fake.rated[0][0]["version"], 1)
        self.page.toggle_compare()
        self.assertEqual(len(self.page.panes()), 1)

    def test_teach_correction(self):
        self.page.correction.insert(0, "Jupiter")
        self.page.teach()
        self.assertEqual(self.fake.taught, [(2, "Jupiter")])


if __name__ == "__main__":
    unittest.main()

"""Builder page on FakeServices: renders, edits update the tray, save shows the warm-start sentence."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import tkinter as tk
import unittest
from tkinter import ttk

from studio_spec import default_spec
from studio_versions import spec_diff
from studio_warmstart import NOTES
from studio_ia.pages import builder as B
from studio_ia.services import FakeServices
from studio_ia.shell import Studio


def texts(widget):
    own = [str(widget.cget("text"))] if isinstance(widget, (tk.Label, ttk.Label)) else []
    return own + [t for child in widget.winfo_children() for t in texts(child)]


class BuilderTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServices()
        self.fake.save_version(default_spec(), "first design")
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)
        self.page = self.app.show("builder")
        self.app.update()

    def test_renders_and_no_changes(self):
        self.assertIn("Build your model", texts(self.page))
        self.assertEqual(self.page.tray_kicker, B.T["no_changes"])
        self.assertEqual(self.page.expect_texts, [B.T["change_something"]])
        self.assertEqual(sorted(self.page.links), sorted(r for r, _ in B.LINKS))

    def test_edit_marks_not_saved_with_chips(self):
        self.page.edit(L=4, H=8)
        self.app.update()
        self.assertEqual(self.page.chip_texts, ["more blocks (2 → 4)", "look-back heads 4 → 8"])
        self.assertEqual(self.page.tray_kicker, B.T["not_saved"])
        self.assertIn("Save as version 2", self.page.actions)
        self.assertEqual(len(self.fake.versions()), 1)  # draft stays on the page
        self.assertEqual(sum("Different look-back heads" in t for t in self.page.expect_texts), 1)
        self.page.discard()
        self.assertEqual(self.page.tray_kicker, B.T["no_changes"])

    def test_save_shows_warm_start_sentence(self):
        self.page.edit(L=4)
        self.page.save()
        self.app.update()
        self.assertEqual(len(self.fake.versions()), 2)
        self.assertEqual(self.fake.versions()[1].change, "more blocks (2 → 4)")
        self.assertEqual(self.page.tray_kicker, B.T["saved_text_needs_catchup"].format(new_version="version 2"))
        self.assertIn(NOTES["same"], self.page.tray_text)
        self.page.queue_it(False)
        self.assertEqual(len(self.fake.queue().up_next), 1)
        self.assertEqual(self.page.phase, "queued")

    def test_word_list_change_needs_scratch(self):
        self.page.edit(V=2)
        self.page.save()
        self.assertIn("FROM SCRATCH", self.page.tray_kicker.upper())
        self.assertIn(NOTES["scratch"], self.page.tray_text)
        self.assertNotIn(B.T["start_from_scratch"], self.page.actions)

    def test_cost_say_matches_formulas(self):
        self.page.edit(L=4)
        d, t = 64, 256
        a, b = (B.cfg(dict(L=n, H=4, W=1, V=0, d=d, T=t)) for n in (2, 4))
        self.assertEqual(self.page.cost_say[B.T["size_label"]], B.say(B.P(a), B.P(b), "bigger", "smaller"))
        self.assertEqual(self.page.cost_say[B.T["time_per_step"]], f"about {B.C_(b) / B.C_(a):.1f}× slower")
        self.assertEqual(self.page.cost_say[B.T["memory_while_training"]], f"about {B.TM(b) / B.TM(a):.1f}× more")
        self.page.discard()
        self.assertEqual(set(self.page.cost_say.values()), {"about the same"})

    def test_last_change_sentence_only_with_history(self):
        self.page.edit(L=4)
        self.assertFalse(any(t.startswith("Last time") for t in self.page.expect_texts))
        spec4 = B.write_spec(default_spec(), dict(self.page.cur, L=4))
        self.assertEqual(spec_diff(default_spec(), spec4), ["more blocks (2 → 4)"])
        self.fake.save_version(spec4, "more blocks (2 → 4)")
        self.page.refresh()
        self.page.edit(L=6)
        self.assertTrue(any(t.startswith("Last time blocks changed (2 → 4, version 1 → 2)") for t in self.page.expect_texts))


if __name__ == "__main__":
    unittest.main()

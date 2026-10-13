"""Compare runs and Quick test render on FakeServices; the queue label counts runs and sweeps; modes change the note."""
import copy
import unittest
import tkinter as tk
from tkinter import ttk

from studio_ia.copy import COMPARE as T
from studio_ia.services import FakeServices
from studio_ia.shell import Studio
from studio_spec import default_spec, validate_spec


def texts(widget):
    own = [str(widget.cget("text"))] if isinstance(widget, (tk.Label, ttk.Label, ttk.Button)) else []
    return own + [t for child in widget.winfo_children() for t in texts(child)]


class CompareGui(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServices()
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)

    def test_both_pages_render(self):
        for route, title in (("compare", "Compare runs"), ("quicktest", "Quick test")):
            page = self.app.show(route)
            self.app.update()
            self.assertIn(title, texts(page))

    def test_queue_label_counts_runs_and_sweeps_and_enqueues(self):
        page = self.app.show("compare")
        for i in range(3):
            v = copy.deepcopy(page.base)
            v["block"]["channel"]["config"]["d_ff"] = 32 * (i + 1)
            page.add_variant(spec=validate_spec(v))
        self.app.update()
        self.assertEqual(page.queue_btn.cget("text"), T["queue_runs"].format(count=12, sweeps=9))
        page.retune.flip()
        self.assertEqual(page.queue_btn.cget("text"), T["queue_runs_no_sweep"].format(count=12))
        page.seeds.pick("5")
        self.assertEqual(page.queue_btn.cget("text"), T["queue_runs_no_sweep"].format(count=20))
        page.queue_it()
        self.assertEqual(self.fake.queue().up_next, [])  # a guess comes first
        page.guess.pick(T["better"])
        page.queue_it()
        job = self.fake.queue().up_next[0]
        self.assertEqual(job.kind, "compare")
        self.assertEqual(job.params["experiment"]["seeds"], 5)

    def test_mode_changes_the_note(self):
        page = self.app.show("compare")
        notes = set()
        for mode, note in (("params_mode", "params_note"), ("compute_mode", "compute_note"), ("tokens_mode", "tokens_note"), ("time_mode", "time_note")):
            page.modes.pick(T[mode])
            self.app.update()
            self.assertEqual(page.mode_note.cget("text"), T[note])
            notes.add(page.mode_note.cget("text"))
        self.assertEqual(len(notes), 4)

    def test_quicktest_needs_a_draft_and_a_guess(self):
        page = self.app.show("quicktest")
        page.start()
        self.assertIn("designer", page.status.cget("text"))
        draft = copy.deepcopy(validate_spec(default_spec()))
        draft["dims"]["n_layer"] = 3
        self.fake.draft_spec = lambda: draft
        page.refresh()
        page.start()
        self.assertIn("guess", page.status.cget("text"))
        page.guess.pick("Lower")
        page.start()
        job = self.fake.queue().up_next[0]
        self.assertEqual((job.kind, job.params["minutes"], job.params["shrink"]), ("quicktest", 3, False))


if __name__ == "__main__":
    unittest.main()

"""Compare runs and Quick test render on FakeServices; the queue label counts runs and sweeps; modes change the note."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import copy
import unittest
import tkinter as tk
from tkinter import ttk

from studio_ia.copy import COMPARE as T
from studio_ia.copy import QUICKTEST as T_QT
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
        self.fake.data = lambda kind: {"kind": kind, "count": 1, "sets": [{"id": "d1", "name": "books", "count": 1}]}

    def variant(self, page, ff=32):
        v = copy.deepcopy(page.base)
        v["block"]["channel"]["config"]["d_ff"] = ff
        return validate_spec(v)

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
        self.assertEqual(job.params["datasets"], ["d1"])

    def test_no_text_means_no_compare_job(self):
        self.fake.data = lambda kind: {"kind": kind, "count": 0, "sets": []}
        page = self.app.show("compare")
        page.add_variant(spec=self.variant(page))
        page.guess.pick(T["better"])
        page.queue_it()
        self.assertEqual(self.fake.queue().up_next, [])
        self.assertIn("no plain text", page.status.cget("text"))

    def test_rows_show_raw_change_and_can_be_removed_or_reopened(self):
        page = self.app.show("compare")
        spec = self.variant(page)
        page.add_variant(spec=spec)
        self.app.update()
        self.assertTrue(any(t.startswith("block.channel.config.d_ff:") for t in texts(page.table)))
        self.assertIn("–", texts(page.table))  # no results yet: a dash, not a made-up number
        page.reopen(1)
        self.assertEqual(self.fake.draft_spec(), spec)
        self.assertEqual(self.app.route, "builder")
        page = self.app.show("compare")
        page.remove_variant(0)
        self.assertEqual(len(page.plan["variants"]), 1)

    def test_results_show_spread_speed_and_the_guess(self):
        page = self.app.show("compare")
        page.add_variant(spec=self.variant(page))
        page.guess.pick(T["better"])
        page.done_guess = T["better"]
        rows = [{"variant": 0, "seeds": [7, 42], "bpc": [3.0, 3.1], "mean": 3.05, "min": 3.0, "max": 3.1, "spread": .1, "delta": None,
                 "inside_spread": None, "lr": 1e-3, "pieces_per_s": 900., "peak_mb": 50., "curve": [[1, 4.], [2, 3.05]]},
                {"variant": 1, "seeds": [7, 42], "bpc": [2.5, 2.6], "mean": 2.55, "min": 2.5, "max": 2.6, "spread": .1, "delta": -.5,
                 "inside_spread": False, "lr": 1e-3, "pieces_per_s": 800., "peak_mb": 60., "curve": [[1, 4.], [2, 2.55]]}]
        page.results = {"variants": rows, "references": {"uniform": 8., "unigram": 4.}}
        page.update_plan()
        self.app.update()
        shown = texts(page.table)
        self.assertIn("2.550 · spread 0.100\n800 pieces/s · 60 MB peak", shown)
        self.assertIn("-0.500 · beyond the spread", shown)
        self.assertIn("You guessed better; B came out better (beyond the spread).", page.finished_text())

    def test_mode_changes_the_note(self):
        page = self.app.show("compare")
        notes = set()
        for mode, note in (("params_mode", "params_note"), ("compute_mode", "compute_note"), ("tokens_mode", "tokens_note"), ("time_mode", "time_note")):
            page.modes.pick(T[mode])
            self.app.update()
            self.assertEqual(page.mode_note.cget("text"), T[note])
            notes.add(page.mode_note.cget("text"))
        self.assertEqual(len(notes), 4)

    def test_x_axis_choice_falls_back_to_steps_when_runs_lack_tokens_or_time(self):
        page = self.app.show("compare")
        page.add_variant(spec=self.variant(page))
        rows = [{"variant": n, "seeds": [7], "bpc": [3.], "mean": 3., "min": 3., "max": 3., "spread": 0., "delta": None, "inside_spread": None,
                 "lr": 1e-3, "pieces_per_s": 900., "peak_mb": 50., "curve": curve} for n, curve in ((0, [[1, 4., 100, 30.], [2, 3., 200, 90.]]), (1, [[1, 4.], [2, 3.]]))]
        page.results = {"variants": rows, "references": {}}
        page.modes.pick(T["params_mode"])
        page.axes.pick(T["axis_time"])
        self.app.update()
        self.assertIn(T["x_axis_steps"], page.curves_title.cget("text"))
        self.assertIn("did not record minutes", page.axis_note.cget("text"))
        rows[1]["curve"] = [[1, 4., 150, 40.], [2, 3., 250, 80.]]
        page.draw_curves()
        self.assertIn(T["x_axis_time"], page.curves_title.cget("text"))
        self.assertEqual(page.axis_note.cget("text"), "")

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

    def test_quicktest_reads_the_designers_staged_draft_and_needs_text_for_a_slice(self):
        draft = copy.deepcopy(validate_spec(default_spec()))
        draft["dims"]["n_layer"] = 3
        self.fake.stage_draft(draft)
        page = self.app.show("quicktest")
        self.assertIn("more blocks (2 → 3)", texts(page.chip_row))
        page.guess.pick("Same")
        self.fake.data = lambda kind: {"kind": kind, "count": 0, "sets": []}
        page.data.pick(T_QT["slice_your_data"])
        page.start()
        self.assertEqual(self.fake.queue().up_next, [])
        self.assertIn("no plain text", page.status.cget("text"))

    def test_quicktest_result_failed_check_and_discard(self):
        draft = copy.deepcopy(validate_spec(default_spec()))
        draft["dims"]["n_layer"] = 3
        self.fake.stage_draft(draft)
        page = self.app.show("quicktest")
        page.result = {"minutes": 3, "broken": "memorises_one_batch", "verdict": None,
                       "checks": [{"id": i, "ok": None, "detail": "not run"} for i in ("starts_near_uniform", "finite", "error_falls")]
                       + [{"id": "memorises_one_batch", "ok": False, "detail": "5.00 -> 4.90"}]}
        page.show_result()
        self.app.update()
        self.assertTrue(any("Look at the parts you changed" in t for t in texts(page.result_card)))
        self.assertIn("✗", texts(page.result_card))
        page.discard()
        self.assertIsNone(self.fake.draft_spec())
        self.assertEqual(self.app.route, "builder")


if __name__ == "__main__":
    unittest.main()

"""Builder page on FakeServices: renders, edits update the tray, save shows the warm-start sentence."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import tkinter as tk
import unittest
from tkinter import ttk

from studio_data_kinds import DataSummary
from studio_spec import default_spec
from studio_versions import spec_diff
from studio_warmstart import NOTES
from studio_ia.copy import EXPECTATION
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
        self.fake.versions()[0].best_checkpoint = "/tmp/ckpt/best.npz"  # trained, so a catch-up can start from it
        self.has_text = lambda kind, query="": DataSummary(kind, 1, sets=[{"id": "d1", "name": "books", "count": 1}])
        self.fake.data = self.has_text
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
        self.assertIn(EXPECTATION["letters"], self.page.guide_text)
        self.assertIn(EXPECTATION["letters"], "\n".join(texts(self.page)))
        self.assertEqual(sorted(self.page.links), sorted(r for r, _ in B.LINKS))

    def test_first_design_can_be_saved_without_a_change(self):
        fake = FakeServices()  # nothing saved yet
        fake.data = self.has_text
        app = Studio(fake)
        self.addCleanup(app.destroy)
        page = app.show("builder")
        app.update()
        self.assertEqual(page.tray_kicker, B.T["first_save"])
        self.assertIn("Save as version 1", page.actions)
        page.save()
        app.update()
        self.assertEqual([(v.number, v.change) for v in fake.versions()], [(1, "first design")])

    def test_nothing_trained_shows_one_line_instead_of_a_chart_and_rows(self):
        fake = FakeServices()  # nothing saved yet
        fake.data = self.has_text
        app = Studio(fake)
        self.addCleanup(app.destroy)
        page = app.show("builder")
        app.update()
        shown = texts(page)
        self.assertIsNone(page.plot)
        self.assertIsNone(page.ask_btn)
        self.assertEqual(page.no_curve, B.T["no_curve_first"])
        self.assertEqual(sum(t.startswith("No error curve yet") for t in shown), 1)
        self.assertIn(B.T["nothing_to_ask_first"], shown)
        self.assertNotIn(B.T["example_prompt"], shown)

    def test_one_trained_version_answers_and_the_other_says_why_not(self):
        self.assertIsNone(self.page.plot)  # trained but no run curve on the fake
        self.assertEqual(sorted(self.page.answer_labels), [1, 2])
        self.assertEqual(self.page.answer_labels[1].cget("text"), B.T["its_answer_appears"])
        self.assertEqual(self.page.answer_labels[2].cget("text"), B.T["compare_later"])
        self.assertIsNotNone(self.page.ask_btn)

    def test_edit_marks_not_saved_with_chips(self):
        self.page.edit(L=4, H=8)
        self.app.update()
        self.assertEqual(self.page.chip_texts, ["more blocks (2 → 4)", "look-back heads 4 → 8"])
        self.assertEqual(self.page.tray_kicker, B.T["not_saved"])
        self.assertIn("Save as version 2", self.page.actions)
        self.assertEqual(len(self.fake.versions()), 1)  # draft stays on the page
        self.assertEqual(self.fake.draft_spec()["dims"]["n_layer"], 4)  # but Quick test and Compare runs can see it
        self.assertEqual(sum("Different look-back heads" in t for t in self.page.expect_texts), 1)
        self.page.discard()
        self.assertEqual(self.page.tray_kicker, B.T["no_changes"])
        self.assertIsNone(self.fake.draft_spec())

    def test_save_shows_warm_start_sentence(self):
        self.page.edit(L=4)
        self.page.save()
        self.app.update()
        self.assertEqual(len(self.fake.versions()), 2)
        self.assertEqual(self.fake.versions()[1].change, "more blocks (2 → 4)")
        self.assertEqual(self.page.tray_kicker, B.T["saved_text_needs_catchup"].format(new_version="version 2"))
        self.assertIn(NOTES["same"], self.page.tray_text)
        self.assertIsNone(self.fake.draft_spec())
        self.page.queue_it(False)
        job = self.fake.queue().up_next[0]
        self.assertEqual((job.kind, job.params["mode"], job.params["version"], job.params["init_from"], job.params["datasets"]),
                         ("pretrain", "catch_up", 2, "/tmp/ckpt/best.npz", ["d1"]))
        self.assertEqual(self.page.phase, "queued")

    def test_no_text_means_no_job(self):
        self.fake.data = lambda kind, query="": DataSummary(kind)
        self.page.edit(L=4)
        self.page.save()
        self.page.queue_it(False)
        self.assertEqual(self.fake.queue().up_next, [])
        self.assertEqual(self.page.phase, "saved")
        self.assertIn("no plain text", self.page.tray_text)

    def test_untrained_parent_trains_from_scratch(self):
        self.fake.versions()[0].best_checkpoint = None
        self.page.refresh()
        self.page.edit(L=4)
        self.page.save()
        self.assertIn("FROM SCRATCH", self.page.tray_kicker.upper())
        self.page.queue_it(False)
        self.assertEqual(self.fake.queue().up_next[0].params["mode"], "from_scratch")

    def test_cost_panel_says_whether_the_weights_carry_over(self):
        self.page.edit(L=4)
        self.assertEqual(self.page.warm["kind"], "same")
        self.assertIn(B.WARM["same"][0], texts(self.page))
        self.page.edit(V=2)
        self.assertFalse(self.page.warm["carries"])
        self.page.discard()
        self.assertIsNone(self.page.warm)

    def test_chart_has_no_example_shape_and_no_invented_numbers(self):
        self.assertNotIn("Example shape", texts(self.page))
        self.assertEqual(self.page.plot_args["smooth"], [])
        self.assertTrue(any(t.startswith("No error curve yet") for t in texts(self.page)))

    def test_a_design_reopened_elsewhere_lands_in_the_picture(self):
        spec = B.write_spec(default_spec(), dict(self.page.cur, L=6))
        self.fake.stage_draft(spec)
        self.page.refresh()
        self.assertEqual(self.page.chip_texts, ["more blocks (2 → 6)"])
        self.fake.stage_draft(None)  # Quick test: discard idea
        self.page.refresh()
        self.assertEqual(self.page.chip_texts, [])

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

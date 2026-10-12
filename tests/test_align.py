"""Align page: vote, undo, preference browser, rubric editor (skipped when no display is available)."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import tkinter as tk

from studio_core import Workspace
from studio_gui import Studio
from studio_model import train
from studio_align import TAB_AUTO, TAB_COMPARE, TAB_RATE, TAB_SAVED

EXAMPLES = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")] * 2
TINY = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4,
            steps=20, learning_rate=.004, warmup=0, validation=0, eval_every=10, objective="assistant")


class AlignPage(unittest.TestCase):
    def test_vote_undo_browser_and_rubric(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            dataset = workspace.create_dataset("Align test")
            workspace.add(dataset, EXAMPLES)
            train(TINY, EXAMPLES, Path(directory) / "base")
            checkpoint = str(Path(directory) / "base" / "last.npz")
            try:
                app = Studio(workspace)
            except tk.TclError as exc:
                if "display" in str(exc).lower():
                    self.skipTest("Tk display is unavailable")
                raise
            app.withdraw()

            def settle():
                deadline = time.monotonic() + 30
                while app.job_kind and time.monotonic() < deadline:
                    app.update()
                    time.sleep(.01)
                self.assertIsNone(app.job_kind, "Worker did not complete")
                app.update()
            try:
                with patch("studio_gui.messagebox.showerror", side_effect=lambda *args, **kwargs: self.fail(str(args))), \
                        patch("studio_align.messagebox.showerror", side_effect=lambda *args, **kwargs: self.fail(str(args))), \
                        patch("studio_align.messagebox.askyesno", return_value=True):
                    app.refresh_datasets(dataset)
                    app.inference_device.set("cpu")
                    app.align_checkpoint.set(checkpoint)
                    app.show_page("align")
                    app.update()
                    self.assertEqual([app.step_widgets[i][1].cget("text") for i in range(4)],
                                     ["Check the model", "Vote", "Train", "Compare"])
                    app.align_tabs.select(TAB_RATE)
                    app.update()
                    # a real comparison is generated from the checkpoint
                    app.next_comparison(None)
                    settle()
                    self.assertIsNotNone(app.comparison)
                    self.assertEqual(len(app.comparison["replies"]), 2)
                    self.assertIn("Aim for at least 20", app.pref_hint.get())

                    def vote(winner, replies=("YES", "yes")):
                        app.comparison = {**app.comparison, "replies": replies}
                        app.vote(winner)
                        settle()
                    vote("A")
                    vote("B", ("cat", "CAT"))
                    vote("A", ("dog", "DOG"))
                    self.assertEqual(len(app.preferences.all()), 3)
                    self.assertEqual(len(app.vote_log), 3)
                    self.assertIn("3 of your votes", app.pref_count.get())
                    # undo restores the previous prompt and removes only the last pair
                    app.undo_vote()
                    self.assertEqual(len(app.preferences.all()), 2)
                    self.assertEqual(app.comparison["replies"], ("dog", "DOG"))
                    self.assertEqual(app.reply_boxes[0].get("1.0", "end-1c"), "dog")
                    vote("B", ("dog", "DOG"))
                    self.assertEqual(app.preferences.all()[-1]["chosen"], "DOG")
                    # the hint flips once enough pairs exist
                    for i in range(17):
                        app.preferences.add([{"role": "user", "content": f"q{i}"}], f"good {i}", f"bad {i}", "teacher:mock", kind="comparison", scores=[8, 3], reason="r")
                    app.refresh_pref_stats()
                    self.assertIn("Enough to start training", app.pref_hint.get())

                    # preference browser
                    app.align_tabs.select(TAB_SAVED)
                    app.update()
                    self.assertEqual(len(app.pref_tree.get_children()), 20)
                    app.pref_filter.set("Your votes")
                    app.refresh_pref_browser()
                    self.assertEqual(len(app.pref_tree.get_children()), 3)
                    first = app.pref_tree.get_children()[0]
                    app.pref_tree.selection_set(first)
                    app.update()
                    self.assertEqual(app.pref_chosen.get("1.0", "end-1c"), "DOG")
                    app.pref_chosen.delete("1.0", "end")
                    app.pref_chosen.insert("1.0", "Dog!")
                    app.save_pref_edit()
                    self.assertEqual(app.preferences.get(first)["chosen"], "Dog!")
                    app.flip_pref()
                    self.assertEqual(app.preferences.get(first)["rejected"], "Dog!")
                    exported = Path(directory) / "export.jsonl"
                    app.export_preferences(str(exported))
                    self.assertEqual(len(exported.read_text(encoding="utf-8").splitlines()), 1)
                    app.pref_tree.selection_set(app.pref_tree.get_children())
                    app.delete_prefs(confirm=False)
                    self.assertEqual(len(app.preferences.all()), 17)
                    self.assertEqual(len(app.pref_tree.get_children()), 0)
                    app.import_preferences(str(exported))
                    self.assertEqual(len(app.preferences.all()), 18)
                    app.import_preferences(str(exported))
                    self.assertEqual(len(app.preferences.all()), 18)  # identical content is not imported twice

                    # rubric editor persists to the workspace settings
                    app.align_tabs.select(TAB_AUTO)
                    before = len(app.rubric)
                    app.rule_text.set("Answers in a single word")
                    app.rule_weight.set("4")
                    app.add_rule()
                    self.assertEqual(len(app.rubric_tree.get_children()), before + 1)
                    stored = json.loads((Path(directory) / "data" / "rlhf_settings.json").read_text(encoding="utf-8"))
                    self.assertEqual(stored["rubric"][-1], {"text": "Answers in a single word", "weight": 4})
                    app.rubric_tree.selection_set(str(before))
                    app.remove_rule()
                    self.assertEqual(len(app.rubric), before)
                    self.assertEqual(app.make_teacher().rubric, app.rubric)
                    cfg = app.auto_config()
                    self.assertAlmostEqual(cfg["win_margin"], .05)
                    self.assertEqual(cfg["candidates"], 4)
                    app.dpo_config()["loss_type"]
                    app.align_tabs.select(TAB_COMPARE)
                    app.update()
            finally:
                app.stop.set()
                app.pause.clear()
                if app.worker:
                    app.worker.join(timeout=20)
                app.destroy()


if __name__ == "__main__":
    unittest.main()

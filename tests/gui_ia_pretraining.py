"""Pretraining and Pretraining-run pages on FakeServices."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import time
import unittest
import tkinter as tk
from tkinter import ttk

from studio_data_kinds import DataSummary
from studio_spec import default_spec
from studio_ia.copy import EXPECTATION, PRETRAINING
from studio_ia.services import FakeServices, Job, QueueState, Version
from studio_ia.shell import Studio
from studio_ia.widgets import tips_in


def texts(widget):
    own = [str(widget.cget("text"))] if isinstance(widget, (tk.Label, ttk.Label, ttk.Button)) else []
    return own + [t for child in widget.winfo_children() for t in texts(child)]


class PretrainingTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServices()
        self.fake.data = lambda kind, query="": DataSummary(kind, 3, sets=[{"id": "d1", "name": "Short stories", "count": 3}])
        self.fake._versions = [Version(4, spec=default_spec(), best_checkpoint="v4.npz", status="trained"),
                               Version(5, change="6 blocks (was 4)", parent=4, spec=default_spec(), status="saved")]
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)

    def test_pretraining_page(self):
        page = self.app.show("pretraining")
        self.app.update()
        t = "\n".join(texts(page))
        self.assertIn("Version 5 · from the Network designer", t)
        tips = "\n".join(tips_in(page))
        self.assertIn("2.00 word pieces per parameter", tips)  # 2,000,000 pieces / 1,000,000 params; the rule of thumb is a tooltip
        self.assertIn("rule of thumb", tips)
        self.assertNotIn("rule of thumb", t)
        self.assertIn("about 10.0× more text", t)
        self.assertIn(EXPECTATION["short_letters"], t)  # 2,000,000 pieces over two passes is little text
        self.assertNotIn(EXPECTATION["continues"], t)  # the detail is behind the ⓘ
        self.assertIn(EXPECTATION["letters"], page.expect_details)
        self.assertIn(EXPECTATION["continues"], page.expect_details)
        self.assertGreater(len(page.grid.find_all()), 100)  # cells of the real first rows

    def test_collection_buttons_call_the_services(self):
        self.fake._collection = [{"id": "tinystories", "name": "TinyStories", "size": "64 MB", "records": "77,941 stories",
                                  "state": "not downloaded", "completed": 0, "total": 0, "error": ""},
                                 {"id": "wikitext2", "name": "WikiText-2", "size": "6 MB", "records": "15,488 paragraphs",
                                  "state": "downloaded", "completed": 0, "total": 0, "error": ""}]
        page = self.app.show("pretraining")
        self.app.update()
        t = "\n".join(texts(page))
        self.assertIn("Download and add all", t)
        self.assertIn("Add to my sources", t)
        page.fetch(None, "download")
        page.fetch("wikitext2", "import")
        self.assertEqual(self.fake.calls, [("download_sources", None, "text", True), ("import_source", "wikitext2")])
        self.fake._collection[0]["error"] = "no network"
        page.on_event("download_failed", {"id": "tinystories", "error": "no network"})
        self.assertIn("Could not finish: no network.", "\n".join(texts(page)))

    def test_download_progress_updates_the_status_line_in_place(self):
        self.fake._collection = [{"id": "tinystories", "name": "TinyStories", "size": "64 MB", "records": "77,941 stories",
                                  "state": "downloading", "completed": 0, "total": 0, "error": ""}]
        page = self.app.show("pretraining")
        self.app.update()
        page.throttle.interval = .3
        label = page.status_labels["tinystories"]
        page.on_event("download", {"id": "tinystories", "text": "Downloading · 1 MB"})
        page.on_event("download", {"id": "tinystories", "text": "Downloading · 2 MB"})
        self.assertEqual(label.cget("text"), "Downloading · 1 MB")  # the first change is drawn at once
        end = time.monotonic() + .6
        while time.monotonic() < end:
            self.app.update()
        self.assertEqual(label.cget("text"), "Downloading · 2 MB")  # the burst's last one a tick later
        self.assertIs(page.status_labels["tinystories"], label)  # the rows were not rebuilt
        page.paint_collection()
        self.assertIs(page.status_labels["tinystories"], label)  # nor are they when nothing about them changed

    def test_opens_at_once_and_waits_only_to_add_to_the_queue(self):
        self.fake.ready = lambda work: False  # a large store: the set list is read in the background
        page = self.app.show("pretraining")
        self.assertEqual(self.app.loader.winfo_manager(), "")  # no panel over the page
        self.assertIn(PRETRAINING["reading_text"], texts(page))
        self.assertTrue(page.add_btn.instate(["disabled"]))
        deadline = time.monotonic() + 10
        while page._warming and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.02)
        t = texts(page)
        self.assertIn("Short stories", t)
        self.assertNotIn(PRETRAINING["reading_text"], t)
        self.assertFalse(page.add_btn.instate(["disabled"]))

    def test_add_to_queue_defaults_to_catch_up(self):
        page = self.app.show("pretraining")
        page.add_to_queue()
        job = self.fake.queue().up_next[-1]
        self.assertEqual((job.kind, job.params["mode"], job.params["init_from"]), ("pretrain", "catch_up", "v4.npz"))
        self.assertEqual(job.params["datasets"], ["d1"])

    def test_word_list_change_forces_scratch(self):
        self.fake._versions[1].spec["tokenizer"] = {"id": "bpe_gpt2", "config": {}}  # differs from version 4's byte word list
        page = self.app.show("pretraining")
        page.pick_mode("catch_up")
        page.add_to_queue()
        self.assertEqual(self.fake.queue().up_next[-1].params["mode"], "from_scratch")

    def test_run_page_status_and_pause(self):
        job = Job("j", "pretrain", "Version 5 catch-up", params={"output_dir": "x", "config": {"batch_size": 16}})
        self.fake.set_queue(QueueState(job=job, step=40, total=200, phase="steady", eta_s=600))
        steps = [(i, 5 - i / 100) for i in range(40)]
        hist = [{"loss": 3 - i / 20, "val_loss": 2 + i / 10} for i in range(6)]
        self.fake.run_curve = lambda d: {"steps": steps, "smooth": steps, "heldout": [(30, 4.7)], "total": 200, "history": hist}
        page = self.app.show("training_run")
        self.app.update()
        t = "\n".join(texts(page))
        self.assertIn("Memorising", t)
        self.assertIn("the first sample appears at the next checkpoint", t)
        self.assertIn("Step 40 of 200", t)
        page.pause_btn.invoke()
        self.assertTrue(self.fake.queue().paused)
        self.assertTrue(page.chip.winfo_ismapped())

    def test_run_page_controls_follow_the_job(self):
        self.fake.set_queue(QueueState())
        self.fake.run_curve = lambda d: {}
        page = self.app.show("training_run")
        self.app.update()
        self.assertTrue(page.pause_btn.instate(["disabled"]) and page.stop_btn.instate(["disabled"]))
        self.assertFalse(page.chip.winfo_ismapped())
        self.fake.set_queue(QueueState(paused=True))
        page.refresh()
        self.assertFalse(page.pause_btn.instate(["disabled"]))  # Resume stays available
        self.assertTrue(page.stop_btn.instate(["disabled"]))


if __name__ == "__main__":
    unittest.main()

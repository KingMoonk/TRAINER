"""Regression tests for the Align page: stopped checks, duplicate votes, Ollama lookup off the UI thread, stopped training."""
import shutil
import tempfile
import time
import unittest
import tkinter as tk
from unittest.mock import patch

from studio_core import Workspace
from studio_gui import Studio

MESSAGES = [{"role": "user", "content": "hi"}]


class AlignFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        try:
            cls.app = Studio(Workspace(cls.directory))
        except tk.TclError as exc:
            shutil.rmtree(cls.directory, ignore_errors=True)
            if "display" in str(exc).lower():
                raise unittest.SkipTest("Tk display is unavailable")
            raise
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def settle(self):
        deadline = time.monotonic() + 10
        while self.app.job_kind and time.monotonic() < deadline:
            self.app.update()
            time.sleep(.01)
        self.assertIsNone(self.app.job_kind)

    def test_stopped_readiness_check_is_not_ready(self):
        app, captured = self.app, []
        with patch.object(app, "checkpoint_path", return_value="/x.npz"), patch.object(app, "selected_dataset", return_value="d"), \
                patch.object(app.workspace, "sample", return_value=[{"user": "a", "assistant": "b"}]), patch.object(app, "start_job", side_effect=lambda k, f, cb: captured.append(cb)):
            app.evaluate_readiness()
        captured[0]({"token_error": 0., "perplexity": 1., "loss": 0., "unknown_tokens": 0, "tokens": 0, "stopped": True})
        self.assertIn("STOPPED", app.ready_verdict.get())
        self.assertNotIn("READY", app.ready_verdict.get())

    def test_a_comparison_takes_only_one_vote(self):
        app = self.app
        before = len(app.preferences.all())
        app.comparison = {"prompt": {"messages": MESSAGES}, "replies": ("good", "bad"), "checkpoint": ""}
        with patch.object(app, "next_comparison"):
            app.vote("A")
            app.vote("A")
        self.assertEqual(len(app.preferences.all()), before + 1)

    def test_finding_models_never_blocks_and_reports_failure(self):
        app, shown = self.app, []
        with patch("studio_align.OllamaTeacher") as teacher, patch("studio_align.messagebox.showerror", side_effect=lambda *a, **k: shown.append(a)):
            teacher.return_value.models.side_effect = OSError("no route")
            app.refresh_ollama()
            self.assertEqual(app.job_kind, "model search")
            self.settle()
        self.assertEqual(len(shown), 1)
        self.assertIn("no route", shown[0][1])
        with patch("studio_align.OllamaTeacher") as teacher:
            teacher.return_value.models.return_value = ["llama3:8b"]
            app.refresh_ollama()
            self.settle()
        self.assertEqual(app.ollama_model.get(), "llama3:8b")

    def test_stopped_training_keeps_the_checkpoint_in_use(self):
        app, captured = self.app, []
        pairs = [{"id": str(i), "messages": MESSAGES, "chosen": "a", "rejected": "b"} for i in range(3)]
        app.align_checkpoint.set("/base.npz")
        with patch.object(app, "checkpoint_path", return_value="/base.npz"), patch.object(app.preferences, "all", return_value=pairs), \
                patch("studio_align.messagebox.askyesno", return_value=True), patch.object(app, "start_job", side_effect=lambda k, f, cb: captured.append(cb)), patch.object(app, "show_page"):
            app.use_human.set(True)
            app.start_dpo()
        captured[0]({"status": "stopped", "checkpoint": "/half/last.npz", "directory": "/half"})
        self.assertEqual(app.align_checkpoint.get(), "/base.npz")
        captured[0]({"status": "completed", "checkpoint": "/done/last.npz", "directory": "/done"})
        self.assertEqual(app.align_checkpoint.get(), "/done/last.npz")

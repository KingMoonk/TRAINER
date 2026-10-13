"""Regression tests for event-loop, run-table and folder-opening fixes in the main window."""
import shutil
import tempfile
import unittest
import tkinter as tk
from unittest.mock import Mock, patch

from studio_core import Workspace
from studio_gui import Studio


class MainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        cls.workspace = Workspace(cls.directory)
        try:
            cls.app = Studio(cls.workspace)
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

    def test_poll_survives_a_failing_handler(self):
        app = self.app
        app.events.put(("training", {}))
        app.events.put(("rlhf", {}))  # a mixin's handler fails too
        app.events.put(("status", "after the failure"))
        failing = Mock(side_effect=ValueError("boom"))
        with patch.dict(app.handlers, training=failing, rlhf=Mock(side_effect=KeyError("mixin"))), patch.object(app, "after") as after:
            app.poll()
        after.assert_called_once()
        self.assertEqual(app.status_var.get(), "after the failure")
        self.assertIn("boom", app.run_log.get("1.0", "end"))
        self.assertIn("mixin", app.run_log.get("1.0", "end"))

    def test_poll_keeps_event_order_past_one_batch(self):
        app, seen = self.app, []
        for index in range(450):
            app.events.put(("rlhf" if index % 3 else "curriculum", index))
        record = Mock(side_effect=seen.append)
        with patch.dict(app.handlers, rlhf=record, curriculum=record), patch.object(app, "after"):
            while not app.events.empty():
                app.poll()
        self.assertEqual(seen, list(range(450)))

    def test_runs_selection_follows_the_run_not_the_row(self):
        app = self.app
        app.show_page("training")
        runs = [{"directory": "/a", "name": "A"}, {"directory": "/b", "name": "B"}]
        with patch.object(self.workspace, "runs", return_value=runs), patch.object(app, "inspect_run"):
            app.refresh_runs()
            app.runs_tree.selection_set("1")
            runs.insert(0, {"directory": "/new", "name": "New"})
            app.refresh_runs()
            self.assertEqual(app.selected_run()["directory"], "/b")

    def test_open_run_folder_does_not_need_startfile(self):
        app = self.app
        with patch.object(app, "selected_run", return_value={"directory": "/x"}), patch("studio_gui.sys.platform", "darwin"), patch("studio_gui.subprocess.Popen") as popen:
            app.open_run_folder()
        popen.assert_called_once_with(["open", "/x"])

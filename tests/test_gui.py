"""Integration check of Tk event delivery and the dataset -> train -> chat flow."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import tkinter as tk

from studio_core import Workspace
from studio_gui import ExampleEditor, Studio


class DesktopWorkflow(unittest.TestCase):
    def test_native_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            dataset = workspace.create_dataset("GUI integration")
            workspace.add(dataset, [{"user": "a", "assistant": "b"}])
            try:
                app = Studio(workspace)
            except tk.TclError as exc:
                if "display" in str(exc).lower():
                    self.skipTest("Tk display is unavailable")
                raise
            app.withdraw()
            def settle():
                deadline = time.monotonic() + 20
                while app.job_kind and time.monotonic() < deadline:
                    app.update()
                    time.sleep(.01)
                self.assertIsNone(app.job_kind, "Worker did not complete")
            try:
                with patch("studio_gui.messagebox.showerror", side_effect=lambda *args, **kwargs: self.fail(str(args))):
                    for page, *_ in app.PAGES:
                        app.show_page(page)
                        app.update()
                    dialog = ExampleEditor(app, workspace.examples(dataset)[0])
                    dialog.withdraw()
                    dialog.save()
                    self.assertEqual(dialog.result["assistant"], "b")
                    app.synth_tabs.select(1)
                    app.synth_count.set("12")
                    app.generate_drafts()
                    settle()
                    self.assertEqual(len(app.drafts), 12)
                    # Synthetic drafts go to a new collection by default; target the working dataset here.
                    app.target_dataset.set(app.dataset_var.get())
                    app.keep_drafts(True)
                    self.assertEqual(len(workspace.examples(dataset)), 13)
                    learning = workspace.create_dataset("Tiny learning check")
                    workspace.add(learning, [{"user": "a", "assistant": "b"}])
                    app.refresh_datasets(learning)
                    # Rotation is the default: runs cycle through the library in order, whatever Curate has selected.
                    self.assertEqual(app.dataset_mode.get(), "rotate")
                    self.assertEqual(app.run_sources(), [dataset])
                    app.workspace.state["rotationIndex"] = 1
                    self.assertEqual(app.run_sources(), [learning])
                    app.workspace.state["rotationIndex"] = 2
                    self.assertEqual(app.run_sources(), [dataset])
                    # Choose by hand with nothing selected falls back to the working dataset.
                    app.dataset_mode.set("manual")
                    app.dataset_mode_changed()
                    self.assertEqual(app.run_sources(), [learning])
                    for key, value in dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4, steps=45, learning_rate=.004, warmup=0, validation=0, eval_every=15, objective="assistant").items():
                        app.config_vars[key].set(str(value))
                    app.start_training()
                    settle()
                    run = workspace.runs()[0]
                    self.assertEqual(run["status"], "completed")
                    self.assertEqual(len({item["step"] for item in app.history}), len(app.history))
                    app.chat_checkpoint.set(str(Path(run["directory"]) / "last.npz"))
                    app.inference_device.set("cpu")
                    app.load_chat_model()
                    settle()
                    self.assertIsNotNone(app.chat_model)
                    app.temperature.set("0")
                    app.chat_input.insert("1.0", "a")
                    app.send_chat()
                    settle()
                    self.assertEqual(app.last_answer, "b")
            finally:
                app.stop.set()
                app.pause.clear()
                if app.worker:
                    app.worker.join(timeout=20)
                app.destroy()


if __name__ == "__main__":
    unittest.main()

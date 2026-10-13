"""Sequential training: stage ordering, weight hand-off, forgetting table, stop/resume and the Tk page."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
import tkinter as tk

from studio_core import Workspace
from studio_curriculum import default_plan, load_plan, resume_plan, run_plan

BASE = dict(width=16, layers=1, heads=2, expansion=2, dropout=0, batch_size=4, learning_rate=.004,
            device="cpu", validation=.1, seed=7)


def make_workspace(directory):
    workspace = Workspace(directory)
    # created out of stage order on purpose
    for name, catalog, rows in (
            ("Reasoning set", "gsm8k", [{"user": f"What is {i} plus {i}?", "assistant": f"{i} plus {i} is {2 * i}."} for i in range(40)]),
            ("Stories", "tinystories", [{"text": f"Once upon a time number {i} a small cat sat on a mat and purred happily."} for i in range(40)]),
            ("Instructions", "dolly15k", [{"user": f"Name color number {i}.", "assistant": f"Color number {i} is blue."} for i in range(40)]),
            ("Long reading", "squad2", [{"user": f"Passage {i} question?", "assistant": f"Answer {i}."} for i in range(40)])):
        dataset = workspace.create_dataset(name, catalogId=catalog)
        workspace.add(dataset, rows)
    return workspace


def small_plan(workspace, steps=15):
    plan = default_plan(workspace, BASE, context=32)
    for stage in plan["stages"]:
        stage.update(steps=steps, max_records=24, eval_records=8)
    return plan


class CurriculumEngine(unittest.TestCase):
    def test_default_plan_orders_stages_and_skips_long_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = make_workspace(directory)
            plan = default_plan(workspace, BASE, context=32)
            self.assertEqual([s["kind"] for s in plan["stages"]], ["language", "instructions", "reasoning"])
            self.assertTrue(all(s["context"] == 32 for s in plan["stages"]))
            self.assertTrue(any("Long reading" in w for w in plan["warnings"]))
            wide = default_plan(workspace, BASE, context=1024)
            self.assertEqual(wide["stages"][-1]["kind"], "knowledge")
            with self.assertRaises(ValueError):
                bad = small_plan(workspace)
                bad["stages"][1]["context"] = 64
                run_plan(workspace, bad, BASE, state_dir=Path(directory) / "s")

    def test_run_chain_eval_table_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = make_workspace(directory)
            events = []
            state = run_plan(workspace, small_plan(workspace), BASE, emit=events.append, plan_id="plan_a")
            self.assertEqual(state["status"], "completed")
            self.assertEqual([s["status"] for s in state["stages"]], ["completed"] * 3)
            runs = [json.loads((Path(s["run_dir"]) / "run.json").read_text()) for s in state["stages"]]
            self.assertIsNone(runs[0]["initialized_from"])
            self.assertTrue(runs[1]["initialized_from"].endswith("npz"))
            self.assertIn("stage1", runs[1]["initialized_from"])
            self.assertIn("stage2", runs[2]["initialized_from"])
            self.assertTrue(all(r["vocabulary"] == 263 for r in runs))
            self.assertEqual([len(row["evals"]) for row in state["eval_table"]], [1, 2, 3])
            self.assertEqual(len(state["forgetting"]), 3)
            self.assertTrue(any(e["type"] == "curriculum" for e in events) and any(e["type"] == "metric" for e in events))
            saved = load_plan(workspace, "plan_a")
            self.assertEqual(saved["status"], "completed")
            self.assertTrue((Path(directory) / "data" / "curriculum" / "plan_a.json").is_file())

            # stop during stage 2, then resume
            stop, seen = threading.Event(), []
            def emit(event):
                seen.append(event)
                if event["type"] == "metric" and event["step"] >= 5 and any(e["type"] == "curriculum" and e.get("stage") == 1 and e["status"] == "running" for e in seen):
                    stop.set()
            partial = run_plan(workspace, small_plan(workspace, 20), BASE, emit=emit, stop=stop, plan_id="plan_b")
            self.assertEqual(partial["status"], "stopped")
            self.assertEqual(partial["stages"][0]["status"], "completed")
            self.assertEqual(partial["stages"][1]["status"], "stopped")
            self.assertEqual(partial["stages"][2]["status"], "pending")
            final = resume_plan(workspace, "plan_b")
            self.assertEqual(final["status"], "completed")
            self.assertEqual([s["status"] for s in final["stages"]], ["completed"] * 3)
            resumed = json.loads((Path(final["stages"][1]["run_dir"]) / "run.json").read_text())
            self.assertTrue(resumed["resumed_from"])
            self.assertEqual(final["stages"][1]["steps_done"], 20)
            self.assertEqual(len(final["eval_table"]), 3)


class CurriculumPage(unittest.TestCase):
    def test_gui_default_plan_runs(self):
        from studio_gui import Studio
        with tempfile.TemporaryDirectory() as directory:
            workspace = make_workspace(directory)
            try:
                app = Studio(workspace)
            except tk.TclError as exc:
                if "display" in str(exc).lower():
                    self.skipTest("Tk display is unavailable")
                raise
            app.withdraw()
            try:
                app.show_page("training")
                app.build_curriculum_plan()
                app.update()
                self.assertEqual(len(app.curriculum_tree.get_children()), 3)
                for stage in app.curriculum["stages"]:
                    stage.update(steps=12, max_records=24, eval_records=8, context=32)
                app.curriculum["context"] = 32
                app.curriculum_settings = lambda: (dict(BASE, context=32), 32, None)
                app.start_curriculum()
                deadline = time.monotonic() + 60
                while app.job_kind and time.monotonic() < deadline:
                    app.update()
                    time.sleep(.01)
                app.update()
                self.assertIsNone(app.job_kind)
                self.assertEqual([s["status"] for s in app.curriculum["stages"]], ["completed"] * 3)
                self.assertEqual(len(app.forget_tree.get_children()), 3)
            finally:
                app.destroy()


if __name__ == "__main__":
    unittest.main()

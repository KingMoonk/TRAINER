"""Mid-training keeps the base design, their own data and a real gate."""
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import studio_model
import studio_training_stages as stages
from studio_core import Workspace
from studio_ia.services import WorkspaceServices
from studio_jobs import _continue_stage
from studio_spec import default_spec


TEXT = [{"text": f"The short answer is yes. Lesson {i}."} for i in range(16)]
CHAT = [{"user": f"Question {i}: say yes", "assistant": "yes"} for i in range(16)]
HELD_CHAT = [{"user": "Question 99: say yes", "assistant": "yes"}]
class StageEvidenceTests(unittest.TestCase):
    def test_a_stage_folder_gives_its_validation_points_and_whether_it_ended_too_small(self):
        import studio_stages
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(stages.run_evidence(folder), {})
            rows = [{"step": 10, "val_loss": 3.0}, {"step": 20}, {"step": 30, "val_loss": 2.5}]
            (folder / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            self.assertEqual(stages.run_evidence(folder), {"curve": [[10.0, 3.0], [30.0, 2.5]]})
            studio_stages.write_journal(folder, {"sources": [], "switches": [], "exhausted": False, "too_small": True})
            self.assertEqual(stages.run_evidence(folder), {"curve": [[10.0, 3.0], [30.0, 2.5]], "too_small": True})


BASE = {"device": "cpu", "width": 32, "layers": 1, "heads": 2, "context": 64, "expansion": 2,
        "dropout": 0, "batch_size": 4, "steps": 8, "learning_rate": .003, "warmup": 0,
        "validation": 0, "eval_every": 4, "objective": "all"}


class Context:
    def __init__(self):
        self.reads = []

    def stage_rows(self, stage, shares, max_chars):
        self.reads.append(stage)
        return TEXT + CHAT

    def post_rows(self, kind):
        return {"heldout_docs": ["A different lesson about the answer yes."],
                "heldout_problems": [], "heldout_conversations": HELD_CHAT}[kind]


class TrainingStages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        studio_model.train(BASE, TEXT, cls.root / "base")
        cls.checkpoint = cls.root / "base" / "last.npz"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_midtrain_trains_every_token_and_sft_is_not_a_second_stage(self):
        cfg = stages.stage_config("midtrain", self.checkpoint, TEXT + CHAT, {"steps": 2})
        self.assertEqual((cfg["objective"], cfg["steps"]), ("all", 2))
        with self.assertRaises(ValueError):
            stages.stage_config("sft", self.checkpoint, CHAT)

    def test_midtrain_reuses_weights_and_frozen_data_on_resume(self):
        context = Context()
        events = []
        interrupted = threading.Event()

        def stop_at_two(event):
            events.append(event)
            if event.get("type") == "metric" and event.get("step") == 2:
                interrupted.set()

        with mock.patch.object(stages, "_gate", return_value=(True, {}, {}, None, [], None)):
            mid = stages.run_stage("midtrain", self.checkpoint, context, self.root / "mid", stop_at_two,
                                   interrupted, threading.Event(), config={"steps": 4, "learning_rate": .0003})
        self.assertFalse(mid["ok"])
        self.assertTrue(mid["stopped"])
        self.assertEqual(context.reads, ["midtrain"])
        self.assertEqual(studio_model.load_checkpoint(mid["checkpoint"], weights_only=True)["step"], 2)
        self.assertTrue((self.root / "mid" / "dataset.jsonl").is_file())

        resumed_context = Context()
        with mock.patch.object(stages, "_gate", return_value=(True, {}, {}, None, [], None)):
            resumed = stages.run_stage("midtrain", self.checkpoint, resumed_context, self.root / "mid-retry", events.append,
                                       threading.Event(), threading.Event(), config={"steps": 4, "learning_rate": .0003},
                                       resume=self.root / "mid" / "last.npz")
        self.assertTrue(resumed["ok"])
        self.assertEqual(resumed_context.reads, [])
        self.assertEqual(studio_model.load_checkpoint(resumed["checkpoint"], weights_only=True)["step"], 4)

    def test_candidate_gate_does_not_retrain(self):
        context = Context()
        with mock.patch.object(stages, "_gate", return_value=(False, {"bpc": 1}, {"bpc": 2}, None, [], None)):
            result = stages.run_stage("midtrain", self.checkpoint, context, self.root / "unused", lambda event: None,
                                      threading.Event(), threading.Event(), candidate=self.checkpoint)
        self.assertFalse(result["ok"])
        self.assertEqual(context.reads, [])

    def test_restart_recovers_completed_gate_or_partial_checkpoint(self):
        done_dir = self.root / "restart-done"
        done_dir.mkdir()
        (done_dir / "run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        shutil.copy2(self.checkpoint, done_dir / "last.npz")
        complete = SimpleNamespace(params={"output_dir": str(done_dir)})
        _continue_stage(complete)
        self.assertEqual(complete.params["candidate"], str(done_dir / "last.npz"))
        partial_dir = self.root / "restart-partial"
        partial_dir.mkdir()
        (partial_dir / "run.json").write_text(json.dumps({"status": "stopped"}), encoding="utf-8")
        (partial_dir / "dataset.jsonl").write_text(json.dumps(CHAT[0]) + "\n", encoding="utf-8")
        shutil.copy2(self.checkpoint, partial_dir / "last.npz")
        partial = SimpleNamespace(params={"output_dir": str(partial_dir)})
        _continue_stage(partial)
        self.assertFalse(partial_dir.exists())
        self.assertTrue(Path(partial.params["resume"]).is_file())

    def test_real_held_back_chat_improves_without_raw_text_regression(self):
        context = Context()
        mid = stages.run_stage("midtrain", self.checkpoint, context, self.root / "measured-mid", lambda event: None,
                               threading.Event(), threading.Event(), config={"steps": 20, "learning_rate": .0003})
        self.assertLess(mid["after"]["conv_loss"], mid["before"]["conv_loss"])
        self.assertTrue(next(c for c in mid["checks"] if c["name"] == "Held-back text: error did not rise")["ok"])
        self.assertTrue(mid["ok"], mid["checks"])

    def test_only_passing_stage_weights_become_the_versions_best(self):
        with tempfile.TemporaryDirectory() as directory:
            svc = WorkspaceServices(Workspace(directory))
            self.addCleanup(svc.close)
            version = svc.store.save_version(default_spec())
            svc.store.set_best_checkpoint(version["number"], self.checkpoint)
            failed = SimpleNamespace(id="failed", params={"version": version["number"], "output_dir": str(self.root / "failed"),
                                                     "result": {"stage": "midtrain", "ok": False, "checkpoint": str(self.root / "measured-mid" / "last.npz")}})
            svc.record_stage(failed)
            self.assertEqual(svc.store.get(version["number"])["best_checkpoint"], str(self.checkpoint))
            passed = SimpleNamespace(id="passed", params={"version": version["number"], "output_dir": str(self.root / "passed"),
                                                     "result": {"stage": "midtrain", "ok": True, "checkpoint": str(self.root / "measured-mid" / "last.npz"),
                                                                "after": {"bpc": 1.2, "pass_k": 0, "pass_n": 0}}})
            with mock.patch.object(svc, "_bench_later"), mock.patch.object(svc, "_note_devices"):
                svc.record_stage(passed)
            self.assertEqual(svc.store.get(version["number"])["best_checkpoint"], passed.params["result"]["checkpoint"])

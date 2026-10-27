"""Mid-training and SFT keep the base design, their own data and a real gate."""
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import studio_model
import studio_training_stages as stages
from studio_core import Workspace
from studio_ia.services import WorkspaceServices
from studio_jobs import _continue_stage
from studio_spec import default_spec


TEXT = [{"text": f"The short answer is yes. Lesson {i}."} for i in range(16)]
CHAT = [{"user": f"Question {i}: say yes", "assistant": "yes"} for i in range(16)]
HELD_CHAT = [{"user": "Question 99: say yes", "assistant": "yes"}]
BASE = {"device": "cpu", "width": 32, "layers": 1, "heads": 2, "context": 64, "expansion": 2,
        "dropout": 0, "batch_size": 4, "steps": 8, "learning_rate": .003, "warmup": 0,
        "validation": 0, "eval_every": 4, "objective": "all"}


class Context:
    def __init__(self):
        self.reads = []

    def stage_rows(self, stage, shares, max_chars):
        self.reads.append(stage)
        return TEXT + CHAT if stage == "midtrain" else CHAT

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

    def test_sft_masks_every_token_except_assistant_content_and_eos(self):
        data = studio_model.load_checkpoint(self.checkpoint, weights_only=True)
        tokenizer = studio_model.Tokenizer.restore(data["tokenizer"])
        cfg = stages.stage_config("sft", self.checkpoint, CHAT, {"steps": 2})
        self.assertEqual(cfg["objective"], "assistant")
        tokens, mask = studio_model.encode_example(CHAT[0], tokenizer, assistant_only=True)
        self.assertFalse(any(mask[:tokens.index(studio_model.ASSISTANT) + 1]))
        self.assertTrue(any(mask[tokens.index(studio_model.ASSISTANT) + 1:]))
        window = studio_model.samples_from([CHAT[0]], tokenizer, cfg)[0]
        self.assertTrue(np.any(window[1] == -100))
        self.assertTrue(np.any(window[1] != -100))

    def test_midtrain_then_sft_reuses_weights_and_frozen_data_on_resume(self):
        context = Context()
        events = []
        with mock.patch.object(stages, "_gate", return_value=(True, {}, {}, None, [], None)):
            mid = stages.run_stage("midtrain", self.checkpoint, context, self.root / "mid", events.append,
                                   threading.Event(), threading.Event(), config={"steps": 4, "learning_rate": .0003})
            interrupted = threading.Event()

            def stop_at_two(event):
                events.append(event)
                if event.get("type") == "metric" and event.get("step") == 2:
                    interrupted.set()

            sft = stages.run_stage("sft", mid["checkpoint"], context, self.root / "sft", stop_at_two,
                                   interrupted, threading.Event(), config={"steps": 4, "learning_rate": .0001})
        self.assertTrue(mid["ok"])
        self.assertFalse(sft["ok"])
        self.assertTrue(sft["stopped"])
        self.assertEqual(context.reads, ["midtrain", "sft"])
        self.assertEqual(studio_model.load_checkpoint(mid["checkpoint"], weights_only=True)["config"]["objective"], "all")
        self.assertEqual(studio_model.load_checkpoint(sft["checkpoint"], weights_only=True)["config"]["objective"], "assistant")
        self.assertEqual(studio_model.load_checkpoint(sft["checkpoint"], weights_only=True)["step"], 2)
        self.assertTrue((self.root / "sft" / "dataset.jsonl").is_file())

        resumed_context = Context()
        with mock.patch.object(stages, "_gate", return_value=(True, {}, {}, None, [], None)):
            resumed = stages.run_stage("sft", mid["checkpoint"], resumed_context, self.root / "sft-retry", events.append,
                                       threading.Event(), threading.Event(), config={"steps": 4, "learning_rate": .0001},
                                       resume=self.root / "sft" / "last.npz")
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
        sft = stages.run_stage("sft", mid["checkpoint"], context, self.root / "measured-sft", lambda event: None,
                               threading.Event(), threading.Event(), config={"steps": 20, "learning_rate": .0003})
        self.assertLess(sft["after"]["conv_loss"], sft["before"]["conv_loss"])
        self.assertTrue(next(c for c in sft["checks"] if c["name"] == "Raw text: error stayed within the drift budget")["ok"])
        self.assertTrue(sft["ok"], sft["checks"])

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
                                                     "result": {"stage": "sft", "ok": True, "checkpoint": str(self.root / "measured-sft" / "last.npz"),
                                                                "after": {"bpc": 1.2, "pass_k": 0, "pass_n": 0}}})
            with mock.patch.object(svc, "_bench_later"), mock.patch.object(svc, "_note_devices"):
                svc.record_stage(passed)
            self.assertEqual(svc.store.get(version["number"])["best_checkpoint"], passed.params["result"]["checkpoint"])

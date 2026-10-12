"""Regression tests for preference-store, teacher-reply, replay, stop and readiness fixes in studio_rlhf."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import studio_rlhf
from studio_model import train
from studio_rlhf import OllamaTeacher, PreferenceStore, TeacherError, auto_rlhf, evaluate_checkpoint

MESSAGES = [{"role": "user", "content": "hi"}]


class Store(unittest.TestCase):
    def test_add_after_file_without_trailing_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PreferenceStore(directory)
            store.add(MESSAGES, "good", "bad", "human")
            store.path.write_bytes(store.path.read_bytes().rstrip(b"\n"))
            store.add(MESSAGES, "better", "worse", "human")
            self.assertEqual(len(store.all()), 2)
            self.assertEqual(len(store.path.read_text(encoding="utf-8").splitlines()), 2)

    def test_invalid_rows_are_ignored_but_kept_in_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PreferenceStore(directory)
            record = store.add(MESSAGES, "good", "bad", "human")
            with store.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": "x", "chosen": "a"}) + "\n")
            self.assertEqual([r["id"] for r in store.all()], [record["id"]])
            store.delete([record["id"]])
            self.assertIn('"id": "x"', store.path.read_text(encoding="utf-8"))

    def test_line_separator_inside_a_record_does_not_split_it(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PreferenceStore(directory)
            store.add(MESSAGES, "good text", "bad", "human")
            self.assertEqual(store.all()[0]["chosen"], "good text")


class Teacher(unittest.TestCase):
    def test_null_content_and_null_answer(self):
        teacher = OllamaTeacher(model="m", retry_delay=0)
        with patch.object(teacher, "_request", return_value={"message": {"content": None}}):
            self.assertEqual(teacher.chat("s", "u", json_mode=False), "")
        with patch.object(teacher, "chat", return_value={"answer": None}):
            self.assertEqual(teacher.rewrite(MESSAGES), "")

    def test_stopped_teacher_makes_no_request(self):
        stop = threading.Event()
        stop.set()
        teacher = OllamaTeacher(model="m", retry_delay=0, stop=stop)
        with patch.object(teacher, "chat") as chat:
            with self.assertRaises(TeacherError):
                teacher.ask("s", "u")
        chat.assert_not_called()

    def test_reason_from_the_swapped_pass_is_reported_in_callers_order(self):
        teacher = OllamaTeacher(model="m", retry_delay=0)
        rng = MagicMock()
        rng.random.return_value = .9  # no swap on the first pass, so the second pass shows the replies swapped
        with patch.object(teacher, "chat", return_value={"score_a": 9, "score_b": 1, "reason": "A is much better"}):
            verdict = teacher.judge(MESSAGES, "first", "second", None, True, rng)
        self.assertEqual(verdict["reason"], "Inconsistent across orderings: A is much better / B is much better")


class Loops(unittest.TestCase):
    def test_replay_zero_replays_nothing(self):
        pair = lambda n: {"messages": MESSAGES, "chosen": f"c{n}", "rejected": f"r{n}", "kind": "comparison", "scores": [9, 1], "reason": "", "reference": None, "source_id": None, "n_candidates": 2}  # noqa: E731
        counter = iter(range(100))
        seen = []

        def fake_train(current, pairs, *args, **kwargs):
            seen.append(len(pairs))
            return {"checkpoint": f"{current}-next"}
        gate = {"wins": 1, "ties": 0, "losses": 0, "n": 1, "win_rate": 1.0}
        prompts = [{"messages": MESSAGES, "reference": None, "source_id": str(i)} for i in range(8)]
        cfg = {"rounds": 2, "prompts_per_round": 3, "eval_prompts": 2, "replay": 0}
        with tempfile.TemporaryDirectory() as directory, patch.object(studio_rlhf, "LocalModel"), patch.object(studio_rlhf, "dpo_train", side_effect=fake_train), \
                patch.object(studio_rlhf, "compare_checkpoints", return_value=gate), patch.object(studio_rlhf, "teacher_pairs", side_effect=lambda *a, **k: ([pair(next(counter))], {})):
            teacher = MagicMock()
            teacher.model = "m"
            auto_rlhf("base", prompts, teacher, PreferenceStore(directory), Path(directory) / "runs", cfg, None, None, None, None, teacher)
        self.assertEqual(seen, [3, 3])

    def test_stopped_evaluation_is_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            examples = [{"user": "say yes", "assistant": "yes"}] * 8
            train(dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4, steps=2, learning_rate=.004,
                       warmup=0, validation=0, eval_every=2, objective="assistant"), examples, Path(directory) / "base")
            stop = threading.Event()
            stop.set()
            report = evaluate_checkpoint(Path(directory) / "base" / "last.npz", examples, "cpu", 8, stop)
            self.assertTrue(report["stopped"])
            self.assertFalse(evaluate_checkpoint(Path(directory) / "base" / "last.npz", examples, "cpu", 8)["stopped"])


if __name__ == "__main__":
    unittest.main()

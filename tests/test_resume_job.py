"""studio_jobs.pretrain with params["resume"]: a job placed again on the node that holds its full checkpoint continues with its optimizer."""
import errno
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import studio_jobs
import studio_model

TINY = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2, steps=6,
            learning_rate=.004, warmup=0, validation=0, eval_every=2, objective="assistant")
EXAMPLES = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog")] * 2


def run_job(root, name, events, **params):
    job = SimpleNamespace(id=name, kind="pretrain", params={"config": dict(TINY), "examples": EXAMPLES, "output_dir": str(root / name), **params})
    studio_jobs.pretrain(job, SimpleNamespace(run_dir=str(root)), events.append, None, None)
    return root / name


def losses(events):
    return {e["step"]: e["loss"] for e in events if e["type"] == "metric"}


def interrupted(root, after=3):
    """A run planned for TINY["steps"] that is stopped after `after` steps, as a lost node leaves it."""
    stop, events = threading.Event(), []

    def emit(event):
        events.append(event)
        if event["type"] == "metric" and event["step"] >= after:
            stop.set()
    studio_model.train(TINY, EXAMPLES, root / "first", emit, stop=stop)
    return root / "first" / "last.npz"


class ResumeJob(unittest.TestCase):
    def test_train_says_it_is_saving_before_it_writes_the_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            studio_model.train(TINY, EXAMPLES, Path(directory) / "run", events.append)
            kinds = [e["type"] for e in events]
            self.assertEqual(kinds.count("saving"), 1)
            self.assertLess(kinds.index("saving"), kinds.index("done"))
            self.assertGreater(kinds.index("saving"), max(i for i, kind in enumerate(kinds) if kind == "metric"))

    def test_a_resumed_job_continues_with_its_optimizer_and_has_no_restart_spike(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whole = []
            run_job(root, "whole", whole)
            last = interrupted(root)
            step = studio_model.load_checkpoint(last)["step"]
            self.assertTrue(0 < step < TINY["steps"], step)
            events = []
            out = run_job(root, "second", events, resume=str(last), resume_steps=TINY["steps"], resume_base=100)
            self.assertEqual([e for e in events if e["type"] == "resumed"], [{"type": "resumed", "base": 100, "step": step}])
            self.assertFalse([e for e in events if e["type"] == "resume_refused"])
            resumed = losses(events)
            self.assertEqual(min(resumed), step + 1)  # it goes on from the checkpoint, not from zero
            reference = losses(whole)
            for number, loss in resumed.items():
                self.assertAlmostEqual(loss, reference[number], places=4, msg=number)  # the same curve as a run that was never interrupted
            self.assertEqual(studio_model.load_checkpoint(out / "last.npz")["step"], TINY["steps"])

    def test_the_original_dataset_is_used_so_a_changed_one_cannot_refuse_the_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = interrupted(root)
            events = []
            run_job(root, "second", events, resume=str(last), resume_steps=TINY["steps"],
                    examples=[{"user": "say new", "assistant": "new"}] * 8)
            self.assertTrue([e for e in events if e["type"] == "resumed"])
            self.assertEqual(max(losses(events)), TINY["steps"])

    def test_a_dataset_whose_text_holds_a_unicode_line_break_still_resumes(self):
        # dataset.jsonl is split on "\n" only: str.splitlines() also breaks at U+0085 and U+2028 inside a record's text.
        examples = [{"user": "say next\x85line", "assistant": "ok done"}, *EXAMPLES]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop = threading.Event()

            def emit(event):
                if event["type"] == "metric" and event["step"] >= 3:
                    stop.set()
            studio_model.train(TINY, examples, root / "first", emit, stop=stop)
            events = []
            run_job(root, "second", events, resume=str(root / "first" / "last.npz"), resume_steps=TINY["steps"])
            self.assertTrue([e for e in events if e["type"] == "resumed"])
            self.assertEqual(max(losses(events)), TINY["steps"])

    def test_a_checkpoint_training_refuses_falls_back_to_the_published_weights_and_says_so(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = interrupted(root)
            events = []
            config = dict(TINY, learning_rate=.01, steps=3)  # the recipe changed: the checkpoint cannot be resumed
            out = run_job(root, "second", events, resume=str(last), resume_steps=TINY["steps"], mode="catch_up", init_from=str(last),
                          config=config)
            refused = [e for e in events if e["type"] == "resume_refused"]
            self.assertEqual(len(refused), 1)
            self.assertIn("Resume must preserve", refused[0]["reason"])
            self.assertFalse([e for e in events if e["type"] == "resumed"])
            self.assertEqual(max(losses(events)), 3)  # a fresh run of the steps that were left, from the weights
            self.assertEqual(studio_model.load_checkpoint(out / "last.npz")["step"], 3)

    def test_without_published_weights_a_refused_checkpoint_fails_the_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = interrupted(root)
            with self.assertRaisesRegex(ValueError, "Resume must preserve"):
                run_job(root, "second", [], resume=str(last), resume_steps=TINY["steps"], config=dict(TINY, learning_rate=.01))

    def resume_failing_with(self, error, stop=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = interrupted(root)
            events, trained = [], []
            job = SimpleNamespace(id="second", kind="pretrain", params={"config": dict(TINY), "examples": EXAMPLES, "output_dir": str(root / "second"),
                                                                       "resume": str(last), "resume_steps": TINY["steps"], "mode": "catch_up",
                                                                       "init_from": str(last)})

            def train(*args, **kwargs):
                trained.append(kwargs.get("resume"))
                if kwargs.get("resume") is not None:
                    if stop is not None:
                        stop.set()
                    raise error
                return {}
            with mock.patch.object(studio_model, "train", train):
                try:
                    studio_jobs.pretrain(job, SimpleNamespace(run_dir=str(root)), events.append, stop, None)
                except Exception as exc:
                    return events, trained, exc
            return events, trained, None

    def test_a_stop_during_resume_preparation_ends_the_job_without_a_refusal_or_a_fresh_start(self):
        events, trained, raised = self.resume_failing_with(InterruptedError("stopped"), threading.Event())
        self.assertIsInstance(raised, InterruptedError)
        self.assertFalse([e for e in events if e["type"] == "resume_refused"])
        self.assertEqual(len(trained), 1)  # only the resume attempt, no training from step 0

    def test_a_full_disk_during_resume_fails_the_job_with_its_error_and_does_not_train_fresh(self):
        events, trained, raised = self.resume_failing_with(OSError(errno.ENOSPC, "No space left on device"))
        self.assertIsInstance(raised, OSError)
        self.assertIn("No space left", str(raised))
        self.assertFalse([e for e in events if e["type"] == "resume_refused"])
        self.assertEqual(len(trained), 1)


if __name__ == "__main__":
    unittest.main()

"""A pretrain job cut short by an app restart (#397): taken again, it continues from its own `last.npz` instead of starting over or
being refused for its own folder."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import studio_jobs
import studio_model
from studio_queue import Job, Plan, Runner

TINY = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2, steps=6,
            learning_rate=.004, warmup=0, validation=0, eval_every=2, objective="assistant")
EXAMPLES = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog")] * 2


def services(root):
    return SimpleNamespace(root=Path(root), workspace=SimpleNamespace(run_dir=Path(root) / "runs"), examples=lambda p: EXAMPLES)


def losses(events):
    return {e["step"]: e["loss"] for e in events if e["type"] == "metric" and e["step"] > 0}  # (step 0 is the first evaluation)


def attempt(root, job, stop_after=None):
    """One attempt of `job` as the queue gives it: prepare in the app, then the handler (here in-process). `stop_after` stops it after
    that many steps, as an app restart does (training saves its checkpoint on the way out)."""
    context = studio_jobs.prepare(job, services(root))
    events, stop = [], threading.Event()

    def emit(event):
        events.append(event)
        if stop_after and event["type"] == "metric" and event["step"] >= stop_after:
            stop.set()
    studio_jobs.pretrain(job, SimpleNamespace(root=context["root"], run_dir=context["run_dir"]), emit, stop, None)
    return events


def make(job_id="e1e18cfe", **config):
    return Job("pretrain", "Pretrain", id=job_id, params={"config": dict(TINY, **config), "examples": EXAMPLES})


def again(job):
    """The job as a new services instance loads it from queue.json: same id and params, waiting."""
    return Job(**{**job.__dict__, "params": dict(job.params), "status": "waiting"})


class RestartResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_job_stopped_mid_run_continues_from_its_checkpoint_with_no_restart_spike(self):
        whole = attempt(self.root, make("whole"))
        job = make()
        attempt(self.root, job, stop_after=3)
        out = Path(job.params["output_dir"])
        step = studio_model.load_checkpoint(out / "last.npz")["step"]
        self.assertTrue(0 < step < TINY["steps"], step)
        second = again(job)  # a new services instance: the folder of the first attempt is there
        events = attempt(self.root, second)
        self.assertEqual([(e["type"], e["base"], e["step"]) for e in events if e["type"] == "resumed"], [("resumed", 0, step)])
        self.assertEqual([e["total"] for e in events if e["type"] == "resumed"], [TINY["steps"]])
        self.assertFalse([e for e in events if e["type"] == "resume_refused"])
        resumed, reference = losses(events), losses(whole)
        self.assertEqual(min(resumed), step + 1)  # on from the checkpoint, not from step 1
        for number, loss in resumed.items():
            self.assertAlmostEqual(loss, reference[number], places=4, msg=number)  # the curve of a run that was never cut short
        self.assertEqual(studio_model.load_checkpoint(out / "last.npz")["step"], TINY["steps"])  # finished in its own folder
        self.assertEqual(second.params["output_dir"], str(out))
        self.assertTrue((out.parent / f"{out.name}-before-restart" / "last.npz").is_file())  # the earlier attempt is kept, not overwritten

    def test_a_second_restart_continues_again(self):
        job = make(steps=12)
        attempt(self.root, job, stop_after=2)
        second = again(job)
        attempt(self.root, second, stop_after=6)
        out = Path(second.params["output_dir"])
        step = studio_model.load_checkpoint(out / "last.npz")["step"]
        self.assertTrue(2 < step < 12, step)
        events = attempt(self.root, again(second))
        self.assertEqual(min(losses(events)), step + 1)
        self.assertEqual(studio_model.load_checkpoint(out / "last.npz")["step"], 12)

    def test_a_job_that_never_ran_is_untouched(self):
        job = make()
        events = attempt(self.root, job)
        self.assertEqual(job.note, "")
        self.assertFalse([e for e in events if e["type"] == "resumed"])
        self.assertEqual(min(losses(events)), 1)

    def test_a_folder_without_a_checkpoint_starts_fresh_with_a_note_and_does_not_fail(self):
        job = make()
        out = self.root / "runs" / f"queue_{job.id}"
        out.mkdir(parents=True)
        (out / "config.json").write_text("{}")  # an attempt cut short before its first checkpoint
        second = again(job)
        events = attempt(self.root, second)
        self.assertEqual(min(losses(events)), 1)
        self.assertIn("starts again from step 0 because it had not saved a usable checkpoint", second.note)
        self.assertEqual(studio_model.load_checkpoint(out / "last.npz")["step"], TINY["steps"])

    def test_an_unreadable_checkpoint_starts_fresh_and_the_note_says_why(self):
        job = make()
        attempt(self.root, job, stop_after=3)
        out = Path(job.params["output_dir"])
        (out / "last.npz").write_bytes(b"not a checkpoint")
        second = again(job)
        events = attempt(self.root, second)
        self.assertEqual(min(losses(events)), 1)
        self.assertIn("starts again from step 0 because it had not saved a usable checkpoint", second.note)
        self.assertEqual(studio_model.load_checkpoint(out / "last.npz")["step"], TINY["steps"])

    def test_a_checkpoint_training_refuses_falls_back_to_a_fresh_start_instead_of_failing(self):
        job = make()
        attempt(self.root, job, stop_after=3)
        out = Path(job.params["output_dir"])
        changed = again(job)
        changed.params["config"] = dict(TINY, learning_rate=.01)  # the recipe changed: the checkpoint cannot be resumed
        events = attempt(self.root, changed)
        self.assertEqual(len([e for e in events if e["type"] == "resume_refused"]), 1)
        self.assertFalse([e for e in events if e["type"] == "resumed"])
        self.assertEqual(min(losses(events)), 1)
        self.assertEqual(studio_model.load_checkpoint(out / "last.npz")["step"], TINY["steps"])

    def test_a_job_another_node_owns_is_left_to_its_owner(self):
        job = make()
        attempt(self.root, job, stop_after=3)
        guest = again(job)
        guest.owner = "owner-box:8787"
        studio_jobs.prepare(guest, services(self.root))
        self.assertNotIn("resume", guest.params)


def resumed_job(job, context, emit, stop, pause):
    """A handler that continues from step 4 of 10 (the checkpoint's) and holds there until the test says so."""
    emit({"type": "resumed", "base": 0, "step": 4, "total": 10})
    while not ((Path(context.root) / "gate").exists() or stop.is_set()):
        time.sleep(.005)
    emit({"type": "metric", "step": 5, "total": 10, "lr": .001})


class QueueState(unittest.TestCase):
    def test_the_queue_shows_the_continued_step_and_the_job_says_so(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            runner = Runner(root / "data" / "queue.json", {"pretrain": resumed_job}, tick=.01)
            runner.set_plan(Plan(hours="any", idle_only=False, pause_on_battery=False))
            self.addCleanup(runner.shutdown)
            job = runner.enqueue(Job("pretrain", "Pretrain", params={"config": {"steps": 10}}))
            runner.start()
            end = time.time() + 30
            while time.time() < end and runner.state().step != 4:
                time.sleep(.01)
            state = runner.state()
            self.assertEqual((state.step, state.total), (4, 10))  # progress starts from the checkpoint's step, not 0
            self.assertIn("continued from its own checkpoint at step 4", job.note)
            (root / "gate").write_text("")
            while time.time() < end and runner.state().job is not None:
                time.sleep(.01)
            self.assertEqual(job.status, "done")


if __name__ == "__main__":
    unittest.main()

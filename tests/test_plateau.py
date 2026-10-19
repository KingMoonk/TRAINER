"""The plateau rule (studio_plateau) and the pretrain job that ends a run with it (studio_jobs)."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import studio_jobs
import studio_model
import studio_plateau as plateau
from studio_ia import pretraining_plan as plan

TINY = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2, steps=40,
            learning_rate=.004, warmup=0, validation=.25, eval_every=2, objective="assistant")
EXAMPLES = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog", "owl", "fox", "bee", "ant")] * 2
EVIDENCE = [(1000, 2.515), (5000, 2.006), (10000, 1.843), (20000, 1.764), (40000, 1.678), (80000, 1.622), (120000, 1.608), (153000, 1.598),
            (180000, 1.595), (200000, 1.593), (230000, 1.592)]


def metric(step, val, seconds=None):
    return {"type": "metric", "step": step, "val_loss": val, "elapsed_seconds": seconds}


def fed(rule, points):
    watcher = plateau.Plateau(rule)
    for step, val in points:
        watcher.feed(metric(step, val))
    return watcher.hit


class RuleTests(unittest.TestCase):
    def test_a_curve_that_still_falls_does_not_fire(self):
        self.assertIsNone(fed({"window_steps": 20000, "min_gain": .005}, EVIDENCE[:8]))

    def test_a_flat_curve_fires_with_the_gain_and_the_window(self):
        hit = fed({"window_steps": 20000, "min_gain": .005}, EVIDENCE)
        self.assertEqual(hit["step"], 180000)  # the first point whose best barely beats the best 20,000 steps earlier
        self.assertLess(hit["gain"], .005)
        self.assertEqual((hit["window"], hit["best"]), ("20,000 steps", 1.595))

    def test_it_does_not_fire_before_min_steps_or_before_a_full_window(self):
        self.assertIsNone(fed({"window_steps": 20000, "min_gain": .005, "min_steps": 500_000}, EVIDENCE))
        self.assertIsNone(fed({"window_steps": 1_000_000, "min_gain": .005}, EVIDENCE))

    def test_the_window_can_be_time(self):
        watcher = plateau.Plateau({"window_seconds": 3600, "min_gain": .01})
        for i, val in enumerate([2., 1.5, 1.4, 1.399, 1.3985, 1.3984]):
            watcher.feed(metric(100 * (i + 1), val, 1800 * i))
        self.assertEqual(watcher.hit["window"], "1.0 h")
        self.assertIsNone(plateau.Plateau({"window_seconds": 3600}).feed(metric(5, 1., None)))  # no clock, no time window

    def test_history_from_a_resumed_run_counts(self):
        watcher = plateau.Plateau({"window_steps": 20000, "min_gain": .005})
        watcher.feed({"type": "history", "history": [{"step": s, "val_loss": v} for s, v in EVIDENCE[:-1]]})
        self.assertIsNotNone(watcher.hit)

    def test_rows_without_a_validation_loss_or_a_step_are_ignored(self):
        watcher = plateau.Plateau({"window_steps": 10, "min_gain": .01})
        for event in ({"type": "metric", "step": 5}, {"type": "metric", "val_loss": 1.}, {"type": "info", "text": "x"}, {"type": "metric", "step": 0, "val_loss": 3.}):
            self.assertIsNone(watcher.feed(event))
        self.assertEqual(watcher.points, [])

    def test_a_rule_needs_a_window_and_a_sane_gain(self):
        with self.assertRaises(ValueError):
            plateau.Plateau({})
        with self.assertRaises(ValueError):
            plateau.Plateau({"window_steps": 5, "min_gain": 2})

    def test_the_stop_event_is_the_users_or_the_rule(self):
        user, watcher = threading.Event(), plateau.Plateau({"window_steps": 20000, "min_gain": .005})
        flag = plateau.Flagged(user, watcher)
        self.assertFalse(flag.is_set())
        for step, val in EVIDENCE:
            watcher.feed(metric(step, val))
        self.assertTrue(flag.is_set())
        self.assertFalse(flag.user_stopped())
        flag.set()
        self.assertTrue(flag.user_stopped())

    def test_the_default_rule_scales_with_the_run_and_the_plan_turns_it_on_for_long_runs(self):
        rule = plateau.default_rule(819_635)
        self.assertEqual((rule["window_steps"], rule["min_steps"]), (40_981, 81_962))
        self.assertEqual(plateau.default_rule(1000)["window_steps"], plateau.MIN_WINDOW)
        self.assertIsNotNone(plan.plateau_rule(1000, 1.5))  # expected to take over an hour
        self.assertIsNone(plan.plateau_rule(1000, .5))
        self.assertIsNone(plan.plateau_rule(1000, 5, False))
        self.assertIsNotNone(plan.plateau_rule(1000, .1, True))
        self.assertIsNotNone(plan.plateau_rule(plan.LONG_STEPS, None))  # not timed yet: judged by its length
        self.assertIsNone(plan.plateau_rule(plan.LONG_STEPS - 1, None))
        self.assertIn("less than 1%", plan.plateau_line(rule))
        self.assertEqual(plan.plateau_line(None), "")


class JobTests(unittest.TestCase):
    def run_job(self, root, rule, stop=None, name="run"):
        events = []
        job = SimpleNamespace(id=name, kind="pretrain", params={"config": dict(TINY), "examples": EXAMPLES, "output_dir": str(root / name),
                                                                  **({"plateau": rule} if rule else {})})
        result = studio_jobs.pretrain(job, SimpleNamespace(run_dir=str(root)), events.append, stop, None)
        return job, root / name, events, result

    def test_a_flat_curve_ends_the_run_early_with_the_reason_and_keeps_the_best(self):
        with tempfile.TemporaryDirectory() as directory:
            # a gain of at least 99.9% is required to go on, which no run delivers: the first window with nothing to beat ends it
            job, out, events, result = self.run_job(Path(directory), {"window_steps": 6, "min_gain": .999, "min_steps": 0})
            step = studio_model.load_checkpoint(out / "last.npz")["step"]
            self.assertTrue(6 < step < TINY["steps"], step)
            self.assertTrue((out / "best.npz").is_file())
            ended = job.params["ended"]
            self.assertEqual((ended["reason"], ended["step"]), ("plateau", step))
            self.assertIn("improved by only", ended["text"])
            self.assertIn("bigger design", ended["text"])
            record = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((record["status"], record["budget_reached"], record["ended"]["text"]), ("completed", "plateau", ended["text"]))
            self.assertEqual((result["status"], result["budget_reached"]), ("completed", "plateau"))
            self.assertIn(ended["text"], [e.get("text") for e in events if e["type"] == "info"])

    def test_a_run_that_keeps_improving_is_not_stopped(self):
        steps = []

        def falling(cfg, examples, out, emit=None, stop=None, **options):
            for step in range(1, 61):
                if stop.is_set():
                    break
                steps.append(step)
                emit({"type": "metric", "step": step, "val_loss": 3. * .97 ** step, "loss": 3.})
            return {"status": "completed", "step": steps[-1]}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(studio_model, "train", falling):
            job, out, _, result = self.run_job(Path(directory), {"window_steps": 10, "min_gain": .01, "min_steps": 0})
        self.assertEqual((steps[-1], result["status"]), (60, "completed"))
        self.assertNotIn("ended", job.params)

    def test_a_curve_that_goes_flat_stops_the_training_loop_at_the_next_step(self):
        steps = []

        def flat(cfg, examples, out, emit=None, stop=None, **options):
            for step in range(1, 100):
                if stop.is_set():
                    break
                steps.append(step)
                emit({"type": "metric", "step": step, "val_loss": 2. + (1. if step < 20 else .0001 / step)})
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "run.json").write_text("{}", encoding="utf-8")
            return {"status": "stopped", "step": steps[-1]}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(studio_model, "train", flat):
            job, out, _, result = self.run_job(Path(directory), {"window_steps": 10, "min_gain": .01, "min_steps": 25})
            self.assertEqual(json.loads((out / "run.json").read_text(encoding="utf-8"))["ended"]["reason"], "plateau")
        self.assertEqual((job.params["ended"]["step"], steps[-1]), (30, 30))  # 10 steps past the drop at step 19, and no step after the flag
        self.assertEqual((result["status"], result["budget_reached"]), ("completed", "plateau"))

    def test_a_run_without_the_rule_is_as_before(self):
        with tempfile.TemporaryDirectory() as directory:
            job, out, _, _ = self.run_job(Path(directory), None)
            self.assertNotIn("ended", job.params)
            self.assertEqual(studio_model.load_checkpoint(out / "last.npz")["step"], TINY["steps"])

    def test_a_stop_the_person_asked_for_is_not_reported_as_a_plateau(self):
        stop, steps = threading.Event(), []

        def stopped(cfg, examples, out, emit=None, stop=None, **options):
            for step in range(1, 100):
                if stop.is_set():
                    break
                steps.append(step)
                emit({"type": "metric", "step": step, "val_loss": 2.})  # flat all along: the rule would fire at step 11
                if step == 5:
                    user.set()
            return {"status": "stopped", "step": steps[-1]}
        user = stop
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(studio_model, "train", stopped):
            job, _, _, result = self.run_job(Path(directory), {"window_steps": 10, "min_gain": .01, "min_steps": 0}, stop)
        self.assertEqual((steps[-1], result["status"]), (5, "stopped"))
        self.assertNotIn("ended", job.params)


if __name__ == "__main__":
    unittest.main()

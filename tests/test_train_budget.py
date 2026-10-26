"""A time or token budget ends the run at the end of the cool-down; dropout masks are seeded on the model's device."""
import itertools
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import studio_model
from studio_model import load_checkpoint, train

EXAMPLES = [{"text": f"Counting up from {n}: {n}, {n + 1}, {n + 2}."} for n in range(12)]
CONFIG = {"width": 16, "layers": 1, "heads": 2, "context": 32, "expansion": 2, "dropout": 0., "batch_size": 2,
          "steps": 100000, "learning_rate": .003, "warmup": 2, "validation": 0, "eval_every": 100000,
          "seed": 5, "device": "cpu", "sample_every": 0}


def run(root, **kwargs):
    tick = itertools.count()
    clock = types.SimpleNamespace(monotonic=lambda: next(tick) * .01, sleep=lambda _s: None)  # every clock read costs 10 ms
    with mock.patch.object(studio_model, "time", clock):
        train(dict(CONFIG, **kwargs.pop("config", {})), EXAMPLES, root, telemetry_every=0, **kwargs)
    rows = [r for r in studio_model.read_metrics(root / "metrics.jsonl") if r["step"]]
    return rows, load_checkpoint(root / "last.npz")


class Budget(unittest.TestCase):
    def test_wall_budget_still_cools_down(self):
        with tempfile.TemporaryDirectory() as d:
            rows, last = run(Path(d) / "r", wall_seconds=12)
        peak = CONFIG["learning_rate"]
        self.assertLess(last["step"], CONFIG["steps"])
        self.assertGreater(max(r["lr"] for r in rows), .9 * peak)  # it did reach the steady phase
        self.assertLessEqual(rows[-1]["lr"], .11 * peak)  # and the run ended on the schedule's floor, not mid-plateau

    def test_token_budget_still_cools_down(self):
        with tempfile.TemporaryDirectory() as d:
            rows, _ = run(Path(d) / "r", token_budget=4000)
        self.assertLessEqual(rows[-1]["lr"], .11 * CONFIG["learning_rate"])


class Resume(unittest.TestCase):
    BUDGET = 4000
    BATCH = CONFIG["batch_size"] * CONFIG["context"]  # the most tokens one step can add

    def test_a_resumed_token_budget_run_trains_the_budget_once(self):
        stop = threading.Event()

        def halfway(event):
            if event.get("type") == "metric" and event.get("trained_tokens", 0) >= self.BUDGET // 2:
                stop.set()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run(root / "a", token_budget=self.BUDGET, emit=halfway, stop=stop)
            half = load_checkpoint(root / "a" / "last.npz")
            _, final = run(root / "b", token_budget=self.BUDGET, resume=root / "a" / "last.npz")
            record = studio_model.read_metrics(root / "b" / "metrics.jsonl")[-1]
        self.assertTrue(self.BUDGET // 2 <= half["trained_tokens"] < self.BUDGET)
        self.assertGreater(half["elapsed_seconds"], 0)
        self.assertGreater(final["step"], half["step"])
        self.assertLessEqual(final["trained_tokens"], self.BUDGET + self.BATCH)  # one batch over at most, not a second budget
        self.assertGreater(final["elapsed_seconds"], half["elapsed_seconds"])  # the clock carries over too
        self.assertEqual(record["trained_tokens"], final["trained_tokens"])

    def test_a_budget_already_spent_stops_the_resume_at_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, done = run(root / "a", token_budget=1500)
            _, again = run(root / "b", token_budget=1500, resume=root / "a" / "last.npz")
        self.assertGreaterEqual(done["trained_tokens"], 1500 - self.BATCH)
        self.assertLessEqual(again["trained_tokens"], 1500 + self.BATCH)  # the resume adds at most the last cool-down steps, not 1500 more


class Dropout(unittest.TestCase):
    def final_weights(self, seed):
        with tempfile.TemporaryDirectory() as d:
            train(dict(CONFIG, steps=4, dropout=.2, seed=seed, eval_every=4), EXAMPLES, Path(d) / "r", telemetry_every=0)
            return load_checkpoint(Path(d) / "r" / "last.npz")["model"]

    def test_masks_are_reproducible_per_seed(self):
        a, b, c = self.final_weights(5), self.final_weights(5), self.final_weights(6)
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        self.assertFalse(all(np.array_equal(a[k], c[k]) for k in a))


if __name__ == "__main__":
    unittest.main()

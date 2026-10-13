"""A time or token budget ends the run at the end of the cool-down; dropout masks are seeded on the model's device."""
import itertools
import tempfile
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

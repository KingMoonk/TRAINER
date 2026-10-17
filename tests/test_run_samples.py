"""Run page samples: the raw-text sampler, train()'s sample events and the page's pure row logic (window-free)."""
import json
import tempfile
import unittest
from pathlib import Path

from studio_ia.services import WorkspaceServices
from studio_model import LocalModel, train

EXAMPLES = [{"text": f"Counting up from {n}: {n}, {n + 1}, {n + 2}."} for n in range(12)]
CONFIG = {"width": 16, "layers": 1, "heads": 2, "context": 32, "expansion": 2, "dropout": 0., "batch_size": 2,
          "steps": 8, "learning_rate": .003, "warmup": 0, "validation": 0, "eval_every": 2, "seed": 3,
          "device": "cpu", "sample_every": 2, "sample_prompt": "Counting up"}


class Samples(unittest.TestCase):
    def test_train_logs_and_emits_samples_and_the_sampler_is_raw_and_greedy(self):
        events = []
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "r"
            train(CONFIG, EXAMPLES, run, emit=events.append, telemetry_every=0)
            logged = [json.loads(line) for line in (run / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["step"] for x in logged], [4, 8])  # every 2nd eval interval of 2 steps
            self.assertEqual([e["step"] for e in events if e["type"] == "sample"], [4, 8])
            model = LocalModel(run / "last.npz", device="cpu")
            self.assertEqual(model.complete("Counting up"), logged[-1]["text"])  # same weights, greedy: same text
            curve = WorkspaceServices._run_samples(run)
            self.assertEqual((curve["sample_every"], curve["prompt"], [x["step"] for x in curve["samples"]]), (4, "Counting up", [4, 8]))

    def test_sampling_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as d:
            train(dict(CONFIG, sample_every=0), EXAMPLES, Path(d) / "r", telemetry_every=0)
            self.assertFalse((Path(d) / "r" / "samples.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

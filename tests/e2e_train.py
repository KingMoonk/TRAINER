"""A tiny real training run on the committed GSM8K sample: it learns, the checkpoint round-trips, it generates, and it resumes.

Not named test_*.py so `unittest discover` skips it; CI runs `python -m unittest tests.e2e_train -v`.
"""
import json
import random
import tempfile
import unittest
from pathlib import Path

from studio_core import normalize, parse_file
from studio_model import ScratchTransformer, Tokenizer, LocalModel, batch_arrays, load_checkpoint, partition_examples, samples_from, train, validate_config

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.jsonl"
CONFIG = {"width": 32, "layers": 1, "heads": 2, "context": 128, "batch_size": 8, "steps": 150, "learning_rate": 0.01, "warmup": 10,
          "eval_every": 25, "dropout": 0.0, "device": "cpu", "seed": 1}
# Test error (held-out cross-entropy, nats per answer token) on the fixture with this config, 150 steps (~1 s on a laptop), seeds 1-8:
# starts at 5.55-5.57 and ends at 2.71-3.46, a drop of 2.09 or more on every seed (seed 1, used here: 5.566 -> 2.975). The margin leaves 0.5 nat of headroom.
MIN_DROP = 1.5


def heldout_error(directory, cfg, checkpoint):
    """The test error again, outside train(): same held-out windows and batching as train()'s own check, weights from the checkpoint file."""
    examples = [json.loads(line) for line in (Path(directory) / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _, held, _ = partition_examples(examples, cfg["validation"], random.Random(cfg["seed"]))
    data = load_checkpoint(checkpoint)
    tokenizer = Tokenizer.restore(data["tokenizer"])
    model = ScratchTransformer(cfg, tokenizer.vocab_size)
    model.load_weights(data["model"])
    samples = samples_from(held, tokenizer, cfg)
    total, count = 0.0, 0
    for start in range(0, min(len(samples), cfg["batch_size"] * 20), cfg["batch_size"]):
        x, y = batch_arrays(samples[start:start + cfg["batch_size"]])
        targets = int((y != -100).sum())
        total += model.loss(x, y) * targets
        count += targets
    return total / count


class TrainEndToEnd(unittest.TestCase):
    def test_train_save_load_generate_resume(self):
        examples = [e for row in parse_file(FIXTURE) if (e := normalize(row))]
        self.assertGreaterEqual(len(examples), 150)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out, errors = Path(tmp.name) / "run", []
        record = train(CONFIG, examples, out, lambda e: errors.append((e["step"], e["val_loss"])) if e["type"] == "metric" and e.get("val_loss") is not None else None)
        self.assertEqual((record["status"], record["step"]), ("completed", 150))
        (first_step, first), (last_step, last) = errors[0], errors[-1]
        print(f"test error: {first:.3f} at step {first_step} -> {last:.3f} at step {last_step}")
        self.assertEqual((first_step, last_step), (0, 150))
        self.assertLess(last, first - MIN_DROP, errors)

        cfg = validate_config(CONFIG)
        for name in ("last.npz", "best.npz"):
            self.assertTrue((out / name).exists(), name)
        reloaded = heldout_error(out, cfg, out / "last.npz")
        print(f"in-memory {last:.6f} vs reloaded checkpoint {reloaded:.6f}")
        self.assertAlmostEqual(reloaded, last, places=5)

        text = LocalModel(out / "last.npz", device="cpu").generate([{"role": "user", "content": examples[0]["user"]}], max_tokens=40, temperature=0)
        self.assertIsInstance(text, str)
        print(f"generated {len(text)} chars: {text[:60]!r}")

        resumed = train({**CONFIG, "steps": 200}, examples, Path(tmp.name) / "resumed", resume=out / "last.npz")
        self.assertEqual((resumed["status"], resumed["step"], resumed["resumed_from"]), ("completed", 200, str(out / "last.npz")))
        self.assertTrue((Path(tmp.name) / "resumed" / "last.npz").exists())
        self.assertEqual(load_checkpoint(Path(tmp.name) / "resumed" / "last.npz")["step"], 200)


if __name__ == "__main__":
    unittest.main()

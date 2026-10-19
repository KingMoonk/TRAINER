"""Fixed evaluation budgets: comparable curves, cheap startup, unchanged training trajectory."""
import json
import os
import statistics
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import studio_model as sm


CONFIG = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2,
              dropout=.2, batch_size=2, steps=2, learning_rate=.003, warmup=0,
              validation=0, eval_every=1, seed=5, sample_every=0)
EXAMPLES = [{"text": f"Example {i}: " + "a useful training sequence " * 4} for i in range(20)]


class EvaluationBudget(unittest.TestCase):
    def test_budget_is_validated_and_defaults_to_64(self):
        self.assertEqual(sm.validate_config({})["eval_windows"], 64)
        self.assertEqual(sm.validate_config({"eval_windows": "400"})["eval_windows"], 400)
        for value in (0, -1, 100001, "bad"):
            with self.assertRaises(ValueError):
                sm.validate_config({"eval_windows": value})

    def test_baseline_and_later_evaluations_use_same_spread_windows(self):
        batches = []
        original = sm.ScratchTransformer.loss

        def observed(model, x, y):
            batches.append((np.asarray(x).copy(), np.asarray(y).copy()))
            return original(model, x, y)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sm.ScratchTransformer, "loss", observed):
            sm.train(dict(CONFIG, eval_windows=7), EXAMPLES, Path(tmp) / "run", telemetry_every=0)
        # No validation split: the baseline and both checkpoint scores evaluate the training subset.
        self.assertEqual(len(batches), 12)
        for offset in (4, 8):
            for baseline, later in zip(batches[:4], batches[offset:offset + 4]):
                for expected, actual in zip(baseline, later):
                    np.testing.assert_array_equal(expected, actual)
        self.assertEqual(sum(len(x) for x, _ in batches[:4]), 7)

    def test_evaluation_budget_does_not_change_weights_or_training_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = []
            for limit in (3, 400):
                folder = Path(tmp) / str(limit)
                sm.train(dict(CONFIG, eval_windows=limit), EXAMPLES, folder, telemetry_every=0)
                checkpoints.append(sm.load_checkpoint(folder / "last.npz"))
        a, b = checkpoints
        self.assertEqual(a["sampler_state"], b["sampler_state"])
        self.assertEqual(a["model_rng"], b["model_rng"])
        for key in a["model"]:
            np.testing.assert_array_equal(a["model"][key], b["model"][key])
        for state in a["optimizer_state"]:
            for key in a["optimizer_state"][state]:
                np.testing.assert_array_equal(a["optimizer_state"][state][key], b["optimizer_state"][state][key])

    def test_validation_baseline_and_checkpoint_scores_use_same_subset(self):
        batches = []
        original = sm.ScratchTransformer.loss

        def observed(model, x, y):
            batches.append((np.asarray(x).copy(), np.asarray(y).copy()))
            return original(model, x, y)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sm.ScratchTransformer, "loss", observed):
            sm.train(dict(CONFIG, eval_windows=7, validation=.2), EXAMPLES, Path(tmp) / "run", telemetry_every=0)
        self.assertEqual(len(batches), 16)  # baseline train + baseline validation + two validation scores
        for offset in (8, 12):
            for baseline, later in zip(batches[4:8], batches[offset:offset + 4]):
                for expected, actual in zip(baseline, later):
                    np.testing.assert_array_equal(expected, actual)

    def test_small_split_evaluates_every_window(self):
        windows, evaluated = [], []
        samples_from, loss = sm.samples_from, sm.ScratchTransformer.loss

        def samples(*args):
            result = samples_from(*args)
            windows.append(len(result))
            return result

        def observed(model, x, y):
            evaluated.append(len(x))
            return loss(model, x, y)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sm, "samples_from", samples), \
                mock.patch.object(sm.ScratchTransformer, "loss", observed):
            sm.train(dict(CONFIG, eval_windows=400), EXAMPLES, Path(tmp) / "run", telemetry_every=0)
        self.assertGreater(windows[0], 0)
        self.assertLess(windows[0], 400)
        self.assertEqual(windows[1], 0)
        self.assertEqual(sum(evaluated), windows[0] * 3)

    def test_resume_keeps_saved_budget_and_legacy_resumes_keep_400(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                first, second = Path(tmp) / "first", Path(tmp) / "second"
                sm.train(dict(CONFIG, eval_windows=7), EXAMPLES, first, telemetry_every=0)
                if legacy:
                    path = first / "last.npz"
                    with np.load(path, allow_pickle=False) as archive:
                        arrays = {key: archive[key] for key in archive.files}
                    metadata = json.loads(str(arrays["metadata"]))
                    del metadata["config"]["eval_windows"]
                    arrays["metadata"] = np.array(json.dumps(metadata))
                    np.savez_compressed(path, **arrays)
                sm.train(dict(CONFIG, steps=3, eval_windows=2), EXAMPLES, second,
                         resume=first / "last.npz", telemetry_every=0)
                checkpoint = sm.load_checkpoint(second / "last.npz")
                self.assertEqual(checkpoint["config"]["eval_windows"], 400 if legacy else 7)
                self.assertEqual(json.loads((second / "config.json").read_text())["eval_windows"], 400 if legacy else 7)

    def test_rebase_may_choose_new_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            sm.train(dict(CONFIG, eval_windows=7), EXAMPLES, first, telemetry_every=0)
            sm.train(dict(CONFIG, steps=1, eval_windows=2), EXAMPLES, second,
                     resume=first / "last.npz", rebase=True, telemetry_every=0)
            self.assertEqual(sm.load_checkpoint(second / "last.npz")["config"]["eval_windows"], 2)


@unittest.skipUnless(os.environ.get("GROUNDWORK_EVAL_BENCHMARK") == "1", "opt-in isolated startup benchmark")
class EvaluationStartupBenchmark(unittest.TestCase):
    def test_repeated_attempts(self):
        """Real training in temporary folders, stopped after first step; never submits a runtime job.

        Set GROUNDWORK_EVAL_BENCHMARK=1 and optionally GROUNDWORK_EVAL_DEVICE=cpu/cuda.
        Both budgets warm their shard cache before three measured attempts. Default-sized model,
        >2M bytes of deterministic text, identical seed, dropout and first-step weight checks.
        """
        device = os.environ.get("GROUNDWORK_EVAL_DEVICE", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        examples = [{"text": f"Document {i}: " + ("Learning representations from language requires useful training data. " * 60)}
                    for i in range(600)]
        config = dict(CONFIG, device=device, width=64, layers=2, heads=4, context=256,
                      expansion=4, batch_size=4, dropout=.1, validation=.1, eval_every=25)
        timings = {400: [], 64: []}
        reference = None
        max_weight_delta = 0.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for attempt in range(4):
                # Alternate order to limit cache/clock bias; attempt zero warms both settings.
                for limit in ((400, 64) if attempt % 2 == 0 else (64, 400)):
                    stop, first_step = threading.Event(), []
                    start = time.perf_counter()

                    def emit(event):
                        if event.get("type") == "metric" and event.get("step") == 1:
                            first_step.append(time.perf_counter() - start)
                            stop.set()

                    folder = root / f"{limit}-{attempt}"
                    sm.train(dict(config, eval_windows=limit), examples, folder, emit=emit, stop=stop,
                             telemetry_every=0, stream_chars=1, shard_dir=root / "shards", snapshot_dir=root / "snapshots")
                    self.assertEqual(len(first_step), 1)
                    checkpoint = sm.load_checkpoint(folder / "last.npz")
                    if reference is None:
                        reference = checkpoint
                    else:
                        self.assertEqual(reference["sampler_state"], checkpoint["sampler_state"])
                        self.assertEqual(reference["model_rng"], checkpoint["model_rng"])
                        for key, expected in reference["model"].items():
                            actual = checkpoint["model"][key]
                            max_weight_delta = max(max_weight_delta, float(np.max(np.abs(expected - actual))))
                            np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-7)
                    if attempt:
                        timings[limit].append(first_step[0])
            result = {"device": device, "characters": sum(len(e["text"]) for e in examples),
                      "threads": torch.get_num_threads(), "max_weight_delta": max_weight_delta,
                      "seconds": timings, "medians": {k: statistics.median(v) for k, v in timings.items()}}
            print("EVALUATION_STARTUP " + json.dumps(result), flush=True)


if __name__ == "__main__":
    unittest.main()

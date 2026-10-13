"""Frozen-engine checkpoint and ModelSpec curriculum compatibility."""
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from studio_core import fingerprint
from studio_curriculum import _stage_config
from studio_model import (LocalModel, Tokenizer, batch_arrays, load_checkpoint, partition_examples,
                          samples_from, train, validate_config)
from studio_spec import config_from_spec, default_spec
from tests import legacy_gpt
from tests.torchcheck import to_np


EXAMPLES = [{"text": f"Counting up from {number}: {number}, {number + 1}, {number + 2}."}
            for number in range(8)]
CONFIG = {"width": 16, "layers": 1, "heads": 2, "context": 32, "expansion": 2,
          "dropout": .2, "batch_size": 2, "accumulation": 2, "steps": 3,
          "learning_rate": .003, "weight_decay": .01, "warmup": 0, "validation": 0,
          "eval_every": 3, "seed": 19, "objective": "all", "device": "cpu"}


class LegacyCheckpoint(unittest.TestCase):
    def test_frozen_engine_checkpoint_inference_and_resume(self):
        # Build a format-1 archive from the frozen engine, including an actual optimizer
        # update and its stochastic stream. Omit fields absent from old checkpoints.
        cfg = validate_config(CONFIG)
        cfg["windowing"] = "sequential"
        tokenizer = Tokenizer("byte")
        sampler = random.Random(cfg["seed"])
        training, _, _ = partition_examples(EXAMPLES, 0, sampler, grouped=False)
        windows = samples_from(training, tokenizer, cfg)
        old = legacy_gpt.ScratchTransformer(cfg, tokenizer.vocab_size)
        optimizer = legacy_gpt.AdamW(old, cfg["weight_decay"])

        def step(number):
            old.zero_grad()
            for _ in range(cfg["accumulation"]):
                batch = [windows[sampler.randrange(len(windows))] for _ in range(cfg["batch_size"])]
                x, y = batch_arrays(batch)
                old.loss_and_backward(x, y, 1 / cfg["accumulation"])
            # warmup=0: the default schedule uses a cosine decay to 10% of peak.
            rate = cfg["learning_rate"] * (.1 + .9 * .5 * (1 + np.cos(np.pi * number / cfg["steps"])))
            optimizer.update(old, rate)

        step(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "legacy.npz"
            legacy_config = {key: value for key, value in cfg.items() if key != "windowing"}
            metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1,
                        "config": legacy_config, "tokenizer": tokenizer.serialize(), "step": 1,
                        "best": None, "optimizer_steps": optimizer.steps, "history": [],
                        "data_hash": hashlib.sha256("".join(fingerprint(row) for row in EXAMPLES).encode()).hexdigest(),
                        "sampler_state": sampler.getstate(), "model_rng": old.rng.bit_generator.state,
                        "device": "cpu"}
            arrays = {"metadata": np.array(json.dumps(metadata))}
            arrays.update({f"weight::{key}": value.copy() for key, value in old.p.items()})
            for state in ("m", "v"):
                arrays.update({f"adam_{state}::{key}": value.copy()
                               for key, value in getattr(optimizer, state).items()})
            np.savez(checkpoint, **arrays)

            restored = load_checkpoint(checkpoint)
            self.assertEqual(restored["optimizer_steps"], 1)
            self.assertEqual(restored["model_rng"], old.rng.bit_generator.state)
            local = LocalModel(checkpoint, device="cpu")
            probe = np.array([[1, 3, 11, 12, 13]], dtype=np.int64)
            np.testing.assert_allclose(to_np(local.model.forward(probe)), old.forward(probe), atol=1e-5)

            step(2)
            step(3)
            train(CONFIG, EXAMPLES, root / "resumed", resume=checkpoint, telemetry_every=0)
            resumed = load_checkpoint(root / "resumed" / "last.npz")
            self.assertEqual(resumed["step"], 3)
            self.assertEqual(resumed["optimizer_steps"], optimizer.steps)
            self.assertEqual(resumed["model_rng"], old.rng.bit_generator.state)
            for key in old.p:
                np.testing.assert_allclose(resumed["model"][key], old.p[key], atol=1e-5, err_msg=key)
                np.testing.assert_allclose(resumed["m"][key], optimizer.m[key], atol=1e-5, err_msg=key)
                np.testing.assert_allclose(resumed["v"][key], optimizer.v[key], atol=1e-5, err_msg=key)

    def test_resume_rejects_changed_optimizer_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(dict(CONFIG, steps=1), EXAMPLES, root / "first", telemetry_every=0)
            with self.assertRaisesRegex(ValueError, "optimizer"):
                train(dict(CONFIG, learning_rate=.004), EXAMPLES, root / "second",
                      resume=root / "first" / "last.npz")


class CurriculumSpec(unittest.TestCase):
    def test_stage_updates_model_spec_without_changing_original(self):
        spec = default_spec()
        state = {"base_config": config_from_spec(spec), "context": 32}
        stage = {"name": "Language", "steps": 10, "learning_rate": .002, "objective": "all"}
        cfg = _stage_config(state, stage, 0)
        normalized = validate_config(cfg)
        self.assertEqual(normalized["architecture"]["dims"]["n_ctx"], 32)
        self.assertEqual(normalized["architecture"]["tokenizer"]["id"], "byte")
        self.assertEqual(normalized["context"], 32)
        self.assertEqual(normalized["learning_rate"], .002)
        self.assertEqual(normalized["objective"], "all")
        self.assertEqual(spec["dims"]["n_ctx"], 256)



if __name__ == "__main__":
    unittest.main()

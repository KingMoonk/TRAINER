"""The compiled default spec must be the same model as the engine it replaced: same seed, same data, same losses."""
import unittest

import numpy as np

from studio_arch import config_from_spec, validate_spec
from studio_model import AdamW, ScratchTransformer, validate_config
from tests import legacy_gpt
from tests.torchcheck import to_np

BASE = {"width": 32, "layers": 2, "heads": 4, "context": 32, "expansion": 4, "dropout": .1, "batch_size": 4,
        "seed": 7, "device": "cpu", "learning_rate": .003}
HYBRID = {"format": "groundwork-arch/1", "name": "hybrid", "embedding": {"tokenizer": "byte", "width": 32, "context": 32, "dropout": .1},
          "blocks": [{"id": "blk_aaaaaa", "attention": {"enabled": True, "heads": 2}, "mlp": {"enabled": True, "expansion": 2, "layers": 2, "activation": "relu"}},
                     {"id": "blk_bbbbbb", "attention": {"enabled": True, "heads": 4}, "mlp": {"enabled": False}},
                     {"id": "blk_cccccc", "attention": {"enabled": False}, "mlp": {"enabled": True, "expansion": 4, "layers": 1, "activation": "gelu"}}]}


def curve(model, optimizer, steps=100, vocab=40):
    data = np.random.default_rng(3)
    losses = []
    for step in range(steps):
        x = (data.integers(0, vocab, (4, 1)) + np.arange(24)) % vocab  # counting upward: learnable in a few steps
        y = np.roll(x, -1, axis=1)
        y[:, -1] = -100
        model.zero_grad()
        losses.append(model.loss_and_backward(x, y))
        optimizer.update(model, .003 * min(1, (step + 1) / 10))
    return np.array(losses)


class Parity(unittest.TestCase):
    def check(self, config):
        old = legacy_gpt.ScratchTransformer(config, 40)
        new = ScratchTransformer(config, 40)
        self.assertEqual(set(old.p), set(new.p))
        for name in old.p:
            np.testing.assert_array_equal(old.p[name], to_np(new.p[name]), err_msg=name)
        before, after = curve(old, legacy_gpt.AdamW(old)), curve(new, AdamW(new))
        self.assertLess(before[-1], before[0] - .2)  # it really trained, so the comparison means something
        # float32 matmul order differs between NumPy and PyTorch, so the two engines drift by rounding over 100 steps
        self.assertLess(np.abs(before - after).max(), 2e-3)
        for name in old.p:
            np.testing.assert_allclose(old.p[name], to_np(new.p[name]), atol=2e-3, err_msg=name)

    def test_uniform_config_trains_the_same_for_100_steps(self):
        self.check(BASE)

    def test_per_block_architecture_trains_the_same_for_100_steps(self):
        self.check(validate_config(dict(config_from_spec(validate_spec(HYBRID)), **{k: BASE[k] for k in ("batch_size", "seed", "device", "learning_rate")})))

    def test_telemetry_has_the_same_shape_and_values(self):
        old, new = legacy_gpt.ScratchTransformer(BASE, 40), ScratchTransformer(BASE, 40)
        x = np.random.default_rng(1).integers(0, 40, (4, 24))
        y = np.roll(x, -1, axis=1)
        results = []
        for model in (old, new):
            model.keep_tape = True
            model.loss_and_backward(x, y)
            results.append(model.telemetry(model.last_tape))
        a, b = results
        self.assertAlmostEqual(a["embedding"]["grad_norm"], b["embedding"]["grad_norm"], places=6)
        self.assertAlmostEqual(a["embedding"]["rms"], b["embedding"]["rms"], places=6)
        self.assertAlmostEqual(a["output"]["grad_norm"], b["output"]["grad_norm"], places=6)
        for left, right in zip(a["blocks"], b["blocks"]):
            self.assertEqual((left["id"], left["index"]), (right["id"], right["index"]))
            self.assertAlmostEqual(left["residual_rms"], right["residual_rms"], places=6)
            np.testing.assert_allclose(left["attention"]["entropy"], right["attention"]["entropy"], atol=1e-6)
            self.assertAlmostEqual(left["attention"]["grad_norm"], right["attention"]["grad_norm"], places=6)
            np.testing.assert_allclose(left["mlp"]["neurons"], right["mlp"]["neurons"], atol=1e-6)
            self.assertAlmostEqual(left["mlp"]["active_fraction"], right["mlp"]["active_fraction"], places=6)
            self.assertAlmostEqual(left["mlp"]["grad_norm"], right["mlp"]["grad_norm"], places=6)


if __name__ == "__main__":
    unittest.main()

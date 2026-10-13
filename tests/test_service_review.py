"""Service regressions found by reviewing inspection, replay and experiments."""
import copy
import unittest

import numpy as np

from studio_compile import CompiledModel
from studio_experiments import seed_stats
from studio_inspect import backward, probe_model
from studio_snapshot import capture, one_step
from studio_spec import validate_spec


def model():
    spec = validate_spec({"dims": {"d_model": 16, "n_layer": 1, "n_ctx": 32, "dropout": 0}})
    return CompiledModel(spec, 29, seed=13, device="cpu")


class ServiceReview(unittest.TestCase):
    def test_selected_backward_keeps_caller_targets_unchanged(self):
        source = model()
        ids = np.array([[1, 2, 3, 4]])
        targets = np.array([[2, 3, 4, -100]])
        before = targets.copy()
        result = backward(source, ids, pos=1, targets=targets)
        np.testing.assert_array_equal(targets, before)
        np.testing.assert_array_equal(result["targets"], [[-100, 3, -100, -100]])

    def test_seed_stats_stay_descriptive(self):
        result = seed_stats([1., 1.1, 1.2], base=[1.2, 1.1, 1.])
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["delta"], 0.)
        self.assertIn("not a significance test", result["interpretation"])

    def test_snapshot_capture_does_not_advance_live_state(self):
        source = model()
        ids = np.array([[1, 2, 3, 4]])
        source.zero_grad()
        source.loss_and_backward(ids, np.array([[2, 3, 4, -100]]))
        original_rng = copy.deepcopy(source.rng.bit_generator.state)
        weights = {key: value.copy() for key, value in source.p.items()}
        gradients = {key: value.copy() for key, value in source.g.items()}
        snapshot = capture(source, ids, step=1)
        self.assertEqual(snapshot["step"], 1)
        self.assertEqual(source.rng.bit_generator.state, original_rng)
        for key in weights:
            np.testing.assert_array_equal(source.p[key], weights[key])
            np.testing.assert_array_equal(source.g[key], gradients[key])

    def test_snapshot_probe_attention_is_bounded_to_32_tokens(self):
        spec = validate_spec({"dims": {"d_model": 16, "n_layer": 1, "n_ctx": 64, "dropout": 0}})
        source = CompiledModel(spec, 29, seed=13, device="cpu")
        ids = np.arange(64, dtype=np.int64)[None] % 29
        snapshot = capture(source, ids, step=1)
        pattern = snapshot["arrays"]["hook::block0.attn.pattern"]
        self.assertLessEqual(pattern.shape[-1], 32)
        self.assertLessEqual(pattern.shape[-2], 32)

    def test_one_step_matches_a_real_private_optimizer_step(self):
        from studio_model import build_optimizer

        source = model()
        ids = np.array([[1, 2, 3, 4]])
        targets = np.array([[2, 3, 4, -100]])
        replay = one_step(source, ids, targets, learning_rate=.001)
        private, _ = probe_model(source)
        optimizer, _ = build_optimizer(private)
        private.zero_grad()
        expected_loss = private.loss_and_backward(ids, targets)
        optimizer.update(private, .001)
        self.assertAlmostEqual(replay["loss"], expected_loss)
        for key, value in private.p.items():
            np.testing.assert_allclose(replay["arrays"][f"weight::{key}"], value, atol=1e-12)

    def test_one_step_rejects_optimizer_with_checkpoint_path(self):
        class Optimizer:
            steps = 0

        with self.assertRaisesRegex(ValueError, "optimizer"):
            one_step("missing-checkpoint.npz", [[1, 2]], [[2, -100]], optimizer=Optimizer())


if __name__ == "__main__":
    unittest.main()

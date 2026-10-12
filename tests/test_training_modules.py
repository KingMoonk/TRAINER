"""Reference and finite-difference tests for training-side modules."""
import unittest

import numpy as np

from studio_registry import get
from studio_slots import Dims


def numeric_gradient(loss_fn, logits, targets, eps=1e-6):
    result = np.empty_like(logits)
    for idx in np.ndindex(logits.shape):
        high, low = logits.copy(), logits.copy()
        high[idx] += eps
        low[idx] -= eps
        result[idx] = (loss_fn(high, targets) - loss_fn(low, targets)) / (2 * eps)
    return result


class TrainingModuleTests(unittest.TestCase):
    def setUp(self):
        self.dims = Dims(4, 1, 5, 3, dtype=np.float64)

    def check_objective(self, short, config):
        spec = get("objective", short)
        cfg = spec.parse(config)
        objective = spec.build(cfg, self.dims, None)
        logits = np.array([[[.4, -.3, 1.1], [.2, .8, -.6]], [[-.2, .5, .1], [3., -2., .4]]])
        targets = np.array([[2, -100], [0, 1]])
        value, gradient = objective(logits, targets, gradient=True)
        np.testing.assert_allclose(gradient, numeric_gradient(objective, logits, targets), rtol=2e-7, atol=2e-8)
        self.assertEqual(value, objective(logits, targets))
        np.testing.assert_array_equal(gradient[0, 1], np.zeros(3))
        with self.assertRaisesRegex(ValueError, "no target tokens"):
            objective(logits, np.full((2, 2), -100), gradient=True)
        return value, gradient

    def test_label_smoothed_cross_entropy_fp64_finite_difference_and_mask(self):
        _, gradient = self.check_objective("ce_ls", {"epsilon": .17})
        # Standard smoothing mixture: (1-epsilon) one-hot + epsilon uniform.
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_z_loss_fp64_finite_difference_and_mask(self):
        value, gradient = self.check_objective("ce_z", {"coefficient": .03})
        self.assertGreater(value, 0)
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_lion_hand_computed_two_steps_and_decay_roles(self):
        spec = get("optimizer", "lion")
        cfg = spec.parse({"beta1": .5, "beta2": .25, "weight_decay": .1})
        optimizer = spec.build(cfg, self.dims, None)
        class Model:
            xp = np
            roles = {"w": "weight", "g": "gain"}
            lr_scale = {}
            p = {"w": np.array([2., -1.]), "g": np.array([2.])}
            grads = {"w": np.array([1., -2.]), "g": np.array([1.])}
            g = grads
        model = Model()
        optimizer.attach(model)
        optimizer.update(model, .1, clip=100)
        # direction is .5*g; decayed w then sign step; gains receive no decay.
        np.testing.assert_allclose(model.p["w"], [1.88, -.89])
        np.testing.assert_allclose(model.p["g"], [1.9])
        np.testing.assert_allclose(optimizer.m["w"], [.75, -1.5])
        model.grads["w"][:] = [-2., 1.]
        model.grads["g"][:] = [-1.]
        optimizer.update(model, .1, clip=100)
        # sign(.5*m + .5*g) = sign([-.625, -.25]); new m=.25*m+.75*g.
        np.testing.assert_allclose(model.p["w"], [1.9612, -.7811])
        np.testing.assert_allclose(model.p["g"], [2.0])
        np.testing.assert_allclose(optimizer.m["w"], [-1.3125, .375])

    def test_wsd_warmup_stability_decay_and_edges(self):
        spec = get("schedule", "wsd")
        cfg = spec.parse({"warmup": 2, "stable_fraction": .5})
        self.assertEqual(spec.parse({"warmup": 2, "stable_fraction": .5}), cfg)
        schedule = spec.build(cfg, self.dims, None)
        rates = [schedule.rate(step, 10, .2) for step in range(1, 11)]
        np.testing.assert_allclose(rates, [.1, .2, .2, .2, .2, .2, .15, .1, .05, 0])
        no_warmup = spec.build(spec.parse({"warmup": 0, "stable_fraction": 0}), self.dims, None)
        self.assertAlmostEqual(no_warmup.rate(1, 4, .2), .15)
        self.assertEqual(no_warmup.rate(4, 4, .2), 0)


if __name__ == "__main__":
    unittest.main()

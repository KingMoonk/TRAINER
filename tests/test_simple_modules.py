"""Focused numerical contracts for the first milestone-2 modules."""
import unittest

import numpy as np

from studio_registry import get
from studio_slots import Ctx, Dims


def finite_difference(layer, x, output_gradient, parameter_name=None, index=None, eps=1e-6):
    if parameter_name is None:
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        value = (np.sum(layer.forward(plus, Ctx()) * output_gradient) -
                 np.sum(layer.forward(minus, Ctx()) * output_gradient)) / (2 * eps)
        layer.forward(x, Ctx())
        return value
    parameter = layer.params[parameter_name]
    original = parameter[index]
    parameter[index] = original + eps
    high = np.sum(layer.forward(x, Ctx()) * output_gradient)
    parameter[index] = original - eps
    low = np.sum(layer.forward(x, Ctx()) * output_gradient)
    parameter[index] = original
    layer.forward(x, Ctx())
    return (high - low) / (2 * eps)


class SimpleModuleTests(unittest.TestCase):
    def setUp(self):
        self.dims = Dims(d_model=4, n_layer=1, n_ctx=5, vocab=7, dtype=np.float64)

    def test_registry_discovers_all_five_modules(self):
        for slot, short in (("position", "sinusoidal"), ("norm", "rmsnorm"), ("head", "untied"),
                            ("schedule", "constant"), ("optimizer", "sgd")):
            self.assertEqual(get(slot, short).id, f"{slot}/{short}")

    def test_sinusoidal_estimate_values_and_transparent_backward(self):
        spec = get("position", "sinusoidal")
        layer = spec.build(spec.parse(), self.dims, np.random.default_rng(1))
        x = np.zeros((2, 3, 4), dtype=np.float64)
        y = layer.forward(x, Ctx())
        np.testing.assert_allclose(y[0], y[1])
        np.testing.assert_allclose(layer.backward(np.ones_like(y)), np.ones_like(y))
        self.assertEqual(spec.estimate(spec.parse(), self.dims).params, 0)
        self.assertEqual(layer.params, {})

    def test_rmsnorm_fp64_gradients_and_accumulation(self):
        spec = get("norm", "rmsnorm")
        cfg = spec.parse({"eps": 1e-4})
        layer = spec.build(cfg, self.dims, np.random.default_rng(2))
        layer.params["g"][:] = [0.7, 1.1, -0.4, 1.4]
        x = np.array([[[.3, -.8, 1.2, .5], [1.1, .2, -.6, .9]]], dtype=np.float64)
        dy = np.array([[[.4, .1, -.7, .2], [-.5, .8, .3, -.2]]], dtype=np.float64)
        layer.forward(x, Ctx())
        dx = layer.backward(dy)
        for idx in np.ndindex(x.shape):
            self.assertAlmostEqual(dx[idx], finite_difference(layer, x, dy, index=idx), places=7)
        for idx in np.ndindex(layer.params["g"].shape):
            self.assertAlmostEqual(layer.grads["g"][idx], finite_difference(layer, x, dy, "g", idx), places=7)
        saved = layer.grads["g"].copy()
        layer.forward(x, Ctx())
        layer.backward(dy)
        np.testing.assert_allclose(layer.grads["g"], 2 * saved)
        self.assertEqual(spec.estimate(cfg, self.dims).params, 4)

    def test_untied_head_fp64_gradients_and_exact_estimate(self):
        spec = get("head", "untied")
        layer = spec.build(spec.parse(), self.dims, np.random.default_rng(3))
        layer.params["output.w"][:] = np.arange(28).reshape(4, 7) / 20
        layer.params["output.b"][:] = np.linspace(-.2, .2, 7)
        x = np.array([[[.2, -.4, .7, 1.], [-.3, .8, .1, -.5]]], dtype=np.float64)
        dy = np.arange(14, dtype=np.float64).reshape(1, 2, 7) / 9 - .5
        layer.forward(x, Ctx())
        dx = layer.backward(dy)
        for idx in np.ndindex(x.shape):
            self.assertAlmostEqual(dx[idx], finite_difference(layer, x, dy, index=idx), places=7)
        for name in layer.params:
            for idx in np.ndindex(layer.params[name].shape):
                self.assertAlmostEqual(layer.grads[name][idx], finite_difference(layer, x, dy, name, idx), places=7)
        self.assertEqual(spec.estimate(spec.parse(), self.dims).params, 35)

    def test_constant_schedule_reference(self):
        schedule = get("schedule", "constant").build(None, self.dims, None)
        for step, total in ((1, 1), (1, 50), (25, 50), (50, 50)):
            self.assertEqual(schedule.rate(step, total, .037), .037)

    def test_sgd_hand_computed_momentum_decay_and_roles(self):
        spec = get("optimizer", "sgd")
        cfg = spec.parse({"lr": .1, "momentum": .5, "weight_decay": .2})
        opt = spec.build(cfg, self.dims, None)
        class Model:
            xp = np
            roles = {"w": "weight", "g": "gain"}
            lr_scale = {}
            p = {"w": np.array([2.]), "g": np.array([2.])}
            grads = {"w": np.array([3.]), "g": np.array([3.])}
            g = grads
        model = Model()
        opt.attach(model)
        opt.update(model, .1, clip=100)
        np.testing.assert_allclose(model.p["w"], [1.66])  # 2 - .1 * (3 + .2*2)
        np.testing.assert_allclose(model.p["g"], [1.7])   # gains are not decayed
        model.grads["w"][:] = 1
        model.grads["g"][:] = 1
        opt.update(model, .1, clip=100)
        # v = .5*(3.4) + (1 + .2*1.66), gain v = .5*3 + 1
        np.testing.assert_allclose(model.p["w"], [1.3568])
        np.testing.assert_allclose(model.p["g"], [1.45])


if __name__ == "__main__":
    unittest.main()

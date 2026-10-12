"""Independent sequence-mixer references, temporal gradients and issue count gates."""
import copy
import math
import unittest

import numpy as np

from studio_compile import CompiledModel
from studio_estimate import estimate
from studio_registry import get
from studio_slots import Ctx, Dims
from studio_spec import validate_spec


MIXERS = ("linear", "conv", "gru", "lstm", "ssm")


class SequenceModules(unittest.TestCase):
    def test_exact_per_block_counts_and_compiler_names(self):
        dims = Dims(256, 4, 256, 50257)
        expected = {"linear": 263_168, "conv": 67_072, "gru": 394_752,
                    "lstm": 526_336, "ssm": 437_760}
        for name, count in expected.items():
            with self.subTest(module=name):
                module = get("mixer", name)
                self.assertEqual(module.estimate(module.parse(), dims).params, count)
                small = validate_spec({"dims": {"d_model": 16, "n_layer": 1, "n_ctx": 32, "dropout": 0},
                                       "block": {"mixer": {"id": name}}})
                model = CompiledModel(small, 23, device="cpu")
                self.assertEqual(estimate(small, 23)["total"], model.parameter_count)
                self.assertEqual(sum(value.size for key, value in model.p.items() if model.owner[key] == ("mixer", 0)),
                                 module.estimate(module.parse(), Dims(16, 1, 32, 23)).params)

    def test_temporal_gradient_flows_back_and_never_forward(self):
        rng = np.random.default_rng(25)
        for name in MIXERS:
            with self.subTest(module=name):
                module = get("mixer", name)
                raw = {"state": 2, "expand": 1, "kernel": 2, "dt_rank": 2} if name == "ssm" else {}
                cfg = module.parse(raw)
                layer = module.build(cfg, Dims(4, 1, 4, 9, dtype=np.float64), rng)
                for parameter in layer.params.values():
                    parameter[...] = rng.normal(0, .25, parameter.shape)
                x = rng.normal(0, .4, (1, 4, 4))
                dy = np.zeros_like(x)
                dy[:, -1] = rng.normal(0, .3, (1, 4))
                layer.forward(x, Ctx())
                dx = layer.backward(dy)
                self.assertGreater(float(np.max(np.abs(dx[:, 0]))), 1e-12)
                index = (0, 0, 0)
                old = x[index]
                x[index] = old + 1e-5
                plus = float(np.sum(layer.forward(x, Ctx()) * dy))
                x[index] = old - 1e-5
                minus = float(np.sum(layer.forward(x, Ctx()) * dy))
                x[index] = old
                self.assertLess(abs(dx[index] - (plus - minus) / 2e-5), 1e-8)
                original = layer.forward(x, Ctx()).copy()
                changed = x.copy()
                changed[:, -1] += 2
                np.testing.assert_allclose(layer.forward(changed, Ctx())[:, :-1], original[:, :-1], atol=1e-12)

    def test_linear_attention_matches_direct_prefix_formula(self):
        module = get("mixer", "linear")
        cfg = module.parse({"heads": 2})
        layer = module.build(cfg, Dims(4, 1, 4, 9, dtype=np.float64), None)
        rng = np.random.default_rng(5)
        for value in layer.params.values():
            value[...] = rng.normal(0, .2, value.shape)
        x = rng.normal(0, .3, (1, 4, 4))
        ctx = Ctx(hooks=True)
        actual = layer.forward(x, ctx)
        q, k, v = (ctx.captured[name] for name in ("linear.q", "linear.k", "linear.v"))
        expected = []
        for t in range(x.shape[1]):
            weights = q[:, :, t, None, :] * k[:, :, :t + 1]
            weights = weights.sum(axis=-1)
            numerator = np.einsum("bht,bhtd->bhd", weights, v[:, :, :t + 1])
            denominator = weights.sum(axis=-1, keepdims=True) + cfg.eps
            expected.append(numerator / denominator)
        stream = np.stack(expected, axis=2).transpose(0, 2, 1, 3).reshape(x.shape)
        reference = stream @ layer.params["linear.proj.w"] + layer.params["linear.proj.b"]
        np.testing.assert_allclose(actual, reference, atol=1e-12)

    def test_ssm_selective_scan_matches_scalar_reference(self):
        module = get("mixer", "ssm")
        cfg = module.parse({"state": 1, "expand": 1, "kernel": 1, "dt_rank": 1})
        layer = module.build(cfg, Dims(1, 1, 4, 9, dtype=np.float64), None)
        for parameter in layer.params.values():
            parameter.fill(0)
        layer.params["ssm.in.w"][...] = [[1., 1.]]
        layer.params["ssm.conv.w"][...] = 1
        layer.params["ssm.x.w"][...] = [[.3, .4, .5]]
        layer.params["ssm.dt.w"][...] = 1
        layer.params["ssm.A_log"][...] = 0
        layer.params["ssm.D"][...] = 1
        layer.params["ssm.out.w"][...] = 1
        values = [0.2, -0.1, 0.3]
        actual = layer.forward(np.array(values, dtype=np.float64).reshape(1, 3, 1), Ctx()).reshape(-1)
        state = 0.
        expected = []
        for value in values:
            signal = value / (1 + math.exp(-value))
            dt = math.log1p(math.exp(.3 * signal))
            state = math.exp(-dt) * state + dt * signal * (.4 * signal)
            y = state * (.5 * signal) + signal
            expected.append(y * value / (1 + math.exp(-value)))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_hybrid_sequence_counts(self):
        spec = validate_spec({"dims": {"d_model": 16, "n_layer": 2, "n_ctx": 32, "dropout": 0}})
        for name in MIXERS:
            with self.subTest(module=name):
                changed = copy.deepcopy(spec)
                changed["blocks"] = [{"index": 1, "mixer": {"id": name}}]
                normalized = validate_spec(changed)
                self.assertEqual(estimate(normalized, 29)["total"],
                                 CompiledModel(normalized, 29, device="cpu").parameter_count)


if __name__ == "__main__":
    unittest.main()

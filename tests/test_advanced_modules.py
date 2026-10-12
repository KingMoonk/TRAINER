"""References and integration gates for rotary/linear positions, GQA and gated/factorized layers."""
import copy
import unittest

import numpy as np

from studio_compile import CompiledModel
from studio_estimate import estimate
from studio_modules.position.alibi import slopes
from studio_registry import get
from studio_slots import Dims
from studio_spec import validate_spec
from studio_summary import bin_edges


def model_spec():
    return validate_spec({"dims": {"d_model": 16, "n_layer": 1, "n_ctx": 32, "dropout": 0},
                          "position": {"id": "rope"}, "embedding": {"id": "factorized", "config": {"rank": 4}},
                          "block": {"mixer": {"id": "gqa", "config": {"heads": 4, "kv_heads": 2}},
                                    "channel": {"id": "swiglu"}},
                          "train": {"precision": {"id": "fp64"}}})


class AdvancedModules(unittest.TestCase):
    def test_exact_issue_parameter_deltas(self):
        baseline = validate_spec({"dims": {"d_model": 256, "n_layer": 4, "n_ctx": 256, "dropout": 0}})
        original = estimate(baseline, 50257)["total"]
        for name in ("rope", "alibi"):
            changed = copy.deepcopy(baseline)
            changed["position"] = {"id": name}
            self.assertEqual(estimate(validate_spec(changed), 50257)["total"] - original, -65_536)
        changed = copy.deepcopy(baseline)
        changed["block"]["channel"] = {"id": "swiglu"}
        self.assertEqual(estimate(validate_spec(changed), 50257)["total"] - original, 11_264)
        for kv_heads, delta in ((2, -263_168), (1, -394_752)):
            changed = copy.deepcopy(baseline)
            changed["block"]["mixer"] = {"id": "gqa", "config": {"heads": 4, "kv_heads": kv_heads}}
            self.assertEqual(estimate(validate_spec(changed), 50257)["total"] - original, delta)
        changed = copy.deepcopy(baseline)
        changed["embedding"] = {"id": "factorized", "config": {"rank": 64}}
        factorized = estimate(validate_spec(changed), 50257)
        embedding = next(row for row in factorized["rows"] if row["slot"] == "embedding")
        self.assertEqual(embedding["params"], 50257 * 64 + 64 * 256)

    def test_rope_is_orthogonal_and_backward_is_inverse(self):
        position = get("position", "rope").build(get("position", "rope").parse(),
                                                  Dims(16, 1, 32, 19, dtype=np.float64), None)
        q = np.random.default_rng(9).normal(size=(2, 4, 7, 4))
        k = np.random.default_rng(10).normal(size=(2, 2, 7, 4))
        rq, rk = position.rotate(q, k)
        np.testing.assert_allclose(np.sum(rq * rq, axis=-1), np.sum(q * q, axis=-1), atol=1e-12)
        np.testing.assert_allclose(np.sum(rk * rk, axis=-1), np.sum(k * k, axis=-1), atol=1e-12)
        bq, bk = position.rotate_back(rq, rk)
        np.testing.assert_allclose(bq, q, atol=1e-12)
        np.testing.assert_allclose(bk, k, atol=1e-12)

    def test_alibi_slopes_bias_and_rows(self):
        self.assertEqual(len(slopes(3)), 3)
        self.assertEqual(slopes(4), [.25, .0625, .015625, .00390625])
        position = get("position", "alibi").build(None, Dims(16, 1, 32, 19, dtype=np.float64), None)
        full = position.score_bias(7, 4)
        self.assertEqual(full.shape, (4, 7, 7))
        np.testing.assert_array_equal(full[:, 2:5], position.score_bias_rows(2, 5, 7, 4))
        np.testing.assert_array_equal(np.diagonal(full, axis1=1, axis2=2), 0)
        self.assertTrue(np.all(full[:, 4, 1] < full[:, 4, 3]))

    def test_tied_factorized_and_gqa_rope_gradients(self):
        spec = model_spec()
        model = CompiledModel(spec, 23, seed=8, device="cpu")
        ids = np.array([[1, 2, 3, 4]])
        targets = np.array([[2, 3, 4, 5]])
        model.zero_grad()
        model.loss_and_backward(ids, targets)
        for name in ("token.low", "token.up", "block0.gqa.q.w", "block0.gqa.k.w", "block0.gqa.v.w",
                     "block0.gate.w", "block0.up.w", "block0.down.w"):
            with self.subTest(parameter=name):
                parameter = model.p[name]
                index = tuple(0 for _ in parameter.shape)
                old = parameter[index]
                parameter[index] = old + 1e-6
                plus = model.loss(ids, targets)
                parameter[index] = old - 1e-6
                minus = model.loss(ids, targets)
                parameter[index] = old
                expected = (plus - minus) / 2e-6
                self.assertLess(abs(model.g[name][index] - expected), 1e-8 + 1e-6 * abs(expected))

    def test_gqa_summary_matches_dense_and_mqa_cache(self):
        for kv_heads, position in ((1, "rope"), (2, "alibi")):
            with self.subTest(kv_heads=kv_heads, position=position):
                spec = model_spec()
                spec["block"]["mixer"]["config"]["kv_heads"] = kv_heads
                spec["position"] = {"id": position}
                model = CompiledModel(spec, 23, seed=3, device="cpu")
                ids = np.array([[1, 2, 3, 4, 5, 6]])
                dense, tape = model.forward(ids, tape=True, hooks={"block0.attn.pattern"})
                summarized, mass = model.forward(ids, attention_summary=3)
                np.testing.assert_allclose(summarized, dense, atol=1e-10)
                self.assertEqual(mass["block0.attn.pattern"].shape, (4, 3, 3))
                pattern = tape.captured["block0.attn.pattern"]
                np.testing.assert_allclose(pattern.sum(axis=-1), 1, atol=1e-12)
                np.testing.assert_array_equal(np.triu(pattern[0, 0], 1), 0)
                edges = bin_edges(ids.shape[1], 3)
                expected = np.empty((4, 3, 3))
                for qi, (qa, qb) in enumerate(zip(edges[:-1], edges[1:])):
                    for ki, (ka, kb) in enumerate(zip(edges[:-1], edges[1:])):
                        expected[:, qi, ki] = pattern[:, :, qa:qb, ka:kb].sum(axis=(0, 2, 3)) / (qb - qa)
                np.testing.assert_allclose(mass["block0.attn.pattern"], expected, atol=1e-12)
                row = next(r for r in estimate(spec, 23)["rows"] if r["slot"] == "mixer")
                self.assertEqual(row["kv_bytes"], 2 * kv_heads * 4 * 8)


if __name__ == "__main__":
    unittest.main()

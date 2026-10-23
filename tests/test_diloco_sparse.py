"""Sparse top-k changes (#593): each member sends only the largest entries of every tensor, int8 plus indices, and keeps the rest for
its next change (residual first, then top-k, then int8). The coordinator averages the sparse changes weighted by tokens; the dense
path is unchanged when delta_sparsity is 0. Window-free."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import studio_diloco
import studio_distributed as runs
import studio_run_health as health
from studio_diloco import Member, Settings, decode_arrays, encode_arrays, encode_feedback, quantize_int8, token_average, top_k
from studio_ia.fake_distributed import FakeCoordinator
from tests import diloco_support as support
from tests.test_diloco import Clock


def tensors(seed=0):
    rng = np.random.default_rng(seed)
    return {"big": rng.normal(size=(200, 200)).astype(np.float32), "bias": rng.normal(size=4).astype(np.float32)}


class RoundTrip(unittest.TestCase):
    def test_a_sparse_body_keeps_the_largest_entries_where_they_were_and_is_smaller(self):
        arrays = tensors()
        dense, sparse = encode_arrays(arrays, "int8"), encode_arrays(arrays, "int8", sparsity=.05)
        out = decode_arrays(sparse)
        self.assertEqual({n: v.shape for n, v in out.items()}, {n: v.shape for n, v in arrays.items()})
        count = int(np.ceil(.05 * arrays["big"].size))
        kept = np.flatnonzero(out["big"].reshape(-1))
        self.assertEqual(len(kept), count)
        self.assertEqual(set(kept), set(np.argsort(-np.abs(arrays["big"].reshape(-1)))[:count]))
        np.testing.assert_allclose(out["big"].reshape(-1)[kept], arrays["big"].reshape(-1)[kept], atol=np.abs(arrays["big"]).max() / 127 / 2 + 1e-6)
        np.testing.assert_allclose(out["bias"], decode_arrays(dense)["bias"])  # too small for sparse to pay: sent dense
        self.assertLess(len(sparse), len(dense) / 3)

    def test_a_non_finite_entry_is_sent_so_the_coordinator_can_refuse_it(self):
        value = np.ones(1000, np.float32)
        value[10] = np.nan
        self.assertIn(10, top_k(value, .01))
        self.assertFalse(np.isfinite(decode_arrays(encode_arrays({"w": value}, "int8", sparsity=.01))["w"]).all())

    def test_a_body_with_indices_out_of_range_is_refused(self):
        buffer = io.BytesIO()
        np.savez(buffer, w=np.ones(2, np.int8), **{"w::scale": np.ones(1, np.float32), "w::idx": np.array([0, 9], np.int32), "w::shape": np.array([5])})
        with self.assertRaises(ValueError):
            decode_arrays(buffer.getvalue())

    def test_setting_checks(self):
        self.assertEqual(Settings().delta_sparsity, 0.)
        Settings(delta_sparsity=.05)
        for bad in (-.1, .5):
            with self.assertRaises(ValueError):
                Settings(delta_sparsity=bad)
        with self.assertRaises(ValueError):
            Settings(delta_sparsity=.05, delta_dtype="float16")


class DenseUnchanged(unittest.TestCase):
    def test_with_sparsity_off_the_body_is_byte_for_byte_the_dense_int8_one(self):
        arrays = tensors()
        expected = io.BytesIO()
        out = {}
        for name, value in arrays.items():
            out[name], out[name + "::scale"] = quantize_int8(value)
        np.savez(expected, **out)
        self.assertEqual(encode_arrays(arrays, "int8"), expected.getvalue())
        self.assertEqual(encode_arrays(arrays, "int8", sparsity=0.), expected.getvalue())
        body, residual = encode_feedback(arrays, "int8", None, sparsity=0.)
        self.assertEqual(body, expected.getvalue())
        plain_body, plain_residual = encode_feedback(arrays, "int8")
        self.assertEqual(body, plain_body)
        for name in arrays:
            np.testing.assert_array_equal(residual[name], plain_residual[name])

    def test_a_member_of_a_run_without_the_setting_encodes_dense(self):
        arrays = tensors()
        for info in ({"delta_dtype": "int8"}, {"delta_dtype": "int8", "delta_sparsity": 0.}):
            member = SimpleNamespace(info=info, residual={})
            self.assertEqual(Member._encode(member, arrays), encode_arrays(arrays, "int8"))
        member = SimpleNamespace(info={"delta_dtype": "int8", "delta_sparsity": .05}, residual={})
        self.assertEqual(Member._encode(member, arrays), encode_arrays(arrays, "int8", sparsity=.05))
        member = SimpleNamespace(info={"delta_dtype": "float16", "delta_sparsity": .05}, residual={})
        self.assertEqual(Member._encode(member, arrays), encode_arrays(arrays, "float16"))  # sparse is int8 only


class ErrorFeedback(unittest.TestCase):
    def test_what_is_not_sent_is_carried_and_added_to_the_next_change(self):
        arrays = {"w": np.full(1000, .01, np.float32)}
        arrays["w"][0] = 1.  # always the largest
        residual, sent = None, np.zeros(1000, np.float32)
        for round_ in range(1, 41):
            body, residual = encode_feedback(arrays, "int8", residual, sparsity=.01)
            sent += decode_arrays(body)["w"]
            self.assertLessEqual(np.abs(sent + residual["w"] - round_ * arrays["w"]).max(), 1e-3 * round_)  # sent + held is everything meant
        self.assertGreater(np.count_nonzero(sent[1:]), 100)  # the .01s that lost to the 1. in round 1 were sent in later rounds
        self.assertLess(np.abs(residual["w"]).max(), 1.)

    def test_the_residual_goes_in_first_then_top_k_then_int8(self):
        arrays = {"w": np.random.default_rng(3).normal(size=200).astype(np.float32)}
        _, residual = encode_feedback(arrays, "int8", None, sparsity=.05)
        body, _ = encode_feedback({"w": np.zeros(200, np.float32)}, "int8", residual, sparsity=.05)  # the next change is nothing: the held entries alone compete
        out = decode_arrays(body)["w"]
        held = residual["w"]
        top = np.argsort(-np.abs(held))[:10]
        self.assertEqual(set(np.flatnonzero(out)), set(top))
        np.testing.assert_allclose(out[top], held[top], atol=np.abs(held[top]).max() / 127)


class Average(unittest.TestCase):
    def test_the_average_of_sparse_changes_is_weighted_by_tokens(self):
        a = {"w": np.zeros(100, np.float32)}
        b = {"w": np.zeros(100, np.float32)}
        a["w"][3], b["w"][3], b["w"][50] = 1., 5., -2.
        sa, sb = (decode_arrays(encode_arrays(x, "int8", sparsity=.02)) for x in (a, b))
        mean = token_average([(10, sa), (30, sb)])["w"]
        self.assertAlmostEqual(float(mean[3]), (10 * 1. + 30 * 5.) / 40, places=1)
        self.assertAlmostEqual(float(mean[50]), 30 * -2. / 40, places=1)
        self.assertEqual(np.count_nonzero(mean), 2)

    def test_the_coordinator_takes_sparse_bodies_and_steps_by_their_token_weighted_mean(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self.addCleanup(studio_diloco.COORDINATORS.clear)
            coordinator = support.build(Path(tmp), inner_steps=2, rounds=2, outer_lr=1., outer_momentum=0., delta_dtype="int8", delta_sparsity=.05,
                                        min_members=2)
            self.assertEqual(coordinator.info()["delta_sparsity"], .05)
            coordinator.join("a")
            coordinator.join("b")
            before = {name: value.copy() for name, value in coordinator.theta.items()}
            name = max(before, key=lambda n: before[n].size)
            deltas = []
            for value in (1., 5.):
                delta = {n: np.zeros_like(v) for n, v in before.items()}
                delta[name].reshape(-1)[:3] = value
                deltas.append(delta)
            sizes = []
            for node, tokens, delta in (("a", 10, deltas[0]), ("b", 30, deltas[1])):
                body = encode_arrays(delta, "int8", sparsity=.05)
                sizes.append(len(body))
                coordinator.submit(0, node, 1, 0, tokens, 1, decode_arrays(body), 2., 3., len(body))
            moved = before[name] - coordinator.theta[name]
            np.testing.assert_allclose(moved.reshape(-1)[:3], 4., rtol=.02)
            np.testing.assert_allclose(moved.reshape(-1)[3:], 0., atol=1e-6)
            row = coordinator.history[0]
            self.assertEqual((row["delta_sparsity"], row["delta_bytes"], row["health"]["sparsity_up"], row["health"]["sparsity_down"]),
                             (.05, sum(sizes), .05, 0.))


class Health(unittest.TestCase):
    def test_a_fake_run_reports_the_bytes_and_the_sparsity_each_way(self):
        def record(settings):
            run = runs.Run("h1", "label", 1, 0., {"rounds": 2, "inner_steps": 100}, "mac", [runs.Machine("mac", "local", device="mps", device_name="M2")])
            clock = Clock()
            coordinator = FakeCoordinator(run, {"rounds": 2, "inner_steps": 100, **settings}, clock=clock, unit=1.)
            coordinator.start()
            clock.now = 100.
            return coordinator.snapshot()["history"][0]["health"]
        dense, sparse = record({"delta_dtype": "int8"}), record({"delta_dtype": "int8", "delta_sparsity": .05})
        self.assertEqual((dense["sparsity_up"], dense["sparsity_down"]), (0., 0.))
        self.assertEqual((sparse["sparsity_up"], sparse["sparsity_down"]), (.05, 0.))
        self.assertLess(sparse["bytes_up"], dense["bytes_up"])
        self.assertEqual(sparse["bytes_down"], dense["bytes_down"])
        self.assertEqual(health.record({"round": 0, "contributed": []}, {}, {}, 0)["sparsity_up"], 0.)


if __name__ == "__main__":
    unittest.main()

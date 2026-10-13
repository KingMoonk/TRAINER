"""Look-inside pages: math and decision logic without Tk, plus the real services over the fixture workspace."""
import math
import tempfile
import unittest

import numpy as np

import studio_inspect as inspect
from studio_ia.pages.head_lab import best_pattern, top_relied
from studio_ia.pages.head_zoom import derive
from studio_ia.pages.loss import fmt_p, loss_rows
from studio_ia.pages.value_map import rank_of
from studio_ia.services import FakeServices, FixtureServices
from studio_modules.tokenizer.byte import Tokenizer
from tests.test_inspect_views import tiny


class PlainFunctions(unittest.TestCase):
    def test_rank_of_counts_nonzero_singular_values(self):
        self.assertEqual(rank_of([3, 2, 1e-12, 0]), 2)
        self.assertEqual(rank_of([0, 0]), 0)

    def test_loss_rows_target_first_and_gradient_is_p_minus_onehot(self):
        p = np.array([.5, .2, .1, .1, .05, .05])
        pieces = list("abcdef")
        rows, rest, n_rest = loss_rows(p, 1, pieces, k=3)
        self.assertEqual([r[0] for r in rows], ["b", "a", "c"])
        self.assertEqual([r[2] for r in rows], [1, 0, 0])
        self.assertAlmostEqual(rows[0][3], .2 - 1)
        self.assertAlmostEqual(rows[1][3], .5)
        self.assertAlmostEqual(rest, 1 - .5 - .2 - .1)
        self.assertEqual(n_rest, 3)
        rows, _, _ = loss_rows(p, 5, pieces, k=3)  # target outside the top three is still shown
        self.assertEqual(rows[0][0], "f")
        self.assertEqual(len(rows), 4)

    def test_fmt_p(self):
        self.assertEqual(fmt_p(.5), "0.50")
        self.assertEqual(fmt_p(1 / 50257), "2.0e-05")

    def test_best_pattern_and_ranking(self):
        self.assertEqual(best_pattern({"previous": .9, "first": .1, "induction": .2}), "previous")
        self.assertEqual(best_pattern({"previous": .2, "first": .7, "induction": .6}), "first")
        self.assertIsNone(best_pattern({"previous": .2, "first": .3, "induction": .4}))
        self.assertEqual(top_relied({(0, 0): .1, (0, 1): .5, (1, 0): .3}, 2), [((0, 1), .5), ((1, 0), .3)])

    def test_bits_per_char_conversion(self):
        self.assertAlmostEqual(inspect.bits_per_char(math.log(2) * 8, 8), 1.0)
        self.assertAlmostEqual(inspect.bits_per_char(math.log(50257), 8), math.log(50257) / (8 * math.log(2)))

    def test_derive_masks_before_softmax_on_synthetic_hooks(self):
        rng = np.random.default_rng(0)
        T, h = 5, 4
        q, k, v = (rng.normal(size=(1, 2, T, h)) for _ in range(3))
        raw = q @ k.transpose(0, 1, 3, 2)
        logits = raw / 2 + rng.normal(size=(1, 2, T, T))  # a position bias: scaled is not just raw / sqrt(d)
        masked = np.where(np.tril(np.ones((T, T), bool)), logits, -1e9)
        e = np.exp(masked - masked.max(-1, keepdims=True))
        hooks = {"block0.attn.q": q, "block0.attn.k": k, "block0.attn.v": v, "block0.attn.scores": raw,
                 "block0.attn.logits": logits, "block0.attn.pattern": e / e.sum(-1, keepdims=True)}
        d = derive({"hooks": hooks, "tokens": [list("abcde")]}, 0, 1)
        self.assertLess(d["error"], 1e-4)
        self.assertFalse(d["plain_scale"])
        np.testing.assert_allclose(d["scaled"], logits[0, 1].T)  # the engine's post-scale, post-bias logits
        self.assertTrue(np.all(np.isinf(d["masked"][np.arange(T)[:, None] > np.arange(T)[None, :]])))
        hooks["block0.attn.pattern"] = hooks["block0.attn.pattern"][..., ::-1].copy()  # a wrong hook is caught
        self.assertGreater(derive({"hooks": hooks, "tokens": [list("abcde")]}, 0, 1)["error"], 1e-4)


class EngineViews(unittest.TestCase):
    def setUp(self):
        self.model = tiny()

    def test_value_map_rank_is_at_most_head_dim(self):
        info = inspect.value_map(self.model, 1, 2)
        self.assertEqual((info["heads"], len(info["singular"])), (4, 16))
        self.assertEqual(rank_of(info["singular"]), 4)  # a 16 x 16 map of rank 16 / 4
        info = inspect.value_map(tiny(heads=2), 0, 0)
        self.assertEqual(rank_of(info["singular"]), 8)

    def test_ablate_set_matches_single_head_ablate_and_zero_differs(self):
        ids = inspect.prepared_text(Tokenizer("byte"), seed=1)
        single = inspect.ablate(self.model, [ids], mode="mean", heads=[(0, 1)])["heads"][0]["delta"]
        got = inspect.ablate_set(self.model, [ids], [(0, 1)], "mean")
        self.assertAlmostEqual(got["delta"], single, places=5)
        self.assertNotAlmostEqual(inspect.ablate_set(self.model, [ids], [(0, 1)], "zero")["delta"], got["delta"], places=4)
        self.assertEqual(inspect.ablate_set(self.model, [ids], [])["delta"], 0.0)

    def test_loss_view_gradient_is_softmax_minus_onehot_and_step_zero_is_ln_v(self):
        ids = list(range(7, 20))
        trained = inspect.loss_view(self.model, ids)
        self.assertLess(trained["grad_error"], 1e-5)
        self.assertAlmostEqual(trained["loss"], -math.log(trained["probabilities"][trained["target"]]), places=4)
        fresh = inspect.loss_view(self.model, ids, trained=False)
        self.assertLess(fresh["grad_error"], 1e-5)
        self.assertAlmostEqual(fresh["loss"], math.log(len(fresh["probabilities"])), delta=.3)
        self.assertEqual(len(fresh["pieces"]), len(fresh["probabilities"]))

    def test_heldout_batch_is_fixed_and_counts_characters(self):
        ids, cpp = inspect.heldout_batch(Tokenizer("byte"), ["hello world " * 6, "abc"], 32)
        self.assertEqual(ids.shape[1], 32)
        self.assertEqual(cpp, 1.0)  # byte pieces are one character each
        with self.assertRaises(ValueError):
            inspect.heldout_batch(Tokenizer("byte"), [], 32)


class FakeServicesContract(unittest.TestCase):
    def test_fake_matches_the_page_contract(self):
        s = FakeServices()
        self.assertEqual(s.inspect_reply(None, "x", [(0, 1)], "zero"), "[reply with 1 heads off]")
        self.assertGreater(s.inspect_heads_off(None, [(0, 1)], "zero")["after"], s.inspect_heads_off(None, [(0, 1)])["after"])


class RealServices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc = FixtureServices(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.svc.close()
        cls.tmp.cleanup()

    def test_every_look_inside_call_returns_real_numbers(self):
        s, v = self.svc, None
        prepared = s.inspect_prepared(v)
        self.assertEqual(prepared["ids"][1:5], prepared["ids"][5:])
        trace = s.inspect_trace(v, "The largest planet")
        self.assertTrue(all(math.isfinite(r["norm"]) for r in trace["rows"]))
        loss = s.inspect_loss(v, "The largest planet is Jupiter")
        self.assertLess(loss["grad_error"], 1e-4)
        info = s.inspect_value_map(v, 99)
        self.assertIsNone(info["singular"])
        heads = info["heads"]
        info = s.inspect_value_map(v, heads)
        self.assertLessEqual(rank_of(info["singular"]), info["d_model"] // heads)
        ranks = s.inspect_rank_heads(v, "mean")
        self.assertEqual([r["rise"] for r in ranks], sorted((r["rise"] for r in ranks), reverse=True))
        off = [(ranks[0]["block"], ranks[0]["head"])]
        bits = s.inspect_heads_off(v, off, "mean")
        self.assertTrue(0 < bits["before"] < 20 and math.isfinite(bits["after"]))
        self.assertIsInstance(s.inspect_reply(v, "Hi", off, "zero"), str)

    def test_no_checkpoint_is_an_honest_error(self):
        class Empty(FixtureServices):
            def __init__(self):
                self.cached = {}

            def _as_version(self, version):
                return None
        with self.assertRaises(ValueError):
            Empty().inspect_trace(None, "x")


if __name__ == "__main__":
    unittest.main()

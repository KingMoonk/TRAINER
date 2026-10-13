import unittest

import numpy as np

import studio_spec
from studio_compile import CompiledModel
from studio_warmstart import warm_start

VOCAB = 263


def make_spec(layers=2, heads=4, channel="mlp_gelu", d_ff=0, tokenizer="byte"):
    spec = studio_spec.default_spec()
    spec["dims"]["n_layer"] = layers
    spec["tokenizer"] = {"id": tokenizer}
    spec["block"]["mixer"] = {"id": "attention", "config": {"heads": heads}}
    spec["block"]["channel"] = {"id": channel, "config": {"d_ff": d_ff} if d_ff else {}}
    return studio_spec.validate_spec(spec)


def trained(spec, seed=0):
    """A model whose every parameter is non-trivial, so a wrongly zeroed or fresh part shows in the logits."""
    model = CompiledModel(spec, VOCAB, device="cpu")
    rng = np.random.default_rng(seed)
    params = {k: (rng.standard_normal(model.backend.to_cpu(v).shape) * .2).astype(np.float32) for k, v in model.p.items()}
    model.load_weights(params)
    return model


def logits(model, ids):
    return model.backend.to_cpu(model.forward(ids))


class WarmStart(unittest.TestCase):
    def setUp(self):
        self.ids = np.random.default_rng(1).integers(0, 256, size=(3, 24))

    def warm(self, old_spec, new_spec):
        old = trained(old_spec)
        params, report = warm_start({k: old.backend.to_cpu(v) for k, v in old.p.items()}, old_spec, new_spec)
        new = CompiledModel(new_spec, VOCAB, device="cpu")
        new.load_weights(params)
        return old, new, report

    def assertPreserved(self, old_spec, new_spec):
        old, new, report = self.warm(old_spec, new_spec)
        np.testing.assert_allclose(logits(new, self.ids), logits(old, self.ids), atol=1e-5)
        self.assertTrue(report["preserving"], report)
        return report

    def test_add_block_preserves_function(self):
        for channel in ("mlp_gelu", "swiglu"):
            report = self.assertPreserved(make_spec(2, channel=channel), make_spec(4, channel=channel))
            self.assertIn("block3.proj.w", report["zeroed"])
            self.assertIn("block3.proj.b", report["zeroed"])
            self.assertTrue(any(n.startswith("block2.") for n in report["fresh"]))
            self.assertIn("block0.qkv.w", report["copied"])

    def test_widen_mlp_preserves_function(self):
        report = self.assertPreserved(make_spec(d_ff=256), make_spec(d_ff=512))
        self.assertIn("block0.f2.w", report["zeroed"])
        self.assertIn("block1.f1.w", report["copied"])
        self.assertIn("starts at zero", report["reason"].replace("start at zero", "starts at zero"))

    def test_widen_swiglu_preserves_function(self):
        self.assertPreserved(make_spec(channel="swiglu", d_ff=128), make_spec(channel="swiglu", d_ff=256))

    def test_add_block_and_widen_together(self):
        self.assertPreserved(make_spec(2, d_ff=256), make_spec(3, d_ff=512))

    def test_new_block_zeroed_weights_are_zero(self):
        _, new, _ = self.warm(make_spec(1), make_spec(2))
        for name in ("block1.proj.w", "block1.proj.b", "block1.f2.w", "block1.f2.b"):
            self.assertFalse(new.backend.to_cpu(new.p[name]).any(), name)

    def test_head_resplit_loads_and_is_not_preserving(self):
        old, new, report = self.warm(make_spec(heads=4), make_spec(heads=8))
        self.assertFalse(report["preserving"])
        self.assertIn("heads", report["reason"])
        self.assertIn("block0.qkv.w", report["copied"])
        self.assertFalse(np.allclose(logits(new, self.ids), logits(old, self.ids), atol=1e-5))

    def test_removing_a_block_drops_weights(self):
        old, new, report = self.warm(make_spec(3), make_spec(2))
        self.assertFalse(report["preserving"])
        self.assertIn("block2.qkv.w", report["dropped"])
        self.assertIn("dropped", report["reason"])
        self.assertEqual(len(report["fresh"]), 0)

    def test_narrowing_mlp_is_not_preserving(self):
        _, _, report = self.warm(make_spec(d_ff=512), make_spec(d_ff=256))
        self.assertFalse(report["preserving"])
        self.assertIn("block0.f1.w", report["dropped"])

    def test_tokenizer_change_is_from_scratch(self):
        old = trained(make_spec())
        params, report = warm_start({k: old.backend.to_cpu(v) for k, v in old.p.items()}, make_spec(), make_spec(tokenizer="character"))
        self.assertEqual(params, {})
        self.assertTrue(report["from_scratch"])
        self.assertFalse(report["preserving"])
        self.assertIn("from scratch", report["reason"])

    def test_identical_spec_copies_everything(self):
        report = self.assertPreserved(make_spec(), make_spec())
        self.assertEqual((report["fresh"], report["zeroed"], report["dropped"]), ([], [], []))


if __name__ == "__main__":
    unittest.main()

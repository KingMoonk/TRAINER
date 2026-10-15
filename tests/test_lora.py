"""LoRA (studio_lora): adapters start as the identity, only they train, the base stays frozen, and merging changes nothing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import studio_lora as lora
import studio_spec
from studio_model import ScratchTransformer, build_optimizer, load_checkpoint, train, validate_config

WORDS = ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")
EXAMPLES = [{"user": f"Copy: {w}", "assistant": w} for w in WORDS] * 3
TINY = dict(device="cpu", width=32, layers=2, heads=2, context=32, expansion=2, dropout=0, batch_size=4,
            steps=20, learning_rate=.004, warmup=0, validation=0, eval_every=10, objective="assistant")
IDS = np.array([[1, 8, 9, 10, 11, 12, 13, 14]])


def tiny_model():
    return ScratchTransformer(dict(TINY), 263)


class Settings(unittest.TestCase):
    def test_defaults_are_filled_in_and_none_stays_none(self):
        self.assertIsNone(lora.lora_settings(None))
        self.assertEqual(lora.lora_settings({}), {"rank": 8, "alpha": 16, "targets": ["qkv", "proj"]})
        self.assertEqual(lora.scale_of(lora.lora_settings({"rank": 4, "alpha": 8})), 2)

    def test_bad_settings_are_refused_in_plain_words(self):
        for bad in ({"rank": 0}, {"rank": 2.5}, {"rank": True}, {"alpha": 0}, {"targets": []}, {"targets": ["qkv", "qkv"]}, {"targets": [3]}, {"x": 1}, "yes"):
            with self.assertRaises(ValueError, msg=bad):
                lora.lora_settings(bad)

    def test_a_spec_and_a_config_carry_it(self):
        spec = studio_spec.default_spec()
        spec["train"]["run"] = {"steps": 10, "lora": {"rank": 4}}
        spec = studio_spec.validate_spec(spec)
        self.assertEqual(spec["train"]["run"]["lora"], {"rank": 4, "alpha": 16, "targets": ["qkv", "proj"]})
        self.assertEqual(validate_config({"architecture": spec})["lora"]["rank"], 4)
        self.assertIsNone(validate_config({})["lora"])
        spec["train"]["run"]["lora"] = {"rank": 0}
        with self.assertRaises(ValueError):
            studio_spec.validate_spec(spec)


class Adapters(unittest.TestCase):
    def test_a_fresh_adapter_is_the_identity_function(self):
        model = tiny_model()
        before = model.forward(IDS).clone()
        view = lora.attach_lora(model, {"rank": 4})
        self.assertTrue(torch.equal(model.forward(IDS), before))
        self.assertTrue(all(float(model.p[key].abs().sum()) == 0 for key in model.p if key.endswith(".b") and key.startswith("lora.")))
        self.assertEqual(sorted(view.p), sorted(model.adapter_names))
        self.assertEqual(len(view.p), 2 * 2 * 2)  # two layers x (qkv, proj) x (a, b)

    def test_only_adapters_get_gradients_and_the_optimizer_sees_only_them(self):
        model = tiny_model()
        base = {key: value.clone() for key, value in model.p.items()}
        view = lora.attach_lora(model, {"rank": 4})
        optimizer, _ = build_optimizer(view)
        self.assertEqual(set(optimizer.m), set(view.p))
        y = np.array([[-100, 9, 10, 11, 12, 13, 14, 15]])
        for step in range(3):
            model.zero_grad()
            model.loss_and_backward(IDS, y)
            frozen = [key for key in base if float(model.g[key].abs().sum()) != 0]
            self.assertEqual(frozen, [], "a frozen weight received a gradient")
            self.assertTrue(any(float(model.g[key].abs().sum()) > 0 for key in view.p if key.endswith(".b")))
            if step == 0:  # B starts at zero, so A has no gradient yet; B does
                self.assertTrue(all(float(model.g[key].abs().sum()) == 0 for key in view.p if key.endswith(".a")))
            optimizer.update(view, .01)
        for key, value in base.items():
            self.assertTrue(torch.equal(model.p[key], value), f"{key} changed")
        self.assertTrue(any(float(model.p[key].abs().sum()) > 0 for key in view.p if key.endswith(".b")))
        self.assertGreater(float(optimizer.steps), 0)

    def test_adapter_gradients_are_the_chain_rule_of_the_plain_models_weight_gradient(self):
        model = tiny_model()
        lora.attach_lora(model, {"rank": 4, "alpha": 8})
        rng = np.random.default_rng(1)
        for key in model.adapter_names:
            model.p[key].copy_(torch.as_tensor(rng.standard_normal(tuple(model.p[key].shape)) * .3, dtype=model.p[key].dtype))
        plain = tiny_model()
        plain.load_weights(model.state_arrays())
        y = np.array([[-100, 9, 10, 11, 12, 13, 14, 15]])
        model.zero_grad()
        model.loss_and_backward(IDS, y)
        plain.zero_grad()
        plain.loss_and_backward(IDS, y)
        for base in ("block0.qkv.w", "block1.proj.w"):
            a, b, scale = model.adapters[base]
            dw = plain.g[base]
            torch.testing.assert_close(model.g[b], scale * model.p[a].T @ dw, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(model.g[a], scale * dw @ model.p[b].T, atol=1e-5, rtol=1e-4)

    def test_a_few_steps_on_one_batch_lower_its_loss(self):
        model = tiny_model()
        view = lora.attach_lora(model, {"rank": 4, "alpha": 8, "targets": ["qkv", "proj", "f1", "f2"]})
        optimizer, _ = build_optimizer(view)
        y = np.array([[-100, 9, 10, 11, 12, 13, 14, 15]])
        first = model.loss(IDS, y)
        for _ in range(40):
            model.zero_grad()
            model.loss_and_backward(IDS, y)
            optimizer.update(view, .05)
        self.assertLess(model.loss(IDS, y), first - .3)

    def test_no_matching_layer_is_a_clear_error_and_leaves_the_model_plain(self):
        model = tiny_model()
        keys = set(model.p)
        with self.assertRaises(ValueError) as caught:
            lora.attach_lora(model, {"targets": ["qkv", "nonsense"]})
        self.assertIn("nonsense", str(caught.exception))
        self.assertEqual(set(model.p), keys)
        self.assertFalse(model.adapters)

    def test_targets_can_be_the_mlp_too(self):
        model = tiny_model()
        view = lora.attach_lora(model, {"targets": ["qkv", "proj", "f1", "f2"]})
        self.assertEqual(len(view.p), 2 * 4 * 2)

    def test_merge_gives_the_same_output_and_a_plain_model(self):
        model = tiny_model()
        lora.attach_lora(model, {"rank": 4, "alpha": 8})
        rng = np.random.default_rng(0)
        for key in model.adapter_names:  # as if trained
            model.p[key].copy_(torch.as_tensor(rng.standard_normal(tuple(model.p[key].shape)) * .3, dtype=model.p[key].dtype))
        unmerged = model.forward(IDS).clone()
        arrays = model.state_arrays()  # effective weights, loadable into a plain model
        plain = tiny_model()
        plain.load_weights(arrays)
        self.assertTrue(torch.allclose(plain.forward(IDS), unmerged, atol=1e-5))
        lora.merge_lora(model)
        self.assertEqual(set(model.p), set(plain.p))
        self.assertFalse(model.adapters)
        self.assertTrue(torch.allclose(model.forward(IDS), unmerged, atol=1e-5))
        self.assertFalse(torch.equal(model.p["block0.qkv.w"], tiny_model().p["block0.qkv.w"]))

    def test_merge_arrays_matches_the_formula(self):
        w = {"x.w": np.ones((3, 2), dtype=np.float32)}
        a, b = np.arange(6, dtype=np.float32).reshape(3, 2), np.ones((2, 2), dtype=np.float32)
        merged = lora.merge_arrays(w, {"x.w.a": a, "x.w.b": b}, 2.0)
        np.testing.assert_allclose(merged["x.w"], 1 + 2.0 * (a @ b))
        self.assertEqual(w["x.w"].tolist(), np.ones((3, 2)).tolist())  # the input is not modified


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        train(dict(TINY, steps=30), EXAMPLES, cls.dir / "base")
        cls.base = cls.dir / "base" / "last.npz"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def metrics(self, name):
        return load_checkpoint(self.dir / name / "last.npz")["history"]

    def test_a_few_lora_steps_lower_the_loss_and_leave_the_base_alone(self):
        different = [{"user": f"Copy: {w}", "assistant": w[::-1]} for w in WORDS] * 3  # a new task for the same model
        cfg = dict(TINY, steps=40, learning_rate=.01, lora={"rank": 4, "alpha": 8})
        record = train(cfg, different, self.dir / "tuned", init_from=self.base)
        self.assertEqual(record["status"], "completed")
        history = self.metrics("tuned")
        self.assertTrue(all(np.isfinite(row["loss"]) for row in history))
        raw = load_checkpoint(self.dir / "tuned" / "last.npz", merged=False)
        original = load_checkpoint(self.base)["model"]
        adapters = [key for key in raw["model"] if key.startswith("lora.")]
        self.assertEqual(len(adapters), 2 * 2 * 2)
        for key, value in original.items():
            np.testing.assert_array_equal(raw["model"][key], value, err_msg=key)
        merged = load_checkpoint(self.dir / "tuned" / "last.npz")["model"]
        self.assertEqual(set(merged), set(original))  # one plain model: chat and export need nothing special
        self.assertFalse(np.array_equal(merged["block0.qkv.w"], original["block0.qkv.w"]))
        self.assertTrue(np.array_equal(merged["token"], original["token"]) if "token" in merged else True)
        self.assertEqual(raw["config"]["lora"]["rank"], 4)
        self.assertTrue(all(key.startswith("lora.") for state in raw["optimizer_state"].values() for key in state))

    def test_the_merged_checkpoint_computes_what_the_adapters_did(self):
        different = [{"user": f"Copy: {w}", "assistant": w[::-1]} for w in WORDS] * 3
        train(dict(TINY, steps=20, learning_rate=.01, lora={}), different, self.dir / "merged_check", init_from=self.base)
        path = self.dir / "merged_check" / "last.npz"
        raw, merged = load_checkpoint(path, merged=False), load_checkpoint(path)
        with_adapters = ScratchTransformer(raw["config"], 263)
        lora.attach_lora(with_adapters, raw["config"]["lora"])
        with_adapters.load_weights(raw["model"])
        plain = ScratchTransformer(merged["config"], 263)
        plain.load_weights(merged["model"])
        self.assertTrue(torch.allclose(with_adapters.forward(IDS), plain.forward(IDS), atol=1e-5))

    def test_resume_continues_a_lora_run(self):
        different = [{"user": f"Copy: {w}", "assistant": w[::-1]} for w in WORDS] * 3
        cfg = dict(TINY, steps=10, learning_rate=.01, lora={"rank": 4})
        train(cfg, different, self.dir / "first", init_from=self.base)
        record = train(dict(cfg, steps=20), different, self.dir / "second", resume=self.dir / "first" / "last.npz")
        self.assertEqual(record["step"], 20)
        with self.assertRaises(ValueError):
            train(dict(cfg, steps=30, lora=None), different, self.dir / "third", resume=self.dir / "first" / "last.npz")

    def test_plain_training_is_unchanged(self):
        data = load_checkpoint(self.base)
        self.assertIsNone(data["config"].get("lora"))
        self.assertFalse(any(key.startswith("lora") for key in data["model"]))


if __name__ == "__main__":
    unittest.main()

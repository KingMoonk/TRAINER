"""Per-block architecture specs: validation, legacy equivalence, gradients, transplant, telemetry."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from studio_arch import (config_from_spec, default_spec, new_block, parameter_breakdown, parameter_names,
                         spec_from_config, spec_hash, transplant, validate_spec)
from studio_compile import CompiledModel
from studio_core import normalize
from studio_model import (DEFAULT_CONFIG, LocalModel, ScratchTransformer, load_checkpoint, parameter_estimate,
                          train, validate_config)
from studio_spec import model_spec
from tests.torchcheck import to_np


def fp64_model(cfg, vocabulary):
    """The configured model compiled in double precision, so finite differences isolate differentiation errors."""
    cfg = validate_config(cfg)
    spec = model_spec(cfg)
    spec["train"]["precision"] = {"id": "fp64"}
    return CompiledModel(spec, vocabulary, seed=cfg["seed"], device=cfg["device"], batch_size=cfg["batch_size"])


def perturb(parameter, delta):
    with torch.no_grad():
        parameter.copy_(torch.as_tensor(delta, dtype=parameter.dtype))


def mixed_spec(width=16):
    attention_only = new_block(heads=2, mlp=False)
    mlp_only = new_block(attention=False, layers=3, activation="relu", expansion=2)
    full = new_block(heads=4, expansion=2)
    spec = default_spec("mixed")
    spec["embedding"].update(width=width, context=32, dropout=0.0)
    spec["blocks"] = [attention_only, mlp_only, full]
    spec["training"].update(device="cpu", batch_size=4, steps=12, learning_rate=.004, warmup=0,
                            validation=0.0, eval_every=6, objective="assistant")
    return validate_spec(spec)


class SpecValidation(unittest.TestCase):
    def test_validate_spec_errors(self):
        def bad(change, pattern):
            spec = default_spec()
            change(spec)
            with self.assertRaisesRegex(ValueError, pattern):
                validate_spec(spec)

        bad(lambda s: s["blocks"].clear(), "between 1 and 24")
        bad(lambda s: s["blocks"].extend(new_block() for _ in range(23)), "between 1 and 24")
        bad(lambda s: s["embedding"].update(width=8), "width")
        bad(lambda s: s["embedding"].update(width=66), "divisible")
        bad(lambda s: s["embedding"].update(context=5000), "Context")
        bad(lambda s: s["blocks"][0]["attention"].update(heads=17), "heads")
        bad(lambda s: s["blocks"][0]["mlp"].update(expansion=9), "expansion")
        bad(lambda s: s["blocks"][0]["mlp"].update(layers=5), "layers")
        bad(lambda s: s["blocks"][0]["mlp"].update(activation="swish"), "activation")
        bad(lambda s: (s["blocks"][0]["attention"].update(enabled=False), s["blocks"][0]["mlp"].update(enabled=False)), "enable")
        bad(lambda s: s["blocks"][1].update(id=s["blocks"][0]["id"]), "repeats")
        bad(lambda s: s["blocks"][0].update(id="nope"), "invalid id")
        bad(lambda s: s["training"].update(learning_rate=5), "Learning rate")
        bad(lambda s: s.update(format="other/9"), "format")
        with self.assertRaises(ValueError):
            validate_spec([])

    def test_disabled_attention_ignores_head_divisibility(self):
        spec = default_spec()
        spec["embedding"]["width"] = 66
        spec["blocks"] = [new_block(heads=4, attention=False)]
        self.assertEqual(validate_spec(spec)["blocks"][0]["attention"]["enabled"], False)

    def test_normalization_is_stable_and_hashable(self):
        spec = mixed_spec()
        again = validate_spec(copy.deepcopy(spec))
        self.assertEqual(spec, again)
        self.assertEqual(spec_hash(spec), spec_hash(again))
        changed = copy.deepcopy(spec)
        changed["blocks"][0]["attention"]["heads"] = 4
        self.assertNotEqual(spec_hash(spec), spec_hash(changed))

    def test_config_roundtrip_and_breakdown(self):
        spec = mixed_spec()
        cfg = validate_config(config_from_spec(spec))
        self.assertEqual((cfg["layers"], cfg["heads"], cfg["width"]), (3, 2, 16))
        names = parameter_names(spec, 40)
        self.assertEqual(parameter_breakdown(spec, 40)["total"], sum(int(np.prod(s)) for s in names.values()))
        self.assertEqual(parameter_estimate(cfg, 40), parameter_breakdown(spec, 40)["total"])
        model = ScratchTransformer(cfg, 40)
        self.assertEqual(model.parameter_count, parameter_breakdown(spec, 40)["total"])
        self.assertEqual({k: v.shape for k, v in model.p.items()}, names)
        kinds = {c["kind"] for c in parameter_breakdown(spec)["components"]}
        self.assertEqual(kinds, {"embedding", "attention", "mlp", "norm", "output"})
        # Form-driven callers may pass the architecture as text.
        self.assertIsNone(validate_config(dict(DEFAULT_CONFIG, architecture="None"))["architecture"])
        self.assertEqual(validate_config(dict(DEFAULT_CONFIG, architecture=__import__("json").dumps(spec)))["layers"], 3)


class LegacyEquivalence(unittest.TestCase):
    def test_legacy_config_and_spec_have_identical_parameters(self):
        legacy = dict(DEFAULT_CONFIG, device="cpu", width=32, layers=3, heads=2, context=32, expansion=2)
        spec = spec_from_config(legacy)
        self.assertEqual([b["id"] for b in spec["blocks"]], ["blk_000000", "blk_000001", "blk_000002"])
        old = ScratchTransformer(legacy, 40)
        new = ScratchTransformer(dict(legacy, architecture=spec), 40)
        self.assertEqual(list(old.p), list(new.p))
        for key in old.p:
            np.testing.assert_array_equal(old.p[key], new.p[key], err_msg=key)
        names = list(old.p)
        self.assertEqual(names[:2], ["token", "position"])
        self.assertIn("block2.f2.b", names)
        x = np.array([[1, 3, 8, 9, 10, 11]])
        np.testing.assert_array_equal(old.forward(x), new.forward(x))
        self.assertEqual(parameter_estimate(legacy, 40), parameter_breakdown(spec, 40)["total"])


class MixedGradients(unittest.TestCase):
    def test_finite_differences_on_mixed_spec(self):
        for scale in (.3, .5):  # tiny init puts pre-activations on the relu kink, so perturb the weights
            cfg = config_from_spec(mixed_spec())
            model = fp64_model(cfg, 12)
            rng = np.random.default_rng(5)
            for k, v in model.p.items():
                if scale and not k.endswith(".g"):
                    perturb(v, to_np(v) + rng.normal(0, scale, tuple(v.shape)))
            x = np.array([[1, 3, 8, 2, 4, 9], [1, 3, 10, 2, 4, 11]])
            y = np.array([[-100, -100, -100, -100, 9, 2], [-100, -100, -100, -100, 11, 2]])
            model.loss_and_backward(x, y)
            self.assertTrue({"block1.fh2.w", "block1.f2.b", "block2.qkv.w"} <= set(model.p))
            self.assertNotIn("block0.f1.w", model.p)
            self.assertNotIn("block1.qkv.w", model.p)
            rng = np.random.default_rng(77)
            for key, parameter in model.p.items():
                for _ in range(2):
                    direction = rng.normal(size=tuple(parameter.shape))
                    direction /= np.linalg.norm(direction)
                    expected = float((to_np(model.g[key]) * direction).sum())
                    original = to_np(parameter).copy()
                    perturb(parameter, original + 1e-5 * direction)
                    positive = model.loss(x, y)
                    perturb(parameter, original - 1e-5 * direction)
                    negative = model.loss(x, y)
                    perturb(parameter, original)
                    self.assertAlmostEqual(expected, (positive - negative) / 2e-5, delta=2e-6 + 1e-4 * abs(expected), msg=f"{key} scale={scale}")

    def test_attention_stays_causal_with_per_block_heads(self):
        model = ScratchTransformer(config_from_spec(mixed_spec()), 40)
        a, b = model.forward(np.array([[1, 3, 8, 9, 10]])), model.forward(np.array([[1, 3, 8, 20, 21]]))
        np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-6, rtol=1e-6)


class Transplant(unittest.TestCase):
    def build(self, spec, seed=1):
        cfg = config_from_spec(spec)
        cfg["seed"] = seed
        return ScratchTransformer(cfg, 40)

    def test_added_block_keeps_existing_weights(self):
        old_spec = mixed_spec()
        old = self.build(old_spec, 1)
        new_spec = copy.deepcopy(old_spec)
        new_spec["blocks"].insert(1, new_block(heads=2))
        new_spec = validate_spec(new_spec)
        new = self.build(new_spec, 2)
        result = transplant(old_spec, old.p, new_spec, new.p)
        np.testing.assert_array_equal(new.p["token"], old.p["token"])
        np.testing.assert_array_equal(new.p["block0.qkv.w"], old.p["block0.qkv.w"])
        np.testing.assert_array_equal(new.p["block2.f1.w"], old.p["block1.f1.w"])  # shifted index, same id
        np.testing.assert_array_equal(new.p["block3.qkv.w"], old.p["block2.qkv.w"])
        self.assertIn("block1.qkv.w", result["fresh"])
        self.assertNotIn("block1.qkv.w", result["copied"])
        self.assertEqual(len(result["copied"]) + len(result["fresh"]), len(new.p))

    def test_expansion_change_reinitializes_only_that_mlp(self):
        old_spec = mixed_spec()
        old = self.build(old_spec, 1)
        new_spec = copy.deepcopy(old_spec)
        new_spec["blocks"][2]["mlp"]["expansion"] = 4
        new_spec["blocks"][2]["attention"]["heads"] = 2  # heads change keeps qkv/proj
        new_spec = validate_spec(new_spec)
        new = self.build(new_spec, 2)
        result = transplant(old_spec, old.p, new_spec, new.p)
        np.testing.assert_array_equal(new.p["block2.qkv.w"], old.p["block2.qkv.w"])
        np.testing.assert_array_equal(new.p["block1.fh1.w"], old.p["block1.fh1.w"])
        self.assertTrue({"block2.f1.w", "block2.f2.w", "block2.n2.g"} <= set(result["fresh"]))
        self.assertFalse(np.array_equal(new.p["block2.f1.w"][:, :32], old.p["block2.f1.w"][:, :32]))

    def test_extra_mlp_layer_is_fresh_but_neighbours_copy(self):
        old_spec = mixed_spec()
        old = self.build(old_spec, 1)
        new_spec = copy.deepcopy(old_spec)
        new_spec["blocks"][2]["mlp"]["layers"] = 2
        new_spec = validate_spec(new_spec)
        new = self.build(new_spec, 2)
        result = transplant(old_spec, old.p, new_spec, new.p)
        self.assertEqual({"block2.fh1.w", "block2.fh1.b"} & set(result["fresh"]), {"block2.fh1.w", "block2.fh1.b"})
        np.testing.assert_array_equal(new.p["block2.f1.w"], old.p["block2.f1.w"])


class TrainingWithSpec(unittest.TestCase):
    EXAMPLES = [normalize({"user": "a", "assistant": "b"}), normalize({"user": "c", "assistant": "d"})]

    def test_telemetry_events_and_localmodel(self):
        spec = mixed_spec()
        events = []
        with tempfile.TemporaryDirectory() as directory:
            result = train(config_from_spec(spec), self.EXAMPLES, Path(directory) / "run", events.append, telemetry_every=3)
            self.assertEqual(result["status"], "completed")
            telemetry = [e for e in events if e["type"] == "telemetry"]
            self.assertEqual([e["step"] for e in telemetry], [1, 3, 6, 9, 12])
            event = telemetry[-1]
            self.assertEqual((event["total"], len(event["blocks"])), (12, 3))
            self.assertTrue(np.isfinite(event["loss"]) and event["lr"] > 0)
            self.assertGreater(event["embedding"]["grad_norm"], 0)
            self.assertGreater(event["embedding"]["rms"], 0)
            self.assertGreater(event["output"]["grad_norm"], 0)
            first, second, third = event["blocks"]
            self.assertEqual([b["id"] for b in event["blocks"]], [b["id"] for b in spec["blocks"]])
            self.assertEqual([b["index"] for b in event["blocks"]], [0, 1, 2])
            self.assertIsNone(first["mlp"])
            self.assertIsNone(second["attention"])
            self.assertEqual(len(first["attention"]["entropy"]), 2)
            self.assertEqual(len(third["attention"]["entropy"]), 4)
            self.assertTrue(all(e >= 0 for e in third["attention"]["entropy"]))
            for block in event["blocks"]:
                self.assertGreater(block["residual_rms"], 0)
            self.assertEqual(len(second["mlp"]["neurons"]), 32)
            self.assertTrue(all(0 <= n <= 1 for n in second["mlp"]["neurons"]) and max(second["mlp"]["neurons"]) in (0, 1))
            self.assertTrue(0 <= second["mlp"]["active_fraction"] <= 1)
            self.assertGreater(second["mlp"]["grad_norm"], 0)
            data = load_checkpoint(Path(directory) / "run" / "last.npz")
            self.assertEqual(data["config"]["architecture"]["blocks"][1]["mlp"]["layers"], 3)
            local = LocalModel(Path(directory) / "run" / "last.npz", device="cpu")
            self.assertTrue(isinstance(local.generate([{"role": "user", "content": "a"}], 4, 0), str))
            events.clear()
            train(config_from_spec(spec), self.EXAMPLES, Path(directory) / "off", events.append, telemetry_every=0)
            self.assertFalse([e for e in events if e["type"] == "telemetry"])

    def test_resume_with_spec_and_legacy_checkpoint_without_architecture(self):
        spec = mixed_spec()
        cfg = config_from_spec(spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(cfg, self.EXAMPLES, root / "a")
            train(dict(cfg, steps=18), self.EXAMPLES, root / "b", resume=root / "a" / "last.npz")
            self.assertEqual(load_checkpoint(root / "b" / "last.npz")["step"], 18)
            changed = copy.deepcopy(spec)
            changed["blocks"][0]["attention"]["heads"] = 4
            with self.assertRaisesRegex(ValueError, "Resume must preserve"):
                train(dict(config_from_spec(changed), steps=18), self.EXAMPLES, root / "c", resume=root / "a" / "last.npz")
            legacy = dict(DEFAULT_CONFIG, device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0.,
                          steps=6, warmup=0, validation=0., eval_every=3, objective="assistant")
            train(legacy, self.EXAMPLES, root / "l1")
            self.assertIsNone(load_checkpoint(root / "l1" / "last.npz")["config"]["architecture"])
            train(dict(legacy, steps=9), self.EXAMPLES, root / "l2", resume=root / "l1" / "last.npz")
            self.assertEqual(load_checkpoint(root / "l2" / "last.npz")["step"], 9)
            # Same structure expressed as an explicit spec resumes an architecture-less checkpoint.
            as_spec = dict(legacy, steps=9, architecture=spec_from_config(legacy))
            train(as_spec, self.EXAMPLES, root / "l3", resume=root / "l1" / "last.npz")

    def test_init_from_transplants_weights(self):
        spec = mixed_spec()
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(config_from_spec(spec), self.EXAMPLES, root / "base")
            grown = copy.deepcopy(spec)
            grown["blocks"].append(new_block(heads=2, expansion=2))
            grown["training"]["steps"] = 6
            train(config_from_spec(validate_spec(grown)), self.EXAMPLES, root / "grown", events.append,
                  init_from=root / "base" / "last.npz")
            base, child = load_checkpoint(root / "base" / "last.npz"), load_checkpoint(root / "grown" / "last.npz")
            self.assertEqual(child["step"], 6)
            self.assertEqual(child["optimizer_steps"], 6)  # fresh optimizer, new run
            self.assertNotIn("block3.qkv.w", base["model"])
            self.assertIn("block3.qkv.w", child["model"])
            info = [e["text"] for e in events if e["type"] == "info" and "Initialized from" in e["text"]]
            self.assertEqual(len(info), 1)
            self.assertIn("copied", info[0])
            first = next(e for e in events if e["type"] == "metric")
            self.assertEqual(first["step"], 0)
            with self.assertRaises(ValueError):
                train(config_from_spec(spec), self.EXAMPLES, root / "both", resume=root / "base" / "last.npz",
                      init_from=root / "base" / "last.npz")


if __name__ == "__main__":
    unittest.main()

"""The modern preset new designs start from, and the guarantee that older specs are untouched."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import studio_preset
import studio_spec
from studio_estimate import estimate
from studio_model import LocalModel, load_checkpoint, train
from studio_spec import config_from_spec, default_spec, validate_spec
from studio_validate import errors

EXAMPLES = [{"text": f"Counting up from {number}: {number}, {number + 1}, {number + 2}."} for number in range(12)]
RUN = {"batch_size": 2, "steps": 4, "validation": 0, "eval_every": 4, "seed": 3, "objective": "all", "device": "cpu"}

# A spec as an older version stored it: nothing but the dimensions, so every module is whatever the registry defaulted to.
OLD_PARTIAL = {"schema": 1, "name": "Old", "dims": {"d_model": 32, "n_layer": 2, "n_ctx": 64, "dropout": .1}}


class Modern(unittest.TestCase):
    def test_a_new_design_is_the_modern_recipe_and_validates(self):
        spec = default_spec(device="cpu")
        self.assertEqual(spec["position"]["id"], "rope")
        self.assertEqual([spec["block"][slot]["id"] for slot in ("norm", "mixer", "channel")], ["rmsnorm", "gqa", "swiglu"])
        self.assertEqual(spec["head"]["id"], "tied")
        self.assertEqual((spec["train"]["schedule"]["id"], spec["train"]["precision"]["id"]), ("wsd", "fp32"))
        optimizer = spec["train"]["optimizer"]["config"]
        self.assertEqual((optimizer["beta1"], optimizer["beta2"]), (.9, .95))
        self.assertFalse(spec["block"]["mixer"]["config"]["bias"])
        self.assertEqual(errors(spec), [])

    def test_the_classic_recipe_stays_selectable(self):
        spec = default_spec(preset="classic")
        self.assertEqual([spec["block"][slot]["id"] for slot in ("norm", "mixer", "channel")], ["layernorm", "attention", "mlp_gelu"])
        self.assertEqual((spec["position"]["id"], spec["train"]["precision"]["id"], spec["train"]["schedule"]["id"]), ("learned", "fp32", "cosine"))
        with self.assertRaises(ValueError):
            default_spec(preset="nonsense")

    def test_parameter_count_matches_the_classic_recipe_at_every_size(self):
        for d_model, n_layer in ((32, 1), (64, 2), (128, 4), (256, 6), (512, 8)):
            dims = {"d_model": d_model, "n_layer": n_layer}
            modern, _ = studio_preset.modern_spec(dims=dims, device="cpu")
            classic = studio_preset.legacy_spec(dims=dims)
            total, target = estimate(modern)["total"], estimate(classic)["total"]
            self.assertLess(abs(total - target) / target, .01, (d_model, total, target))
            self.assertEqual(errors(modern), [], d_model)

    def test_trains_and_round_trips_a_checkpoint(self):
        spec = default_spec(device="cpu")
        spec["dims"].update(d_model=32, n_layer=2, n_ctx=32)
        spec["block"]["mixer"]["config"].update(heads=2, kv_heads=1)
        cfg = {**config_from_spec(validate_spec(spec)), **RUN}
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            train(cfg, EXAMPLES, first)
            train(dict(cfg, steps=6, eval_every=6), EXAMPLES, second, resume=first / "last.npz")
            a, b = load_checkpoint(first / "last.npz"), load_checkpoint(second / "last.npz")
            self.assertEqual((a["step"], b["step"]), (4, 6))
            self.assertEqual(a["model"].keys(), b["model"].keys())
            self.assertEqual(b["config"]["architecture"], validate_spec(spec))
            self.assertTrue(all(np.isfinite(value).all() for value in b["model"].values()))
            reply = LocalModel(second / "last.npz", device="cpu").generate([{"role": "user", "content": "Counting"}], max_tokens=4)
            self.assertIsInstance(reply, str)


class Precision(unittest.TestCase):
    def test_bf16_where_the_probe_says_it_runs(self):
        with mock.patch("studio_modules.precision.policy.autocast_works", return_value=True):
            self.assertEqual(studio_preset.precision_for("cuda"), ("bf16", ""))
            self.assertEqual(studio_preset.precision_for("mps")[0], "bf16")
            spec, note = studio_preset.modern_spec(device="cuda")
        self.assertEqual((spec["train"]["precision"]["id"], note), ("bf16", ""))

    def test_falls_back_to_fp32_and_says_why_when_it_does_not(self):
        with mock.patch("studio_modules.precision.policy.autocast_works", return_value=False):
            spec, note = studio_preset.modern_spec(device="cuda")
            self.assertEqual(studio_preset.precision_for("mps")[0], "fp32")
        self.assertEqual(spec["train"]["precision"]["id"], "fp32")
        self.assertIn("bfloat16 does not run on this GPU", note)

    def test_the_cpu_trains_in_fp32_without_probing(self):
        with mock.patch("studio_modules.precision.policy.autocast_works", side_effect=AssertionError("probed")):
            self.assertEqual(studio_preset.precision_for("cpu")[0], "fp32")
            self.assertIn("CPU", studio_preset.precision_for("cpu")[1])

    def test_heads_divide_the_width_with_an_even_depth(self):
        for d_model in (16, 24, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024):
            heads, kv = studio_preset.attention_heads(d_model)
            self.assertEqual((d_model % heads, heads % kv, (d_model // heads) % 2), (0, 0, 0), d_model)


class OlderSpecs(unittest.TestCase):
    def test_a_stored_spec_that_omits_modules_still_reads_as_the_classic_recipe(self):
        spec = validate_spec(OLD_PARTIAL)
        self.assertEqual({slot: spec[slot]["id"] for slot in ("tokenizer", "embedding", "position", "head")},
                         {"tokenizer": "byte", "embedding": "learned", "position": "learned", "head": "tied"})
        self.assertEqual({slot: spec["block"][slot]["id"] for slot in ("norm", "mixer", "channel")},
                         {"norm": "layernorm", "mixer": "attention", "channel": "mlp_gelu"})
        self.assertEqual({slot: entry["id"] for slot, entry in spec["train"].items()},
                         {"objective": "ce", "optimizer": "adamw", "schedule": "cosine", "init": "gpt2", "precision": "fp32"})
        self.assertEqual(spec["train"]["optimizer"]["config"]["beta2"], .999)

    def test_the_frozen_defaults_and_hashes_did_not_move(self):
        self.assertEqual(studio_spec.DEFAULTS, {"tokenizer": "byte", "embedding": "learned", "position": "learned", "norm": "layernorm",
                                                "mixer": "attention", "channel": "mlp_gelu", "head": "tied", "objective": "ce",
                                                "optimizer": "adamw", "schedule": "cosine", "init": "gpt2", "precision": "fp32"})
        self.assertEqual(studio_spec.spec_hash(validate_spec(default_spec(preset="classic"))),
                         studio_spec.spec_hash(studio_preset.legacy_spec()))
        self.assertNotEqual(studio_spec.structure_key(default_spec(device="cpu")), studio_spec.structure_key(default_spec(preset="classic")))
        self.assertEqual(validate_spec(validate_spec(OLD_PARTIAL)), validate_spec(OLD_PARTIAL))
        # Computed on the code before the preset existed: the network an old spec describes did not change.
        self.assertEqual(hashlib.sha256(studio_spec.structure_key(OLD_PARTIAL).encode()).hexdigest(),
                         "384440ee1edb758a208c7f3f47ec21ed4be88b663fa52088ade4be94207ae379")

    def test_an_old_spec_trains_to_the_same_weights_as_the_classic_preset_and_resumes(self):
        classic = default_spec(preset="classic")
        classic["dims"].update(OLD_PARTIAL["dims"])
        self.assertEqual(validate_spec(classic)["block"], validate_spec(OLD_PARTIAL)["block"])
        cfg = {**config_from_spec(validate_spec(OLD_PARTIAL)), **RUN}
        with tempfile.TemporaryDirectory() as tmp:
            one, two, resumed = (Path(tmp) / name for name in ("one", "two", "resumed"))
            train(cfg, EXAMPLES, one)
            train({**config_from_spec(validate_spec(classic)), **RUN}, EXAMPLES, two)
            train(dict(cfg, steps=6, eval_every=6), EXAMPLES, resumed, resume=one / "last.npz")
            a, b, c = (load_checkpoint(path / "last.npz") for path in (one, two, resumed))
            self.assertEqual(c["step"], 6)
            self.assertEqual(c["config"]["architecture"]["block"]["norm"]["id"], "layernorm")
            for key in a["model"]:
                self.assertEqual(a["model"][key].tolist(), b["model"][key].tolist(), key)


class Api(unittest.TestCase):
    def test_start_and_presets_lead_with_modern_and_keep_classic(self):
        from fastapi.testclient import TestClient
        from studio_api.app import create_app
        from studio_api.hosts import ANY
        from studio_ia.services import FakeServices
        client = TestClient(create_app(FakeServices(), open=True, hosts=ANY))
        start = client.get("/design/start").json()
        self.assertEqual(start["spec"]["position"]["id"], "rope")
        self.assertIsInstance(start["note"], str)
        keys = [preset["key"] for preset in client.get("/design/presets").json()]
        self.assertEqual(keys[:2], ["modern", "classic"])
        self.assertIn("starter", keys)


if __name__ == "__main__":
    unittest.main()

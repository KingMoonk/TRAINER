"""Independent references for MoE, Muon, width scaling and BPE persistence."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from studio_estimate import estimate
from studio_modules.init.mup import MuPInit
from studio_modules.optimizer.muon import zeropower_newton_schulz
from studio_modules.tokenizer.bpe_gpt2 import GPT2Config, bytes_to_unicode, load_assets, pretokenize
from studio_registry import get
from studio_slots import Ctx, Dims, OFFSET
from studio_spec import validate_spec


class RemainingModules(unittest.TestCase):
    def test_moe_exact_total_and_active_delta(self):
        spec = validate_spec({"dims": {"d_model": 256, "n_layer": 4, "n_ctx": 256, "dropout": 0}})
        baseline = estimate(spec, 50257)
        spec["block"]["channel"] = {"id": "moe"}
        result = estimate(validate_spec(spec), 50257)
        self.assertEqual(result["total"] - baseline["total"], 14_724_096)
        self.assertEqual(result["active"] - baseline["active"], 2_110_464)

    def test_moe_auxiliary_gradient_and_routing(self):
        module = get("channel", "moe")
        cfg = module.parse({"experts": 3, "top_k": 2, "d_ff": 4, "aux_weight": .2})
        layer = module.build(cfg, Dims(4, 1, 4, 9, dtype=np.float64), None)
        rng = np.random.default_rng(24)
        for parameter in layer.params.values():
            parameter[...] = rng.normal(0, .15, parameter.shape)
        x = rng.normal(0, .3, (1, 4, 4))
        dy = rng.normal(0, .2, x.shape)

        def scalar():
            ctx = Ctx()
            output = layer.forward(x, ctx)
            return float(np.sum(output * dy) + ctx.aux)

        ctx = Ctx(hooks=True)
        output = layer.forward(x, ctx)
        self.assertEqual(output.shape, x.shape)
        self.assertGreater(ctx.aux, 0)
        chosen = ctx.captured["moe.router.top"]
        self.assertEqual(chosen.shape, (1, 4, 2))
        self.assertTrue(np.all(chosen[..., 0] != chosen[..., 1]))
        layer.backward(dy)
        router = layer.params["moe.router.w"]
        analytic = layer.grads["moe.router.w"].copy()
        for index in np.ndindex(router.shape):
            old = router[index]
            router[index] = old + 1e-5
            plus = scalar()
            router[index] = old - 1e-5
            minus = scalar()
            router[index] = old
            expected = (plus - minus) / 2e-5
            self.assertLess(abs(analytic[index] - expected), 1e-8 + 1e-6 * abs(expected))

    def test_muon_matrix_and_adamw_fallback_reference(self):
        cfg = get("optimizer", "muon").parse({"momentum": 0, "ns_steps": 5, "weight_decay": .1,
                                                "adam_beta1": .8, "adam_beta2": .5})

        class Model:
            xp = np
            p = {"block0.w": np.array([[1., 2.], [3., 4.]]),
                 "token": np.array([[1., 2.], [3., 4.]]), "head.w": np.array([[1., 2.], [3., 4.]]),
                 "bias": np.array([1., 2.])}
            g = {"block0.w": np.array([[.3, -.2], [.1, .4]]),
                 "token": np.array([[.3, -.2], [.1, .4]]), "head.w": np.array([[.3, -.2], [.1, .4]]),
                 "bias": np.array([.3, -.2])}
            roles = {"block0.w": "weight", "token": "embedding", "head.w": "weight", "bias": "bias"}
            owner = {"block0.w": ("mixer", 0), "token": ("embedding", None),
                     "head.w": ("head", None), "bias": ("norm", None)}
            lr_scale = {}

        model = Model()
        original = {name: value.copy() for name, value in model.p.items()}
        optimizer = get("optimizer", "muon").build(cfg, None, None).attach(model)
        optimizer.update(model, .02, clip=10)
        matrix_step = zeropower_newton_schulz(model.g["block0.w"], cfg.ns_steps, np)
        np.testing.assert_allclose(model.p["block0.w"], original["block0.w"] * (1 - .02 * .1) - .02 * matrix_step)
        for name in ("token", "head.w", "bias"):
            decay = 1 - .02 * .1 if name != "bias" else 1
            expected = original[name] * decay - .02 * model.g[name] / (np.abs(model.g[name]) + cfg.adam_eps)
            np.testing.assert_allclose(model.p[name], expected)
        self.assertTrue(all(set(getattr(optimizer, state)) == set(model.p) for state in optimizer.states))

    def test_mup_reference_width_and_roles(self):
        class Ones:
            def normal(self, mean, std, shape):
                return np.full(shape, std)

        module = get("init", "mup")
        cfg = module.parse({"base_width": 4, "std": .02, "scale_residual": False})
        initializer = MuPInit(cfg)
        mlp = get("channel", "mlp_gelu")
        layer = mlp.build(mlp.parse(), Dims(8, 1, 4, 9, dtype=np.float64), None)
        initializer.apply(layer, Ones(), layer.dims)
        self.assertAlmostEqual(layer.params["f1.w"][0, 0], .02 / math.sqrt(2))
        self.assertAlmostEqual(layer.lr_scale["f1.w"], .5)
        self.assertAlmostEqual(layer.params["f1.b"][0], 0)
        embedding = get("embedding", "learned").build(None, layer.dims, None)
        initializer.apply(embedding, Ones(), layer.dims)
        self.assertAlmostEqual(embedding.params["token"][0, 0], .02)
        self.assertEqual(embedding.lr_scale, {})

    def test_gpt2_fixture_unicode_and_offline_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = {symbol: byte for byte, symbol in bytes_to_unicode().items()}
            encoder["<|endoftext|>"] = 50256
            Path(directory, "encoder.json").write_text(json.dumps(encoder), encoding="utf-8")
            Path(directory, "vocab.bpe").write_text("#version: 0.2\n", encoding="utf-8")
            tokenizer = load_assets(GPT2Config(directory))
            text = "Hello  Greek Ελληνικά 😀\n it's"
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertEqual(tokenizer.vocab_size, 50257 + OFFSET)
            self.assertEqual(tokenizer.restore(json.loads(json.dumps(tokenizer.serialize()))).encode(text), tokenizer.encode(text))
            self.assertEqual(pretokenize("it's  nice!"), ["it", "'s", " ", " nice", "!"])
            # The original ordered pattern treats only ASCII space as the optional
            # prefix and leaves the final whitespace character for the next token.
            fixtures = {
                "αβ  42🎈's": ["αβ", " ", " 42", "🎈'", "s"],
                "A\u0085\u200b B": ["A", "\u0085", "\u200b", " B"],
                "x\u00a0\u00a0y": ["x", "\u00a0", "\u00a0", "y"],
                "'S 're": ["'", "S", " '", "re"],
                "漢字\u0301😀": ["漢字", "\u0301😀"],
            }
            for sample, expected in fixtures.items():
                with self.subTest(text=sample):
                    self.assertEqual(pretokenize(sample), expected)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "studio_modules.tokenizer.bpe_gpt2.urllib.request.urlopen", side_effect=OSError("offline")):
            with self.assertRaisesRegex(ValueError, "connect to the internet"):
                load_assets(GPT2Config(directory))

    def test_custom_bpe_determinism_and_json_restore(self):
        module = get("tokenizer", "bpe_custom")
        cfg = module.parse({"vocab_size": 275})
        examples = [{"text": "café café cat"}, {"text": "cat café cat"}]
        first = module.fit(cfg, examples)
        second = module.fit(cfg, list(reversed(examples)))
        self.assertEqual(first.serialize(), second.serialize())
        self.assertLessEqual(first.vocab_size, cfg.vocab_size)
        text = "café cat 🦊"
        ids = first.encode(text)
        self.assertEqual(first.decode(ids), text)
        self.assertLess(len(first.encode("café café")), len("café café".encode("utf-8")))
        restored = module.restore(json.loads(json.dumps(first.serialize())))
        self.assertEqual(restored.encode(text), ids)
        self.assertEqual(restored.decode(ids), text)


if __name__ == "__main__":
    unittest.main()

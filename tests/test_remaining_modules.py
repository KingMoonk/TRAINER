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

    def test_mup_complete_m1_identical_to_base(self):
        """m == 1 (base width): parameters and lr_scale identical to before μP completion."""
        class Ones:
            def normal(self, mean, std, shape):
                return np.full(shape, std)

        module = get("init", "mup")
        cfg = module.parse({"base_width": 8, "std": .02, "scale_residual": False})
        initializer = MuPInit(cfg)
        mlp = get("channel", "mlp_gelu")
        layer = mlp.build(mlp.parse(), Dims(8, 1, 4, 9, dtype=np.float64), None)
        initializer.apply(layer, Ones(), layer.dims)
        # At base width (m=1), hidden matrices should have std .02 and lr_scale 1.0
        self.assertAlmostEqual(layer.params["f1.w"][0, 0], .02)  # not scaled
        self.assertAlmostEqual(layer.lr_scale["f1.w"], 1.0)  # 1/m where m=1

    def test_mup_complete_adam_family_width_mult_4(self):
        """m == 4 with Adam family: correct std and lr_scale ratios."""
        class Ones:
            def normal(self, mean, std, shape):
                return np.full(shape, std)

        module = get("init", "mup")
        cfg = module.parse({"base_width": 4, "std": .02, "scale_residual": False, "optimizer_family": "adam"})
        initializer = MuPInit(cfg)
        dims = Dims(16, 1, 4, 9, dtype=np.float64)  # d_model=16, base_width=4, so m=4

        # Test hidden matrix: should have std .02/sqrt(4) = .01
        mlp = get("channel", "mlp_gelu")
        layer = mlp.build(mlp.parse(), Dims(16, 1, 4, 9, dtype=np.float64), None)
        initializer.apply(layer, Ones(), dims)
        self.assertAlmostEqual(layer.params["f1.w"][0, 0], .01, places=5)  # .02 / sqrt(4)
        self.assertAlmostEqual(layer.lr_scale["f1.w"], .25, places=5)  # 1/4

        # Test input-like (embedding): should have std .02, no lr_scale
        embedding = get("embedding", "learned").build(None, dims, None)
        initializer.apply(embedding, Ones(), dims)
        self.assertAlmostEqual(embedding.params["token"][0, 0], .02)
        self.assertEqual(embedding.lr_scale, {})

        # Test untied readout: should have std .02/4 = .005, lr_scale 1/4
        head = get("head", "untied").build(None, dims, None)
        initializer.apply(head, Ones(), dims)
        self.assertAlmostEqual(head.params["output.w"][0, 0], .005, places=5)  # .02 / 4
        self.assertAlmostEqual(head.lr_scale["output.w"], .25, places=5)  # 1/4

    def test_mup_complete_sgd_family_width_mult_4(self):
        """m == 4 with SGD family: different lr_scale than Adam."""
        class Ones:
            def normal(self, mean, std, shape):
                return np.full(shape, std)

        module = get("init", "mup")
        cfg = module.parse({"base_width": 4, "std": .02, "scale_residual": False, "optimizer_family": "sgd"})
        initializer = MuPInit(cfg)
        dims = Dims(16, 1, 4, 9, dtype=np.float64)  # d_model=16, base_width=4, so m=4

        # Test hidden matrix: should have std .02/sqrt(4) = .01, no lr_scale (default 1)
        mlp = get("channel", "mlp_gelu")
        layer = mlp.build(mlp.parse(), Dims(16, 1, 4, 9, dtype=np.float64), None)
        initializer.apply(layer, Ones(), dims)
        self.assertAlmostEqual(layer.params["f1.w"][0, 0], .01, places=5)  # .02 / sqrt(4)
        self.assertNotIn("f1.w", layer.lr_scale)  # no lr_scale for hidden matrices with SGD

        # Test input-like (embedding): should have std .02, lr_scale=m=4
        embedding = get("embedding", "learned").build(None, dims, None)
        initializer.apply(embedding, Ones(), dims)
        self.assertAlmostEqual(embedding.params["token"][0, 0], .02)
        self.assertAlmostEqual(embedding.lr_scale["token"], 4.0, places=5)  # m

        # Test untied readout: should have std .02/4 = .005, lr_scale 1/4 (same as Adam)
        head = get("head", "untied").build(None, dims, None)
        initializer.apply(head, Ones(), dims)
        self.assertAlmostEqual(head.params["output.w"][0, 0], .005, places=5)  # .02 / 4
        self.assertAlmostEqual(head.lr_scale["output.w"], .25, places=5)  # 1/4

    def test_mup_tied_readout_logit_scale(self):
        """Tied readout gets logit_scale attribute set by init/mup."""
        class Ones:
            def normal(self, mean, std, shape):
                return np.full(shape, std)

        module = get("init", "mup")
        cfg = module.parse({"base_width": 4, "std": .02, "scale_residual": False})
        initializer = MuPInit(cfg)
        dims = Dims(16, 1, 4, 9, dtype=np.float64)  # m=4

        head = get("head", "tied").build(None, dims, None)
        initializer.apply(head, Ones(), dims)
        # logit_scale should be 1/m = 0.25
        self.assertAlmostEqual(head.logit_scale, 0.25, places=5)

    def test_mup_attention_score_scale(self):
        """Attention layer gets score_scale attribute set by init/mup."""
        class Ones:
            def normal(self, mean, std, shape):
                return np.full(shape, std)

        module = get("init", "mup")
        cfg = module.parse({"base_width": 8, "std": .02, "scale_residual": False})
        initializer = MuPInit(cfg)
        dims = Dims(16, 1, 4, 256, dtype=np.float64)  # d_model=16, base_width=8, so m=2

        attn = get("mixer", "attention").build(get("mixer", "attention").parse({"heads": 4}), dims, None)
        initializer.apply(attn, Ones(), dims)
        # With 4 heads: d_head=4, base_d_head=2 (base 8, 4 heads), m_head=2
        # score_scale should be 1/sqrt(m_head) = 1/sqrt(2) ≈ 0.707
        expected_scale = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(attn.score_scale, expected_scale, places=5)

    def test_mup_tied_readout_forward_backward(self):
        """Tied readout forward/backward applies logit_scale correctly."""
        from studio_modules.embedding.learned import LearnedEmbedding

        class RNG:
            def __init__(self):
                self.rng = np.random.default_rng(42)
            def normal(self, mean, std, shape):
                return self.rng.normal(mean, std, shape)

        module = get("init", "mup")
        cfg = module.parse({"base_width": 8, "std": .02, "scale_residual": False})
        initializer = MuPInit(cfg)
        dims = Dims(16, 1, 4, 9, dtype=np.float64)  # m=2

        # Create embedding and tied head
        embedding = LearnedEmbedding(dims)
        tied_head = get("head", "tied").build(None, dims, None)
        rng = RNG()
        initializer.apply(embedding, rng, dims)
        initializer.apply(tied_head, rng, dims)

        # Forward pass through tied head
        x = np.random.randn(2, 3, 16)  # batch, seq, d_model

        # The tied head needs access to embedding
        ctx = Ctx(embedding=embedding)
        logits = tied_head.forward(x, ctx)

        # Logits should be scaled by logit_scale (1/m = 0.5)
        expected_scale = 1.0 / 2.0
        self.assertAlmostEqual(tied_head.logit_scale, expected_scale, places=5)

        # Backward pass
        dlogits = np.random.randn(*logits.shape)
        dx = tied_head.backward(dlogits)
        self.assertEqual(dx.shape, x.shape)

    def test_mup_numeric_gradient_check(self):
        """Numerical gradient check verifies forward/backward consistency with μP scaling."""
        class RNG:
            def __init__(self):
                self.rng = np.random.default_rng(42)
            def normal(self, mean, std, shape):
                return self.rng.normal(mean, std, shape)

        module = get("init", "mup")
        cfg = module.parse({"base_width": 4, "std": .02, "scale_residual": False})
        initializer = MuPInit(cfg)

        # Create a small model with m != 1
        mlp_spec = get("channel", "mlp_gelu")
        dims = Dims(12, 1, 4, 9, dtype=np.float64)  # m=3
        layer = mlp_spec.build(mlp_spec.parse({"d_ff": 12}), dims, None)
        rng = RNG()
        initializer.apply(layer, rng, dims)

        # Gradient check: compare analytic gradients to numerical gradients
        x = np.random.randn(2, 3, 12)
        epsilon = 1e-5

        ctx = Ctx()
        y = layer.forward(x, ctx)
        dy = np.random.randn(*y.shape)

        # Analytic gradient
        layer.backward(dy)
        analytic_grad_f1_w = layer.grads["f1.w"].copy()

        # Numerical gradient
        layer.grads["f1.w"].fill(0)
        f1_w_orig = layer.params["f1.w"].copy()
        numerical_grad = np.zeros_like(f1_w_orig)
        for i in range(min(3, f1_w_orig.shape[0])):
            for j in range(min(3, f1_w_orig.shape[1])):
                layer.params["f1.w"][...] = f1_w_orig.copy()
                layer.params["f1.w"][i, j] += epsilon
                y_plus = layer.forward(x, Ctx())
                f_plus = np.sum(y_plus * dy)

                layer.params["f1.w"][i, j] = f1_w_orig[i, j] - epsilon
                y_minus = layer.forward(x, Ctx())
                f_minus = np.sum(y_minus * dy)

                numerical_grad[i, j] = (f_plus - f_minus) / (2 * epsilon)

        layer.params["f1.w"][...] = f1_w_orig.copy()

        # Check relative error (should be small)
        rel_error = np.abs(analytic_grad_f1_w[:3, :3] - numerical_grad[:3, :3]) / (np.abs(analytic_grad_f1_w[:3, :3]) + np.abs(numerical_grad[:3, :3]) + 1e-8)
        self.assertLess(np.max(rel_error), 0.01, msg=f"Gradient check failed; max rel error: {np.max(rel_error)}")

    def test_mup_other_init_modules_unaffected(self):
        """Other init modules (gpt2) are completely unaffected by μP changes."""
        class Ones:
            def normal(self, mean, std, shape):
                return np.full(shape, std)

        gpt2_module = get("init", "gpt2")
        cfg = gpt2_module.parse({"std": .02, "scale_residual": False})
        initializer = gpt2_module.build(cfg, Dims(8, 1, 4, 9, dtype=np.float64), None)

        mlp = get("channel", "mlp_gelu")
        layer = mlp.build(mlp.parse(), Dims(8, 1, 4, 9, dtype=np.float64), None)
        initializer.apply(layer, Ones(), layer.dims)

        # GPT2 should not set any lr_scale
        self.assertEqual(layer.lr_scale, {})

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

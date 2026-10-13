"""Executable contracts for every module discovered by the registry."""
import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import studio_registry as registry
from studio_slots import Ctx, Dims, LAYER_SLOTS, OFFSET, UNK, config_dict


DIMS = Dims(4, 1, 3, 9, xp=np, dtype=np.float64)
STEP = 1e-5


def fixture(module):
    """Small, smooth fp64 inputs with all positions and experts reachable."""
    cfg = module.parse()
    dims = DIMS
    if module.short == "attention":
        cfg = module.parse({"heads": 2})
    elif module.short == "linear":
        cfg = module.parse({"heads": 2})
    elif module.short == "ssm":
        cfg = module.parse({"state": 2, "expand": 1, "kernel": 2, "dt_rank": 2})
    elif module.short == "moe":
        cfg = module.parse({"experts": 2, "top_k": 2, "d_ff": 4, "aux_weight": 0})
    layer = module.build(cfg, dims, np.random.default_rng(11))
    rng = np.random.default_rng(12)
    for name, value in layer.params.items():
        value[...] = rng.normal(0, .2, value.shape)
        if layer.roles[name] == "gain":
            value[...] += 1
    embedding = None
    if module.slot == "head" and "unembed" in module.requires:
        embedding = registry.get("embedding", "learned").build(None, dims, None)
        embedding.params["token"][...] = rng.normal(0, .2, embedding.params["token"].shape)
    x = np.array([[1, 2, 1]], dtype=np.int64) if module.slot == "embedding" else rng.normal(0, .4, (1, 3, 4))
    if module.slot == "head":
        dy = rng.normal(0, .3, (1, 3, dims.vocab))
    else:
        dy = rng.normal(0, .3, (1, 3, dims.d_model))
    return cfg, layer, embedding, x, dy


def context(embedding=None):
    return Ctx(np, hooks=True, embedding=embedding)


def scalar(layer, x, dy, embedding=None):
    return float(np.sum(layer.forward(x, context(embedding)) * dy))


def assert_gradient(test, actual, expected, label):
    """Relative error, with a strict absolute rule around zero."""
    scale = max(abs(actual), abs(expected))
    error = abs(actual - expected)
    test.assertLessEqual(error, 1e-8 if scale < 1e-5 else 1e-6 * scale, label)


class LayerContracts(unittest.TestCase):
    def test_shapes_hooks_counts_and_determinism(self):
        for module in registry.every():
            if module.slot not in LAYER_SLOTS:
                continue
            with self.subTest(module=module.id):
                cfg = module.parse()
                if module.skip:
                    self.assertEqual(module.estimate(cfg, DIMS).params, 0)
                    self.assertFalse(module.hooks)
                    continue
                cfg, layer, embedding, x, _ = fixture(module)
                expected_count = module.estimate(cfg, DIMS).params
                self.assertEqual(sum(p.size for p in layer.params.values()), expected_count)
                for length in (1, 3):
                    sample = x[:, :length]
                    ctx = context(embedding)
                    out = layer.forward(sample, ctx)
                    width = DIMS.vocab if module.slot == "head" else DIMS.d_model
                    self.assertEqual(out.shape, (1, length, width))
                    np.testing.assert_array_equal(out, layer.forward(sample, context(embedding)))
                    symbols = {"B": 1, "T": length, "d": DIMS.d_model, "V": DIMS.vocab,
                               **module.symbols(cfg, DIMS)}
                    self.assertEqual(set(ctx.captured), {name for name, _ in module.hooks})
                    for name, shape in module.hooks:
                        self.assertEqual(ctx.captured[name].shape, tuple(symbols[s] for s in shape.split()), name)
                # A hook patch must feed the value used downstream and retain dtype.
                if module.hooks:
                    unpatched = layer.forward(x, context(embedding)).copy()
                    for name, _ in module.hooks:
                        patched = Ctx(np, hooks=True, patches={name: lambda value: np.zeros_like(value)}, embedding=embedding)
                        changed = layer.forward(x, patched)
                        expected_dtype = np.int64 if name == "moe.router.top" else np.float64
                        self.assertEqual(patched.captured[name].dtype, np.dtype(expected_dtype))
                        self.assertFalse(np.any(patched.captured[name]))
                        self.assertGreater(float(np.max(np.abs(changed - unpatched))), 1e-10,
                                           f"{module.id}: patching {name} did not affect the output")

    def test_fp64_central_differences_and_gradient_accumulation(self):
        for module in registry.every():
            if module.slot not in LAYER_SLOTS or module.skip:
                continue
            with self.subTest(module=module.id):
                _, layer, embedding, x, dy = fixture(module)
                layer.forward(x, context(embedding))
                dx = layer.backward(dy)
                grads = {name: grad.copy() for name, grad in layer.grads.items()}
                if embedding is not None:
                    grads.update({"embedding." + name: grad.copy() for name, grad in embedding.grads.items()})
                for name, grad in grads.items():
                    if name in getattr(module, "zero_grad_params", ()):
                        continue  # e.g. a key bias cancels under softmax shift invariance
                    self.assertGreater(float(np.max(np.abs(grad))), 1e-12,
                                       f"{module.id} {name}: trainable parameter received no gradient")
                if dx is not None:
                    for index in np.ndindex(x.shape):
                        original = x[index]
                        x[index] = original + STEP
                        plus = scalar(layer, x, dy, embedding)
                        x[index] = original - STEP
                        minus = scalar(layer, x, dy, embedding)
                        x[index] = original
                        assert_gradient(self, dx[index], (plus - minus) / (2 * STEP), f"{module.id} input {index}")
                parameters = dict(layer.params)
                if embedding is not None:
                    parameters.update({"embedding." + name: value for name, value in embedding.params.items()})
                for name, parameter in parameters.items():
                    for index in np.ndindex(parameter.shape):
                        original = parameter[index]
                        parameter[index] = original + STEP
                        plus = scalar(layer, x, dy, embedding)
                        parameter[index] = original - STEP
                        minus = scalar(layer, x, dy, embedding)
                        parameter[index] = original
                        assert_gradient(self, grads[name][index], (plus - minus) / (2 * STEP), f"{module.id} {name}{index}")
                layer.forward(x, context(embedding))
                layer.backward(dy)
                for name, grad in layer.grads.items():
                    np.testing.assert_allclose(grad, 2 * grads[name], atol=1e-12, err_msg=module.id + " " + name)
                if embedding is not None:
                    for name, grad in embedding.grads.items():
                        np.testing.assert_allclose(grad, 2 * grads["embedding." + name], atol=1e-12)

    def test_causal_token_mixers(self):
        for module in registry.for_slot("mixer"):
            if module.skip:
                continue
            with self.subTest(module=module.id):
                _, layer, embedding, x, _ = fixture(module)
                original = layer.forward(x, context(embedding)).copy()
                changed = x.copy()
                changed[:, -1] += 3
                result = layer.forward(changed, context(embedding))
                np.testing.assert_allclose(result[:, :-1], original[:, :-1], atol=1e-12)


class NonLayerContracts(unittest.TestCase):
    def test_registry_config_json_round_trip_and_validation(self):
        for module in registry.every():
            with self.subTest(module=module.id):
                cfg = module.parse()
                raw = json.loads(json.dumps(config_dict(cfg)))
                self.assertEqual(module.parse(raw), cfg)
                with self.assertRaises(ValueError):
                    module.parse({"unknown_contract_field": 1})
                for field in dataclasses.fields(cfg) if cfg is not None else ():
                    if "min" in field.metadata:
                        bad = dict(raw, **{field.name: field.metadata["min"] - 1})
                        with self.assertRaises(ValueError):
                            module.parse(bad)

    def test_objective_logits_gradient_and_ignored_targets(self):
        for module in registry.for_slot("objective"):
            with self.subTest(module=module.id):
                objective = module.build(module.parse(), DIMS, None)
                logits = np.random.default_rng(3).normal(size=(1, 3, DIMS.vocab))
                labels = np.array([[1, -100, 4]])
                loss, gradient = objective(logits, labels, gradient=True, xp=np)
                self.assertTrue(math.isfinite(loss))
                np.testing.assert_array_equal(gradient[:, 1], 0)
                for index in np.ndindex(logits.shape):
                    old = logits[index]
                    logits[index] = old + STEP
                    plus = objective(logits, labels, xp=np)
                    logits[index] = old - STEP
                    minus = objective(logits, labels, xp=np)
                    logits[index] = old
                    assert_gradient(self, gradient[index], (plus - minus) / (2 * STEP), str(index))
                with self.assertRaises(ValueError):
                    objective(logits, np.full_like(labels, -100), xp=np)

    def test_tokenizer_round_trip(self):
        for module in registry.for_slot("tokenizer"):
            with self.subTest(module=module.id):
                text = "héllo 🍂"
                if module.short == "bpe_gpt2":
                    from studio_modules.tokenizer.bpe_gpt2 import bytes_to_unicode
                    with tempfile.TemporaryDirectory() as directory:
                        encoder = {symbol: byte for byte, symbol in bytes_to_unicode().items()}
                        encoder["<|endoftext|>"] = 50256
                        Path(directory, "encoder.json").write_text(json.dumps(encoder), encoding="utf-8")
                        Path(directory, "vocab.bpe").write_text("#version: 0.2\n", encoding="utf-8")
                        fitted = module.fit(module.parse({"asset_dir": directory}), [{"text": text}])
                else:
                    fitted = module.fit(module.parse(), [{"text": text}])
                tokens = fitted.encode(text)
                self.assertEqual(fitted.decode(tokens), text)
                self.assertTrue(all(OFFSET <= token < fitted.vocab_size for token in tokens))
                restored = module.restore(json.loads(json.dumps(fitted.serialize())))
                self.assertEqual(restored.encode(text), tokens)
                self.assertEqual(restored.decode(tokens), text)
                if module.vocab(module.parse()) is not None:
                    self.assertEqual(module.vocab(module.parse()), fitted.vocab_size)
                if module.short == "character":
                    self.assertEqual(fitted.encode("∉"), [UNK])

    def test_precision_initializer_schedule_and_optimizer(self):
        for module in registry.for_slot("precision"):
            with self.subTest(module=module.id):
                dtype = module.build(module.parse(), DIMS, None)
                self.assertEqual(np.dtype(dtype).name, "float" + module.short[2:])
        for module in registry.for_slot("init"):
            with self.subTest(module=module.id):
                layer = registry.get("norm", "layernorm").build(registry.get("norm", "layernorm").parse(), DIMS, None)
                module.build(module.parse(), DIMS, None).apply(layer, np.random.default_rng(4), DIMS)
                np.testing.assert_array_equal(layer.params["g"], 1)
                np.testing.assert_array_equal(layer.params["b"], 0)
                embedding = registry.get("embedding", "learned")
                left, right = (embedding.build(None, DIMS, None) for _ in range(2))
                initializer = module.build(module.parse(), DIMS, None)
                initializer.apply(left, np.random.default_rng(7), DIMS)
                initializer.apply(right, np.random.default_rng(7), DIMS)
                for name in left.params:
                    np.testing.assert_array_equal(left.params[name], right.params[name])
                    self.assertGreater(float(np.max(np.abs(left.params[name]))), 0)
        for module in registry.for_slot("schedule"):
            with self.subTest(module=module.id):
                schedule = module.build(module.parse(), DIMS, None)
                rates = [schedule.rate(step, 100, .01) for step in range(1, 101)]
                self.assertTrue(all(math.isfinite(rate) and rate >= 0 for rate in rates))
        for module in registry.for_slot("optimizer"):
            with self.subTest(module=module.id):
                self.assertTrue(module.build(module.parse(), DIMS, None).states is not None)

    def test_adamw_two_steps_and_state_against_scalar_reference(self):
        module = registry.get("optimizer", "adamw")
        cfg = module.parse({"beta1": .8, "beta2": .5, "weight_decay": .1, "eps": 1e-8})

        class Model:
            xp = np
            p = {"weight": np.array([2.], dtype=np.float64), "bias": np.array([2.], dtype=np.float64)}
            g = {"weight": np.array([.5]), "bias": np.array([.5])}
            roles = {"weight": "weight", "bias": "bias"}
            lr_scale = {}

        model = Model()
        optimizer = module.build(cfg, DIMS, None).attach(model)
        for step in (1, 2):
            norm = optimizer.update(model, .1, clip=10)
            self.assertAlmostEqual(norm, math.sqrt(.5))
            self.assertEqual(optimizer.steps, step)
            np.testing.assert_allclose(optimizer.m["weight"], [.5 * (1 - .8 ** step)])
            np.testing.assert_allclose(optimizer.v["weight"], [.25 * (1 - .5 ** step)])
        self.assertAlmostEqual(model.p["weight"][0], (2 * .99 - .1) * .99 - .1)
        self.assertAlmostEqual(model.p["bias"][0], 1.8)


if __name__ == "__main__":
    unittest.main()

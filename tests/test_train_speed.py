"""Faster runs: mixed precision, fused attention and gradient checkpointing change speed and memory, never the result."""
import copy
import unittest
from unittest import mock

import numpy as np
import torch

import studio_registry as registry
from studio_compile import CompiledModel
from studio_estimate import estimate
from studio_model import build_optimizer
from studio_modules.precision import policy
from studio_spec import validate_spec
from studio_validate import validate
from tests.torchcheck import to_np

IDS = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
TARGETS = np.array([[2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2]])


def spec_of(precision="fp64", position="learned", mixer="attention", channel="mlp_gelu", dropout=0., layers=3, checkpointing=False):
    spec = {"dims": {"d_model": 16, "n_layer": layers, "n_ctx": 32, "dropout": dropout},
            "position": {"id": position}, "block": {"mixer": {"id": mixer, "config": {"heads": 4}}, "channel": {"id": channel}},
            "train": {"precision": {"id": precision}, "run": {"checkpointing": checkpointing}}}
    if channel == "moe":
        spec["block"]["channel"] = {"id": "moe", "config": {"experts": 2, "top_k": 2, "d_ff": 16, "aux_weight": .1}}
    return validate_spec(spec)


def build(spec, seed=5):
    return CompiledModel(spec, 12, seed=seed, device="cpu")


def gradients(model, training=True):
    model.zero_grad()
    loss = model.loss_and_backward(IDS, TARGETS) if training else model.loss(IDS, TARGETS)
    return loss, {key: to_np(value).copy() for key, value in model.g.items()}


def same_weights(source, target):
    target.load_weights({key: to_np(value) for key, value in source.p.items()})


class FusedAttention(unittest.TestCase):
    def pair(self, **kwargs):
        model = build(spec_of(**kwargs))
        return model

    def test_fused_and_exact_paths_agree_in_outputs_and_gradients(self):
        for position in ("learned", "alibi", "rope"):
            for mixer in ("attention", "gqa"):
                with self.subTest(position=position, mixer=mixer):
                    model = self.pair(position=position, mixer=mixer)
                    with mock.patch("torch.nn.functional.scaled_dot_product_attention", wraps=torch.nn.functional.scaled_dot_product_attention) as kernel:
                        fast = to_np(model.forward(IDS))
                        self.assertEqual(kernel.call_count, 3, "every block should take the fused kernel")
                        kernel.reset_mock()
                        exact = to_np(model.forward(IDS, hooks=True))   # an inspection page listening: the hand-written path
                        self.assertEqual(kernel.call_count, 0)
                    np.testing.assert_allclose(fast, exact, atol=1e-10, rtol=1e-9)
                    # Gradients: a run with hooks on the scores takes the exact path for the same step.
                    model.zero_grad()
                    logits, tape = model.forward(IDS, training=True, tape=True)
                    model.backward_loss(model.objective(logits, TARGETS), tape)
                    fast_grads = {key: to_np(value).copy() for key, value in model.g.items()}
                    model.zero_grad()
                    logits, tape = model.forward(IDS, training=True, tape=True, hooks={"block0.attn.pattern", "block1.attn.pattern", "block2.attn.pattern"})
                    model.backward_loss(model.objective(logits, TARGETS), tape)
                    for key, value in model.g.items():
                        np.testing.assert_allclose(to_np(value), fast_grads[key], atol=1e-10, rtol=1e-7, err_msg=key)

    def test_each_listened_hook_forces_the_exact_path(self):
        model = self.pair(position="alibi")
        for hook in ("attn.scores", "attn.logits", "attn.pattern"):
            with self.subTest(hook=hook), mock.patch("torch.nn.functional.scaled_dot_product_attention", wraps=torch.nn.functional.scaled_dot_product_attention) as kernel:
                model.forward(IDS, hooks={f"block1.{hook}"})
                self.assertEqual(kernel.call_count, 2, "only the block being watched is exact")
        with mock.patch("torch.nn.functional.scaled_dot_product_attention", wraps=torch.nn.functional.scaled_dot_product_attention) as kernel:
            model.forward(IDS, patches={"block0.attn.logits": lambda value: value})
            self.assertEqual(kernel.call_count, 2)

    def test_a_patch_still_changes_the_result(self):
        model = self.pair()
        baseline = to_np(model.forward(IDS))
        patched = to_np(model.forward(IDS, patches={"block0.attn.pattern": lambda value: torch.zeros_like(value)}))
        self.assertGreater(np.abs(baseline - patched).max(), 1e-6)

    def test_kv_cached_decoding_matches_the_full_pass(self):
        for position in ("learned", "alibi"):
            with self.subTest(position=position):
                model = self.pair(position=position)
                full = to_np(model.forward(IDS[:1]))
                cache = {"n": 0}
                head = to_np(model.forward(IDS[:1, :5], cache=cache))   # a prefill (fused) then single tokens (offset, exact)
                steps = [to_np(model.forward(IDS[:1, i:i + 1], cache=cache)) for i in range(5, 8)]
                np.testing.assert_allclose(np.concatenate([head, *steps], axis=1), full, atol=1e-10)

    def test_dropout_is_reproducible_and_active_in_the_fused_path(self):
        spec = spec_of(dropout=.3, precision="fp32")

        def run(seed):
            return to_np(build(spec, seed).forward(IDS, training=True))

        np.testing.assert_array_equal(run(1), run(1))
        self.assertGreater(np.abs(run(1) - run(2)).max(), 1e-4)
        evaluated = build(spec, 1)
        np.testing.assert_array_equal(to_np(evaluated.forward(IDS)), to_np(evaluated.forward(IDS)))


class Checkpointing(unittest.TestCase):
    def test_gradients_match_without_checkpointing(self):
        for options in (dict(), dict(position="alibi"), dict(dropout=.3), dict(channel="moe"), dict(mixer="gqa", dropout=.2, channel="moe")):
            with self.subTest(**options):
                plain = build(spec_of(**options))
                saving = build(spec_of(checkpointing=True, **options))
                same_weights(plain, saving)
                self.assertTrue(saving.checkpointing and not plain.checkpointing)
                loss, expected = gradients(plain)
                saved_loss, got = gradients(saving)
                self.assertAlmostEqual(saved_loss, loss, places=10)
                for key, value in expected.items():
                    np.testing.assert_allclose(got[key], value, atol=1e-10, rtol=1e-8, err_msg=key)
                self.assertGreater(max(np.abs(value).max() for value in got.values()), 0)

    def test_nothing_is_recomputed_outside_training_or_while_inspecting(self):
        model = build(spec_of(checkpointing=True))
        with mock.patch("studio_compile.checkpoint", wraps=lambda fn, x, **kw: fn(x)) as wrapped:
            model.forward(IDS)
            model.forward(IDS, training=True, tape=True, hooks=True)
            self.assertEqual(wrapped.call_count, 0)
            model.forward(IDS, training=True, tape=True)
            self.assertEqual(wrapped.call_count, 3)

    def test_spec_option_is_validated_and_kept_only_when_on(self):
        self.assertTrue(validate_spec({"train": {"run": {"checkpointing": True}}})["train"]["run"]["checkpointing"])
        self.assertNotIn("run", validate_spec({"train": {"run": {"checkpointing": False}}})["train"])
        with self.assertRaises(ValueError):
            validate_spec({"train": {"run": {"checkpointing": "yes"}}})


class MixedPrecision(unittest.TestCase):
    def setUp(self):
        # Whether this machine's CPU has fast half-precision kernels varies (Apple CPUs do not); the logic is tested as if it did.
        patcher = mock.patch.object(policy, "autocast_works", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_modules_keep_float32_weights_and_declare_their_compute_dtype(self):
        for short, compute, scaled in (("bf16", torch.bfloat16, False), ("fp16", torch.float16, True)):
            module = registry.get("precision", short)
            self.assertEqual((module.compute, module.scaled), (compute, scaled))
            self.assertEqual(module.build(module.parse(), None, None), torch.float32)
        self.assertIsNone(registry.get("precision", "fp32").compute)

    def test_autocast_forward_tracks_float32_and_keeps_master_weights(self):
        for precision in ("bf16", "fp16"):
            for options in (dict(), dict(position="alibi", mixer="gqa"), dict(channel="moe", dropout=.2)):
                with self.subTest(precision=precision, **options):
                    reference = build(spec_of(precision="fp32", **options))
                    mixed = build(spec_of(precision=precision, **options))
                    same_weights(reference, mixed)
                    self.assertIsNotNone(mixed.compute)
                    logits = mixed.forward(IDS)
                    self.assertEqual(logits.dtype, torch.float32, "the loss sees float32 logits")
                    self.assertTrue(all(value.dtype == torch.float32 for value in mixed.p.values()))
                    if not options.get("dropout"):
                        np.testing.assert_allclose(to_np(logits), to_np(reference.forward(IDS)), atol=.15)
                    loss, expected = gradients(reference)
                    mixed_loss, got = gradients(mixed)
                    self.assertTrue(all(value.dtype == torch.float32 for value in mixed.g.values()))
                    if not options.get("dropout"):
                        self.assertAlmostEqual(mixed_loss, loss, delta=.05)
                        for key, value in expected.items():   # the gradient points the same way even where bf16 rounds
                            a, b = got[key].ravel(), value.ravel()
                            if np.linalg.norm(b) > 1e-6:
                                self.assertGreater(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30), .9, key)

    def test_every_mixer_and_channel_runs_under_autocast(self):
        for slot in ("mixer", "channel"):
            for module in registry.for_slot(slot):
                if module.skip:
                    continue
                with self.subTest(module=module.id):
                    spec = {"dims": {"d_model": 16, "n_layer": 1, "n_ctx": 32, "dropout": 0}, "position": {"id": "learned"},
                            "block": {slot: {"id": module.short}}, "train": {"precision": {"id": "bf16"}}}
                    if module.short in ("attention", "gqa", "linear"):
                        spec["block"][slot]["config"] = {"heads": 4}
                    model = build(validate_spec(spec))
                    loss = model.loss_and_backward(IDS, TARGETS)
                    self.assertTrue(np.isfinite(loss))
                    self.assertTrue(all(np.isfinite(to_np(value)).all() for value in model.g.values()))

    def test_bf16_training_reduces_loss(self):
        model = build(spec_of(precision="bf16", layers=2))
        optimizer, _ = build_optimizer(model)
        first = None
        for _ in range(25):
            model.zero_grad()
            loss = model.loss_and_backward(IDS, TARGETS)
            first = loss if first is None else first
            optimizer.update(model, 3e-3)
        self.assertLess(loss, first * .7)

    def test_attention_scores_and_softmax_stay_float32_in_the_exact_path(self):
        model = build(spec_of(precision="bf16"))
        model.forward(IDS, hooks=True)
        ctx_hooks = {}
        original = torch.softmax

        def spy(x, *args, **kwargs):
            ctx_hooks[x.dtype] = True
            return original(x, *args, **kwargs)

        with mock.patch("torch.softmax", spy):
            model.forward(IDS, hooks=True)
        self.assertEqual(set(ctx_hooks), {torch.float32})
        captured = model.forward(IDS, tape=True, hooks={"block0.attn.q", "block0.attn.pattern", "block0.mlp.act"})[1].captured
        self.assertTrue(all(value.dtype == torch.float32 for value in captured.values()), "inspection reads full precision")
        self.assertIsInstance(model.backend.to_cpu(captured["block0.attn.q"]), np.ndarray)

    def test_float16_loss_scaling_skips_an_overflowing_step_and_recovers(self):
        model = build(spec_of(precision="fp16", layers=2))
        optimizer, _ = build_optimizer(model)
        reference = build(spec_of(precision="fp32", layers=2))
        same_weights(reference, model)
        _, expected = gradients(reference)
        _, got = gradients(model)
        for key, value in expected.items():   # the scale is divided out again: gradients are those of the unscaled loss
            if np.linalg.norm(value) > 1e-6:
                np.testing.assert_allclose(np.linalg.norm(got[key]), np.linalg.norm(value), rtol=.1, err_msg=key)
        self.assertTrue(model.accept_gradients())
        before = {key: to_np(value).copy() for key, value in model.p.items()}
        model.scaler.scale = 2.0 ** 60       # far past float16's range: backward overflows
        model.zero_grad()
        model.loss_and_backward(IDS, TARGETS)
        self.assertTrue(model.overflowed)
        self.assertEqual(optimizer.update(model, 1e-3), 0.)
        self.assertEqual(optimizer.steps, 0)
        for key, value in model.p.items():
            np.testing.assert_array_equal(to_np(value), before[key])
        self.assertEqual(model.scaler.scale, 2.0 ** 59)
        model.scaler.scale = 2.0 ** 10
        model.zero_grad()
        model.loss_and_backward(IDS, TARGETS)
        self.assertGreater(optimizer.update(model, 1e-3), 0)
        self.assertEqual(optimizer.steps, 1)

    def test_unsupported_device_falls_back_to_float32_with_a_clear_note(self):
        with mock.patch.object(policy, "autocast_works", return_value=False):
            model = build(spec_of(precision="bf16"))
            self.assertIsNone(model.compute)
            self.assertIn("training in float32 instead", model.precision_note)
            self.assertEqual(model.forward(IDS).dtype, torch.float32)
            issues = [issue for issue in validate(spec_of(precision="bf16")) if issue.slot == "precision"]
            self.assertEqual([issue.level for issue in issues], ["WARN"])
        self.assertEqual(build(spec_of(precision="bf16")).precision_note, "")
        self.assertEqual([issue for issue in validate(spec_of(precision="bf16")) if issue.slot == "precision"], [])

    def test_cpu_half_precision_must_be_about_as_fast_as_float32(self):
        for half_seconds, fast in ((.001, True), (.002, True), (.1, False)):
            with self.subTest(half_seconds=half_seconds), mock.patch.object(policy, "_best_time", side_effect=[half_seconds, .001]):
                self.assertEqual(policy._cpu_half_is_fast(torch.bfloat16), fast)

    def test_estimate_counts_bf16_activations_and_checkpointing(self):
        base = spec_of(precision="fp32", layers=4)
        full = estimate(base, vocab=12)
        mixed = estimate(spec_of(precision="bf16", layers=4), vocab=12)
        self.assertEqual(mixed["total"], full["total"])
        self.assertEqual(mixed["compute_dtype"], "bfloat16")
        self.assertEqual(mixed["activation_bytes_per_token"] * 2, full["activation_bytes_per_token"])
        self.assertEqual(mixed["kv_bytes_per_token"] * 2, full["kv_bytes_per_token"])
        saving = estimate(spec_of(precision="fp32", layers=4, checkpointing=True), vocab=12)
        self.assertTrue(saving["checkpointing"])
        self.assertLess(saving["activation_bytes_per_token"], full["activation_bytes_per_token"])
        deep = copy.deepcopy(base)
        deep["dims"]["n_layer"] = 12
        deeper, deeper_saving = estimate(deep, vocab=12), estimate({**deep, "train": {**deep["train"], "run": {"checkpointing": True}}}, vocab=12)
        self.assertGreater(deeper["activation_bytes_per_token"] / deeper_saving["activation_bytes_per_token"],
                           full["activation_bytes_per_token"] / saving["activation_bytes_per_token"])


if __name__ == "__main__":
    unittest.main()

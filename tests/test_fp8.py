"""precision/fp8: the quantize and padding maths on the CPU, the policy and API gates, and a CUDA loss-parity run (skipped where
the device does not run float8)."""
import unittest
from unittest import mock

import numpy as np
import torch

import studio_registry as registry
from studio_compile import CompiledModel
from studio_ia import design as D
from studio_model import build_optimizer
from studio_modules import ops
from studio_modules.precision import fp8_ops, policy
from studio_spec import validate_spec
from studio_validate import validate

FP8 = registry.get("precision", "fp8")


def reference_mm(a, a_scale, b_t, b_scale):
    """What torch._scaled_mm computes, in float32 on any device: the dequantized operands multiplied."""
    return fp8_ops.dequantize(a, a_scale) @ fp8_ops.dequantize(b_t, b_scale).t()


class QuantizeMaths(unittest.TestCase):
    def test_the_largest_value_lands_on_the_formats_largest_number(self):
        x = torch.tensor([[0.5, -2.0], [1.0, 0.25]])
        for dtype in (fp8_ops.E4M3, fp8_ops.E5M2):
            q, inverse = fp8_ops.quantize(x, dtype)
            self.assertEqual(q.dtype, dtype)
            self.assertEqual(float(q.float().abs().max()), float(torch.finfo(dtype).max))
            self.assertAlmostEqual(float(inverse), 2.0 / float(torch.finfo(dtype).max), places=7)

    def test_dequantizing_restores_the_values_within_the_formats_rounding(self):
        torch.manual_seed(0)
        x = torch.randn(40, 24) * 3
        for dtype, tolerance in ((fp8_ops.E4M3, 2 ** -3), (fp8_ops.E5M2, 2 ** -2)):    # 3 and 2 mantissa bits: half a step is 2^-4 and 2^-3
            q, inverse = fp8_ops.quantize(x, dtype)
            back = fp8_ops.dequantize(q, inverse, x.shape)
            self.assertEqual(back.shape, x.shape)
            relative = (back - x).abs() / x.abs().clamp(min=float(x.abs().max()) * 2 ** -6)
            self.assertLessEqual(float(relative.max()), tolerance)

    def test_an_all_zero_tensor_keeps_scale_one_and_stays_zero(self):
        q, inverse = fp8_ops.quantize(torch.zeros(3, 5), fp8_ops.E4M3)
        self.assertEqual(float(inverse), 1.0)
        self.assertEqual(float(q.float().abs().sum()), 0.0)

    def test_a_huge_or_tiny_tensor_is_scaled_into_range_without_overflow(self):
        for factor in (1e-30, 1e6):
            x = torch.linspace(-1, 1, 64).reshape(8, 8) * factor
            q, inverse = fp8_ops.quantize(x, fp8_ops.E4M3)
            self.assertTrue(torch.isfinite(q.float()).all())
            self.assertLess(float((fp8_ops.dequantize(q, inverse, x.shape) - x).abs().max()), 0.07 * factor)

    def test_sizes_are_padded_with_zeros_to_multiples_of_sixteen(self):
        self.assertEqual([fp8_ops.padded(n) for n in (1, 16, 17, 648, 650)], [16, 16, 32, 656, 656])
        x = torch.ones(5, 19)
        q, _ = fp8_ops.quantize(x, fp8_ops.E4M3)
        self.assertEqual(tuple(q.shape), (16, 32))
        self.assertEqual(float(q.float()[5:].abs().sum() + q.float()[:, 19:].abs().sum()), 0.0)
        self.assertTrue(q.is_contiguous())

    def test_a_padded_product_equals_the_unpadded_one(self):
        torch.manual_seed(1)
        x, w = torch.randn(7, 21), torch.randn(21, 13)
        xq, xs = fp8_ops.quantize(x, fp8_ops.E4M3)
        wq, ws = fp8_ops.quantize(w.t(), fp8_ops.E4M3)
        got = reference_mm(xq, xs, wq, ws)[:7, :13]
        want = fp8_ops.dequantize(xq, xs, x.shape) @ fp8_ops.dequantize(wq, ws, w.t().shape).t()
        self.assertTrue(torch.allclose(got, want, atol=1e-5))
        self.assertLess(float((got - x @ w).abs().max()), 0.1 * float((x @ w).abs().max()))

    def test_only_wide_enough_layers_on_a_gpu_with_gradients_run_in_float8(self):
        wide, narrow = torch.zeros(64, 64), torch.zeros(64, 8)
        self.assertFalse(fp8_ops.applies(wide, wide))                  # the policy is off
        with fp8_ops.enabled():
            self.assertFalse(fp8_ops.applies(wide, wide))               # on the CPU
        self.assertFalse(fp8_ops._active.get())
        fake = mock.Mock(is_cuda=True)
        with fp8_ops.enabled(), mock.patch.object(torch, "is_grad_enabled", return_value=True):
            self.assertTrue(fp8_ops.applies(fake, wide))
            self.assertFalse(fp8_ops.applies(fake, narrow))             # narrower than 16 stays in the compute dtype
        with fp8_ops.enabled(), mock.patch.object(torch, "is_grad_enabled", return_value=False):
            self.assertFalse(fp8_ops.applies(fake, wide))               # scoring and chat

    def test_linear_is_the_plain_product_when_float8_is_off(self):
        layer = mock.Mock(params={"f.w": torch.eye(4), "f.b": torch.ones(4)})
        self.assertTrue(torch.equal(ops.linear(layer, torch.arange(8.).reshape(2, 4), "f"), torch.arange(8.).reshape(2, 4) + 1))


class Policy(unittest.TestCase):
    def test_the_module_is_registered_between_bf16_and_fp64_and_stores_float32(self):
        self.assertEqual([m.short for m in registry.for_slot("precision")], ["fp32", "bf16", "fp16", "fp8", "fp64"])
        self.assertEqual((FP8.compute, FP8.fp8, FP8.scaled, FP8.build({}, None, None)), (torch.bfloat16, True, False, torch.float32))

    def test_without_float8_hardware_it_is_unavailable_with_a_reason_and_falls_back_to_bfloat16(self):
        with mock.patch.object(fp8_ops, "supported", return_value=False), mock.patch.object(policy, "autocast_works", return_value=True):
            self.assertIn("RTX 40", policy.unavailable(FP8, "cpu"))
            self.assertIn("the CPU", policy.unavailable(FP8, "cpu"))
            self.assertFalse(policy.uses_fp8(FP8, "cuda"))
            compute, note = policy.plan(FP8, "cuda")
        self.assertEqual(compute, torch.bfloat16)
        self.assertIn("Training in bfloat16 instead.", note)

    def test_it_falls_back_to_float32_where_bfloat16_does_not_run_either(self):
        with mock.patch.object(fp8_ops, "supported", return_value=False), mock.patch.object(policy, "autocast_works", return_value=False):
            self.assertEqual(policy.plan(FP8, "cpu")[0], None)
            self.assertIn("Training in float32 instead.", policy.plan(FP8, "cpu")[1])

    def test_with_float8_hardware_it_is_available_and_plans_bfloat16_with_no_note(self):
        with mock.patch.object(fp8_ops, "supported", return_value=True), mock.patch.object(policy, "autocast_works", return_value=True):
            self.assertEqual(policy.unavailable(FP8, "cuda"), "")
            self.assertTrue(policy.uses_fp8(FP8, "cuda"))
            self.assertEqual(policy.plan(FP8, "cuda"), (torch.bfloat16, ""))

    def test_the_other_modules_are_never_refused(self):
        for short in ("fp32", "bf16", "fp16", "fp64"):
            self.assertEqual(policy.unavailable(registry.get("precision", short), "cpu"), "")

    def test_a_probe_on_the_cpu_says_no(self):
        self.assertFalse(fp8_ops.supported("cpu"))
        self.assertIn("RTX 40", fp8_ops.unsupported_reason("cpu"))

    def test_an_fp8_design_on_a_machine_without_it_warns_and_still_compiles_as_bfloat16(self):
        spec = validate_spec({"dims": {"d_model": 16, "n_layer": 1, "n_ctx": 32}, "train": {"precision": {"id": "fp8"}}})
        with mock.patch.object(fp8_ops, "supported", return_value=False):
            self.assertTrue(any(i.level == "WARN" and "8-bit float" in i.text for i in validate(spec)))
            model = CompiledModel(spec, 12, device="cpu")
        self.assertFalse(model.fp8)
        self.assertIn("8-bit float", model.precision_note)


class DesignGate(unittest.TestCase):
    def setUp(self):
        self.base = validate_spec({"dims": {"d_model": 16, "n_layer": 1, "n_ctx": 32}})

    def choices(self, **patch):
        return {**D.read_spec(self.base), **patch}

    def test_picking_fp8_is_refused_plainly_where_it_cannot_run(self):
        with mock.patch.object(fp8_ops, "supported", return_value=False), self.assertRaises(ValueError) as caught:
            D.write_spec(self.base, self.choices(precision="fp8"))
        self.assertIn("8-bit float needs an NVIDIA GPU", str(caught.exception))

    def test_picking_fp8_is_accepted_where_it_runs_and_other_picks_are_unchanged(self):
        with mock.patch.object(fp8_ops, "supported", return_value=True), mock.patch.object(policy, "autocast_works", return_value=True):
            spec = D.write_spec(self.base, self.choices(precision="fp8"))
        self.assertEqual(spec["train"]["precision"], {"id": "fp8", "config": {}})
        with mock.patch.object(fp8_ops, "supported", return_value=False):
            self.assertEqual(D.write_spec(spec, self.choices(precision="fp8"), D.read_spec(spec)), spec)    # already saved: not asked again
            self.assertEqual(D.write_spec(self.base, self.choices(precision="bf16"))["train"]["precision"]["id"], "bf16")

    def test_the_not_saved_summary_names_what_fp8_trades(self):
        cur, draft = self.choices(), self.choices(precision="fp8")
        chips = D.training_changes(cur, draft)
        self.assertEqual(chips[0], "precision fp32 → fp8")
        self.assertIn("8-bit matrix multiplies and smaller saved linear-layer inputs", chips[1])
        self.assertIn("loss may end higher", chips[1])
        self.assertEqual(D.training_changes(cur, self.choices(precision="bf16")), ["precision fp32 → bf16"])


@unittest.skipUnless(torch.cuda.is_available() and fp8_ops.supported("cuda"), "needs a CUDA GPU of compute capability 8.9 or newer that runs float8")
class CudaParity(unittest.TestCase):
    """A tiny model trains in float8 and tracks bfloat16: the loss falls and stays within a stated tolerance of bfloat16's."""

    STEPS, TOLERANCE = 30, 0.35      # nats: float8 may end higher than bfloat16 by up to this much on this tiny corpus

    def run_losses(self, precision):
        # d_model 48 and a feed-forward of 192 are not both multiples of 16 only by luck: 48 is, so also use width 40 (padded).
        spec = validate_spec({"dims": {"d_model": 40, "n_layer": 2, "n_ctx": 32, "dropout": 0.}, "block": {"mixer": {"id": "attention", "config": {"heads": 4}}},
                              "train": {"precision": {"id": precision}}})
        model = CompiledModel(spec, 64, seed=3, device="cuda")
        self.assertEqual(model.fp8, precision == "fp8")
        optimizer, _ = build_optimizer(model)
        rng = np.random.default_rng(0)
        ids = rng.integers(3, 20, size=(8, 17))
        losses = []
        for _ in range(self.STEPS):
            model.zero_grad()
            losses.append(model.loss_and_backward(ids[:, :-1], ids[:, 1:]))
            optimizer.update(model, 3e-3)
        self.assertTrue(all(np.isfinite(model.g[key].cpu().numpy()).all() for key in model.g))
        self.assertTrue(all(value.dtype == torch.float32 for value in model.p.values()))    # the weights stay wide
        return model, losses

    def test_loss_falls_and_stays_close_to_bfloat16(self):
        _, wide = self.run_losses("bf16")
        model, narrow = self.run_losses("fp8")
        self.assertLess(narrow[-1], narrow[0] * 0.8)
        self.assertLess(abs(narrow[-1] - wide[-1]), self.TOLERANCE)
        weights = {key: value.detach().cpu().numpy() for key, value in model.p.items()}     # a checkpoint loads back for resume and chat
        again = CompiledModel(model.spec, 64, seed=9, device="cuda")
        again.load_weights(weights)
        self.assertTrue(np.isfinite(again.loss(np.array([[3, 4, 5, 6]]), np.array([[4, 5, 6, 7]]))))

    def test_the_float8_product_matches_the_wide_one_and_its_gradients_point_the_same_way(self):
        torch.manual_seed(2)
        x = torch.randn(10, 40, device="cuda", requires_grad=True)       # 10 and 40 are not multiples of 16
        w = torch.randn(40, 24, device="cuda", requires_grad=True)
        y = fp8_ops.Fp8Linear.apply(x, w, torch.float32)
        y.square().sum().backward()
        want_x, want_w = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
        (want_x @ want_w).square().sum().backward()
        self.assertEqual((y.shape, x.grad.shape, w.grad.shape), ((10, 24), (10, 40), (40, 24)))
        for got, want in ((y, (x.detach() @ w.detach())), (x.grad, want_x.grad), (w.grad, want_w.grad)):
            cosine = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
            self.assertGreater(float(cosine), 0.98)


if __name__ == "__main__":
    unittest.main()

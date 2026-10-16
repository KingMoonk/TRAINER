"""Causal linear attention: finite outputs, cached decoding equal to the full window, and the Apple MPS cumsum memory corruption (issue #242)."""
import unittest

import numpy as np
import torch

from studio_backend import gpu_status
from studio_model import LocalModel, ScratchTransformer, validate_config
from studio_modules.mixer.linear import prefix_sums
from studio_modules.tokenizer.byte import Tokenizer

ODD_STRIDES = (8192, 1024, 8192, 32, 1)  # [1, 8, 1, 32, 32] as k[..., :, None] * v[..., None, :] lays it out: a stride on the length-1 axis


def local(device, d_model=64, layers=3, heads=4, seed=1):
    """A LocalModel with random weights and no file, linear attention with learned positions."""
    spec = {"schema": 1, "dims": {"d_model": d_model, "n_layer": layers, "n_ctx": 128, "dropout": 0}, "position": "learned",
            "block": {"mixer": {"id": "linear", "config": {"heads": heads}}}}
    config = validate_config({"architecture": spec, "device": device, "seed": seed})
    result = LocalModel.__new__(LocalModel)
    result.config, result.tokenizer = config, Tokenizer("byte")
    result.model = ScratchTransformer(config, result.tokenizer.vocab_size)
    result.rng, result.step = np.random.default_rng(1), 0
    return result


def decode_logits(model, ids, prefill):
    """Logits after every prefix from `prefill` tokens on: one prefill pass, then one cached token per step."""
    cache = {"n": 0}
    with torch.no_grad():
        rows = [model.forward(ids[None, :prefill], cache=cache)[0, -1]]
        rows += [model.forward(ids[None, i:i + 1], cache=cache)[0, -1] for i in range(prefill, len(ids))]
    return torch.stack(rows).cpu()


def full_window_logits(model, ids, prefill):
    with torch.no_grad():
        return torch.stack([model.forward(ids[None, :i])[0, -1] for i in range(prefill, len(ids) + 1)]).cpu()


class PrefixSums(unittest.TestCase):
    def test_matches_cumsum_and_leaves_a_single_token_alone(self):
        x = torch.randn(2, 3, 5, 4, 4, dtype=torch.float64)
        torch.testing.assert_close(prefix_sums(x), torch.cumsum(x, dim=2))
        one = x[:, :, :1]
        torch.testing.assert_close(prefix_sums(one), one)

    def test_odd_stride_on_a_length_one_axis(self):
        x = torch.empty_strided((1, 8, 1, 32, 32), ODD_STRIDES).normal_()
        torch.testing.assert_close(prefix_sums(x), x)


class CachedDecoding(unittest.TestCase):
    def test_cached_logits_equal_the_full_window(self):
        ids = np.random.default_rng(0).integers(6, 200, size=40)
        model = local("cpu").model
        for prefill in (1, 7, 20):
            with self.subTest(prefill=prefill):
                torch.testing.assert_close(decode_logits(model, ids, prefill), full_window_logits(model, ids, prefill), atol=1e-5, rtol=1e-4)

    def test_large_magnitude_inputs_stay_finite(self):
        # The sums grow with the text length and with the feature size; the normaliser keeps the quotient bounded.
        ids = np.random.default_rng(0).integers(6, 200, size=100)
        model = local("cpu").model
        for key, value in model.p.items():
            if key.endswith(".qkv.w"):
                with torch.no_grad():
                    value.mul_(40.)
        logits = decode_logits(model, ids, 10)
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits, full_window_logits(model, ids, 10), atol=1e-3, rtol=1e-3)


class MpsLinearAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = gpu_status()
        if not status["available"] or status["kind"] != "mps":
            raise unittest.SkipTest("Apple MPS is not available")

    def test_prefix_sums_of_one_token_do_not_touch_other_tensors(self):
        for _ in range(40):
            guards = [torch.full((8192,), 7., device="mps") for _ in range(3)]
            x = torch.empty_strided((1, 8, 1, 32, 32), ODD_STRIDES, device="mps").normal_()
            out = prefix_sums(x)
            torch.mps.synchronize()
            self.assertTrue(all(bool((guard == 7.).all()) for guard in guards))
            torch.testing.assert_close(out.cpu(), x.cpu())

    def test_cached_decoding_matches_the_cpu(self):
        # d_model 256, 6 layers, 8 heads: the model from the issue. The logits were off by 1 to 1e27 before.
        ids = np.random.default_rng(0).integers(6, 200, size=40)
        for seed in (1, 2):
            with self.subTest(seed=seed):
                cpu, gpu = local("cpu", 256, 6, 8, seed), local("mps", 256, 6, 8, seed)
                gpu.model.load_weights(cpu.model.state_arrays())
                cached = decode_logits(gpu.model, ids, 20)
                self.assertTrue(torch.isfinite(cached).all())
                torch.testing.assert_close(cached, decode_logits(cpu.model, ids, 20), atol=1e-4, rtol=1e-4)

    def test_generation_has_finite_probabilities(self):
        prompt = [{"role": "user", "content": "Copy: yes please"}]
        for seed in range(4):
            with self.subTest(seed=seed):
                local("mps", 256, 6, 8, seed).generate(prompt, max_tokens=32, temperature=.8)  # multinomial raises on inf or NaN

    def test_training_values_and_gradients_match_the_cpu(self):
        cpu, gpu = local("cpu", 64, 2, 4).model, local("mps", 64, 2, 4).model
        gpu.load_weights(cpu.state_arrays())
        x = np.random.default_rng(0).integers(6, 200, size=(4, 32))
        y = np.roll(x, -1, axis=1)
        self.assertAlmostEqual(cpu.loss_and_backward(x, y), gpu.loss_and_backward(x, y), delta=1e-4)
        for key in cpu.g:
            expected, got = cpu.g[key].cpu(), gpu.g[key].cpu()
            self.assertTrue(torch.isfinite(got).all(), key)
            torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-3, msg=key)


if __name__ == "__main__":
    unittest.main()

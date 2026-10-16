"""Issue #236: the KV cache is stored in half precision while the weights stay float32. Attention and GQA only; linear attention keeps a fixed-size state."""
import unittest
from unittest import mock

import numpy as np
import torch

import studio_model
from studio_compile import CompiledModel
from studio_estimate import estimate
from studio_modules.mixer.attention import KVCache, new_cache, reduced_dtype
from studio_spec import validate_spec
from tests.test_memory_p0 import local

IDS = np.random.default_rng(0).integers(6, 200, size=(1, 30))
PREFILL = 6
POSITIONS = ("learned", "rope", "alibi", "sinusoidal", "none")
MIXERS = ("attention", "gqa")

# Largest |logit difference| allowed between the float32 cache and a reduced one, on the fixed prompt above, over the 25 positions
# generated after the prefill. bfloat16 keeps 8 significant bits (relative error 2^-8 per cached value), float16 keeps 11.
# The matrices are scaled up (SHARPEN) so the logits spread over about +-2 (standard deviation 0.6) and attention is not near uniform.
# The worst measured differences over every mixer and position are 3.7e-3 (bfloat16) and 4.1e-4 (float16); the bounds are 5x and
# 7x that, so another PyTorch build rounding differently does not flake, and still only a few percent of the logits' spread.
TOLERANCE = {torch.bfloat16: 2e-2, torch.float16: 3e-3}
SHARPEN = 8.


def spec_of(mixer, position, precision="fp32"):
    return validate_spec({"dims": {"d_model": 16, "n_layer": 2, "n_ctx": 32, "dropout": 0.},
                          "position": {"id": position},
                          "block": {"mixer": {"id": mixer, "config": {"heads": 4, "kv_heads": 2} if mixer == "gqa" else {"heads": 4}}},
                          "train": {"precision": {"id": precision}}})


def model_of(mixer, position, precision="fp32"):
    model = CompiledModel(spec_of(mixer, position, precision), 200, seed=5, device="cpu")
    with torch.no_grad():
        for weight in model.p.values():
            if weight.ndim == 2:
                weight.mul_(SHARPEN)
    return model


def decode(model, cache):
    """Logits at every position after a 6-token prefill, one token per step, through `cache`."""
    out = [model.forward(IDS[:, :PREFILL], cache=cache)[0, -1]]
    out += [model.forward(IDS[:, i:i + 1], cache=cache)[0, -1] for i in range(PREFILL, IDS.shape[1] - 1)]
    return torch.stack(out).float()


def cache_bytes(cache):
    """Bytes held by the cache's tensors, counted from the tensors themselves."""
    return sum(t.numel() * t.element_size() for key, state in cache.items() if key != "n" for t in state)


class ReducedCache(unittest.TestCase):
    def test_logits_stay_close_to_the_float32_cache(self):
        for mixer in MIXERS:
            for position in POSITIONS:
                model = model_of(mixer, position)
                exact = decode(model, KVCache())
                for dtype, tolerance in TOLERANCE.items():
                    with self.subTest(mixer=mixer, position=position, dtype=dtype):
                        got = decode(model, KVCache(dtype))
                        difference = (got - exact).abs().max().item()
                        self.assertGreater(difference, 0, "the reduced cache should actually round the keys and values")
                        self.assertLess(difference, tolerance)
                        self.assertEqual(got.dtype, exact.dtype)

    def test_bytes_per_cached_token_halve(self):
        for mixer in MIXERS:
            with self.subTest(mixer=mixer):
                model = model_of(mixer, "rope")
                per_token = estimate(model.spec, model.dims.vocab)["kv_bytes_per_token"]
                sizes = {}
                for name, cache in (("full", KVCache()), ("reduced", new_cache("cpu"))):
                    model.forward(IDS[:, :PREFILL], cache=cache)
                    for i in range(PREFILL, 20):
                        model.forward(IDS[:, i:i + 1], cache=cache)
                    self.assertEqual(cache["n"], 20)
                    sizes[name] = cache_bytes(cache) / cache["n"]
                    self.assertTrue(all(t.dtype == (torch.float32 if name == "full" else reduced_dtype("cpu")) for k, s in cache.items() if k != "n" for t in s))
                self.assertEqual(sizes["full"], per_token)   # the estimate counts float32 here
                self.assertEqual(sizes["reduced"] * 2, sizes["full"])

    def test_the_float32_option_gives_exactly_the_current_results(self):
        for mixer in MIXERS:
            for position in POSITIONS:
                with self.subTest(mixer=mixer, position=position):
                    model = model_of(mixer, position)
                    current = decode(model, {"n": 0})   # a plain dict is what callers passed before this change
                    for cache in (KVCache(), new_cache("cpu", reduced=False)):
                        torch.testing.assert_close(decode(model, cache), current, atol=0, rtol=0)
                        self.assertTrue(all(t.dtype == torch.float32 for k, s in cache.items() if k != "n" for t in s))

    def test_works_under_autocast(self):
        """Under bfloat16 autocast the projections come out float32 (the bias add promotes them) and the matrix multiplies round their inputs
        to bfloat16 anyway, so a bfloat16 cache gives the same logits bit for bit, in half the bytes."""
        model = model_of("gqa", "rope")
        model.compute = torch.bfloat16   # as the bf16 precision module sets it; forced because plan() drops it on a CPU without fast bfloat16
        plain, reduced = {"n": 0}, new_cache("cpu")
        torch.testing.assert_close(decode(model, reduced), decode(model, plain), atol=0, rtol=0)
        self.assertTrue(all(t.dtype == torch.bfloat16 for k, s in reduced.items() if k != "n" for t in s))
        self.assertEqual(cache_bytes(reduced) * 2, cache_bytes(plain))

    def test_a_float32_pass_upcasts_what_it_reads(self):
        """The cache holds half precision, but attention reads it in the compute dtype: the outputs are float32."""
        model = model_of("attention", "learned")
        cache = new_cache("cpu")
        self.assertEqual(model.forward(IDS[:, :PREFILL], cache=cache).dtype, torch.float32)
        self.assertEqual(model.forward(IDS[:, PREFILL:PREFILL + 1], cache=cache).dtype, torch.float32)
        self.assertEqual(cache[0][0].dtype, reduced_dtype("cpu"))

    def test_copying_and_selecting_rows_keeps_the_reduced_dtype(self):
        """Batch code rebuilds the cache as a plain dict of the same tensors, repeated or indexed along the batch axis."""
        model = model_of("gqa", "rope")
        cache = new_cache("cpu")
        model.forward(IDS[:, :PREFILL], cache=cache)
        copy = {key: value if key == "n" else tuple(t.repeat_interleave(2, dim=0) for t in value) for key, value in cache.items()}
        selected = {key: value if key == "n" else tuple(t[[1]] for t in value) for key, value in copy.items()}
        for plain in (copy, selected):
            result = model.forward(np.repeat(IDS[:, PREFILL:PREFILL + 1], plain[0][0].shape[0], axis=0), cache=plain)
            self.assertTrue(all(t.dtype == reduced_dtype("cpu") for k, s in plain.items() if k != "n" for t in s))
            self.assertEqual(plain["n"], PREFILL + 1)
        expected = model.forward(IDS[:, PREFILL:PREFILL + 1], cache=cache)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_linear_attention_ignores_the_option(self):
        model = CompiledModel(validate_spec({"dims": {"d_model": 16, "n_layer": 2, "n_ctx": 32, "dropout": 0.}, "position": {"id": "learned"},
                                             "block": {"mixer": {"id": "linear", "config": {"heads": 4}}}}), 200, seed=5, device="cpu")
        full, reduced = {"n": 0}, new_cache("cpu")
        torch.testing.assert_close(decode(model, reduced), decode(model, full), atol=0, rtol=0)
        self.assertEqual(cache_bytes(reduced), cache_bytes(full))


class DeviceChoice(unittest.TestCase):
    def test_bfloat16_on_the_cpu(self):
        self.assertEqual(reduced_dtype("cpu"), torch.bfloat16)

    def test_cuda_needs_bfloat16_support(self):
        reduced_dtype.cache_clear()
        try:
            with mock.patch("torch.cuda.is_available", return_value=True):
                with mock.patch("torch.cuda.is_bf16_supported", return_value=True):
                    self.assertEqual(reduced_dtype("cuda"), torch.bfloat16)
                reduced_dtype.cache_clear()
                with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
                    self.assertEqual(reduced_dtype("cuda"), torch.float16)
        finally:
            reduced_dtype.cache_clear()

    def test_a_device_that_cannot_hold_bfloat16_gets_float16(self):
        reduced_dtype.cache_clear()
        try:
            with mock.patch("torch.ones", side_effect=RuntimeError("bfloat16 is not supported on this device")):
                self.assertEqual(reduced_dtype("mps"), torch.float16)
        finally:
            reduced_dtype.cache_clear()


class Chat(unittest.TestCase):
    def reply(self, reduced):
        model, seen = local("gqa", "rope"), []
        del model.reduced_kv_cache   # tests.test_memory_p0.local turns it off; this is the class default chat gets
        if reduced is not None:
            model.reduced_kv_cache = reduced
        real = studio_model.last_logits
        with mock.patch.object(studio_model, "last_logits", lambda m, window, sliding, cache, stop=None: seen.append(cache) or real(m, window, sliding, cache, stop)):
            model.generate([{"role": "user", "content": "Count from 4 to 9."}], temperature=0, max_tokens=8)
        return seen

    def test_chat_reduces_the_cache_on_cuda_only_by_default(self):
        self.assertIsNone(studio_model.LocalModel.reduced_kv_cache)
        seen = self.reply(None)
        self.assertTrue(all(cache is seen[0] for cache in seen))
        self.assertTrue(all(t.dtype == torch.float32 for k, s in seen[0].items() if k != "n" for t in s))
        model = local("gqa", "rope")
        del model.reduced_kv_cache
        for device, reduced in (("cpu", False), ("mps", False), ("cuda", True)):
            with mock.patch.object(model.model, "device", device):
                self.assertEqual(model.reduced_cache(), reduced, device)

    def test_reduced_cache_when_asked(self):
        seen = self.reply(True)
        self.assertTrue(all(t.dtype == reduced_dtype("cpu") for k, s in seen[0].items() if k != "n" for t in s))

    def test_a_group_of_replies_uses_the_same_cache_choice(self):
        model = local("gqa", "rope")
        model.reduced_kv_cache = True
        made = []
        real = studio_model.new_cache
        with mock.patch.object(studio_model, "new_cache", lambda device, reduced=True: made.append(reduced) or real(device, reduced)):
            model.generate_group([{"role": "user", "content": "Hi"}], 3, temperature=.8, max_tokens=4)
        self.assertEqual(made, [True])

    def test_float32_stays_available(self):
        seen = self.reply(False)
        self.assertTrue(all(t.dtype == torch.float32 for k, s in seen[0].items() if k != "n" for t in s))

    def test_reduced_and_float32_greedy_replies_are_both_complete(self):
        """A fixed prompt; the replies may differ by a token where two logits are within rounding, so only their shape is compared."""
        for reduced in (True, False):
            model = local("attention", "learned")
            model.reduced_kv_cache = reduced
            self.assertIsInstance(model.generate([{"role": "user", "content": "Hi"}], temperature=0, max_tokens=6), str)
            self.assertIsInstance(model.complete("The ", 6), str)


if __name__ == "__main__":
    unittest.main()

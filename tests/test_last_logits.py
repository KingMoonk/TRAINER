"""Generation projects one vocabulary row without changing the transformer or its KV cache (#548)."""
import os
import statistics
import time
import unittest
from unittest import mock

import numpy as np
import torch

from studio_compile import CompiledModel
from studio_model import batch_logits
from tests.test_memory_p0 import local


class LastLogits(unittest.TestCase):
    def test_last_logits_match_full_output_for_tied_and_untied_heads(self):
        ids = np.arange(7, 31)[None].repeat(2, axis=0)
        for head in ("tied", "untied"):
            for mixer in ("attention", "gqa", "linear", "gru"):
                with self.subTest(head=head, mixer=mixer):
                    spec = {"dims": {"d_model": 16, "n_layer": 2, "n_ctx": 32, "dropout": 0},
                            "head": head, "block": {"mixer": {"id": mixer, "config": {"heads": 4, "kv_heads": 2} if mixer == "gqa" else {}}}}
                    model = CompiledModel(spec, 64, seed=5, device="cpu")
                    expected = model.forward(ids)[:, -1:]
                    with mock.patch.object(model.head, "forward", wraps=model.head.forward) as project:
                        got = model.forward(ids, last_only=True)
                        self.assertEqual(tuple(project.call_args.args[0].shape), (2, 1, 16))
                    self.assertEqual(tuple(got.shape), (2, 1, 64))
                    torch.testing.assert_close(got, expected)

    def test_cached_prefill_preserves_every_position_and_next_decode(self):
        for mixer, position in (("attention", "learned"), ("gqa", "rope"), ("linear", "none")):
            with self.subTest(mixer=mixer):
                model = local(mixer, position).model
                ids = np.arange(7, 28)[None]
                cache = {"n": 0}
                got = batch_logits(model, ids.tolist(), False, cache, chunk=6)
                torch.testing.assert_close(got, model.forward(ids)[:, -1], atol=1e-5, rtol=1e-4)
                self.assertEqual(cache["n"], ids.shape[1])
                extended = np.concatenate((ids, [[29]]), axis=1)
                got = batch_logits(model, extended.tolist(), False, cache)
                torch.testing.assert_close(got, model.forward(extended)[:, -1], atol=1e-5, rtol=1e-4)
                self.assertEqual(cache["n"], extended.shape[1])

    def test_greedy_generation_matches_old_full_output_even_after_context_fills(self):
        original = CompiledModel.forward

        def full(model, *args, **kwargs):
            kwargs["last_only"] = False
            return original(model, *args, **kwargs)

        for mixer in ("attention", "gqa", "gru"):
            with self.subTest(mixer=mixer):
                model = local(mixer)
                with mock.patch.object(CompiledModel, "forward", full):
                    expected = model.generate([{"role": "user", "content": "A long prompt " * 4}], max_tokens=12, temperature=0)
                got = model.generate([{"role": "user", "content": "A long prompt " * 4}], max_tokens=12, temperature=0)
                self.assertEqual(got, expected)

    def test_training_and_inspection_cannot_silently_drop_output_positions(self):
        model = local().model
        for option in ({"training": True}, {"tape": True}, {"hooks": {"logits"}}, {"trace": True},
                       {"patches": {"final": lambda x: x}}, {"attention_summary": 4}):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "unobserved inference"):
                model.forward([[7, 8]], last_only=True, **option)
        self.assertEqual(tuple(model.forward([[7, 8]]).shape[:2]), (1, 2))


@unittest.skipUnless(os.environ.get("GROUNDWORK_CHAT_BENCHMARK") == "1", "opt-in CPU/CUDA latency measurement")
class LastLogitsBenchmark(unittest.TestCase):
    def test_prompt_latency_and_memory(self):
        """Same weights and prompt, interleaved full/last projection; timings are reported, never asserted."""
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for vocab in (264, 32000):
                spec = {"dims": {"d_model": 64, "n_layer": 2, "n_ctx": 256, "dropout": 0},
                        "train": {"precision": "fp32"}}
                model = CompiledModel(spec, vocab, seed=5, device=device)
                ids = np.random.default_rng(5).integers(7, vocab, size=(1, 256))
                synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
                expected = model.forward(ids)[:, -1:].clone()
                torch.testing.assert_close(model.forward(ids, last_only=True), expected, atol=1e-5, rtol=1e-4)
                times, peaks = {False: [], True: []}, {False: [], True: []}
                for iteration in range(12):
                    for last_only in ((False, True) if iteration % 2 == 0 else (True, False)):
                        synchronize()
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                            allocated = torch.cuda.memory_allocated()
                        start = time.perf_counter()
                        result = model.forward(ids, cache={"n": 0}, last_only=last_only)
                        synchronize()
                        elapsed = time.perf_counter() - start
                        if iteration >= 3:
                            times[last_only].append(elapsed * 1000)
                            if device == "cuda":
                                peaks[last_only].append(torch.cuda.max_memory_allocated() - allocated)
                        del result
                print(f"CHAT_PREFILL device={device} vocab={vocab} context=256 width=64 layers=2 "
                      f"full_ms={statistics.median(times[False]):.3f} last_ms={statistics.median(times[True]):.3f} "
                      f"speedup={statistics.median(times[False]) / statistics.median(times[True]):.2f} "
                      f"peak_full_bytes={max(peaks[False], default=0)} peak_last_bytes={max(peaks[True], default=0)}", flush=True)
                del model

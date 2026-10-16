"""Issue #235: a KV-cached prefill runs in chunks through the same cache and checks `stop` between them. Same logits, same replies."""
import threading
import unittest
from unittest import mock

import numpy as np
import torch

import studio_model
from studio_compile import CompiledModel
from studio_model import batch_logits
from studio_modules.mixer.attention import KVCache
from tests.test_memory_p0 import CACHED, local, positions

SHORT = [{"role": "user", "content": "Hi"}]                                       # 6 prompt tokens
LONG = [{"role": "user", "content": "Count from 4 to 9, slowly."}]                # 30 prompt tokens: nearly the 32-token window
IDS = np.random.default_rng(0).integers(6, 200, size=30)
CHUNKS = (4, 7, 29, 100)   # 29 leaves a last chunk of one token (linear attention's running sums must carry across it); 100 is the whole prompt in one pass


def chunked(size):
    return mock.patch.object(studio_model, "PREFILL_CHUNK", size)


class Recorder:
    """Wraps CompiledModel.forward: records the (rows, new tokens) shape of every pass; `after` calls stop.set() once that many passes ran."""

    def __init__(self, stop=None, after=None):
        self.shapes, self.stop, self.after = [], stop, after

    def __enter__(self):
        real = CompiledModel.forward

        def forward(model, ids, *args, **kwargs):
            self.shapes.append(tuple(np.shape(ids)))
            out = real(model, ids, *args, **kwargs)
            if self.stop is not None and len(self.shapes) == self.after:
                self.stop.set()
            return out
        self.patch = mock.patch.object(CompiledModel, "forward", forward)
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()


def same_cache(test, got, want):
    test.assertEqual(got["n"], want["n"])
    test.assertEqual(set(got), set(want))
    for key in want:
        if key != "n":
            for a, b in zip(got[key], want[key]):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


class SameLogits(unittest.TestCase):
    def test_every_cached_mixer_and_position(self):
        for mixer in CACHED:
            for position in positions(mixer):
                model = local(mixer, position).model
                whole = model.forward(IDS[None])[0, -1]
                exact = KVCache()
                torch.testing.assert_close(batch_logits(model, [list(IDS)], False, exact)[0], whole, atol=1e-5, rtol=1e-4)
                for size in CHUNKS:
                    with self.subTest(mixer=mixer, position=position, chunk=size), chunked(size), Recorder() as recorder:
                        cache = KVCache()
                        got = batch_logits(model, [list(IDS)], False, cache)[0]
                        torch.testing.assert_close(got, whole, atol=1e-5, rtol=1e-4)
                        same_cache(self, cache, exact)   # n, keys and values, or linear attention's running sums
                        self.assertEqual(cache["n"], len(IDS))
                        self.assertEqual(sum(shape[1] for shape in recorder.shapes), len(IDS))
                        self.assertTrue(all(shape[1] <= size for shape in recorder.shapes), recorder.shapes)

    def test_a_cache_that_already_holds_part_of_the_text_runs_only_the_rest(self):
        model = local("linear", "learned").model
        cache = KVCache()
        model.forward(IDS[None, :10], cache=cache)
        with chunked(5), Recorder() as recorder:
            got = batch_logits(model, [list(IDS[:25])], False, cache)[0]
        torch.testing.assert_close(got, model.forward(IDS[None, :25])[0, -1], atol=1e-5, rtol=1e-4)
        self.assertEqual([shape[1] for shape in recorder.shapes], [5, 5, 5])

    def test_a_sliding_window_is_cleared_and_prefilled_in_chunks(self):
        model = local("gqa", "rope").model
        cache = KVCache()
        window = np.concatenate((IDS, IDS[:2]))   # 32 tokens: the whole context
        model.forward(window[None, :20], cache=cache)
        with chunked(8), Recorder() as recorder:
            got = batch_logits(model, [list(window)], True, cache)[0]
        torch.testing.assert_close(got, model.forward(window[None])[0, -1], atol=1e-5, rtol=1e-4)
        self.assertEqual([shape[1] for shape in recorder.shapes], [8, 8, 8, 8])
        self.assertEqual(cache["n"], 32)

    def test_a_model_without_a_cache_runs_the_whole_window(self):
        model = local("conv", "learned")
        self.assertFalse(model.model.can_cache or model.model.can_cache is None)
        with chunked(4), Recorder() as recorder:
            reply = model.generate(SHORT, temperature=0, max_tokens=5)
        self.assertEqual(recorder.shapes[0], (1, 6))
        self.assertTrue(all(shape[1] >= 6 for shape in recorder.shapes))
        with chunked(1000):
            self.assertEqual(model.generate(SHORT, temperature=0, max_tokens=5), reply)


class SameReplies(unittest.TestCase):
    def test_greedy_generate_is_identical_and_survives_a_sliding_window(self):
        for mixer in CACHED:
            for position in positions(mixer):
                with self.subTest(mixer=mixer, position=position):
                    model = local(mixer, position)
                    with chunked(1000):
                        want = model.generate(LONG, temperature=0, max_tokens=48)   # prompt + 48 new tokens passes the 32-token context
                    for size in (4, 7):
                        with chunked(size), Recorder() as recorder:
                            self.assertEqual(model.generate(LONG, temperature=0, max_tokens=48), want, size)
                        self.assertTrue(any(shape[1] == size for shape in recorder.shapes), "the prompt really was chunked")

    def test_sampled_generate_group_is_identical(self):
        for mixer in CACHED:
            for prompt, max_tokens in ((SHORT, 40), (LONG, 24)):
                with self.subTest(mixer=mixer, prompt=len(prompt)):
                    a, b = local(mixer, positions(mixer)[0]), local(mixer, positions(mixer)[0])
                    with chunked(1000):
                        want = a.generate_group(prompt, 4, max_tokens=max_tokens, temperature=.9)
                    with chunked(5), Recorder() as recorder:
                        got = b.generate_group(prompt, 4, max_tokens=max_tokens, temperature=.9)
                    self.assertEqual(got, want)
                    self.assertTrue(any(shape[1] == 5 for shape in recorder.shapes))

    def test_complete_is_identical(self):
        for mixer in CACHED:
            for position in positions(mixer):
                with self.subTest(mixer=mixer, position=position):
                    model = local(mixer, position)
                    prompt = "Counting up from 4: 4, 5, 6, 7"
                    with chunked(1000):
                        want = model.complete(prompt, max_tokens=24)
                    with chunked(6), Recorder() as recorder:
                        self.assertEqual(model.complete(prompt, max_tokens=24), want)
                    self.assertTrue(any(shape[1] == 6 for shape in recorder.shapes))


class StopBetweenChunks(unittest.TestCase):
    def test_generate_returns_after_at_most_one_chunk(self):
        for mixer in CACHED:
            with self.subTest(mixer=mixer):
                stop = threading.Event()
                seen = []
                with chunked(8), Recorder(stop, after=1) as recorder:
                    reply = local(mixer).generate(LONG, temperature=0, max_tokens=20, stop=stop, on_text=seen.append)
                self.assertEqual(reply, "")
                self.assertEqual(recorder.shapes, [(1, 8)])
                self.assertEqual(seen, [""])

    def test_stop_during_a_sliding_reprefill(self):
        stop = threading.Event()
        model = local("attention", "learned")
        with chunked(8), Recorder() as warm:
            model.generate(LONG, temperature=0, max_tokens=6)   # the window slides after two tokens
        passes = len(warm.shapes)
        self.assertGreater(passes, 6)
        with chunked(8), Recorder(stop, after=passes + 1) as recorder:   # the first chunk of the next re-prefill
            reply = model.generate(LONG, temperature=0, max_tokens=30, stop=stop)
        self.assertEqual(len(recorder.shapes), passes + 1)
        self.assertLessEqual(len(reply), 30)

    def test_a_stop_set_before_the_prefill_runs_nothing(self):
        stop = threading.Event()
        stop.set()
        with chunked(8), Recorder() as recorder:
            self.assertEqual(local().generate(LONG, stop=stop), "")
        self.assertEqual(recorder.shapes, [])

    def test_generate_group_returns_empty_replies(self):
        for temperature in (.9, [.5, .9, 1.2]):
            with self.subTest(temperature=temperature):
                stop = threading.Event()
                with chunked(8), Recorder(stop, after=1) as recorder:
                    replies = local("linear", "learned").generate_group(LONG, 3, max_tokens=20, temperature=temperature, stop=stop)
                self.assertEqual(replies, ["", "", ""])
                self.assertEqual(recorder.shapes, [(1, 8)])

    def test_a_stop_set_after_the_prefill_still_ends_the_reply(self):
        stop = threading.Event()
        with chunked(8), Recorder(stop, after=4) as recorder:   # the 30-token prompt takes four chunks: the last one sets stop
            reply = local().generate(LONG, temperature=0, max_tokens=20, stop=stop)
        self.assertEqual(len(recorder.shapes), 4)
        self.assertLessEqual(len(reply), 1)   # the first token is sampled from the finished prefill, then the loop sees stop


if __name__ == "__main__":
    unittest.main()

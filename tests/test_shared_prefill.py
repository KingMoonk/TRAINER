"""Issue #234: a group of answers to one prompt shares one prefill. Same seeds give the same answers as separate generate() calls."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import studio_rl
import studio_rlhf
from studio_compile import CompiledModel
from studio_model import EOS, LocalModel, train
from tests.test_memory_p0 import CACHED, local, positions

SHORT = [{"role": "user", "content": "Hi"}]                                       # 6 prompt tokens: 26 more fit in the 32-token window
LONG = [{"role": "user", "content": "Count from 4 to 9, slowly."}]                # 30 prompt tokens: the text outgrows the window at once
N = 5


class CountingForward:
    """Wraps CompiledModel.forward: records (rows, new tokens) per pass, and nudges the end token up so replies end at different lengths."""

    def __init__(self, bonus=3.):
        self.calls, self.bonus = [], bonus

    def __enter__(self):
        real = CompiledModel.forward

        def forward(model, ids, *args, **kwargs):
            self.calls.append(tuple(np.shape(ids)))
            logits = real(model, ids, *args, **kwargs)
            if self.bonus:
                logits[..., EOS] += self.bonus   # the same nudge in the grouped and the separate run
            return logits
        self.patch = mock.patch.object(CompiledModel, "forward", forward)
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()

    def prefills(self, prompt_tokens):
        return [c for c in self.calls if c == (1, prompt_tokens)]


def pair(mixer="attention", position="learned"):
    """Two models with the same weights and the same sampling seed."""
    return local(mixer, position), local(mixer, position)


class SameAnswers(unittest.TestCase):
    def check(self, mixer, position, messages, max_tokens, temperature=.9):
        grouped, separate = pair(mixer, position)
        with CountingForward():
            together = grouped.generate_group(messages, N, max_tokens=max_tokens, temperature=temperature)
            apart = [separate.generate(messages, max_tokens=max_tokens, temperature=temperature) for _ in range(N)]
        self.assertEqual(together, apart, (mixer, position))
        self.assertEqual(grouped.rng.integers(2 ** 62), separate.rng.integers(2 ** 62), "both draw the same seeds from the model's stream")
        return together

    def test_every_cached_mixer_and_position_matches_separate_calls(self):
        for mixer in CACHED:
            for position in positions(mixer):
                with self.subTest(mixer=mixer, position=position):
                    answers = self.check(mixer, position, SHORT, 20)
                    self.assertGreater(len(set(answers)), 1, "the rows really differ")
                    self.assertGreater(len(set(map(len, answers))), 1, "and end at different lengths")

    def test_text_that_outgrows_the_window_matches(self):
        for mixer in CACHED:
            with self.subTest(mixer=mixer, start="mid-answer"):
                self.check(mixer, positions(mixer)[0], SHORT, 50)   # 6 + 50 tokens slide a 32-token window
            with self.subTest(mixer=mixer, start="prompt already too long"):
                self.check(mixer, positions(mixer)[0], LONG + [{"role": "assistant", "content": "4, 5, 6"}, {"role": "user", "content": "More, please."}], 24)

    def test_one_temperature_per_answer_matches(self):
        temperatures = studio_rlhf.temperature_spread(N)
        grouped, separate = pair()
        with CountingForward():
            together = grouped.generate_group(SHORT, N, max_tokens=20, temperature=temperatures, top_k=40)
            apart = [separate.generate(SHORT, max_tokens=20, temperature=t, top_k=40) for t in temperatures]
        self.assertEqual(together, apart)

    def test_greedy_answers_are_one_decode(self):
        grouped, separate = pair()
        with CountingForward() as counted:
            together = grouped.generate_group(SHORT, N, max_tokens=12, temperature=0)
            passes = len(counted.calls)
            alone = separate.generate(SHORT, max_tokens=12, temperature=0)
        self.assertEqual(together, [alone] * N)
        self.assertEqual(passes, len(counted.calls) - passes, "the same passes as one generate() call")
        self.assertTrue(all(rows == 1 for rows, _ in counted.calls))

    def test_stop_returns_what_was_written(self):
        grouped = local()
        stop = threading.Event()
        stop.set()
        self.assertEqual(grouped.generate_group(SHORT, N, max_tokens=12, stop=stop), [""] * N)


class OneForwardPerPrompt(unittest.TestCase):
    def test_the_prompt_is_run_once_per_group(self):
        grouped, separate = pair()
        with CountingForward() as counted:
            grouped.generate_group(SHORT, N, max_tokens=20)
        self.assertEqual(counted.prefills(6), [(1, 6)], "one pass over the prompt, at batch 1")
        self.assertTrue(all(tokens == 1 for _, tokens in counted.calls[1:]), "every later pass is one new token per row")
        with CountingForward() as counted:
            for _ in range(N):
                separate.generate(SHORT, max_tokens=20)
        self.assertEqual(len(counted.prefills(6)), N)

    def test_finished_answers_leave_the_batch(self):
        grouped = local()
        with CountingForward() as counted:
            answers = grouped.generate_group(SHORT, N, max_tokens=20)
        rows = [r for r, _ in counted.calls[1:]]
        self.assertEqual(rows, sorted(rows, reverse=True), "rows only ever leave")
        self.assertLess(min(rows), N, "some answer ended early")
        self.assertLess(sum(rows), N * len(rows), "and stopped costing compute")
        self.assertGreater(len(set(map(len, answers))), 1)

    def test_a_model_without_a_cache_loops_over_generate(self):
        grouped, separate = pair("gru", "learned")
        self.assertFalse(grouped.model.can_cache)
        with mock.patch.object(LocalModel, "generate", autospec=True, side_effect=LocalModel.generate) as spy:
            together = grouped.generate_group(SHORT, 3, max_tokens=10, temperature=.9)
        self.assertEqual(spy.call_count, 3)
        self.assertEqual(together, [separate.generate(SHORT, max_tokens=10, temperature=.9) for _ in range(3)])


class UsedWhereAGroupIsSampled(unittest.TestCase):
    def test_pass_at_k_round_2_and_compare_share_the_prefill(self):
        tiny = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4, steps=2,
                    learning_rate=.004, warmup=0, validation=0, eval_every=10, objective="assistant")
        with tempfile.TemporaryDirectory() as directory:
            train(tiny, [{"user": "Copy: yes", "assistant": "yes"}] * 8, Path(directory) / "base")
            checkpoint = Path(directory) / "base" / "last.npz"
            problems = [{"prompt": "Copy: yes", "rule": "copy"}]
            prompt_tokens = 2 + len("Copy: yes") + 1 + 1   # BOS, role, bytes, EOS, ASSISTANT
            with CountingForward(bonus=0.) as counted:
                result = studio_rl.pass_at_k(checkpoint, problems, count=1, k=4)
            self.assertEqual(result["answers"], 4)
            self.assertEqual(len(counted.prefills(prompt_tokens)), 1, "pass@k: one prefill for its 4 answers")
            with CountingForward(bonus=0.) as counted:
                studio_rl.practice_with_checks(checkpoint, problems, Path(directory) / "r2", group=4, steps=2, prompts_per_step=1, max_tokens=4)
            self.assertEqual(len(counted.prefills(prompt_tokens)), 2, "round 2: one prefill per group, one group per step")

            class Fixed:
                def judge(self, *args, **kwargs):
                    return {"scores": [5, 5], "reason": ""}
            prompts = [{"messages": [{"role": "user", "content": "Copy: yes"}]}] * 2
            with CountingForward(bonus=0.) as counted:
                studio_rlhf.compare_checkpoints(checkpoint, checkpoint, prompts, Fixed(), samples=3, max_tokens=4, temperature=1., device="cpu")
            self.assertEqual(len(counted.prefills(prompt_tokens)), 4, "compare: one prefill per prompt per model, not per sample")

    def test_best_of_n_candidates_share_the_prefill(self):
        model = local()
        cfg = {**studio_rlhf.AUTO_DEFAULTS, "candidates": 4, "max_tokens": 10}
        with CountingForward(bonus=0.) as counted:
            candidates = studio_rlhf.sample_candidates(model, SHORT, cfg, threading.Event())
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(len(counted.prefills(6)), 1, "four candidates at four temperatures, one pass over the prompt")


class CacheRows(unittest.TestCase):
    def test_expanded_rows_are_independent_copies(self):
        from studio_model import expand_cache, keep_cache_rows
        cache = {"n": 3, 0: (torch.arange(6.).reshape(1, 2, 3, 1), torch.zeros(1, 2, 3, 1))}
        expand_cache(cache, 3)
        self.assertEqual(cache[0][0].shape, (3, 2, 3, 1))
        cache[0][0][0] += 100
        self.assertEqual(float(cache[0][0][1].max()), 5.)
        keep_cache_rows(cache, [2])
        self.assertEqual(cache[0][0].shape, (1, 2, 3, 1))
        self.assertEqual(cache["n"], 3)


if __name__ == "__main__":
    unittest.main()

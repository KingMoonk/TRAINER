"""Issue #237: pinned, prefetched batches on CUDA; every other device takes the plain path, and the batches never change."""
import random
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import studio_model
from studio_model import BatchPrefetcher, ScratchTransformer, batch_arrays, batch_stage, load_checkpoint, train

EXAMPLES = [{"text": f"Counting up from {n}: {n}, {n + 1}, {n + 2}."} for n in range(12)]
CONFIG = {"width": 16, "layers": 2, "heads": 2, "context": 32, "expansion": 2, "dropout": 0., "batch_size": 4,
          "steps": 5, "learning_rate": .003, "warmup": 1, "validation": 0, "eval_every": 5, "seed": 5,
          "device": "cpu", "sample_every": 0}
WINDOWS = [(list(range(1, 3 + n % 5)), list(range(2, 4 + n % 5))) for n in range(30)]
calls = []


def stage(x, y):
    """A CPU stand-in for cuda_stage: no pinning, but the same (x, y, ready) shape and a record of what was consumed."""
    return x, y, lambda: calls.append(1)


def prefetching(_device):
    return stage


def source(seed, staged, size):
    rng = random.Random(seed)
    return rng, BatchPrefetcher(rng, lambda: batch_arrays([WINDOWS[rng.randrange(len(WINDOWS))] for _ in range(size[0])]), stage if staged else None)


def run(**options):
    """(metric losses, last checkpoint) of a tiny CPU run, with the prefetch path forced on or left to the device."""
    events, forced = [], options.pop("prefetch", False)
    with tempfile.TemporaryDirectory() as d, mock.patch.object(studio_model, "batch_stage", prefetching if forced else batch_stage):
        record = train(dict(CONFIG, **options.pop("config", {})), EXAMPLES, Path(d) / "r", emit=events.append, telemetry_every=0, **options)
        checkpoint = load_checkpoint(Path(d) / "r" / "last.npz")
    return [e["loss"] for e in events if e["type"] == "metric"], checkpoint, record, events


class Batches(unittest.TestCase):
    def test_prefetched_batches_equal_plain_ones_for_a_fixed_seed(self):
        for seed in (0, 7):
            plain, ahead = source(seed, False, [3]), source(seed, True, [3])
            for _ in range(8):
                a, b = plain[1].next(), ahead[1].next()
                self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
                self.assertEqual(a[2], b[2])
                self.assertEqual(plain[1].state(), ahead[1].state())  # a checkpoint stores where the consumed batches stopped

    def test_discard_puts_the_drawn_ahead_batch_back_so_a_new_size_sees_the_same_draws(self):
        size = [4]
        plain, ahead = source(1, False, size)[1], source(1, True, size)[1]
        for batch in (plain, ahead):
            batch.next()
        self.assertEqual(plain.state(), ahead.state())
        self.assertNotEqual(ahead.rng.getstate(), ahead.state())  # it really did draw one ahead
        size[0] = 2
        ahead.discard()
        self.assertEqual(ahead.rng.getstate(), plain.rng.getstate())
        self.assertTrue(all(np.array_equal(p, a) for p, a in zip(plain.next()[:2], ahead.next()[:2])))
        self.assertEqual(plain.next()[0].shape[0], 2)

    def test_ready_runs_once_per_consumed_batch(self):
        calls.clear()
        batch = source(2, True, [2])[1]
        for count in (1, 2, 3):
            batch.next()
            self.assertEqual(len(calls), count)

    def test_cpu_and_mps_take_the_plain_path_and_cuda_stages(self):
        self.assertIsNone(batch_stage("cpu"))
        self.assertIsNone(batch_stage("mps"))
        with mock.patch.object(studio_model, "cuda_stage", return_value="staged"):
            self.assertEqual(batch_stage("cuda"), "staged")

    def test_a_plain_path_never_pins_or_draws_ahead(self):
        rng, batch = source(3, False, [2])
        before = rng.getstate()
        with mock.patch.object(torch.Tensor, "pin_memory", side_effect=AssertionError("pinned")), \
                mock.patch.object(studio_model, "cuda_stage", side_effect=AssertionError("staged")):
            batch.next()
            self.assertIsNone(batch.ahead)
            self.assertNotEqual(rng.getstate(), before)
            self.assertEqual(batch.state(), rng.getstate())
            losses, *_ = run()  # a whole CPU run, through train()
        self.assertTrue(losses and all(np.isfinite(losses)))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
    def test_cuda_stage_copies_the_same_values_to_the_gpu(self):
        x, y = batch_arrays(WINDOWS[:4])
        staged, target, ready, *_ = (*studio_model.cuda_stage()(x, y),)
        ready()
        self.assertEqual((staged.device.type, target.device.type), ("cuda", "cuda"))
        self.assertTrue(np.array_equal(staged.cpu().numpy(), x) and np.array_equal(target.cpu().numpy(), y))


class Training(unittest.TestCase):
    def test_a_run_with_prefetch_matches_one_without(self):
        plain, plain_checkpoint, _, _ = run()
        ahead, ahead_checkpoint, _, _ = run(prefetch=True)
        self.assertEqual(plain, ahead)
        self.assertEqual(plain_checkpoint["sampler_state"], ahead_checkpoint["sampler_state"])
        for key, value in plain_checkpoint["model"].items():
            self.assertTrue(np.array_equal(value, ahead_checkpoint["model"][key]))

    def test_accumulation_matches_too(self):
        self.assertEqual(run(config={"batch_size": 2, "accumulation": 2})[0], run(prefetch=True, config={"batch_size": 2, "accumulation": 2})[0])

    def test_an_oom_shrink_drops_the_prefetched_batch_and_the_run_is_unchanged(self):
        def failing():
            real, seen = ScratchTransformer.forward, []

            def forward(self, ids, *args, **kwargs):
                if kwargs.get("training"):
                    seen.append(len(ids))
                    if len(seen) == 4:
                        raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
                return real(self, ids, *args, **kwargs)
            return mock.patch.object(ScratchTransformer, "forward", forward), seen

        with failing()[0]:
            _, plain_checkpoint, _, plain_events = run(config={"steps": 6, "eval_every": 6})
        patch, seen = failing()
        with patch:
            _, ahead_checkpoint, record, ahead_events = run(prefetch=True, config={"steps": 6, "eval_every": 6})
        self.assertEqual(record["status"], "completed")
        self.assertEqual(seen[:4], [4, 4, 4, 4])
        self.assertEqual(set(seen[4:]), {2})  # nothing of the old size survives the shrink
        # a metric row is also written when a second has passed since the last, so a slow machine adds rows: compare the steps always reported
        always = lambda events: {e["step"]: e["loss"] for e in events if e["type"] == "metric" and (e["step"] in (1, 6) or e["step"] % 5 == 0)}
        self.assertEqual(always(plain_events), always(ahead_events))
        self.assertEqual(plain_checkpoint["sampler_state"], ahead_checkpoint["sampler_state"])

    def test_resume_continues_from_the_consumed_batches_not_the_prefetched_one(self):
        def two_parts(forced):
            with tempfile.TemporaryDirectory() as d, mock.patch.object(studio_model, "batch_stage", prefetching if forced else batch_stage):
                root, events = Path(d), []
                train(dict(CONFIG, steps=3, eval_every=3), EXAMPLES, root / "a", telemetry_every=0)
                first = load_checkpoint(root / "a" / "last.npz")["sampler_state"]
                train(dict(CONFIG, steps=6, eval_every=6), EXAMPLES, root / "b", resume=root / "a" / "last.npz", emit=events.append, telemetry_every=0)
                return first, [e["loss"] for e in events if e["type"] == "metric"], load_checkpoint(root / "b" / "last.npz")["sampler_state"]
        self.assertEqual(two_parts(False), two_parts(True))

    def test_stop_saves_the_state_before_the_prefetched_batch(self):
        def stopped(forced):
            stop = threading.Event()
            events = []

            def emit(event):
                events.append(event)
                if event["type"] == "metric" and event["step"] >= 2:
                    stop.set()
            with tempfile.TemporaryDirectory() as d, mock.patch.object(studio_model, "batch_stage", prefetching if forced else batch_stage):
                record = train(dict(CONFIG, steps=6, eval_every=6), EXAMPLES, Path(d) / "r", emit=emit, stop=stop, telemetry_every=0)
                return record["status"], load_checkpoint(Path(d) / "r" / "last.npz")["sampler_state"]
        self.assertEqual(stopped(False), stopped(True))
        self.assertEqual(stopped(True)[0], "stopped")


if __name__ == "__main__":
    unittest.main()

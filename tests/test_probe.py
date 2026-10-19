"""The measured device choice: fake devices and clocks, no GPU needed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_probe
from studio_backend import get_backend
from studio_compute import step_seconds

GPU = {"available": True, "kind": "cuda", "name": "Fake GPU", "reason": "", "memory_bytes": 8 << 30}
NO_GPU = {"available": False, "kind": "", "name": "", "reason": "none", "memory_bytes": 0}
TOKENS = 4 * 64


class Clock:
    """A clock the fake steps advance: each device costs `seconds[device]` per step at batch multiple 1 (linear in the multiple)."""
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


def fake_runner(clock, seconds, fail=(), calls=None, big=None):
    def runner(device, multiple):
        if calls is not None:
            calls.append((device, multiple))
        if device in fail:
            raise RuntimeError("CUDA error: out of memory")
        cost = (big if multiple > 1 and big is not None else seconds[device] * multiple)
        closed = []

        def step():
            clock.now += cost
        return step, lambda: None, lambda: closed.append(1)
    return runner


def spec(batch_layers=2, precision="fp32"):
    return {"name": "x", "dims": {"d_model": 32, "n_layer": batch_layers, "n_ctx": 64, "dropout": 0.},
            "train": {"precision": precision, "optimizer": {"id": "adamw", "config": {"lr": 1e-3}}, "run": {}}}


class TimeSteps(unittest.TestCase):
    def test_median_after_warmup(self):
        clock, costs = Clock(), iter([9., 9., .3, .1, .2])  # two slow warm-up steps, then three timed
        def step():
            clock.now += next(costs)
        self.assertAlmostEqual(studio_probe.time_steps(step, lambda: None, clock, budget=100), .2)

    def test_slow_device_is_bounded(self):
        clock, calls = Clock(), []
        def step():
            calls.append(1)
            clock.now += 10.
        self.assertEqual(studio_probe.time_steps(step, lambda: None, clock, budget=4.), 10.)
        self.assertEqual(len(calls), 2)  # one warm-up, one timed: not the five of a fast device


class Choice(unittest.TestCase):
    def pick(self, cpu, gpu, **kwargs):
        clock = Clock()
        return studio_probe.probe(fake_runner(clock, {"cpu": cpu, "cuda": gpu}, **kwargs), ["cpu", "cuda"], TOKENS, clock, larger=4)

    def test_gpu_faster(self):
        result = self.pick(.3, .1)
        self.assertEqual(result["device"], "cuda")
        self.assertAlmostEqual(result["speedup"], 3.)
        self.assertIn("GPU 3.0x faster, using it", result["reason"])

    def test_cpu_faster_both_ways(self):
        result = self.pick(.1, .3)
        self.assertEqual(result["device"], "cpu")
        self.assertIn("CPU is faster for this size", result["reason"])
        self.assertAlmostEqual(result["speedup"], 3.)

    def test_gpu_that_is_barely_faster_is_not_worth_it(self):
        self.assertEqual(self.pick(.1, .098)["device"], "cpu")

    def test_gpu_faster_only_with_a_larger_batch_says_so(self):
        result = self.pick(.1, .3, big=.4)  # 4x the tokens in .4s on the GPU against .4s on the CPU: not better
        self.assertIsNone(result["larger_batch"])
        result = self.pick(.1, .3, big=.2)  # 4x the tokens in .2s: twice the CPU's rate
        self.assertEqual(result["device"], "cpu")
        self.assertAlmostEqual(result["larger_batch"]["speedup"], 2.)
        self.assertIn("4x larger", result["reason"])

    def test_gpu_failure_falls_back_with_a_plain_reason(self):
        result = self.pick(.2, .1, fail=("cuda",))
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["failed"], {"cuda": "ran out of memory"})
        self.assertIn("cuda probe ran out of memory; using the CPU", result["reason"])

    def test_every_device_failing_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no device could be timed"):
            self.pick(.2, .1, fail=("cpu", "cuda"))


class Cache(unittest.TestCase):
    def setUp(self):
        studio_probe.forget()
        self.addCleanup(studio_probe.forget)

    def measure(self, directory, calls, batch=4, design=None, status=GPU):
        clock = Clock()
        return studio_probe.measure(design or spec(), 263, batch, status, path=str(Path(directory) / "p.json"),
                                    runner=fake_runner(clock, {"cpu": .3, "cuda": .1}, calls=calls), clock=clock)

    def test_probe_runs_once_per_shape_and_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            first = self.measure(directory, calls)
            ran = len(calls)
            self.assertGreater(ran, 0)
            self.assertEqual(self.measure(directory, calls), first)
            self.assertEqual(len(calls), ran)
            studio_probe.forget()  # a new process: the file serves it
            self.assertEqual(self.measure(directory, calls)["device"], "cuda")
            self.assertEqual(len(calls), ran)
            self.assertEqual(studio_probe.cached(spec(), 263, 4, GPU, path=str(Path(directory) / "p.json"))["device"], "cuda")

    def test_batch_precision_and_design_each_get_their_own_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            self.measure(directory, calls)
            for kwargs in ({"batch": 8}, {"design": spec(precision="bf16")}, {"design": spec(batch_layers=4)}):
                before = len(calls)
                self.measure(directory, calls, **kwargs)
                self.assertGreater(len(calls), before, kwargs)

    def test_learning_rate_does_not_change_the_key(self):
        other = spec()
        other["train"]["optimizer"]["config"]["lr"] = 5e-2
        self.assertEqual(studio_probe.shape_key(spec(), 263, 4), studio_probe.shape_key(other, 263, 4))

    def test_no_gpu_means_no_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            self.assertIsNone(self.measure(directory, calls, status=NO_GPU))
            self.assertEqual(calls, [])

    def test_a_corrupt_cache_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "p.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(self.measure(directory, [])["device"], "cuda")


class AutoBackend(unittest.TestCase):
    def backend(self, result, work=10_000_000, status=GPU):
        with patch("studio_backend.gpu_status", return_value=status):
            return get_backend("auto", work=work, measure=lambda: result)

    def result(self, device, cpu, gpu):
        clock = Clock()
        return studio_probe.probe(fake_runner(clock, {"cpu": cpu, "cuda": gpu}), ["cpu", "cuda"], TOKENS, clock)

    def test_picks_the_faster_device_both_ways_and_records_the_speeds(self):
        gpu = self.backend(self.result("cuda", .3, .1), work=1)  # the size rule would say CPU: the measurement wins
        self.assertEqual(gpu.device, "cuda")
        self.assertIn("GPU 3.0x faster, using it", gpu.name)
        self.assertEqual(gpu.probe["device"], "cuda")
        cpu = self.backend(self.result("cpu", .1, .3), work=10**9)  # the size rule would say GPU
        self.assertEqual(cpu.device, "cpu")
        self.assertIn("CPU is faster for this size", cpu.reason)
        self.assertEqual(set(cpu.probe["tokens_per_second"]), {"cpu", "cuda"})

    def test_failed_probe_falls_back_to_the_size_rule_and_says_so(self):
        def broken():
            raise RuntimeError("driver error")
        with patch("studio_backend.gpu_status", return_value=GPU):
            small = get_backend("auto", work=1, measure=broken)
            big = get_backend("auto", work=10**9, measure=broken)
        self.assertEqual(small.device, "cpu")
        self.assertIn("the device probe failed (RuntimeError: driver error)", small.reason)
        self.assertEqual(big.device, "cuda")
        self.assertIn("driver error", big.name)

    def test_explicit_devices_and_missing_gpu_never_probe(self):
        def never():
            raise AssertionError("must not probe")
        with patch("studio_backend.gpu_status", return_value=NO_GPU):
            self.assertEqual(get_backend("auto", work=10**9, measure=never).device, "cpu")
        with patch("studio_backend.gpu_status", return_value=GPU):
            self.assertEqual(get_backend("cuda", measure=never).device, "cuda")
        self.assertEqual(get_backend("cpu", measure=never).device, "cpu")

    def test_without_a_probe_the_old_rule_holds(self):
        with patch("studio_backend.gpu_status", return_value=GPU):
            self.assertEqual(get_backend("auto", work=1).device, "cpu")
            self.assertEqual(get_backend("auto", work=10**9).device, "cuda")


class StepSeconds(unittest.TestCase):
    cfg = {"batch_size": 4, "context": 64, "layers": 2, "accumulation": 2}

    def test_unchanged_without_a_probe(self):
        self.assertEqual(step_seconds(self.cfg, 1e5, "cpu"), step_seconds(self.cfg, 1e5, "cpu", probe={}))

    def test_measured_seconds_replace_the_fit_at_the_probed_shape(self):
        probe = {"seconds": {"cpu": .5, "cuda": .02}, "tokens": 4 * 64}
        self.assertAlmostEqual(step_seconds(self.cfg, 1e5, "cpu", probe=probe), 1.0)  # two micro-batches of .5 s
        self.assertAlmostEqual(step_seconds(self.cfg, 1e5, "cuda", probe=probe), .04)

    def test_other_batch_scales_by_the_fit(self):
        probe = {"seconds": {"cpu": .5}, "tokens": 2 * 64}
        self.assertAlmostEqual(step_seconds(dict(self.cfg, accumulation=1), 1e5, "cpu", probe=probe), 1.0)  # CPU is linear in tokens

    def test_device_the_probe_did_not_time_uses_the_fit(self):
        probe = {"seconds": {"cpu": .5}, "tokens": 256}
        self.assertEqual(step_seconds(self.cfg, 1e5, "cuda", probe=probe), step_seconds(self.cfg, 1e5, "cuda"))


if __name__ == "__main__":
    unittest.main()

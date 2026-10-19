"""The training-speed benchmark: a fake training run and a fake clock, no real training."""
import dataclasses
import tempfile
import threading
import unittest
from unittest.mock import patch

import studio_mlperf as m
from studio_ia.services import FakeServices
from tests.sse_harness import ApiCase

INFO = {"kind": "cpu", "name": "Fake CPU", "node": "box", "threads": 4}
TASK = m.TASKS["small"]


def fake_runner(bpcs, per_step=2048, step_seconds=.5, first_extra=2.):
    """A runner that reports one metric row per evaluation: `bpcs` is the held-out bits per character at each one."""
    def runner(task, seed, device, emit, stop, directory):
        emit({"type": "history", "history": []})
        for number, bpc in enumerate(bpcs, 1):
            if stop.is_set():
                return
            step = number * task.eval_every
            emit({"type": "metric", "step": step, "val_loss": bpc * m.LN2, "loss": 1., "tokens_per_second": per_step / step_seconds,
                  "trained_tokens": step * per_step, "elapsed_seconds": first_extra + step * step_seconds})
    return runner


def bench(runner, **kw):
    return m.run_benchmark("small", "cpu", runner=runner, info=INFO, measure=lambda d: 1e12, **kw)


class TaskTests(unittest.TestCase):
    def test_frozen_checksum_holds_and_an_edit_is_caught(self):
        self.assertEqual(m.verify(TASK), TASK.checksum)
        with self.assertRaises(ValueError):
            m.verify(dataclasses.replace(TASK, target_bpc=1.9))
        with self.assertRaises(ValueError):
            m.verify(dataclasses.replace(TASK, text_seed=3))

    def test_text_is_the_same_every_time(self):
        self.assertEqual(m.corpus(5, 1), m.corpus(5, 1))
        self.assertNotEqual(m.corpus(5, 1), m.corpus(5, 2))
        self.assertEqual(len(TASK.examples()), TASK.rows)


class RunTests(unittest.TestCase):
    def test_reaches_target_and_records_time_throughput_and_mfu(self):
        r = bench(fake_runner([5., 3., 1.9, 1.5]), attempts=3)
        self.assertEqual(r["status"], "converged")
        step = 3 * TASK.eval_every  # the third evaluation is the first at or below 2.0
        self.assertAlmostEqual(r["time_to_target"]["median"], 2. + step * .5)
        self.assertEqual(r["time_to_target"]["n"], 3)
        self.assertTrue(all(a["steps"] == step for a in r["attempts"]))
        self.assertAlmostEqual(r["tokens_per_s"], 4096.)
        self.assertAlmostEqual(r["mfu"], 4096. * r["flops_per_token"] / 1e12)
        self.assertEqual(r["peak"]["kind"], "measured_gemm")
        self.assertEqual(r["seeds"], list(TASK.seeds[:3]))
        self.assertEqual(r["checksum"], TASK.checksum)

    def test_flops_count_the_output_projection(self):
        import studio_estimate
        import studio_model
        vocab = studio_model.vocabulary_guess(studio_model.validate_config(TASK.config(0)))
        self.assertEqual(m.flops_per_token(TASK), studio_estimate.estimate(TASK.spec(), vocab)["train_flops_per_token"] + 6 * 64 * vocab)

    def test_datasheet_peak_is_used_when_known(self):
        r = m.run_benchmark("small", "cuda", runner=fake_runner([1.]), info={"kind": "cuda", "name": "NVIDIA GeForce RTX 4060 Laptop GPU"},
                            attempts=1, measure=lambda d: self.fail("measured"))
        self.assertEqual((r["peak"]["kind"], r["peak"]["flops"]), ("datasheet", 15.11e12))

    def test_missing_the_target_is_not_converged(self):
        r = bench(fake_runner([5., 4., 3., 2.5]), attempts=2)
        self.assertEqual(r["status"], "did_not_converge")
        self.assertIsNone(r["time_to_target"])
        self.assertAlmostEqual(r["best_bpc"], 2.5)
        self.assertIsNotNone(r["tokens_per_s"])  # throughput is still real

    def test_one_attempt_missing_means_not_converged(self):
        runs = iter([fake_runner([1.]), fake_runner([5.])])
        r = bench(lambda *a: next(runs)(*a), attempts=2)
        self.assertEqual(r["status"], "did_not_converge")

    def test_a_crash_is_a_failed_result_with_its_reason(self):
        def boom(*a):
            raise RuntimeError("CUDA error: out of memory")
        r = bench(boom, attempts=1)
        self.assertEqual(r["status"], "failed")
        self.assertIn("out of memory", r["error"])

    def test_stop_ends_the_run_as_stopped(self):
        stop = threading.Event()
        inner = fake_runner([5., 4.])

        def runner(task, seed, device, emit, s, directory):
            stop.set()
            inner(task, seed, device, emit, s, directory)
        self.assertEqual(bench(runner, stop=stop, attempts=2)["status"], "stopped")

    def test_setup_uses_the_clock(self):
        ticks = iter(range(0, 1000, 3))
        r = bench(fake_runner([1.]), clock=lambda: next(ticks), attempts=1)
        self.assertEqual(r["setup_s"], 3)

    def test_energy_from_nvml_when_present(self):
        class Nvml:
            reads = iter([1000, 6000])
            nvmlInit = staticmethod(lambda: None)
            nvmlDeviceGetHandleByIndex = staticmethod(lambda i: 0)
            nvmlDeviceGetTotalEnergyConsumption = staticmethod(lambda h: next(Nvml.reads))
        with patch.dict("sys.modules", {"pynvml": Nvml}):
            energy = m.Energy("cuda")
            self.assertEqual(energy.joules(), 5.)
        self.assertIsNone(m.Energy("cpu").joules())


class ViewTests(unittest.TestCase):
    def result(self, ident, seconds, version=1, label="a", at=1.):
        base = bench(fake_runner([1.]), attempts=1)
        base.update(id=ident, version=version, label=label, created_at=at)
        base["time_to_target"] = {"median": seconds, "min": seconds, "max": seconds, "n": 1} if seconds else None
        base["status"] = "converged" if seconds else "did_not_converge"
        return base

    def test_leaderboard_ranks_fastest_first_and_holds_other_versions_apart(self):
        results = [self.result("slow", 50), self.result("miss", None), self.result("fast", 10), self.result("old", 1, version=0)]
        board = m.leaderboard(results, "small")
        self.assertEqual([r["id"] for r in board["divisions"]["single"]], ["fast", "slow", "miss"])
        self.assertEqual(board["other_versions"], [{"version": 0, "results": 1}])

    def test_compare_refuses_different_versions(self):
        results = [self.result("a", 10), self.result("b", 20), self.result("c", 5, version=0)]
        rows = m.compare(results, ["a", "b"])["rows"]
        self.assertEqual({r["id"]: r["relative_time"] for r in rows}, {"a": 1., "b": 2.})
        with self.assertRaises(ValueError):
            m.compare(results, ["a", "c"])

    def test_history_follows_app_versions_in_time_order(self):
        results = [self.result("b", 8, at=2.), self.result("a", 10, at=1.)]
        results[0]["app_version"], results[1]["app_version"] = "v2", "v1"
        points = m.history(results, "small")["points"]
        self.assertEqual([(p["id"], p["app_version"]) for p in points], [("a", "v1"), ("b", "v2")])

    def test_results_are_stored_and_read_back(self):
        with tempfile.TemporaryDirectory() as root:
            r = bench(fake_runner([1.]), attempts=1)
            m.store_result(root, r)
            self.assertEqual(m.load_results(root)[0]["id"], r["id"])
            self.assertEqual(m.load_results(root + "/none"), [])

    def test_queue_job_stores_the_result(self):
        with tempfile.TemporaryDirectory() as root:
            job = type("J", (), {"params": {"task": "small", "device": "cpu", "attempts": 1}})()
            ctx = type("C", (), {"root": root})()
            with patch.object(m, "train_runner", fake_runner([1.])), patch.object(m, "describe_device", lambda kind: INFO), \
                    patch.object(m, "measured_peak", lambda d: 1e12):
                m.job(job, ctx, lambda e: None, threading.Event(), threading.Event())
            self.assertEqual(m.load_results(root)[0]["status"], "converged")
            self.assertEqual(job.params["result"]["status"], "converged")


class RouterTests(ApiCase):
    prefix = "/mlperf"

    def make_services(self):
        services = FakeServices()
        services._mlperf = [dict(bench(fake_runner([1.]), attempts=1), id="r1"), dict(bench(fake_runner([1.]), attempts=1), id="r2")]
        return services

    def test_board_run_and_compare(self):
        board = self.call("GET", "/mlperf/board?task=small", 200)
        self.assertEqual(board["tasks"][0]["id"], "small")
        self.assertEqual(len(board["board"]["divisions"]["single"]), 2)
        self.assertEqual(len(self.call("POST", "/mlperf/compare", 200, json={"ids": ["r1", "r2"]})["rows"]), 2)
        self.call("GET", "/mlperf/board?task=nope", 404)
        self.call("POST", "/mlperf/run", 201, json={"task": "small", "device": "cpu"})
        self.call("POST", "/mlperf/run", 422, json={"task": "nope"})


if __name__ == "__main__":
    unittest.main()

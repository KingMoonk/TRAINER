"""bench_cuda.py: the CPU-runnable parts (size picking, timing, table formatting) and its refusal to run without CUDA."""
import contextlib
import io
import unittest
from unittest import mock

import bench_cuda
import studio_memory


class HelperChecks(unittest.TestCase):
    def test_step_times_are_the_gaps_after_the_warmup(self):
        self.assertEqual(bench_cuda.step_times([0., 1., 3., 6., 10.], warmup=1), [2., 3., 4.])
        self.assertEqual(bench_cuda.step_times([0.], warmup=0), [])

    def test_summarize_reports_mean_and_median_in_milliseconds(self):
        row = bench_cuda.summarize([.001, .002, .009])
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["mean_ms"], 4.)
        self.assertAlmostEqual(row["median_ms"], 2.)
        self.assertEqual(bench_cuda.summarize([])["n"], 0)

    def test_ratio_is_nan_without_a_measurement(self):
        self.assertEqual(bench_cuda.ratio(3., 2.), 1.5)
        self.assertNotEqual(bench_cuda.ratio(float("nan"), 2.), bench_cuda.ratio(float("nan"), 2.))
        self.assertNotEqual(bench_cuda.ratio(1., 0.), bench_cuda.ratio(1., 0.))

    def test_markdown_table_has_a_header_a_rule_and_one_line_per_row(self):
        table = bench_cuda.markdown_table(["a", "b"], [[1, "x"], [2, "y"]]).splitlines()
        self.assertEqual(table, ["| a | b |", "| --- | --- |", "| 1 | x |", "| 2 | y |"])

    def test_synthetic_examples_repeat_for_a_seed_and_differ_between_seeds(self):
        self.assertEqual(bench_cuda.synthetic_examples(5, seed=1), bench_cuda.synthetic_examples(5, seed=1))
        self.assertNotEqual(bench_cuda.synthetic_examples(5, seed=1), bench_cuda.synthetic_examples(5, seed=2))


class SizingChecks(unittest.TestCase):
    def test_the_widest_model_is_a_valid_config(self):
        cfg = bench_cuda.scaled_config(1024, 24, batch_size=4)
        self.assertEqual((cfg["width"], cfg["layers"]), (1024, 24))

    def test_plan_leaves_room_for_the_offloaded_run_but_not_the_plain_one(self):
        plan = bench_cuda.plan_offload_case(80 * bench_cuda.GIB)
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(plan["need"], bench_cuda.MIN_STATE_NEED)
        self.assertLess(plan["target_free"], plan["need"])
        self.assertGreater(plan["target_free"], plan["need_offload"])
        self.assertEqual(plan["ballast"], 80 * bench_cuda.GIB - plan["target_free"])
        # the helpers the training job asks agree: with that much free memory the state is offloaded
        free = plan["target_free"]
        self.assertTrue(studio_memory.offload_wanted(plan["config"], 1., "cuda", gpu=free))
        self.assertGreater(studio_memory.host_bytes(plan["config"]), 0)

    def test_plan_takes_the_smallest_model_that_works_whatever_the_gpu_size(self):
        small, big = (bench_cuda.plan_offload_case(n * bench_cuda.GIB) for n in (16, 80))
        self.assertEqual((small["width"], small["layers"]), (big["width"], big["layers"]))
        self.assertIn((big["width"], big["layers"]), bench_cuda.LADDER)
        self.assertEqual(small["ballast"], 16 * bench_cuda.GIB - small["target_free"])

    def test_plan_is_none_when_the_gpu_has_too_little_free(self):
        self.assertIsNone(bench_cuda.plan_offload_case(bench_cuda.GIB // 4))


class ClockChecks(unittest.TestCase):
    def test_a_cpu_run_ticks_once_per_step_and_offload_does_not_change_the_count(self):
        examples = bench_cuda.synthetic_examples(40)
        cfg = dict(bench_cuda.scaled_config(32, 1, batch_size=2), device="cpu", context=32)
        for kwargs in ({}, {"plain": True}, {"offload": True}):
            seconds, _ = bench_cuda.timed_run(cfg, examples, 4, 1, **kwargs)
            self.assertEqual(len(seconds), 2)  # 4 steps give 3 gaps; the first is warmup
            self.assertTrue(all(s >= 0 for s in seconds))


class RefusalChecks(unittest.TestCase):
    def test_main_refuses_without_cuda(self):
        err = io.StringIO()
        with mock.patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stderr(err):
            self.assertEqual(bench_cuda.main([]), 2)
        self.assertIn("CUDA", err.getvalue())


if __name__ == "__main__":
    unittest.main()

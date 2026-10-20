"""studio_scaling (the fitted curve, the flat point, the measured step rate) and studio_ia.sizing (what the plan says about size)."""
import threading
import unittest

import studio_scaling as scaling
import studio_spec
from studio_ia import sizing

# Version 1's validation loss against steps (133k parameters, byte tokenizer, five catalog sets), as observed on a live run.
EVIDENCE = [(1000, 2.515), (5000, 2.006), (10000, 1.843), (20000, 1.764), (40000, 1.678), (80000, 1.622), (120000, 1.608), (153000, 1.598)]
MEASURED = {"seconds_per_step": .2275, "tokens_per_second": 18_000., "device": "cpu"}


class FakeClock:
    """A clock that moves `tick` seconds each time it is read."""

    def __init__(self, tick):
        self.now, self.tick = 0., tick

    def __call__(self):
        self.now += self.tick
        return self.now


class FitTests(unittest.TestCase):
    def test_the_fit_follows_the_observed_curve_and_finds_where_it_goes_flat(self):
        fit = scaling.fit_curve(EVIDENCE)
        self.assertLess(fit["rmse"], .02)
        self.assertTrue(1.4 < fit["floor"] < 1.6, fit)
        for step, loss in EVIDENCE:
            self.assertAlmostEqual(scaling.predict(fit, step), loss, delta=.03)
        flat = scaling.flat_point(fit, 1000, 819_635)
        self.assertTrue(100_000 < flat < 180_000, flat)  # the person saw it flat from about step 80,000 to 120,000
        self.assertEqual(scaling.flat_point(fit, 1000, 5000), 5000)  # still falling fast: the limit is returned
        self.assertGreater(scaling.predict(fit, 819_635), fit["floor"])

    def test_too_few_points_or_a_curve_that_does_not_fall_give_no_fit(self):
        self.assertIsNone(scaling.fit_curve(EVIDENCE[:4]))
        self.assertIsNone(scaling.fit_curve([(s, 2. + s / 1000) for s in range(1000, 9000, 1000)]))
        self.assertIsNone(scaling.fit_curve([(s, 2.) for s in range(1000, 9000, 1000)]))
        self.assertIsNone(scaling.fit_curve([(0, 5.), (-1, 4.)]))

    def test_a_densely_logged_start_does_not_outweigh_the_rest(self):
        dense = [(s, 2.5 - .0002 * s) for s in range(1, 300)] + EVIDENCE[2:]
        fit = scaling.fit_curve(dense)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(scaling.predict(fit, 153000), 1.598, delta=.05)

    def test_the_curve_points_run_from_start_to_stop(self):
        points = scaling.curve_points(scaling.fit_curve(EVIDENCE), 1000, 100_000, 10)
        self.assertEqual((len(points), round(points[0][0]), round(points[-1][0])), (10, 1000, 100_000))
        self.assertEqual([y for _, y in points], sorted((y for _, y in points), reverse=True))


class StepRateTests(unittest.TestCase):
    def test_resizing_gqa_preserves_valid_query_and_key_value_groups(self):
        original = studio_spec.default_spec()
        original["block"]["mixer"] = {"id": "gqa", "config": {"heads": 4, "kv_heads": 2}}
        for width in (16, 32, 48, 64, 256, 512):
            with self.subTest(width=width):
                resized = sizing.resized_spec(original, width, 1)
                config = resized["block"]["mixer"]["config"]
                self.assertEqual(width % config["heads"], 0)
                self.assertEqual(config["heads"] % config["kv_heads"], 0)
                self.assertGreaterEqual(config["kv_heads"], 1)
        self.assertEqual(original["block"]["mixer"]["config"]["kv_heads"], 2)

    def test_the_rate_is_the_clocks_mean_over_the_timed_steps_and_warmup_is_not_timed(self):
        calls = []

        def work():
            calls.append(1)
        clock = FakeClock(.25)  # read twice around every timed call: each reads as .25 s
        self.assertAlmostEqual(scaling.step_rate(work, steps=4, warmup=3, clock=clock), .25)
        self.assertEqual(len(calls), 7)

    def test_a_stop_ends_the_measure(self):
        stop = threading.Event()
        stop.set()
        self.assertIsNone(scaling.step_rate(lambda: None, stop=stop))

    def test_a_design_is_timed_on_the_device_a_run_would_use(self):
        spec = sizing.resized_spec(studio_spec.default_spec(), 16, 1)
        spec["dims"]["n_ctx"] = 32
        found = scaling.measure_step_rate(spec, 2, 64, steps=2, warmup=1, clock=FakeClock(.5))
        self.assertEqual((found["seconds_per_step"], found["device"], found["rows"], found["context"]), (.5, "cpu", 2, 32))
        self.assertAlmostEqual(found["tokens_per_second"], 2 * 32 / .5)
        self.assertIn("small", found["note"])  # auto chose the CPU for a model this small, and says so


class SizingTests(unittest.TestCase):
    spec = studio_spec.default_spec()
    PIECES, PARAMS, ROWS = 545_000_000, 133_312, 16
    STEPS = PIECES // (ROWS * 256)

    def plan(self, hours=4., evidence=None, gpu=None, steps=None, pieces=None, params=None, spec=None, measured=MEASURED):
        return sizing.sizing(spec or self.spec, 263, self.ROWS, 1, pieces or self.PIECES, params or self.PARAMS, steps or self.STEPS, hours,
                             measured, {"points": evidence} if evidence else None, gpu)

    def test_a_tiny_model_on_much_text_is_too_small_and_a_design_for_the_hours_is_suggested(self):
        out = self.plan()
        self.assertEqual(out["verdict"], "model_too_small")
        self.assertIn("too small", out["warning"])
        found = out["suggestion"]
        self.assertGreater(found["params"], self.PARAMS)
        self.assertLessEqual(found["hours"], 4.01)
        self.assertGreaterEqual(found["tokens_per_param"], sizing.RULE)
        self.assertLess(found["steps"], self.STEPS)
        # more hours buy a bigger design, never a smaller one
        self.assertGreaterEqual(self.plan(hours=24)["suggestion"]["params"], found["params"])
        self.assertLessEqual(self.plan(hours=.5)["suggestion"]["params"], found["params"])

    def test_the_time_is_the_measured_step_time_times_the_steps(self):
        out = self.plan()
        self.assertAlmostEqual(out["run_hours"], self.STEPS * .2275 / 3600)
        self.assertEqual((out["device"], out["seconds_per_step"]), ("cpu", .2275))
        self.assertIsNone(self.plan(measured=None)["run_hours"])
        self.assertIsNone(self.plan(measured=None)["suggestion"])

    def test_the_earlier_runs_curve_says_how_many_steps_are_wasted(self):
        points = [(s * 4096, v) for s, v in EVIDENCE]
        out = self.plan(evidence=points, steps=819_635)  # what the live plan asked for
        self.assertIn("its own last run", out["warning"])
        flat = out["expected"]["flat_steps"]
        self.assertTrue(100_000 < flat < 180_000, flat)
        self.assertIn(f"{flat:,}", out["warning"])
        self.assertLess(out["expected"]["loss_planned"], out["expected"]["loss_flat"])

    def test_a_plan_with_little_text_for_the_model_says_so(self):
        out = self.plan(pieces=300_000, steps=73)
        self.assertEqual(out["verdict"], "data_too_small")
        self.assertIn("memorising", out["warning"])

    def test_a_model_sized_for_its_text_is_balanced(self):
        out = self.plan(pieces=3_000_000, steps=733)
        self.assertEqual(out["verdict"], "balanced")

    def test_a_gpu_is_mentioned_only_when_the_run_stays_on_the_cpu_and_the_gpu_would_be_faster(self):
        gpu = {"available": True, "kind": "cuda", "name": "RTX 4090"}
        cpu_only = studio_spec.default_spec()
        cpu_only["train"]["run"] = {"device": "cpu"}
        note = self.plan(gpu=gpu, spec=cpu_only)["gpu_note"]
        self.assertIn("RTX 4090", note)
        self.assertIn("cuda", note)
        self.assertEqual(self.plan(gpu=None, spec=cpu_only)["gpu_note"], "")
        self.assertEqual(self.plan(gpu={"available": False}, spec=cpu_only)["gpu_note"], "")
        on_gpu = studio_spec.default_spec()
        on_gpu["train"]["run"] = {"device": "cuda"}
        self.assertEqual(self.plan(gpu=gpu, spec=on_gpu, measured=dict(MEASURED, device="cuda"))["gpu_note"], "")

    def test_auto_moves_a_wide_design_to_the_gpu_and_keeps_a_small_one_on_the_cpu(self):
        gpu = {"available": True, "kind": "cuda"}
        small, wide = self.spec, sizing.resized_spec(self.spec, 512, 8)
        self.assertEqual((sizing.device_kind(small, sizing._work(small, 16), gpu), sizing.device_kind(wide, sizing._work(wide, 16), gpu)),
                         ("cpu", "cuda"))
        self.assertEqual(sizing.device_kind(wide, sizing._work(wide, 16), None), "cpu")

    def test_a_hand_built_design_gets_a_verdict_and_no_suggested_size(self):
        built = studio_spec.default_spec()
        built["blocks"] = [dict(built["block"]) for _ in range(built["dims"]["n_layer"])]
        self.assertIsNone(sizing.resized_spec(built, 128, 4))
        self.assertIsNone(self.plan(spec=built)["suggestion"])

    def test_every_rung_of_the_ladder_is_a_valid_design_and_they_grow(self):
        import studio_estimate
        sizes = []
        for width, layers in sizing.LADDER:
            spec = sizing.resized_spec(self.spec, width, layers)
            self.assertIsNotNone(spec, (width, layers))
            self.assertEqual((spec["dims"]["d_model"], spec["dims"]["n_layer"]), (width, layers))
            sizes.append(studio_estimate.estimate(spec, 263)["total"])
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(self.spec["dims"]["d_model"], 64)  # the design given is not changed

    def test_words_for_hours_and_counts(self):
        self.assertEqual([sizing.hours_text(h) for h in (.25, 1.5, 12, 60, None)], ["15 min", "1.5 h", "12 h", "2.5 days", ""])
        self.assertEqual([sizing.count_text(n) for n in (950, 133_312, 4_200_000, 545_000_000)], ["950", "133k", "4.2M", "545M"])


if __name__ == "__main__":
    unittest.main()

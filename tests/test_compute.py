"""The compute planner, the learning-rate range test and the Builder cost panel's lines about them."""
import math
import threading
import unittest

import studio_compute as sc
import studio_quicktest as quicktest
from studio_ia import design as D
from studio_ia.pages import builder as B
from studio_ia.services import FakeServices
from studio_spec import default_spec, validate_spec


def tiny_spec():
    s = default_spec()
    s["dims"].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
    return validate_spec(s)


EXAMPLES = [{"text": t} for t in quicktest.STORIES]
CPU = {"device": "cpu"}


class Plan(unittest.TestCase):
    def test_twenty_tokens_per_parameter(self):
        p = sc.plan(1_000_000, 100_000_000)
        self.assertEqual(p["optimal_tokens"], 20_000_000)
        self.assertEqual(p["planned_tokens"], 20_000_000)
        self.assertEqual(p["data_params"], 5_000_000)  # 100M tokens support 5M parameters
        self.assertEqual(p["recommended_params"], 5_000_000)
        self.assertEqual(p["tokens_per_param"], 20)
        self.assertAlmostEqual(p["epochs"], .2)
        self.assertIsNone(p["limit"])
        self.assertFalse(p["under_trained"] or p["over_repeated"])

    def test_steps_come_from_batch_times_context(self):
        p = sc.plan(1_000_000, 100_000_000, batch_tokens=16 * 256)
        self.assertEqual(p["steps"], math.ceil(20_000_000 / 4096))
        self.assertIsNone(sc.plan(1_000_000, 100_000_000)["steps"])

    def test_data_caps_the_run_at_four_epochs(self):
        p = sc.plan(1_000_000, 2_000_000)  # 2M tokens support 100K parameters; the design has ten times that
        self.assertEqual(p["planned_tokens"], 8_000_000)  # four epochs, short of 20M
        self.assertEqual(p["limit"], "data")
        self.assertAlmostEqual(p["epochs"], 4)
        self.assertAlmostEqual(p["epochs_for_optimal"], 10)
        self.assertTrue(p["over_repeated"])
        self.assertFalse(p["under_trained"])  # 8 tokens per parameter is not far below 20

    def test_far_below_the_rule_is_under_trained(self):
        p = sc.plan(10_000_000, 1_000_000)  # 4M tokens against 200M wanted: 0.4 per parameter
        self.assertTrue(p["under_trained"])
        self.assertTrue(p["over_repeated"])
        self.assertAlmostEqual(p["tokens_per_param"], .4)

    def test_four_epochs_exactly_is_not_over_repeated(self):
        p = sc.plan(1_000_000, 5_000_000)  # 20M wanted = four passes over 5M
        self.assertFalse(p["over_repeated"])
        self.assertTrue(sc.plan(1_000_000, 4_999_999)["over_repeated"])

    def test_time_budget_limits_tokens_and_size(self):
        p = sc.plan(1_000_000, 1_000_000_000, seconds=100, tokens_per_second=50_000)
        self.assertEqual(p["time_tokens"], 5_000_000)
        self.assertEqual(p["planned_tokens"], 5_000_000)  # 5 per parameter, not 20
        self.assertEqual(p["limit"], "time")
        self.assertEqual(p["recommended_params"], 250_000)  # the size the time budget can train properly
        self.assertAlmostEqual(p["seconds_needed"], 100)
        self.assertFalse(p["under_trained"])  # exactly at the 5-per-parameter cut-off
        self.assertTrue(sc.plan(1_000_000, 1_000_000_000, seconds=10, tokens_per_second=50_000)["under_trained"])

    def test_a_generous_budget_changes_nothing(self):
        p = sc.plan(1_000_000, 100_000_000, seconds=3600, tokens_per_second=50_000)
        self.assertIsNone(p["limit"])
        self.assertEqual(p["planned_tokens"], 20_000_000)
        self.assertAlmostEqual(p["seconds_needed"], 400)

    def test_recommended_size_takes_the_tighter_of_data_and_time(self):
        p = sc.plan(1_000_000, 10_000_000, seconds=100, tokens_per_second=50_000)
        self.assertEqual(p["recommended_params"], 250_000)  # time (5M tokens) is tighter than data (10M)

    def test_unknown_data_raises_no_flags(self):
        p = sc.plan(1_000_000, None)
        self.assertIsNone(p["data_params"])
        self.assertIsNone(p["recommended_params"])
        self.assertFalse(p["under_trained"] or p["over_repeated"])
        self.assertEqual(p["planned_tokens"], 20_000_000)

    def test_throughput_agrees_with_the_time_model_the_pretraining_page_uses(self):
        from studio_ux import step_seconds
        cfg = {"batch_size": 16, "context": 256, "layers": 4}
        self.assertAlmostEqual(sc.throughput(16, 256, 1_000_000, 4), 16 * 256 / step_seconds(cfg, 1_000_000, "cpu"))
        self.assertGreater(sc.throughput(16, 256, 1_000_000, 4, "cuda"), sc.throughput(16, 256, 1_000_000, 4, "cpu"))


class Suggest(unittest.TestCase):
    def curve(self):
        rates = [1e-6 * 10 ** (i * 6 / 99) for i in range(100)]
        # flat, then a fall that is steepest around index 60, then a floor
        losses = [5. - 2.5 / (1 + math.exp(-(i - 60) / 5)) for i in range(100)]
        return rates, losses

    def test_a_tenth_of_the_steepest_rate(self):
        rates, losses = self.curve()
        rate, index = sc.suggest(rates, losses)
        self.assertAlmostEqual(rate, rates[index] / 10)
        self.assertTrue(55 <= index <= 75)

    def test_a_noisy_dip_at_the_start_is_not_read_as_the_steepest_fall(self):
        rates, losses = self.curve()
        losses[2] -= 3.  # one bad batch while the rate is still tiny
        _, index = sc.suggest(rates, losses)
        self.assertTrue(55 <= index <= 75)

    def test_no_fall_gives_no_suggestion(self):
        rates = [1e-6 * 10 ** (i * 6 / 99) for i in range(50)]
        self.assertEqual(sc.suggest(rates, [3. + i * .01 for i in range(50)]), (None, None))
        self.assertEqual(sc.suggest(rates[:2], [3., 2.]), (None, None))

    def test_the_suggestion_stays_inside_the_sweep(self):
        rates = [1e-6 * 10 ** (i * 6 / 99) for i in range(50)]
        rate, _ = sc.suggest(rates, [5. - .1 * i for i in range(50)])
        self.assertGreaterEqual(rate, rates[0])
        self.assertLessEqual(rate, rates[-1])

    def test_smoothing_has_no_start_bias(self):
        self.assertAlmostEqual(sc.smooth([2., 2., 2.])[0], 2.)
        self.assertAlmostEqual(sc.smooth([2., 2., 2.])[-1], 2.)


class RangeTest(unittest.TestCase):
    def test_a_tiny_model_gets_a_suggestion_inside_the_sweep(self):
        events = []
        r = sc.lr_range_test(tiny_spec(), EXAMPLES, config=CPU, emit=events.append)
        self.assertEqual(len(r["rates"]), sc.SWEEP_STEPS)
        self.assertTrue(math.isfinite(r["suggestion"]))
        self.assertLessEqual(r["lr_min"], r["suggestion"])
        self.assertLessEqual(r["suggestion"], r["lr_max"])
        self.assertAlmostEqual(r["suggestion"], r["rates"][r["steepest"]] / 10)
        self.assertFalse(r["cancelled"])
        self.assertEqual(r["rates"][0], 1e-6)
        self.assertTrue(all(b > a for a, b in zip(r["rates"], r["rates"][1:])))  # the rate only rises
        self.assertEqual(len(events), sc.SWEEP_STEPS)
        self.assertEqual(events[0]["type"], "lr_range")

    def test_the_sweep_ends_early_when_the_loss_diverges(self):
        r = sc.lr_range_test(tiny_spec(), EXAMPLES, lr_min=1e-3, lr_max=1e3, config=CPU)
        self.assertTrue(r["diverged"])
        self.assertLess(len(r["rates"]), sc.SWEEP_STEPS)
        self.assertTrue(math.isfinite(r["suggestion"]))
        self.assertLess(r["suggestion"], r["rates"][-1])  # a tenth of the steepest point, well before the blow-up
        self.assertGreaterEqual(r["failed_at"], r["rates"][-1])  # the rate that blew up: the last one tried, never an earlier one

    def test_the_studio_runs_it_on_the_processor(self):
        from unittest import mock
        from studio_ia.services import WorkspaceServices
        svc = WorkspaceServices.__new__(WorkspaceServices)  # only the text sample is needed
        svc.workspace = mock.Mock(state={"datasets": [{"id": "a"}]})
        svc._heldout = lambda: []
        with mock.patch("studio_data_kinds.text_sample", return_value=(["some text"], None)), \
                mock.patch.object(sc, "lr_range_test", return_value={}) as run:
            svc.lr_range_test(tiny_spec())
        self.assertEqual(run.call_args.kwargs["config"], {"device": "cpu"})  # never a GPU probe or Metal work off the Tk thread

    def test_stopping_cancels_the_test(self):
        stop = threading.Event()
        stop.set()
        r = sc.lr_range_test(tiny_spec(), EXAMPLES, stop=stop, config=CPU)
        self.assertTrue(r["cancelled"])
        self.assertEqual(r["rates"], [])
        self.assertIsNone(r["suggestion"])

    def test_stopping_part_way_keeps_what_was_measured_but_suggests_nothing(self):
        stop = threading.Event()
        r = sc.lr_range_test(tiny_spec(), EXAMPLES, stop=stop, config=CPU, emit=lambda e: stop.set() if e["step"] == 10 else None)
        self.assertTrue(r["cancelled"])
        self.assertEqual(len(r["rates"]), 10)
        self.assertIsNone(r["suggestion"])

    def test_the_design_is_untouched(self):
        spec = tiny_spec()
        before = repr(spec)
        sc.lr_range_test(spec, EXAMPLES, steps=5, config=CPU)
        self.assertEqual(repr(spec), before)

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            sc.lr_range_test(tiny_spec(), EXAMPLES, steps=2, config=CPU)
        with self.assertRaises(ValueError):
            sc.lr_range_test(tiny_spec(), EXAMPLES, lr_min=1., lr_max=.1, config=CPU)


class CostPanel(unittest.TestCase):
    def test_counting_and_no_text(self):
        self.assertEqual(D.compute_lines(1_000_000, 256, 4, None), [(B.T["compute_counting"], False)])
        self.assertEqual(D.compute_lines(1_000_000, 256, 4, {}), [(B.T["compute_none"], False)])
        self.assertEqual(D.compute_lines(1_000_000, 256, 4, {"pieces": None}), [(B.T["compute_none"], False)])

    def test_a_design_far_too_big_for_the_text_says_so_and_is_flagged(self):
        lines = D.compute_lines(10_000_000, 256, 4, {"pieces": 1_000_000, "tokens_per_second": 5000.})
        text, bad = lines[0]
        self.assertIn("supports about 50.0K parameters at 20 word pieces each", text)
        self.assertIn("this design has 10.0M", text)
        self.assertIn("a smaller design or more text", text)  # the remedy is size or text, not training longer
        self.assertTrue(bad)
        self.assertTrue(lines[-1][1])  # and the repeat warning
        self.assertIn("200 times", lines[-1][0])

    def test_steps_and_time_for_a_design_the_text_fits(self):
        lines = D.compute_lines(100_000, 256, 2, {"pieces": 100_000_000, "tokens_per_second": 10_000.})
        self.assertEqual(len(lines), 2)
        self.assertIn("so your text could train a bigger one", lines[0][0])
        self.assertFalse(lines[0][1])
        self.assertIn(f"about {math.ceil(2_000_000 / (D.ROWS * 256)):,} steps", lines[1][0])
        self.assertIn("under one pass", lines[1][0])
        self.assertIn("3 minutes", lines[1][0])  # 2M tokens at 10K a second is 200 s

    def test_supported_estimate_says_about_once_whether_or_not_the_count_is_estimated(self):
        for preview in ({"pieces": 2_000_000, "tokens_per_second": 10_000.},
                        {"pieces": 2_000_000, "tokens_per_second": 10_000., "estimated": True}):
            text = D.compute_lines(1_000_000, 256, 2, preview)[0][0]
            self.assertEqual(text.count("about"), 1, text)
            self.assertIn("supports about 100.0K parameters", text)
            self.assertNotIn("about about", text)

    def test_repeated_text_is_named_but_not_called_under_trained(self):
        lines = D.compute_lines(1_000_000, 256, 2, {"pieces": 2_000_000, "tokens_per_second": 10_000., "estimated": True})
        self.assertIn("about 100.0K parameters", lines[0][0])
        self.assertIn("repeated", lines[0][0])
        self.assertNotIn("under-trained", lines[0][0])
        self.assertIn("4.0 passes", lines[1][0])

    def test_a_design_between_one_and_four_passes_says_the_text_is_reread(self):
        lines = D.compute_lines(300_000, 256, 2, {"pieces": 2_000_000, "tokens_per_second": 10_000.})  # one pass supports 100K
        self.assertIn("reads your text about 3.0 times", lines[0][0])
        self.assertNotIn("enough text", lines[0][0])
        self.assertFalse(lines[0][1])
        self.assertEqual(len(lines), 2)  # within 4 passes: no repeat warning

    def test_without_a_measured_speed_the_time_model_is_used(self):
        lines = D.compute_lines(100_000, 256, 2, {"pieces": 100_000_000})
        self.assertIn("At this computer's speed", lines[1][0])

    def test_passes_in_words(self):
        self.assertEqual(D.passes_text(.2), "under one pass")
        self.assertEqual(D.passes_text(1.), "one pass")
        self.assertEqual(D.passes_text(2.5), "2.5 passes")
        self.assertEqual(D.passes_text(12.2), "12 passes")

    def test_rate_sentence(self):
        result = FakeServices().lr_range_test(tiny_spec())
        text = D.lr_text(result, .0006)
        self.assertIn(f"{result['suggestion']:.2g}", text)
        self.assertIn("0.0006", text)
        self.assertNotIn("blew up", text)
        self.assertIn("blew up", D.lr_text(dict(result, diverged=True), .0006))
        self.assertEqual(D.lr_text(None), "")
        self.assertIn("no suggestion", D.lr_text(dict(result, suggestion=None)))
        self.assertIn("Stopped", D.lr_text(dict(result, cancelled=True)))
        self.assertIn("could not run: Boom", D.lr_text({"error": "Boom"}))

    def test_a_sweep_that_blew_up_before_falling_says_so_at_the_failing_rate(self):
        early = dict(rates=[], losses=[], suggestion=None, steepest=None, diverged=True, failed_at=1e-6, cancelled=False)
        self.assertEqual(D.lr_text(early), B.T["lr_blew_up_early"].format(rate="1e-06"))
        result = dict(FakeServices().lr_range_test(tiny_spec()), diverged=True, failed_at=.25)
        self.assertIn("blew up past 0.25", D.lr_text(result, .0006))

    def test_every_new_sentence_exists(self):
        for key in ("compute_supports", "compute_under", "compute_enough", "compute_reread", "compute_roomy", "compute_repeats", "compute_steps", "compute_time",
                    "compute_repeat", "compute_recount", "compute_counting", "compute_none", "compute_tip", "lr_find", "lr_stop", "lr_tip",
                    "lr_running", "lr_result", "lr_blew_up", "lr_blew_up_early", "lr_flat", "lr_stopped", "lr_failed"):
            self.assertIn(key, B.T)


if __name__ == "__main__":
    unittest.main()

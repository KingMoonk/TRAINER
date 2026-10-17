"""The compute planner, the learning-rate range test and the Builder cost panel's lines about them."""
import math
import threading
import unittest

import studio_compute as sc
import studio_quicktest as quicktest
from studio_spec import default_spec, validate_spec


def tiny_spec():
    s = default_spec()
    s["dims"].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
    return validate_spec(s)


EXAMPLES = [{"text": t} for t in quicktest.STORIES]
CPU = {"device": "cpu"}


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


if __name__ == "__main__":
    unittest.main()

"""No idle GPUs: members download the outer update quantized (every theta stays bit-identical), int8 changes carry their rounding
error into the next change, rounds can be time-boxed (the outer step weighs each change by the tokens it trained)."""
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

import studio_diloco
from studio_diloco import (add_update, decode_arrays, dequantize_int8, encode_arrays, encode_feedback, quantize_int8, timed_steps,
                           token_average, train_seconds, update_origin)
from tests import diloco_support as support
from tests.test_diloco import Clock


class Int8(unittest.TestCase):
    def test_each_block_has_its_own_scale_so_an_outlier_spoils_only_its_block(self):
        rng = np.random.default_rng(1)
        value = (rng.normal(size=1024) * .01).astype(np.float32)
        value[3] = 50.  # an outlier in the first block
        values, scale = quantize_int8(value)
        self.assertEqual((values.dtype, values.shape, scale.shape), (np.int8, value.shape, (1024 // studio_diloco.INT8_BLOCK,)))
        out = dequantize_int8(values, scale)
        first, rest = slice(0, studio_diloco.INT8_BLOCK), slice(studio_diloco.INT8_BLOCK, None)
        self.assertLessEqual(np.abs(out[first] - value[first]).max(), 50. / 127 / 2 + 1e-6)
        self.assertLessEqual(np.abs(out[rest] - value[rest]).max(), np.abs(value[rest]).max() / 127 / 2 + 1e-9)
        self.assertLess(np.abs(out[rest] - value[rest]).max(), .001)  # one scale for the tensor would have rounded all of these to 0

    def test_a_tensor_that_is_not_a_whole_number_of_blocks_and_a_legacy_per_tensor_scale_round_trip(self):
        value = np.linspace(-1, 1, 300, dtype=np.float32).reshape(10, 30)
        values, scale = quantize_int8(value)
        self.assertEqual(scale.shape, (2,))
        np.testing.assert_allclose(dequantize_int8(values, scale), value, atol=1. / 127 / 2 + 1e-6)
        legacy = dequantize_int8(np.array([127, -127, 0], np.int8), np.float32(.5 / 127))
        np.testing.assert_allclose(legacy, [.5, -.5, 0.])
        with self.assertRaises(ValueError):
            dequantize_int8(values, scale[:1])

    def test_the_rounding_error_of_a_change_is_added_to_the_next_one(self):
        small = np.full(256, .3, np.float32)
        small[0] = 100.  # the block's step is .79, so .3 rounds to 0 on its own
        arrays = {"w": small}
        _, plain = encode_feedback(arrays, "int8")
        self.assertGreater(np.abs(plain["w"][1:]).max(), .29)  # everything the encoding dropped is kept...
        sent = np.zeros(256, np.float32)
        residual = None
        for _ in range(10):
            body, residual = encode_feedback(arrays, "int8", residual)
            sent += decode_arrays(body)["w"]
        np.testing.assert_allclose(sent[1:], 10 * .3, atol=.79)  # ...and goes out with a later change: ten .3s add up to ~3, not 0
        self.assertGreater(sent[1:].mean(), 2.)
        self.assertLessEqual(np.abs(sent + residual["w"] - 10 * small).max(), 1e-3)  # nothing is lost: what went out plus what is held is what was meant

    def test_a_non_finite_value_is_not_carried_over(self):
        body, residual = encode_feedback({"w": np.array([1., np.inf, 2.], np.float32)}, "int8")
        self.assertFalse(np.isfinite(decode_arrays(body)["w"]).all())  # the coordinator refuses it
        self.assertTrue(np.isfinite(residual["w"]).all())


class Directory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.addCleanup(studio_diloco.COORDINATORS.clear)


class UpdateDownloads(Directory):
    def chain(self, dtype, rounds=4, **changes):
        """A member that downloads theta once and then only the outer update, against a coordinator; returns both after `rounds` rounds."""
        coordinator = support.build(self.dir / dtype, dtype, inner_steps=2, rounds=rounds + 1, outer_lr=.7, outer_momentum=.9, delta_dtype=dtype, **changes)
        coordinator.join("a")
        mine = decode_arrays(coordinator.params_bytes(0, "a"))
        rng = np.random.default_rng(3)
        sizes = []
        for number in range(rounds):
            delta = {name: (rng.normal(size=array.shape) * .01).astype(np.float32) for name, array in coordinator.theta.items()}
            coordinator.submit(number, "a", 1, 0, 10, 2, delta, 2., 3., 1234)
            body = coordinator.params_bytes(number + 1, "a", since=number)
            self.assertEqual(update_origin(body), number)
            sizes.append(len(body))
            mine = add_update(mine, decode_arrays(body))
        return coordinator, mine, sizes

    def test_a_member_that_downloads_fp16_updates_holds_exactly_the_coordinators_theta(self):
        coordinator, mine, sizes = self.chain("float16")
        for name, value in coordinator.theta.items():
            np.testing.assert_array_equal(mine[name], value, err_msg=name)
        self.assertLess(max(sizes), len(encode_arrays(coordinator.theta, "float32")))  # (the test model is tiny: headers weigh as much as values)

    def test_a_member_that_downloads_int8_updates_holds_exactly_the_coordinators_theta(self):
        coordinator, mine, sizes = self.chain("int8")
        for name, value in coordinator.theta.items():
            np.testing.assert_array_equal(mine[name], value, err_msg=name)
        self.assertLess(max(sizes), len(encode_arrays(coordinator.theta, "float16")))
        self.assertTrue(any(np.abs(r).max() > 0 for r in coordinator.update_residual.values()))  # the coordinator holds what the update lost

    def test_the_coordinators_theta_is_the_update_that_went_out_so_the_rounding_is_not_lost(self):
        coordinator, _, _ = self.chain("int8", rounds=1)
        unrounded = support.build(self.dir / "plain", "plain", inner_steps=2, rounds=2, outer_lr=.7, outer_momentum=.9, delta_dtype="float32")
        self.assertEqual(coordinator.settings.delta_dtype, "int8")
        for name in coordinator.theta:  # the update differs from the exact one by what int8 rounds, and the residual is that difference
            np.testing.assert_allclose(coordinator.theta[name], unrounded.theta[name], atol=1.)

    def test_a_fresh_joiner_or_a_member_out_of_step_or_after_a_pivot_gets_the_whole_theta(self):
        coordinator, mine, _ = self.chain("float16", rounds=3)
        coordinator.join("b")
        for since in (None, 0, 5):  # none held, a round that is not the one before, a round from the future
            body = coordinator.params_bytes(3, "b", since=since)
            self.assertIsNone(update_origin(body))
            for name, value in decode_arrays(body).items():
                np.testing.assert_array_equal(value, coordinator.theta[name])
        self.assertIsNotNone(update_origin(coordinator.params_bytes(3, "a", since=2)))
        coordinator._theta_replaced()  # a pivot or an accepted polish replaced theta by something that is not an update
        self.assertIsNone(update_origin(coordinator.params_bytes(3, "a", since=2)))

    def test_fp32_runs_keep_downloading_the_whole_theta(self):
        coordinator = support.build(self.dir / "f32", "f32", inner_steps=2, rounds=3, outer_lr=1., delta_dtype="float32")
        coordinator.join("a")
        coordinator.submit(0, "a", 1, 0, 10, 2, support.constant_delta(coordinator), 2., 3., 1234)
        self.assertIsNone(update_origin(coordinator.params_bytes(1, "a", since=0)))

    def test_the_round_records_what_the_members_downloaded(self):
        coordinator, _, sizes = self.chain("float16", rounds=2)
        self.assertEqual(coordinator.history[1]["theta_bytes"], sizes[0])  # round 1's members took round 0's update


class Weighted(Directory):
    def test_the_outer_step_weighs_each_change_by_the_tokens_it_trained(self):
        slow, fast = {"w": np.array([1., 1.], np.float32)}, {"w": np.array([5., -3.], np.float32)}
        np.testing.assert_allclose(token_average([(10, slow), (30, fast)])["w"], [4., -2.])
        coordinator = support.build(self.dir, inner_steps=2, rounds=2, outer_lr=1., outer_momentum=0., delta_dtype="float32", min_members=2)
        coordinator.join("a")
        coordinator.join("b")
        before = {name: value.copy() for name, value in coordinator.theta.items()}
        name = next(iter(before))
        shape = before[name].shape
        coordinator.submit(0, "a", 1, 0, 10, 1, {n: np.full(v.shape, 1., np.float32) for n, v in before.items()}, 2., 3., 1)
        coordinator.submit(0, "b", 1, 0, 30, 3, {n: np.full(v.shape, 5., np.float32) for n, v in before.items()}, 2., 3., 1)
        np.testing.assert_allclose(coordinator.theta[name], before[name] - np.full(shape, 4., np.float32), rtol=1e-6)


class Late(Directory):
    def start(self):
        clock = Clock()
        c = support.build(self.dir, clock=clock, inner_steps=2, rounds=5, min_members=2, missed_limit=3, outer_lr=1., outer_momentum=0., delta_dtype="float32")
        c.join("a")
        c.join("b")
        full = lambda value: {n: np.full(v.shape, value, np.float32) for n, v in c.theta.items()}  # noqa: E731
        return clock, c, full

    def test_a_change_that_comes_in_after_its_round_closed_counts_in_the_next_outer_step_instead_of_being_dropped(self):
        clock, c, full = self.start()
        c.submit(0, "a", 1, 0, 10, 2, full(1.), 2., 3., 1)
        clock.now += 2000
        c.snapshot()  # round 0 closes without b
        self.assertEqual(c.history[0]["dropped"], ["b"])
        before = {n: v.copy() for n, v in c.theta.items()}
        c.submit(0, "b", 1, 0, 30, 2, full(5.), 2., 3., 1)  # b's change for the closed round: kept, not refused
        c.submit(1, "a", c.current.generation, 0, 10, 2, full(1.), 2., 3., 1)
        c.submit(1, "b", c.current.generation, 0, 10, 2, full(1.), 2., 3., 1)
        self.assertEqual(c.history[1]["late"], ["b"])
        name = next(iter(before))
        np.testing.assert_allclose(c.theta[name], before[name] - (10 * 1. + 10 * 1. + 30 * 5.) / 50, rtol=1e-6)  # weighted by tokens, stale or not
        self.assertEqual(c.history[0]["late"], [])

    def test_a_change_two_rounds_late_or_twice_over_or_from_a_stranger_is_still_refused(self):
        clock, c, full = self.start()
        c.submit(0, "a", 1, 0, 10, 2, full(1.), 2., 3., 1)
        clock.now += 2000
        c.snapshot()
        with self.assertRaises(studio_diloco.Stale):
            c.submit(0, "stranger", 1, 0, 10, 2, full(1.), 2., 3., 1)
        with self.assertRaises(studio_diloco.Stale):
            c.submit(0, "a", 1, 0, 10, 2, full(1.), 2., 3., 1)  # it contributed already
        c.submit(0, "b", 1, 0, 30, 2, full(5.), 2., 3., 1)
        with self.assertRaises(studio_diloco.Stale):
            c.submit(0, "b", 1, 0, 30, 2, full(5.), 2., 3., 1)  # only once


class TimeBox(Directory):
    def test_the_members_train_for_the_fastest_ones_time_for_h_steps(self):
        on, off = support.settings(inner_steps=100, time_box=True), support.settings(inner_steps=100)
        self.assertEqual(train_seconds(on, ["a", "b"], {"a": .2, "b": .5}), 100 * .2)
        self.assertEqual(train_seconds(on, ["a", "b"], {"a": .2}), 20.)  # an untimed member does not set it
        self.assertEqual(train_seconds(on, ["a"], {}), 0.)  # nobody timed yet: the first round runs fixed steps
        self.assertEqual(train_seconds(off, ["a", "b"], {"a": .2, "b": .5}), 0.)

    def test_the_schedule_is_spread_over_the_steps_a_member_is_expected_to_do(self):
        s = support.settings(inner_steps=100, time_box=True)
        steps = timed_steps(s, ["fast", "slow", "new"], {"fast": .2, "slow": .5}, 20.)
        self.assertEqual(steps, {"fast": 100, "slow": 40, "new": 100})
        self.assertEqual(timed_steps(s, ["x"], {"x": .001}, 20.)["x"], int(100 * studio_diloco.MAX_SHARE))

    def member(self, now, train_deadline):
        clock = [now]
        member = studio_diloco.Member(None, "r", "n", 0, self.dir / "s", threading.Event(), clock=lambda: clock[0], progress=False)
        member.info = {"inner_steps": 10, "rounds": 3}
        member.first_step, member.deadline, member.train_deadline = 0, 100., train_deadline
        return member, clock

    def test_a_time_boxed_member_steps_until_the_clock_says_stop_not_for_a_count(self):
        member, clock = self.member(0., 30.)
        self.assertFalse(member.due(10, 0))  # H steps are not the end
        clock[0] = 29.
        self.assertFalse(member.due(15, 0))
        clock[0] = 30.
        self.assertTrue(member.due(15, 0))
        clock[0] = 0.
        self.assertTrue(member.due(int(10 * studio_diloco.MAX_SHARE), 0))  # a bound on the steps
        member.train_deadline, clock[0] = 500., 100.
        self.assertTrue(member.due(3, 0))  # the round's deadline still ends it
        plain, _ = self.member(0., None)
        self.assertFalse(plain.due(9, 0))
        self.assertTrue(plain.due(10, 0))


if __name__ == "__main__":
    unittest.main()

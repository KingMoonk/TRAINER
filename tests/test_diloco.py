"""studio_diloco: the outer optimizer, the shard deal, the coordinator's rules on a fake clock, restart from diloco.json, and one
member against the real routes giving the weights plain training gives."""
import json
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import studio_diloco
import studio_model
from studio_diloco import Stale, Windows, decode_arrays, encode_arrays
from tests import diloco_support as support


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Rules(unittest.TestCase):
    def test_outer_step_is_nesterov_on_the_pseudo_gradient(self):
        theta, v = {"w": np.array([1., 2.], np.float32)}, {"w": np.zeros(2, np.float32)}
        delta = {"w": np.array([.5, -.5], np.float32)}
        studio_diloco.outer_step(theta, v, delta, .7, .9)  # v = delta; theta -= .7 (delta + .9 v)
        np.testing.assert_allclose(v["w"], [.5, -.5])
        np.testing.assert_allclose(theta["w"], [1 - .7 * .95, 2 + .7 * .95], rtol=1e-6)
        studio_diloco.outer_step(theta, v, delta, .7, .9)  # v = .9 * .5 + .5 = .95
        np.testing.assert_allclose(v["w"], [.95, -.95], rtol=1e-6)

    def test_lr_one_and_no_momentum_adopts_the_average_weights(self):
        theta, v = {"w": np.array([1., 2.], np.float32)}, {"w": np.zeros(2, np.float32)}
        delta = studio_diloco.token_average([(1, {"w": np.array([.2, 0.], np.float32)}), (3, {"w": np.array([.6, 4.], np.float32)})])
        np.testing.assert_allclose(delta["w"], [.5, 3.])  # weighted by target tokens: a slower node counts for its work
        studio_diloco.outer_step(theta, v, delta, 1., 0.)
        np.testing.assert_allclose(theta["w"], [.5, -1.])

    def test_a_member_with_no_target_tokens_counts_for_nothing(self):
        self.assertIsNone(studio_diloco.token_average([(0, {"w": np.ones(1, np.float32)})]))

    def test_the_deal_is_deterministic_complete_and_ordered_for_one_member(self):
        deal = studio_diloco.deal_shards(7, 2, ["a", "b", "c"], 10)
        self.assertEqual(deal, studio_diloco.deal_shards(7, 2, ["a", "b", "c"], 10))
        self.assertEqual(sorted(sum(deal.values(), [])), list(range(10)))
        self.assertNotEqual(deal, studio_diloco.deal_shards(7, 3, ["a", "b", "c"], 10))
        self.assertEqual(studio_diloco.deal_shards(7, 5, ["a"], 6), {"a": [0, 1, 2, 3, 4, 5]})

    def test_sampler_seed_depends_on_seed_rank_and_round(self):
        seeds = {studio_diloco.sampler_seed(s, k, r) for s in (1, 2) for k in (0, 1) for r in (0, 1)}
        self.assertEqual(len(seeds), 8)

    def test_deadline_follows_the_fastest_step_time(self):
        s = support.settings(inner_steps=100, slack_ratio=.5, slack_seconds=10., first_round_seconds=900.)
        self.assertEqual(studio_diloco.round_seconds(s, None), 900.)
        self.assertEqual(studio_diloco.round_seconds(s, .2), 100 * .2 * 1.5 + 10)

    def test_arrays_round_trip_and_fp16_is_clipped(self):
        arrays = {"a.b": np.array([1e9, -1e9, .5], np.float32)}
        out = decode_arrays(encode_arrays(arrays, "float16"))["a.b"]
        np.testing.assert_allclose(out, [65504, -65504, .5])
        np.testing.assert_array_equal(decode_arrays(encode_arrays(arrays, "float32"))["a.b"], arrays["a.b"])
        with self.assertRaises(ValueError):
            decode_arrays(b"not an archive")

    def test_int8_arrays_are_a_quarter_of_fp32_and_within_half_a_step_of_the_original(self):
        rng = np.random.default_rng(0)
        arrays = {"w": rng.normal(size=(64, 32)).astype(np.float32) * .01, "zero": np.zeros(5, np.float32), "b.c": np.array([3., -3., 0.], np.float32)}
        body = encode_arrays(arrays, "int8")
        out = decode_arrays(body)
        self.assertEqual(set(out), set(arrays))  # the scales are not arrays of the model
        for name, value in arrays.items():
            self.assertEqual(out[name].dtype, np.float32)
            np.testing.assert_allclose(out[name], value, atol=np.abs(value).max() / 127 / 2 + 1e-9)
        np.testing.assert_array_equal(out["zero"], arrays["zero"])
        self.assertLess(len(body), len(encode_arrays(arrays, "float16")))
        big = {"w": rng.normal(size=(256, 256)).astype(np.float32)}
        self.assertLess(len(encode_arrays(big, "int8")), len(encode_arrays(big, "float16")) * .55)
        self.assertLess(len(encode_arrays(big, "int8")), len(encode_arrays(big, "float32")) * .27)

    def test_a_non_finite_int8_tensor_decodes_non_finite(self):
        out = decode_arrays(encode_arrays({"w": np.array([1., np.inf])}, "int8"))["w"]
        self.assertFalse(np.isfinite(out).all())

    def test_settings_refuse_nonsense(self):
        for bad in ({"inner_steps": 0}, {"delta_dtype": "int4"}, {"outer_momentum": 1.}, {"min_members": 3, "max_members": 2}):
            with self.assertRaises(ValueError):
                support.settings(**bad)

    def test_windows_read_several_stores_as_one(self):
        class Store(list):
            nbytes = 0

            def describe(self):
                return {"from_start": 1.}
        view = Windows([Store([1, 2]), Store([3])])
        self.assertEqual([view[i] for i in range(3)] + [view[-1]], [1, 2, 3, 3])
        self.assertEqual(view[:2], [1, 2])
        view.set([])
        self.assertEqual(len(view), 0)


class Directory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)


class Rounds(Directory):
    def setUp(self):
        super().setUp()
        self.clock = Clock()

    def coordinator(self, **changes):
        return support.build(self.dir, clock=self.clock, **changes)

    def send(self, c, node, number, steps=2, seconds=1., value=.01, tokens=10):
        view = c.round_view(number, node)
        self.assertEqual(view["state"], "open", view)
        c.submit(number, node, view["generation"], 0, tokens, steps, support.constant_delta(c, value), seconds)

    def test_a_member_that_misses_the_deadline_is_dropped_for_the_round_and_rejoins_the_next_with_the_new_weights(self):
        c = self.coordinator(inner_steps=2, rounds=5, outer_lr=1., min_members=2, grace_seconds=10., first_round_seconds=100.)
        c.join("a")
        c.join("b")
        name = next(iter(c.theta))
        start = c.theta[name].copy()
        self.send(c, "a", 0, steps=2, seconds=1.)
        self.clock.now = 109.
        self.assertEqual(c.round_view(0, "b")["state"], "open")  # still inside deadline + grace
        self.clock.now = 111.
        late = c.round_view(0, "b")
        self.assertEqual(late["state"], "closed")
        self.assertEqual(late["current"], 1)
        self.assertEqual(c.history[0]["dropped"], ["b"])
        self.assertEqual(c.history[0]["contributed"], ["a"])
        with self.assertRaises(Stale):
            c.submit(0, "b", 1, 0, 10, 2, support.constant_delta(c), 1.)  # too late: round 0 is closed, nothing applied
        np.testing.assert_allclose(c.theta[name], start - .01, rtol=1e-5)  # only a's delta, at outer lr 1
        view = c.round_view(1, "b")
        self.assertEqual(view["state"], "open")
        self.assertEqual(view["deadline_s"], .5 * 2 * 1.25 + 30.)  # from the fastest measured step time
        np.testing.assert_array_equal(decode_arrays(c.params_bytes(1))[name], c.theta[name])  # the new weights
        self.send(c, "a", 1)
        self.send(c, "b", 1)
        self.assertEqual(c.history[1]["contributed"], ["a", "b"])
        self.assertEqual(c.history[1]["dropped"], [])

    def test_a_member_silent_for_missed_limit_rounds_is_removed_and_must_join_again(self):
        c = self.coordinator(inner_steps=2, rounds=9, missed_limit=2, grace_seconds=1., first_round_seconds=5.)
        c.join("a")
        c.join("b")
        self.send(c, "a", 0)  # b joined mid-round 0, so round 0 is a's alone
        generation = c.round_view(1, "a")["generation"]
        for number in (1, 2):  # b is in these rounds and sends nothing
            self.send(c, "a", number)
            self.clock.now += 100.
            c.round_view(number, "a")
        self.assertNotIn("b", c.members)
        view = c.round_view(3, "a")
        self.assertGreater(view["generation"], generation)  # the membership changed
        self.assertFalse(c.round_view(3, "b")["member"])
        c.join("b")
        self.send(c, "a", 3)  # b joins at the boundary: round 3 is a's alone
        self.assertEqual(c.round_view(4, "b")["state"], "open")

    def test_stale_generation_round_and_epoch_are_refused_and_change_nothing(self):
        c = self.coordinator(inner_steps=2)
        c.join("a", epoch=4)
        view = c.round_view(0, "a")
        before = {k: v.copy() for k, v in c.theta.items()}
        delta = support.constant_delta(c)
        for number, generation, epoch, why in ((1, view["generation"], 4, "round"), (0, view["generation"] + 1, 4, "generation"),
                                               (0, view["generation"], 3, "epoch")):
            with self.subTest(why):
                with self.assertRaises(Stale):
                    c.submit(number, "a", generation, epoch, 10, 2, delta)
        with self.assertRaises(Stale):
            c.join("a", epoch=3)
        with self.assertRaises(Stale):
            c.submit(0, "stranger", view["generation"], 0, 10, 2, delta)
        self.assertEqual(c.current.contributions, {})
        for name, value in before.items():
            np.testing.assert_array_equal(c.theta[name], value)
        c.submit(0, "a", view["generation"], 4, 10, 2, delta)
        self.assertTrue(any((c.theta[k] != before[k]).any() for k in before))

    def test_a_second_contribution_to_a_round_is_refused(self):
        c = self.coordinator(inner_steps=2, min_members=2)
        c.join("a")
        c.join("b")
        self.send(c, "a", 0)
        with self.assertRaises(Stale):
            self.send(c, "a", 0)

    def test_a_delta_that_does_not_fit_the_model_is_a_value_error(self):
        c = self.coordinator()
        c.join("a")
        view = c.round_view(0, "a")
        with self.assertRaises(ValueError):
            c.submit(0, "a", view["generation"], 0, 10, 2, {"nope": np.zeros(1, np.float32)})
        bad = support.constant_delta(c)
        bad[next(iter(bad))][...] = np.nan
        with self.assertRaises(ValueError):
            c.submit(0, "a", view["generation"], 0, 10, 2, bad)

    def test_joins_and_leaves_take_effect_at_the_round_boundary(self):
        c = self.coordinator(inner_steps=2, min_members=1)
        c.join("a")
        first = c.round_view(0, "a")["generation"]
        c.join("b")  # mid-round: pending
        self.assertEqual(c.round_view(0, "b")["state"], "pending")
        self.assertEqual(len(c.current.members), 1)
        self.send(c, "a", 0)
        self.assertEqual(c.round_view(0, "b")["state"], "closed")
        view = c.round_view(1, "b")
        self.assertEqual((view["state"], view["members"]), ("open", 2))
        self.assertGreater(view["generation"], first)
        c.leave("b", view["generation"], 0)  # the open round stops waiting for b at once
        self.send(c, "a", 1)
        self.assertEqual(c.round_view(2, "a")["members"], 1)
        self.assertEqual(c.round_view(2, "a")["generation"], view["generation"] + 1)

    def test_the_run_pauses_below_min_members_and_continues_when_they_return(self):
        c = self.coordinator(inner_steps=2, min_members=2)
        c.join("a")
        self.assertEqual(c.round_view(0, "a")["state"], "pending")  # no round opens for one member
        c.join("b")
        self.assertEqual(c.round_view(0, "a")["state"], "open")
        self.send(c, "a", 0)
        c.leave("b", c.generation, 0)  # b leaves with its round-0 work unsent: the round closes on a alone
        self.assertIsNone(c.current)
        self.assertEqual(c.round_view(1, "a")["state"], "pending")
        c.join("c")
        self.assertEqual(c.round_view(1, "a")["state"], "open")

    def test_a_long_poll_returns_when_the_round_opens(self):
        c = self.coordinator(inner_steps=2, min_members=2)
        c.join("a")
        timer = threading.Timer(.1, c.join, ("b",))
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertEqual(c.round_view(0, "a", wait=5.)["state"], "open")

    def test_a_long_poll_on_an_open_round_answers_at_once(self):
        c = self.coordinator(inner_steps=2)
        c.join("a")
        started = time.monotonic()
        self.assertEqual(c.round_view(0, "a", wait=5.)["state"], "open")
        self.assertLess(time.monotonic() - started, 1.)

    def test_the_last_round_finishes_the_run_and_writes_the_checkpoint(self):
        c = self.coordinator(inner_steps=2, rounds=2, outer_lr=1.)
        c.join("a")
        self.send(c, "a", 0)
        self.send(c, "a", 1)
        self.assertEqual(c.round_view(2, "a")["state"], "finished")
        saved = studio_model.load_checkpoint(self.dir / "last.npz")
        self.assertEqual(saved["step"], 4)
        self.assertEqual(set(saved["model"]), set(c.theta))
        name = next(iter(c.theta))
        np.testing.assert_array_equal(saved["model"][name], c.theta[name])
        self.assertEqual(support.read_json(self.dir / "run.json")["status"], "completed")
        self.assertEqual(len(studio_model.read_metrics(self.dir / "metrics.jsonl")), 2)

    def test_a_restarted_coordinator_continues_at_the_next_round_with_the_same_momentum(self):
        c = self.coordinator(inner_steps=2, rounds=4, outer_lr=.7, outer_momentum=.9)
        c.join("a")
        for number in range(2):
            self.send(c, "a", number, value=.01 * (number + 1))
        saved = support.read_json(self.dir / "diloco.json")
        self.assertEqual((saved["round"], saved["generation"]), (2, c.generation))
        self.assertEqual(len(saved["history"]), 2)
        theta, momentum, generation = c.theta, c.momentum, c.generation
        again = self.coordinator(inner_steps=2, rounds=4, outer_lr=.7, outer_momentum=.9)  # the process died; a new one reads the folder
        self.assertEqual(again.next_round, 2)
        for name in momentum:
            np.testing.assert_array_equal(again.momentum[name], momentum[name])
            np.testing.assert_array_equal(again.theta[name], theta[name])
        self.assertTrue(any(v.any() for v in again.momentum.values()))
        again.join("a")
        view = again.round_view(2, "a")
        self.assertEqual(view["state"], "open")
        self.assertGreater(view["generation"], generation)  # a contribution from before the restart is stale
        self.send(again, "a", 2)
        self.assertEqual(again.next_round, 3)
        self.assertEqual(len(list(self.dir.glob("diloco_state_*.npz"))), 1)
        with self.assertRaises(ValueError):
            support.build(self.dir, inner_steps=5)  # another run's folder

    def test_the_shards_cover_every_window_once(self):
        c = self.coordinator(shards=5)
        windows = []
        for shard in range(c.shard_count):
            store = studio_diloco.install_shard(self.dir / "cache", shard, c.shard_bytes(shard), c.tokenizer.vocab_size)
            windows += [(x.tolist(), y.tolist()) for x, y in store]
        self.assertEqual(windows, [(x.tolist(), y.tolist()) for x, y in c.train])


class LiveView(Directory):
    """What the Training tab's live view reads: member reports, the round timeline, the event log and the run's phase."""

    def setUp(self):
        super().setUp()
        self.clock = Clock()

    def coordinator(self, **changes):
        return support.build(self.dir, clock=self.clock, **changes)

    def report(self, c, node, number, phase, step=0, loss=None, tps=None, epoch=0):
        view = c.round_view(number, node)
        return c.progress(number, node, view.get("generation", c.generation), epoch, phase, step, loss, tps)

    def test_the_latest_report_per_member_is_kept_with_its_samples(self):
        c = self.coordinator(inner_steps=4, first_round_seconds=100.)
        c.join("a")
        self.assertTrue(self.report(c, "a", 0, "downloading"))
        self.clock.now = 3.
        self.assertTrue(self.report(c, "a", 0, "training", 2, 3.5, 900.))
        self.clock.now = 5.
        self.assertTrue(self.report(c, "a", 0, "training", 3, 3.2, 950.))
        self.clock.now = 6.
        node = c.snapshot()["nodes"]["a"]
        self.assertEqual({k: node["progress"][k] for k in ("phase", "step", "steps", "loss", "tokens_per_second")},
                         {"phase": "training", "step": 3, "steps": 4, "loss": 3.2, "tokens_per_second": 950.})
        self.assertEqual(node["progress"]["reported_ago"], 1.)
        self.assertEqual([(s["step"], s["t"]) for s in node["progress"]["samples"]], [(2, 3.), (3, 5.)])
        self.assertEqual(node["shards"], c.shard_count)

    def test_a_training_or_uploading_report_for_a_closed_round_or_generation_is_stale_but_waiting_is_not(self):
        c = self.coordinator(inner_steps=2, first_round_seconds=100.)
        c.join("a", epoch=2)
        generation = c.round_view(0, "a")["generation"]
        with self.assertRaises(Stale):
            c.progress(0, "a", generation + 1, 2, "training", 1)
        with self.assertRaises(Stale):
            c.progress(0, "a", generation, 1, "training", 1)  # another lease epoch
        with self.assertRaises(Stale):
            c.progress(0, "stranger", generation, 0, "waiting")
        self.assertTrue(c.progress(0, "a", generation, 2, "uploading", 2))
        self.send_as(c, "a", 0, epoch=2)
        with self.assertRaises(Stale):
            c.progress(0, "a", generation, 2, "training", 1)  # round 0 closed
        self.assertTrue(c.progress(0, "a", generation, 2, "waiting"))  # but saying "waiting" is always fine
        with self.assertRaises(ValueError):
            c.progress(1, "a", generation, 2, "sleeping")

    def send_as(self, c, node, number, epoch=0):
        view = c.round_view(number, node)
        c.submit(number, node, view["generation"], epoch, 10, 2, support.constant_delta(c), 1.)

    def test_the_timeline_marks_each_members_download_first_step_upload_and_receipt_on_the_coordinators_clock(self):
        c = self.coordinator(inner_steps=2, rounds=3, min_members=2, first_round_seconds=100., grace_seconds=10.)
        c.join("a")
        c.join("b")
        self.clock.now = 1.
        c.params_bytes(0, "a")
        self.clock.now = 2.
        c.params_bytes(0, "b")
        self.clock.now = 3.
        self.report(c, "a", 0, "training", 0)
        self.clock.now = 5.
        self.report(c, "a", 0, "uploading", 2)
        self.clock.now = 6.
        self.send_as(c, "a", 0)
        line = c.snapshot()["timeline"][-1]
        self.assertEqual((line["round"], line["open"], line["deadline"]), (0, True, 100.))
        self.assertEqual({k: line["lanes"]["a"][k] for k in ("theta_taken", "first_step", "delta_sent", "delta_received", "counted")},
                         {"theta_taken": 1., "first_step": 3., "delta_sent": 5., "delta_received": 6., "counted": True})
        self.assertEqual(line["lanes"]["b"]["theta_taken"], 2.)
        self.assertIsNone(line["lanes"]["b"]["counted"])  # not in yet
        self.clock.now = 111.  # b misses the deadline and the grace period
        c.round_view(0, "b")
        closed = c.snapshot()["timeline"][0]
        self.assertEqual((closed["round"], closed["open"], closed["closed"]), (0, False, 111.))
        self.assertFalse(closed["lanes"]["b"]["counted"])  # counted for nothing this round
        self.assertTrue(closed["lanes"]["a"]["counted"])

    def test_a_finished_round_s_timeline_is_appended_to_metrics_jsonl_and_only_the_last_few_stay_in_memory(self):
        c = self.coordinator(inner_steps=2, rounds=studio_diloco.TIMELINES_KEPT + 3, first_round_seconds=100.)
        c.join("a")
        for number in range(studio_diloco.TIMELINES_KEPT + 3):
            self.clock.now += 1.
            c.params_bytes(number, "a")
            self.send_as(c, "a", number)
        lines = [json.loads(line) for line in (self.dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), studio_diloco.TIMELINES_KEPT + 3)
        self.assertEqual(lines[0]["timeline"]["round"], 0)
        self.assertEqual(lines[0]["timeline"]["lanes"]["a"]["theta_taken"], 1.)
        self.assertEqual(len(c.snapshot()["timeline"]), studio_diloco.TIMELINES_KEPT)
        self.assertEqual(c.snapshot()["timeline"][-1]["round"], studio_diloco.TIMELINES_KEPT + 2)

    def test_the_event_log_records_joins_leaves_drops_late_members_refused_changes_outer_steps_and_checkpoints(self):
        c = self.coordinator(inner_steps=2, rounds=5, min_members=1, missed_limit=2, grace_seconds=1., first_round_seconds=5.)
        c.join("a")
        c.join("b")
        self.send_as(c, "a", 0)  # b joined mid-round 0
        with self.assertRaises(Stale):
            c.submit(0, "a", c.generation, 0, 10, 2, support.constant_delta(c), 1.)  # round 0 is closed
        for number in (1, 2):
            self.send_as(c, "a", number)
            self.clock.now += 100.
            c.round_view(number, "a")
        c.drop("a")
        kinds = [(event["kind"], event["node"]) for event in c.snapshot()["events"]]
        for expected in (("joined", "a"), ("joined", "b"), ("outer_step", ""), ("checkpoint", ""), ("stale", "a"), ("late", "b"), ("dropped", "b"), ("left", "a")):
            self.assertIn(expected, kinds)
        events = c.snapshot()["events"]
        self.assertEqual([e["seq"] for e in events], sorted(e["seq"] for e in events))
        outer = next(e for e in events if e["kind"] == "outer_step")
        self.assertIn("size", outer["text"])
        self.assertGreater(c.history[0]["outer_step_norm"], 0.)

    def test_the_event_log_is_bounded(self):
        c = self.coordinator(inner_steps=2, first_round_seconds=100.)
        for index in range(studio_diloco.EVENTS_KEPT + 20):
            c._event("joined", f"n{index}", 0, "x")
        events = c.snapshot()["events"]
        self.assertEqual(len(events), studio_diloco.EVENTS_KEPT)
        self.assertEqual(events[-1]["seq"], studio_diloco.EVENTS_KEPT + 20)

    def test_the_phase_follows_the_round_from_taking_theta_to_collecting_changes(self):
        c = self.coordinator(inner_steps=2, rounds=2, min_members=2, first_round_seconds=100.)
        self.assertEqual(c.snapshot()["phase"], "waiting")
        c.join("a")
        self.assertEqual(c.snapshot()["phase"], "waiting")  # below min_members
        c.join("b")
        self.assertEqual(c.snapshot()["phase"], "next_round")
        self.report(c, "a", 0, "downloading")
        self.assertEqual(c.snapshot()["phase"], "next_round")
        self.report(c, "a", 0, "training", 1)
        self.assertEqual(c.snapshot()["phase"], "training")
        self.report(c, "b", 0, "uploading", 2)
        view = c.snapshot()
        self.assertEqual(view["phase"], "collecting")
        self.send_as(c, "a", 0)
        view = c.snapshot()
        self.assertEqual((view["contributed"], view["expected"], view["deadline_s"]), (1, 2, 100.))
        self.send_as(c, "b", 0)
        self.assertEqual(c.snapshot()["phase"], "next_round")  # round 1 is open
        self.send_as(c, "a", 1)
        self.send_as(c, "b", 1)
        self.assertEqual(c.snapshot()["phase"], "finished")

    def test_a_snapshot_during_a_closing_round_answers_with_its_phase_instead_of_waiting_for_it(self):
        c = self.coordinator(inner_steps=2, rounds=2, first_round_seconds=100.)
        c.join("a")
        c.snapshot()  # a first answer to fall back on
        entered, release, answers = threading.Event(), threading.Event(), []

        def slow(theta):
            entered.set()
            release.wait(10)
            return 1.
        c.evaluate = slow
        closing = threading.Thread(target=lambda: self.send_as(c, "a", 0))
        closing.start()
        self.assertTrue(entered.wait(10))
        with mock.patch.object(studio_diloco, "PROGRESS_WAIT", .05):
            began = time.monotonic()
            answers.append(c.snapshot())
            answers.append(c.progress(0, "a", c.generation, 0, "waiting"))
            self.assertLess(time.monotonic() - began, 5.)
        release.set()
        closing.join(10)
        self.assertEqual(answers[0]["phase"], "validating")
        self.assertFalse(answers[1])  # nothing taken; the next report says it all again


class Hooked:
    """studio_model.train's `distributed=` hook with no network: the member's data, per-round sampler and schedule, and the weights
    never replaced. What DiLoCo with one member, outer lr 1 and no momentum must equal."""

    def __init__(self, coordinator, inner, rounds):
        self.tokenizer, self.train, self.val, self.sampler = coordinator.tokenizer, coordinator.train, Windows(), random.Random()
        self.seed, self.inner, self.total_steps = coordinator.cfg["seed"], inner, inner * rounds

    def rate_step(self, number):
        return number

    def due(self, step, trained_tokens):
        return step % self.inner == 0

    def sync(self, step, trained_tokens, model):
        self.model = model
        self.sampler.seed(studio_diloco.sampler_seed(self.seed, 0, step // self.inner))


class Equivalence(Directory):
    def test_one_member_with_outer_lr_one_and_no_momentum_trains_exactly_like_studio_model_train(self):
        inner, rounds = 3, 4
        coordinator = support.build(self.dir / "diloco", inner_steps=inner, rounds=rounds)
        studio_diloco.COORDINATORS["equivalence"] = coordinator
        self.addCleanup(studio_diloco.COORDINATORS.pop, "equivalence", None)
        with TestClient(support.app()) as client:
            studio_diloco.run_member(support.Peer(client), "equivalence", "solo", 0, self.dir, self.dir / "member", lambda event: None,
                                     threading.Event(), wait=.2)
        self.assertTrue(coordinator.finished)
        self.assertEqual([len(r["contributed"]) for r in coordinator.history], [1] * rounds)
        hook = Hooked(coordinator, inner, rounds)
        plain = {**support.CONFIG, "steps": inner * rounds, "eval_every": 100000}
        studio_model.train(plain, [], self.dir / "plain", distributed=hook, telemetry_every=0)
        expected = {k: v.astype(np.float32) for k, v in hook.model.state_arrays().items()}
        self.assertEqual(set(expected), set(coordinator.theta))
        for name, value in expected.items():
            np.testing.assert_allclose(coordinator.theta[name], value, atol=1e-5, rtol=1e-5, err_msg=name)
        fresh = studio_model.ScratchTransformer(coordinator.cfg, coordinator.tokenizer.vocab_size).state_arrays()
        self.assertGreater(max(float(np.abs(expected[k] - fresh[k]).max()) for k in expected), 1e-3)  # training moved the weights
        self.assertFalse((self.dir / "data" / "remote_shards" / "equivalence").exists())  # the shard cache is deleted at the end

    def test_the_equivalence_is_not_vacuous_another_outer_lr_ends_elsewhere(self):
        inner, rounds = 3, 4
        coordinator = support.build(self.dir / "diloco", inner_steps=inner, rounds=rounds, outer_lr=.5)
        studio_diloco.COORDINATORS["control"] = coordinator
        self.addCleanup(studio_diloco.COORDINATORS.pop, "control", None)
        with TestClient(support.app()) as client:
            studio_diloco.run_member(support.Peer(client), "control", "solo", 0, self.dir, self.dir / "member", lambda event: None,
                                     threading.Event(), wait=.2)
        hook = Hooked(coordinator, inner, rounds)
        studio_model.train({**support.CONFIG, "steps": inner * rounds, "eval_every": 100000}, [], self.dir / "plain", distributed=hook, telemetry_every=0)
        gap = max(float(np.abs(coordinator.theta[k] - v.astype(np.float32)).max()) for k, v in hook.model.state_arrays().items())
        self.assertGreater(gap, 1e-3)


if __name__ == "__main__":
    unittest.main()

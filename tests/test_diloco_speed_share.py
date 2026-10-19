"""Heterogeneous members (#430): the coordinator deals each member inner steps in proportion to its measured speed, on a fake clock
and fake members whose step takes a fixed time."""
import dataclasses
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import studio_diloco
from tests import diloco_support as support

TOKENS_PER_STEP = 8


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Fleet(unittest.TestCase):
    """Members named by their seconds per step. `play` runs one round the way Member does: the dealt steps, cut at the deadline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()

    def coordinator(self, speeds, **changes):
        folder = Path(tempfile.mkdtemp(dir=self.tmp.name))  # one folder per run: a test may make two
        self.clock.now = 0.
        c = support.build(folder, clock=self.clock, **{"inner_steps": 100, "rounds": 12, "min_members": len(speeds), **changes})
        self.speeds = dict(speeds)
        for node in self.speeds:
            c.join(node)
        return c

    def play(self, c, nodes=None, pull=None):
        """One round for `nodes`. Returns (dealt steps, steps done, seconds the round took, tokens)."""
        number, start = c.next_round, self.clock.now
        views = {node: c.round_view(number, node) for node in (nodes or self.speeds)}
        dealt = {node: view["steps"] for node, view in views.items()}
        done, finish = {}, {}
        for node, view in views.items():
            step_time = self.speeds[node]
            done[node] = min(dealt[node], int(view["deadline_s"] / step_time))
            finish[node] = done[node] * step_time
        for node in sorted(views, key=finish.get):  # each member sends when its steps are done
            self.clock.now = start + finish[node]
            delta = pull(done[node], c) if pull else support.constant_delta(c, .001 * done[node])
            c.submit(number, node, views[node]["generation"], 0, done[node] * TOKENS_PER_STEP, done[node], delta, finish[node])
        return dealt, done, max(finish.values()), sum(done.values()) * TOKENS_PER_STEP


class Dealing(Fleet):
    def test_speeds_1_2_and_4_are_dealt_steps_in_that_proportion_and_finish_together(self):
        c = self.coordinator({"slow": 4., "mid": 2., "fast": 1.})
        dealt, _, _, _ = self.play(c)
        self.assertEqual(set(dealt.values()), {100})  # nobody is measured yet: H each, as before
        dealt, done, seconds, _ = self.play(c)
        self.assertAlmostEqual(dealt["mid"] / dealt["slow"], 2., delta=.1)
        self.assertAlmostEqual(dealt["fast"] / dealt["slow"], 4., delta=.2)
        finishes = [dealt[node] * self.speeds[node] for node in dealt]
        self.assertLess((max(finishes) - min(finishes)) / max(finishes), .1)
        self.assertEqual(done, dealt)  # all finished inside the deadline, none cut
        self.assertEqual(sum(dealt.values()), 300)  # the work three equal-H members would do
        view = c.round_view(c.next_round, "fast")
        self.assertAlmostEqual(view["deadline_s"], max(finishes) * 1.25 + 30., delta=2)  # from the slowest predicted finish, plus slack

    def test_equal_speeds_deal_h_to_everyone_every_round(self):
        c = self.coordinator({"a": 1., "b": 1., "c": 1.})
        for _ in range(4):
            dealt, done, _, _ = self.play(c)
            self.assertEqual(dealt, {"a": 100, "b": 100, "c": 100})
            self.assertEqual(done, dealt)
        self.assertEqual(studio_diloco.round_seconds(c.settings, 1.), c.round_view(c.next_round, "a")["deadline_s"])
        self.assertEqual([row["dealt"] for row in c.history], [{"a": 100, "b": 100, "c": 100}] * 4)

    def test_a_new_member_gets_a_small_first_round_and_then_a_measured_share(self):
        c = self.coordinator({"a": 1., "b": 2.}, rounds=20)
        self.play(c)
        self.play(c)
        self.speeds["new"] = 1.
        c.join("new")
        self.play(c, nodes=["a", "b"])  # the join takes effect at the round boundary
        dealt, *_ = self.play(c)
        self.assertEqual(dealt["new"], 25)  # a quarter of H: a probe
        self.assertGreater(dealt["a"], 100)
        dealt, done, *_ = self.play(c)
        self.assertEqual(dealt["new"], dealt["a"])  # as fast as a, so the same share now
        self.assertEqual(done, dealt)

    def test_shares_are_clamped_and_the_deadline_follows_the_clamped_steps(self):
        c = self.coordinator({"s1": 40., "s2": 40., "s3": 40., "s4": 40., "fast": .1})
        self.play(c)
        dealt, *_ = self.play(c)
        self.assertEqual(dealt["s1"], 10)  # not below a tenth of H
        self.assertEqual(dealt["fast"], 300)  # nor above three times H
        view = c.round_view(c.next_round, "s1")
        self.assertAlmostEqual(view["deadline_s"], 10 * 40. * 1.25 + 30., delta=1)  # the slow members' clamped steps set the deadline

    def test_a_member_that_left_is_dealt_around(self):
        c = self.coordinator({"slow": 4., "mid": 2., "fast": 1.})
        self.play(c)
        self.play(c)
        c.settings = dataclasses.replace(c.settings, min_members=2)  # the run goes on with two
        c.leave("slow", c.generation, 0)
        self.play(c, nodes=["mid", "fast"])  # the round in flight; slow leaves at its end
        dealt, *_ = self.play(c, nodes=["mid", "fast"])
        self.assertEqual(dealt, {"mid": 67, "fast": 133})  # the two left share the work of two equal-H members


class Forgetful(dict):
    def __setitem__(self, key, value):
        pass


class Totals(Fleet):
    def run_rounds(self, equal, rounds=6):
        """Tokens and seconds over `rounds` rounds of members at speeds 1, 2 and 4; equal=True deals H to everyone, as the run did before."""
        c = self.coordinator({"slow": 4., "mid": 2., "fast": 1.}, rounds=rounds + 1)
        if equal:
            c.step_times = Forgetful()  # no member is ever timed: H for everyone, and a deadline from the fastest step time
        tokens = 0
        for _ in range(rounds):
            tokens += self.play(c)[3]
        return tokens, self.clock.now

    def test_total_tokens_per_round_rise_against_equal_h(self):
        equal_tokens, equal_seconds = self.run_rounds(True)
        tokens, seconds = self.run_rounds(False)
        self.assertGreater(tokens, equal_tokens)  # slow members are no longer cut at the deadline mid-schedule
        self.assertGreater(tokens / seconds, equal_tokens / equal_seconds)  # and more tokens per second of the round

    def test_a_two_member_run_with_unequal_speeds_still_converges(self):
        c = self.coordinator({"slow": 3., "fast": 1.}, inner_steps=20, rounds=30, outer_lr=1.)
        target = {name: np.zeros_like(value) for name, value in c.theta.items()}
        rng = np.random.default_rng(0)
        for name, value in c.theta.items():
            c.theta[name] = value + rng.normal(size=value.shape).astype(np.float32)

        def distance():
            return float(np.sqrt(sum(((c.theta[name] - target[name]) ** 2).sum() for name in c.theta)))

        def pull(steps, coordinator):  # steps of gradient descent on |theta - target|^2: more steps, a longer way
            return {name: (1 - .95 ** steps) * (coordinator.theta[name] - target[name]) for name in coordinator.theta}
        before = distance()
        for _ in range(12):
            self.play(c, pull=pull)
        self.assertLess(distance(), before * .01)
        self.assertTrue(all(np.isfinite(value).all() for value in c.theta.values()))


class Timing(unittest.TestCase):
    def test_the_reported_seconds_are_the_timed_steps_without_warm_up_or_pauses(self):
        member = studio_diloco.Member.__new__(studio_diloco.Member)
        member.warm, member.began, member.clock = False, 0., lambda: 1000.  # a wall clock that ran through a long pause
        member.busy, member.timed = 0., 0
        member.stepped(30.)  # the first step of the attempt: warm-up, not timed
        for _ in range(4):
            member.stepped(.5)
        self.assertEqual(member._work_seconds(5), 2.5)
        member.busy, member.timed = 0., 0  # a new round, before any step was timed: the wall time
        self.assertEqual(member._work_seconds(5), 1000.)


class Schedule(unittest.TestCase):
    def test_the_rate_follows_the_position_in_the_round_not_the_step_count(self):
        member = studio_diloco.Member.__new__(studio_diloco.Member)
        member.info, member.round, member.first_step = {"inner_steps": 100}, 3, 40
        member.entered = studio_diloco.Entered(3, {}, 0, 0, 1., [], steps=200)
        self.assertEqual(member.rate_step(40), 300)
        self.assertEqual(member.rate_step(140), 350.)  # half way through its 200 steps: half way through the round's 100 positions
        self.assertEqual(member.rate_step(240), 400.)  # the end of the round, where every member ends
        member.entered = studio_diloco.Entered(3, {}, 0, 0, 1., [], steps=100)
        self.assertEqual(member.rate_step(90), 350)  # H steps: as it always was
        self.assertIsInstance(member.rate_step(90), int)
        member.entered = studio_diloco.Entered(3, {}, 0, 0, 1., [])
        self.assertEqual(member.rate_step(90), 350)


class Rule(unittest.TestCase):
    """deal_steps on its own, including the numbers a real run could hand it."""

    def deal(self, h, times, members=None):
        settings = support.settings(inner_steps=h)
        return studio_diloco.deal_steps(settings, members or list(times), times)

    def test_the_total_stays_h_per_measured_member_whatever_the_rounding(self):
        for h in (1, 2, 3, 7, 10, 100):
            for times in ({"a": 1., "b": 2., "c": 4.}, {"a": 1., "b": 3.}, {"a": 1., "b": 1.5, "c": 7., "d": 2.}):
                dealt = self.deal(h, times)
                if min(dealt.values()) > max(1, math.ceil(h * studio_diloco.MIN_SHARE)) and max(dealt.values()) < int(h * studio_diloco.MAX_SHARE):
                    self.assertEqual(sum(dealt.values()), h * len(times), (h, times, dealt))
                self.assertTrue(all(value >= 1 for value in dealt.values()), (h, times, dealt))

    def test_small_h_still_gives_every_member_a_step(self):
        for h in (1, 2, 3):
            dealt = self.deal(h, {"a": 1., "b": 50.})
            self.assertTrue(all(1 <= value <= max(h, int(h * studio_diloco.MAX_SHARE)) for value in dealt.values()), (h, dealt))
        self.assertEqual(self.deal(1, {"a": 1., "b": 1.}), {"a": 1, "b": 1})

    def test_nothing_measured_one_member_and_a_probe_only_group_deal_h(self):
        self.assertEqual(self.deal(8, {"a": None, "b": None}), {"a": 8, "b": 8})
        self.assertEqual(self.deal(8, {}, ["a", "b"]), {"a": 8, "b": 8})
        self.assertEqual(self.deal(8, {"a": 3.}), {"a": 8})
        self.assertEqual(self.deal(8, {"a": 3.}, ["a", "new"]), {"a": 8, "new": 2})

    def test_non_finite_and_non_positive_times_count_as_not_measured(self):
        for bad in (math.inf, math.nan, 0., -1.):
            self.assertEqual(self.deal(8, {"a": bad, "b": bad}), {"a": 8, "b": 8}, bad)
            self.assertEqual(self.deal(8, {"a": bad, "b": 1.}, ["a", "b"]), {"a": 2, "b": 8}, bad)  # a is the probe

    def test_speeds_within_five_percent_keep_h_for_everyone(self):
        self.assertEqual(self.deal(100, {"a": 1., "b": 1.03, "c": .99}), {"a": 100, "b": 100, "c": 100})
        self.assertNotEqual(self.deal(100, {"a": 1., "b": 1.2})["a"], 100)

    def test_a_contribution_with_non_finite_seconds_is_refused(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        c = support.build(Path(tmp.name), inner_steps=3)
        c.join("a")
        view = c.round_view(0, "a")
        for bad in (math.inf, math.nan, -1.):
            with self.assertRaises(ValueError):
                c.submit(0, "a", view["generation"], 0, 10, 3, support.constant_delta(c), bad)
        self.assertEqual(c.current.contributions, {})


class RealMembers(unittest.TestCase):
    """Real Member threads over the real routes with studio_model.train, and an uneven deal: node0 gets 2, 4, 6, ... steps a round and
    node1 gets 2. A fast member that is dealt more than H on average must not run out of steps, and each member runs what it was dealt."""

    ROUNDS, H = 4, 4

    def run_members(self, overlap):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        coordinator = support.build(folder / "c", "uneven", inner_steps=self.H, rounds=self.ROUNDS, min_members=2, outer_lr=.7, outer_momentum=0.)
        studio_diloco.COORDINATORS["uneven"] = coordinator
        self.addCleanup(studio_diloco.COORDINATORS.pop, "uneven", None)
        opened, errors = [], []

        def deal(settings, members, step_times):
            opened.append(len(opened))
            return {node: 2 * (opened[-1] + 1) if node == "node0" else 2 for node in members}

        with mock.patch.object(studio_diloco, "deal_steps", deal), TestClient(support.app()) as client:
            def member(index):
                try:
                    studio_diloco.run_member(support.Peer(client), "uneven", f"node{index}", 0, folder / f"m{index}", folder / f"out{index}",
                                             lambda event: None, threading.Event(), wait=.5, overlap=overlap)
                except Exception as exc:  # noqa: BLE001 - reported below
                    errors.append(exc)
            threads = [threading.Thread(target=member, args=(i,)) for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(240)
        self.assertEqual(errors, [])
        self.assertTrue(coordinator.finished)
        return coordinator

    def test_each_member_runs_its_dealt_steps_and_the_run_finishes(self):
        c = self.run_members(False)
        self.assertEqual([row["dealt"] for row in c.history], [{"node0": 2 * (r + 1), "node1": 2} for r in range(self.ROUNDS)])
        self.assertEqual([row["steps"] for row in c.history], [row["dealt"] for row in c.history])
        self.assertEqual([row["tokens"] > 0 for row in c.history], [True] * self.ROUNDS)

    def test_an_overlapped_member_runs_the_deal_it_has_the_round_after_it_was_dealt(self):
        c = self.run_members(True)
        self.assertEqual([row["steps"]["node1"] for row in c.history], [2] * self.ROUNDS)
        # the exchange that brings round n's deal ends after round n's steps: round n runs on the deal of round n - 1 (round 0 and 1 on the first)
        self.assertEqual([row["steps"]["node0"] for row in c.history], [2, 2, 4, 6])


if __name__ == "__main__":
    unittest.main()

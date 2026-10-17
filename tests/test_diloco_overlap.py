"""studio_diloco.Member with overlap=True: the delta uploads and the next weights download while the next round's steps run, and the
outer result is applied a round late (w' = theta + (w - w_start)). Speed is measured on a fake clock; correctness on real members."""
import tempfile
import threading
import types
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import studio_diloco
import studio_model
from tests import diloco_support as support
from tests import test_diloco

CALL_SECONDS = .05  # what every call to the coordinator costs in the speed test
STEP_SECONDS = .1
INNER, ROUNDS = 5, 6


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Lane:
    """studio_diloco.Member's `spawn` on the fake clock: runs the exchange at once on a branch of time, puts the clock back where
    it was, and lets result() move it on to when the exchange ended, if the caller got there first."""

    def __init__(self, clock, fn):
        self.clock, self.value, self.error = clock, None, None
        begun = clock.now
        try:
            self.value = fn()
        except Exception as exc:  # noqa: BLE001 - raised again by result(), as Background does
            self.error = exc
        self.end, clock.now = clock.now, begun

    def result(self):
        self.clock.now = max(self.clock.now, self.end)
        if self.error is not None:
            raise self.error
        return self.value


class Delayed:
    """A client that makes every call take `seconds` of the fake clock."""

    def __init__(self, peer, clock, seconds):
        self.peer, self.clock, self.seconds = peer, clock, seconds

    def __getattr__(self, name):
        call = getattr(self.peer, name)

        def delayed(*args, **kwargs):
            self.clock.now += self.seconds
            return call(*args, **kwargs)
        return delayed


class Weights:
    """The part of a model the member touches: load_weights and state_arrays."""

    def load_weights(self, weights):
        self.weights = {name: np.array(value, dtype=np.float32) for name, value in weights.items()}

    def state_arrays(self):
        return {name: value.copy() for name, value in self.weights.items()}


class Directory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.addCleanup(studio_diloco.COORDINATORS.clear)


class Speed(Directory):
    def run_member(self, overlap, seconds):
        """A member stepping a stub model (STEP_SECONDS a step) against a real coordinator through a client with `seconds` per call.
        Returns (steps per second between the first step and the last round's last step, the coordinator, the model)."""
        clock = Clock()
        coordinator = support.build(self.dir / f"c{overlap}{seconds}", clock=clock, inner_steps=INNER, rounds=ROUNDS, outer_lr=1.)
        studio_diloco.COORDINATORS["speed"] = coordinator
        with TestClient(support.app()) as client:
            member = studio_diloco.Member(Delayed(support.Peer(client), clock, seconds), "speed", "solo", 0, self.dir / f"m{overlap}{seconds}",
                                          threading.Event(), clock=clock, wait=.1, progress=False, overlap=overlap,
                                          spawn=lambda fn: Lane(clock, fn))
            member.shard_dir.mkdir(parents=True)
            member.join()
            member._enter(member.info["round"])
            model = Weights()
            member.sync(0, 0, model)
            begun, last = clock.now, None
            for step in range(1, INNER * ROUNDS + 1):
                clock.now += STEP_SECONDS
                for value in model.weights.values():
                    value -= .001
                if member.due(step, step * 10):
                    if step == INNER * ROUNDS:
                        last = clock.now
                    member.sync(step, step * 10, model)
            member.halt.set()
        self.assertTrue(coordinator.finished)
        self.assertEqual([len(r["contributed"]) for r in coordinator.history], [1] * ROUNDS)
        return INNER * ROUNDS / (last - begun), coordinator, model

    def test_a_slow_link_does_not_slow_the_steps_down_when_communication_is_overlapped(self):
        undelayed, _, _ = self.run_member(True, 0.)
        delayed, _, _ = self.run_member(True, CALL_SECONDS)
        print(f"steps per second, overlapped: undelayed {undelayed:.3f}, {CALL_SECONDS} s per call {delayed:.3f}")
        self.assertAlmostEqual(undelayed, 1 / STEP_SECONDS, places=6)
        self.assertGreaterEqual(delayed, undelayed * .9)

    def test_the_same_link_slows_the_steps_down_without_overlap_so_the_test_is_not_vacuous(self):
        undelayed, _, _ = self.run_member(False, 0.)
        delayed, _, _ = self.run_member(False, CALL_SECONDS)
        print(f"steps per second, not overlapped: undelayed {undelayed:.3f}, {CALL_SECONDS} s per call {delayed:.3f}")
        self.assertLess(delayed, undelayed * .9)

    def test_a_link_slower_than_the_steps_is_waited_for_at_the_end_of_the_round_and_no_longer(self):
        seconds = INNER * STEP_SECONDS  # three calls per exchange: 3x a round of steps
        rate, _, _ = self.run_member(True, seconds)
        self.assertLess(rate, .5 / STEP_SECONDS)
        self.assertGreater(rate, INNER / (3 * seconds) * .9)  # the exchange is the bottleneck: a round of steps per exchange, three calls

    def test_a_single_member_ends_with_the_weights_it_trained_whatever_the_link(self):
        _, coordinator, model = self.run_member(True, CALL_SECONDS)
        expected = {name: value - np.float32(.001 * INNER * ROUNDS) for name, value in self.initial().items()}
        for name, value in model.weights.items():
            np.testing.assert_allclose(value, expected[name], atol=1e-5, err_msg=name)
        for name, value in coordinator.theta.items():  # lr 1, no momentum: the last round's weights, all but the rounds still in flight
            np.testing.assert_allclose(value, expected[name], atol=1e-5, err_msg=name)

    def initial(self):
        coordinator = support.build(self.dir / "initial", inner_steps=INNER, rounds=ROUNDS)
        return coordinator.theta


class Boundary(Directory):
    """The end of a round's steps, with the exchange already in flight handed over by hand."""

    def member(self, entered, pending, anchor, round_=2):
        member = studio_diloco.Member(None, "r", "n", 0, self.dir / "s", threading.Event(), progress=False, overlap=True, spawn=lambda fn: Lane(Clock(), fn))
        member.info = {"seed": 1, "rounds": 10, "inner_steps": 3, "delta_dtype": "float32"}
        member.started, member.round, member.rank, member.anchor, member.deadline_s = True, round_, 0, anchor, 100.
        member.entered, member.pending, member.first_step, member.first_tokens, member.began = entered, pending, 0, 0, 0.
        member.emit = lambda event: None
        return member

    def test_the_new_theta_takes_the_work_done_since_the_round_started_and_the_delta_goes_out_on_a_new_exchange(self):
        theta, anchor, sent = {"w": np.array([10., 20.], np.float32)}, {"w": np.array([1., 2.], np.float32)}, []
        member = self.member(None, types.SimpleNamespace(result=lambda: studio_diloco.Entered(2, theta, 5, 0, 100., [])), anchor)
        member.spawn = lambda fn: sent.append(fn) or types.SimpleNamespace(result=lambda: None)
        model = Weights()
        model.load_weights({"w": np.array([1.5, 1.], np.float32)})  # trained: +.5 and -1 from the round's start
        member.sync(3, 30, model)
        np.testing.assert_allclose(model.weights["w"], [10.5, 19.])  # theta + (w - w_start)
        np.testing.assert_allclose(member.anchor["w"], [10.5, 19.])
        self.assertEqual((member.round, member.first_step, member.report_round, len(sent)), (3, 3, 3, 1))
        self.assertEqual(member.generation, 5)

    def test_a_round_that_closed_without_the_member_drops_its_work_and_starts_over_from_the_new_theta(self):
        theta = {"w": np.array([7., 8.], np.float32)}
        member = self.member(None, types.SimpleNamespace(result=lambda: studio_diloco.Entered(4, theta, 6, 0, 100., [], skipped=True)),
                             {"w": np.zeros(2, np.float32)})
        member.spawn = lambda fn: self.fail("nothing is sent for a round that closed")
        model = Weights()
        model.load_weights({"w": np.array([-3., -3.], np.float32)})
        member.sync(3, 30, model)
        np.testing.assert_array_equal(model.weights["w"], [7., 8.])
        self.assertEqual((member.round, member.first_step, member.pending), (4, 3, None))
        np.testing.assert_array_equal(member.anchor["w"], [7., 8.])

    def test_a_run_that_finished_or_a_fenced_member_stops_instead_of_training_on(self):
        member = self.member(None, types.SimpleNamespace(result=lambda: None), {"w": np.zeros(2, np.float32)})
        member.finished = True
        member.sync(3, 30, Weights())
        self.assertTrue(member.stop.is_set())


class Training(Directory):
    def train(self, members, run, **changes):
        """`members` overlapped member threads against one coordinator over the routes; the finished coordinator."""
        coordinator = support.build(self.dir / run, run, **changes)
        studio_diloco.COORDINATORS[run] = coordinator
        errors = []
        with TestClient(support.app()) as client:
            def member(index):
                try:
                    studio_diloco.run_member(support.Peer(client), run, f"node{index}", 0, self.dir / f"{run}-m{index}", self.dir / f"{run}-out{index}",
                                             lambda event: None, threading.Event(), wait=.5, overlap=True)
                except Exception as exc:  # noqa: BLE001 - reported below
                    errors.append(exc)
            threads = [threading.Thread(target=member, args=(i,)) for i in range(members)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(240)
        self.assertEqual(errors, [])
        self.assertTrue(coordinator.finished)
        return coordinator

    def test_one_overlapped_member_with_outer_lr_one_and_no_momentum_trains_exactly_like_studio_model_train(self):
        inner, rounds = 3, 4
        coordinator = self.train(1, "solo", inner_steps=inner, rounds=rounds, outer_lr=1., outer_momentum=0.)
        self.assertEqual([len(r["contributed"]) for r in coordinator.history], [1] * rounds)
        hook = test_diloco.Hooked(coordinator, inner, rounds)
        studio_model.train({**support.CONFIG, "steps": inner * rounds, "eval_every": 100000}, [], self.dir / "plain", distributed=hook, telemetry_every=0)
        for name, value in hook.model.state_arrays().items():
            np.testing.assert_allclose(coordinator.theta[name], value.astype(np.float32), atol=1e-5, rtol=1e-5, err_msg=name)

    def test_two_overlapped_members_beat_one_after_the_same_steps_per_member(self):
        shape = dict(inner_steps=4, rounds=6, outer_lr=.7, outer_momentum=.9, min_members=1, delta_dtype="float16")
        one = self.train(1, "one", **shape)
        two = self.train(2, "two", **{**shape, "min_members": 2})
        self.assertEqual([len(r["contributed"]) for r in two.history], [2] * 6)
        losses = (one.history[-1]["val_loss"], two.history[-1]["val_loss"])
        print(f"overlapped, final validation loss after 6 rounds x 4 steps per member: 1 member {losses[0]:.4f}, 2 members {losses[1]:.4f}")
        self.assertLess(losses[1], losses[0])
        self.assertLess(two.history[-1]["val_loss"], two.history[0]["val_loss"])
        self.assertFalse((self.dir / "two-m0" / "data" / "remote_shards" / "two").exists())

    def test_the_shard_cache_and_the_exchange_are_gone_when_a_member_is_stopped_mid_run(self):
        coordinator = support.build(self.dir / "stop", "stop", inner_steps=3, rounds=50, outer_lr=1.)
        studio_diloco.COORDINATORS["stop"] = coordinator
        stop, seen = threading.Event(), []

        def emit(event):
            if event.get("type") == "metric" and event.get("step", 0) >= 8:
                stop.set()
            seen.append(event)
        with TestClient(support.app()) as client:
            member = studio_diloco.Member(support.Peer(client), "stop", "solo", 0, self.dir / "stop-shards", stop, emit=emit, wait=.2, overlap=True)
            rounds = member.work(self.dir / "stop-out", emit)
        self.assertIsNotNone(rounds)
        self.assertFalse(coordinator.finished)
        self.assertGreaterEqual(len(coordinator.history), 1)
        self.assertIsNone(member.pending)
        self.assertFalse((self.dir / "stop-shards").exists())


if __name__ == "__main__":
    unittest.main()

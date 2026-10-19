"""Pause and resume of a DiLoCo run (#490), and starting a run from another run's weights: tiny runs on an injected clock."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import studio_diloco
from tests import diloco_support as support


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Pausing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "run"
        self.clock = Clock()

    def coordinator(self, **changes):
        return support.build(self.dir, clock=self.clock, **{"inner_steps": 2, "rounds": 8, **changes})

    def send(self, c, node="a", seconds=1.):
        number = c.next_round
        view = c.round_view(number, node)
        self.assertEqual(view["state"], "open", view)
        self.clock.now += seconds
        c.submit(number, node, view["generation"], 0, 10, 2, support.constant_delta(c), seconds)

    def hold(self, c, node="a"):
        """Pause while a round is in flight, then let that round close: the run is paused between rounds."""
        number = c.next_round
        view = c.round_view(number, node)
        c.pause()
        self.clock.now += 1.
        c.submit(number, node, view["generation"], 0, 10, 2, support.constant_delta(c), 1.)

    def test_a_paused_run_opens_no_round_and_says_so(self):
        c = self.coordinator()
        c.join("a")
        self.hold(c)
        view = c.round_view(1, "a")
        self.assertEqual((view["state"], view["paused"], view["member"]), ("pending", True, True))
        snap = c.snapshot()
        self.assertEqual((snap["state"], snap["phase"], snap["paused"], snap["round"]), ("waiting", "paused", True, 1))
        self.assertEqual([e["kind"] for e in snap["events"] if e["kind"] in ("paused", "resumed")], ["paused"])

    def test_a_round_in_flight_finishes_after_a_pause(self):
        c = self.coordinator()
        c.join("a")
        number = c.next_round
        view = c.round_view(number, "a")
        c.pause()
        self.assertEqual((c.snapshot()["phase"], c.snapshot()["paused"]), ("next_round", True))
        self.clock.now += 1.
        c.submit(number, "a", view["generation"], 0, 10, 2, support.constant_delta(c), 1.)
        self.assertEqual(c.next_round, 1)
        self.assertEqual((c.current, c.round_view(1, "a")["state"]), (None, "pending"))
        self.assertEqual(c.snapshot()["phase"], "paused")

    def test_resume_opens_the_next_round(self):
        c = self.coordinator()
        c.join("a")
        self.hold(c)
        c.resume()
        view = c.round_view(1, "a")
        self.assertEqual((view["state"], view["paused"]), ("open", False))
        self.assertEqual([e["kind"] for e in c.snapshot()["events"] if e["kind"] in ("paused", "resumed")], ["paused", "resumed"])

    def test_a_waiting_member_is_woken_by_a_resume(self):
        c = self.coordinator()
        c.join("a")
        self.hold(c)
        woken = []
        import threading
        thread = threading.Thread(target=lambda: woken.append(c.round_view(1, "a", wait=5.)))
        thread.start()
        c.resume()
        thread.join(5)
        self.assertEqual(woken[0]["state"], "open")

    def test_pausing_twice_or_resuming_a_run_that_is_going_changes_nothing(self):
        c = self.coordinator()
        c.join("a")
        c.resume()
        c.pause()
        c.pause()
        self.assertEqual([e["kind"] for e in c.snapshot()["events"] if e["kind"] in ("paused", "resumed")], ["paused"])

    def test_a_finished_run_cannot_be_paused_or_resumed(self):
        c = self.coordinator()
        c.join("a")
        c.stop()
        with self.assertRaises(studio_diloco.Stale):
            c.pause()
        with self.assertRaises(studio_diloco.Stale):
            c.resume()

    def test_paused_time_does_not_count_toward_max_seconds(self):
        c = self.coordinator(max_seconds=30., rounds=40)
        c.join("a")
        self.hold(c)
        self.clock.now += 1000.
        c.snapshot()
        c.resume()
        self.assertFalse(c.finished)
        self.send(c)
        self.assertFalse(c.finished)
        self.assertLess(c.rules.active_seconds, 30.)
        for _ in range(6):
            if c.finished:
                break
            self.send(c, seconds=5.)
        self.assertEqual(c.rules.stop_rule, "max_seconds")

    def test_the_pause_survives_a_restart(self):
        c = self.coordinator()
        c.join("a")
        self.send(c)
        self.hold(c)
        saved = support.read_json(self.dir / "diloco.json")
        self.assertTrue(saved["paused"])
        self.assertEqual(saved["round"], 2)
        again = self.coordinator()
        self.assertTrue(again.paused)
        self.assertEqual(again.next_round, 2)
        again.join("a")
        self.assertEqual(again.round_view(2, "a")["state"], "pending")
        again.resume()
        self.assertEqual(again.round_view(2, "a")["state"], "open")
        self.assertFalse(support.read_json(self.dir / "diloco.json")["paused"])

    def test_a_pause_before_the_first_round_is_kept_too(self):
        c = self.coordinator()
        c.pause()
        c.join("a")
        self.assertEqual(c.round_view(0, "a")["state"], "pending")
        again = self.coordinator()
        self.assertTrue(again.paused)
        self.assertEqual(again.next_round, 0)

    def test_the_next_round_close_writes_the_flag_too(self):
        c = self.coordinator()
        c.join("a")
        number = c.next_round
        view = c.round_view(number, "a")
        c.pause()
        c.submit(number, "a", view["generation"], 0, 10, 2, support.constant_delta(c), 1.)
        self.assertTrue(support.read_json(self.dir / "diloco.json")["paused"])


class StartingFrom(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_new_run_starts_from_the_weights_of_a_checkpoint(self):
        old = support.build(self.root / "old")
        old.join("a")
        view = old.round_view(0, "a")
        old.submit(0, "a", view["generation"], 0, 10, 3, support.constant_delta(old), 1.)
        self.assertTrue((self.root / "old" / "last.npz").is_file())
        fresh = support.build(self.root / "fresh")
        started = studio_diloco.Coordinator.build("run2", self.root / "again", support.CONFIG, support.EXAMPLES, support.settings(),
                                                  start_from=self.root / "old" / "last.npz")
        for name, value in old.theta.items():
            np.testing.assert_array_equal(started.theta[name], value)
            self.assertFalse(np.array_equal(fresh.theta[name], value))
            self.assertFalse(started.momentum[name].any())
        self.assertEqual(started.next_round, 0)

    def test_a_checkpoint_of_another_model_is_refused_with_a_sentence(self):
        old = support.build(self.root / "old")
        old.join("a")
        view = old.round_view(0, "a")
        old.submit(0, "a", view["generation"], 0, 10, 3, support.constant_delta(old), 1.)
        wider = {**support.CONFIG, "width": 32}
        with self.assertRaisesRegex(ValueError, "different model"):
            studio_diloco.Coordinator.build("run2", self.root / "again", wider, support.EXAMPLES, support.settings(),
                                            start_from=self.root / "old" / "last.npz")

    def test_a_missing_checkpoint_is_refused_with_a_sentence(self):
        with self.assertRaisesRegex(ValueError, "could not be read"):
            studio_diloco.Coordinator.build("run2", self.root / "again", support.CONFIG, support.EXAMPLES, support.settings(),
                                            start_from=self.root / "nowhere.npz")


if __name__ == "__main__":
    unittest.main()

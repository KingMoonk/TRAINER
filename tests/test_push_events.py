"""The events pages refresh on instead of polling: replica value changes, the coalescer, coordinator watchers, self-test."""
import unittest

import studio_diloco
import studio_state
from studio_remote import Coalescer, _fold_facts


class ReplicaWatch(unittest.TestCase):
    def setUp(self):
        self.replica = studio_state.Replica("a", 1, clock=lambda: 0.)
        self.seen = []
        self.stop = self.replica.watch(self.seen.append)

    def test_a_new_value_is_told_and_the_same_value_is_not(self):
        self.replica.put("run", "r1", {"round": 1})
        self.replica.put("run", "r1", {"round": 1})
        self.replica.put("run", "r1", {"round": 2})
        self.assertEqual(self.seen, [{("a", "run", "r1")}, {("a", "run", "r1")}])

    def test_a_merge_tells_only_the_values_that_changed(self):
        other = studio_state.Replica("b", 1, clock=lambda: 0.)
        other.put("progress", "r1", {"step": 3})
        self.replica.merge(other.delta({}))
        self.replica.merge(other.delta({}))  # the same rows again: nothing new
        other.put("progress", "r1", {"step": 3})  # a newer version of the same value
        self.replica.merge(other.delta({}))
        self.assertEqual(self.seen, [{("b", "progress", "r1")}])

    def test_stop(self):
        self.stop()
        self.replica.put("run", "r1", 1)
        self.assertEqual(self.seen, [])


class CoalescerTest(unittest.TestCase):
    def setUp(self):
        self.now, self.sent, self.pending = 0., [], []
        self.coalesce = Coalescer(lambda key, payload: self.sent.append((key, payload)), 1., fold=_fold_facts,
                                  clock=lambda: self.now, later=lambda delay, call: self.pending.append((delay, call)))

    def facts(self, kind):
        return {"kinds": [kind], "runs": [], "origins": ["a"]}

    def test_first_at_once_then_one_folded_trailing_call(self):
        self.coalesce("facts", self.facts("run"))
        self.now = .2
        self.coalesce("facts", self.facts("queue"))
        self.coalesce("facts", self.facts("node"))
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(self.pending), 1)
        self.assertAlmostEqual(self.pending[0][0], .8)
        self.now = 1.
        self.pending.pop()[1]()
        self.assertEqual(self.sent[1], ("facts", {"kinds": ["node", "queue"], "runs": [], "origins": ["a"]}))

    def test_keys_are_apart_and_a_quiet_key_sends_at_once(self):
        self.coalesce("r1", self.facts("run"))
        self.coalesce("r2", self.facts("run"))
        self.now = 5.
        self.coalesce("r1", self.facts("run"))
        self.assertEqual([key for key, _ in self.sent], ["r1", "r2", "r1"])
        self.assertEqual(self.pending, [])


class CoordinatorWatch(unittest.TestCase):
    def test_an_event_tells_the_watchers_until_stopped(self):
        seen = []
        stop = studio_diloco.watch(seen.append)
        try:
            coordinator = studio_diloco.Coordinator.__new__(studio_diloco.Coordinator)
            coordinator.event_seq, coordinator.events, coordinator.started = 0, [], 0.
            coordinator.clock = lambda: 1.
            coordinator._event("joined", "n1", 1, "n1 joined.")
        finally:
            stop()
        coordinator._event("left", "n1", 1, "n1 left.")
        self.assertEqual(seen, [coordinator])


if __name__ == "__main__":
    unittest.main()

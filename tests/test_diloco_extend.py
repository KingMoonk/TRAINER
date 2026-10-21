"""Adding rounds (and changing H) to a DiLoCo run while it goes: the plan, the coordinator's extend(), a member that is already in the
run training the new rounds with the learning rate carrying on, and a successor coordinator that keeps the extended settings."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_diloco
from tests import diloco_support as support

PLAN = [{"round": 0, "q": 0., "h": 3, "rounds": 4}]


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Plan(unittest.TestCase):
    def test_a_plan_without_changes_is_the_plain_schedule(self):
        self.assertEqual((studio_diloco.plan_total(PLAN), studio_diloco.plan_frame(PLAN, 2), studio_diloco.plan_rounds(PLAN)), (12, 12, 4))
        self.assertEqual(studio_diloco.plan_position(PLAN, 2, 1), 7)
        self.assertEqual(studio_diloco.plan_position(PLAN, 3, 99), 12)  # never past the end

    def test_the_rate_carries_on_across_an_extension_without_a_jump(self):
        plan = [*PLAN, {"round": 3, "q": .75, "h": 3, "rounds": 8}]
        self.assertEqual(studio_diloco.plan_total(plan), 24)
        before = studio_diloco.plan_position(plan, 2, 3) / studio_diloco.plan_frame(plan, 2)  # the end of round 2, on the old frame
        after = studio_diloco.plan_position(plan, 3) / studio_diloco.plan_frame(plan, 3)  # the start of round 3, on the new one
        self.assertAlmostEqual(before, .75)
        self.assertAlmostEqual(after, .75)
        self.assertAlmostEqual(studio_diloco.plan_position(plan, 7, 3) / 24, 1.)

    def test_a_new_h_counts_its_own_steps_from_its_round_on(self):
        plan = [*PLAN, {"round": 3, "q": .75, "h": 5, "rounds": 6}]
        self.assertEqual([studio_diloco.plan_h(plan, n) for n in range(6)], [3, 3, 3, 5, 5, 5])
        self.assertEqual((studio_diloco.plan_before(plan, 3), studio_diloco.plan_before(plan, 5), studio_diloco.plan_total(plan)), (9, 19, 24))


class Extending(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "run"
        self.clock = Clock()

    def coordinator(self, **changes):
        return support.build(self.dir, clock=self.clock, **{"inner_steps": 3, "rounds": 4, **changes})

    def play(self, c, rounds, node="a"):
        c.join(node)
        for _ in range(rounds):
            if c.finished:
                break
            number = c.next_round
            view = c.round_view(number, node)
            self.assertEqual(view["state"], "open", view)
            self.clock.now += 1.
            c.submit(number, node, view["generation"], 0, 10, view["steps"], support.constant_delta(c, .01), 1.)

    def test_a_run_extended_midway_does_not_finish_at_the_old_count(self):
        c = self.coordinator()
        self.play(c, 1)
        answer = c.extend(rounds=8)
        self.assertEqual(answer, {"rounds": 8, "inner_steps": 3, "from_round": 3})
        self.play(c, 3)  # rounds 1, 2, 3: the old run would have ended with the one after
        self.assertEqual((c.finished, c.next_round), (False, 4))
        self.play(c, 4)
        self.assertEqual((c.finished, len(c.history)), (True, 8))
        self.assertEqual(c.info()["total_steps"], 24)
        self.assertIn("Extended to 8 rounds at round 1.", [e["text"] for e in c.snapshot()["events"] if e["kind"] == "extended"])
        self.assertEqual(c.snapshot()["rounds"], 8)

    def test_a_new_h_applies_from_a_later_round_only(self):
        c = self.coordinator(rounds=6)
        self.play(c, 1)
        answer = c.extend(inner_steps=5)
        self.assertEqual(answer, {"rounds": 6, "inner_steps": 5, "from_round": 3})
        steps = []
        for _ in range(5):
            self.play(c, 1)
            steps.append(c.history[-1]["step"])
        self.assertEqual(steps, [6, 9, 14, 19, 24])  # rounds 1 and 2 ran 3 steps each, rounds 3 to 5 run 5
        self.assertTrue(c.finished)

    def test_the_view_of_an_open_round_carries_the_plan_and_the_rounds(self):
        c = self.coordinator()
        c.join("a")
        c.extend(rounds=6)
        view = c.round_view(0, "a")
        self.assertEqual(view["rounds"], 6)
        self.assertEqual([entry["round"] for entry in view["plan"]], [0, 2])

    def test_a_second_extension_before_the_first_applies_replaces_it(self):
        c = self.coordinator()
        c.extend(rounds=6)
        c.extend(rounds=9, inner_steps=2)
        self.assertEqual([(e["round"], e["h"], e["rounds"]) for e in c.plan], [(0, 3, 4), (2, 2, 9)])

    def test_extensions_chain(self):
        c = self.coordinator()
        self.play(c, 1)
        c.extend(rounds=6)
        self.play(c, 3)
        c.extend(rounds=10)
        self.play(c, 20)
        self.assertEqual((c.finished, len(c.history)), (True, 10))

    def test_numbers_that_shorten_or_change_nothing_are_refused_with_a_sentence(self):
        c = self.coordinator()
        with self.assertRaisesRegex(ValueError, "can only add rounds"):
            c.extend(rounds=3)
        with self.assertRaisesRegex(ValueError, "Nothing changes"):
            c.extend(rounds=4)
        with self.assertRaisesRegex(ValueError, "Nothing changes"):
            c.extend(rounds=4, inner_steps=3)
        with self.assertRaisesRegex(ValueError, "at most 100000 rounds"):
            c.extend(rounds=100_001)
        self.assertEqual(c.plan, PLAN)

    def test_a_run_that_is_over_cannot_be_extended(self):
        c = self.coordinator()
        self.play(c, 4)
        self.assertTrue(c.finished)
        with self.assertRaises(studio_diloco.Stale):
            c.extend(rounds=9)

    def test_the_extension_is_saved_so_a_restart_keeps_it(self):
        c = self.coordinator()
        self.play(c, 1)
        c.extend(rounds=8)
        saved = support.read_json(self.dir / "diloco.json")
        self.assertEqual((saved["settings"]["rounds"], saved["plan"]), (8, c.plan))
        self.assertEqual(support.read_json(self.dir / "run.json")["rounds"], 8)

    def test_a_coordinator_made_from_a_replica_keeps_the_extended_settings(self):
        c = self.coordinator()
        self.play(c, 2)
        c.extend(rounds=8, inner_steps=4)
        self.play(c, 1)  # a closed round writes the extension to diloco.json as the replicas fetch it
        replica, shards = Path(self.tmp.name) / "replica", Path(self.tmp.name) / "shards"
        replica.mkdir()
        (replica / "theta.npz").write_bytes(studio_diloco.encode_arrays(c.theta))
        (replica / "momentum.npz").write_bytes(studio_diloco.encode_arrays(c.momentum))
        (replica / "diloco.json").write_bytes((self.dir / "diloco.json").read_bytes())
        (replica / "info.json").write_text(json.dumps(c.info()), encoding="utf-8")
        (replica / "validation.npz").write_bytes(c.validation_bytes())
        for shard in range(c.shard_count):
            studio_diloco.install_shard(shards, shard, c.shard_bytes(shard), c.tokenizer.vocab_size)
        taken = studio_diloco.Coordinator.from_replica("run1", Path(self.tmp.name) / "taken", replica, shards, 2, clock=Clock())
        self.assertEqual((taken.settings.rounds, taken.settings.inner_steps), (8, 4))
        self.assertEqual(taken.plan, c.plan)
        self.assertEqual(taken.info()["total_steps"], c.info()["total_steps"])
        taken.join("a")
        self.assertEqual(taken.round_view(taken.next_round, "a")["rounds"], 8)
        self.assertEqual(studio_diloco.plan_h(taken.plan, 7), 4)


class Member(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_a_member_already_in_the_run_trains_the_new_rounds_and_the_rate_carries_on(self):
        coordinator = support.build(self.dir / "diloco", inner_steps=3, rounds=4)
        studio_diloco.COORDINATORS["extend"] = coordinator
        self.addCleanup(studio_diloco.COORDINATORS.pop, "extend", None)
        original = studio_diloco.Member.rate_step
        seen, extended = [], []

        def watch(member, number):
            if member.round == 1 and not extended:  # the member is in round 1, as the run is: round 3 is the first the extension reaches
                extended.append(coordinator.extend(rounds=8))
            position = original(member, number)
            seen.append((member.round, position / member.total_steps, member.total_steps))
            return position

        with TestClient(support.app()) as client, mock.patch.object(studio_diloco.Member, "rate_step", autospec=True, side_effect=watch):
            studio_diloco.run_member(support.Peer(client), "extend", "solo", 0, self.dir, self.dir / "member", lambda event: None,
                                     threading.Event(), wait=.2)
        self.assertEqual(extended, [{"rounds": 8, "inner_steps": 3, "from_round": 3}])
        self.assertTrue(coordinator.finished)
        self.assertEqual([len(r["contributed"]) for r in coordinator.history], [1] * 8)
        self.assertEqual({round_ for round_, _, _ in seen}, set(range(8)))
        self.assertEqual({total for round_, _, total in seen if round_ < 3}, {12})
        self.assertEqual({total for round_, _, total in seen if round_ >= 3}, {24})
        shares = [share for _, share, _ in seen]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(shares, shares[1:])), shares)  # the rate never steps back
        self.assertLess(max(b - a for a, b in zip(shares, shares[1:])), .15, shares)  # nor does it jump on at the extension
        self.assertGreater(shares[-1], .9)


if __name__ == "__main__":
    unittest.main()

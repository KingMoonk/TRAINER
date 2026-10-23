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
import studio_distributed as runs
from studio_ia.fake_distributed import FakeCoordinator
from studio_modules.schedule.cosine import Cosine, CosineConfig
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

    def rate_at(self, c, number, warmup):
        return Cosine(CosineConfig(warmup=warmup)).rate(studio_diloco.plan_position(c.plan, number), studio_diloco.plan_frame(c.plan, number), 1.)

    def test_the_learning_rate_is_continuous_when_the_extension_lands_inside_or_past_the_warmup(self):
        for warmup in (8, 2, 0):  # round 2 starts at step 6: inside the first warmup, past the others
            with self.subTest(warmup=warmup):
                c = support.build(Path(self.tmp.name) / f"warmup{warmup}", clock=self.clock, inner_steps=3, rounds=4)
                c.cfg = {**c.cfg, "warmup": warmup}
                before = self.rate_at(c, 2, warmup)
                self.assertEqual(c.extend(rounds=8)["from_round"], 2)
                self.assertAlmostEqual(self.rate_at(c, 2, warmup), before)

    def test_a_longer_run_keeps_the_end_of_the_warmup_where_it_is(self):
        self.assertEqual(studio_diloco.plan_carry(5, 12, 24, 8), 5)  # inside the warmup: the same step
        self.assertEqual(studio_diloco.plan_carry(8, 12, 24, 8), 8)  # its end
        self.assertEqual(studio_diloco.plan_carry(10, 12, 24, 8), 8 + 2 * 16 / 4)  # past it: the same share of what is left
        self.assertEqual(studio_diloco.plan_carry(6, 12, 24, 0), 12)  # no warmup: the same share

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

    def test_an_extension_before_the_first_round_closes_survives_a_restart(self):
        c = self.coordinator()
        c.extend(rounds=8, inner_steps=4)
        self.assertFalse((self.dir / "diloco.json").is_file())
        again = self.coordinator()  # the app restarted: the folder holds no round yet, and the run is asked for with its first numbers
        self.assertEqual((again.settings.rounds, again.settings.inner_steps), (8, 4))
        self.assertEqual(again.plan, c.plan)
        self.assertEqual((again.info()["rounds"], again.info()["total_steps"]), (8, studio_diloco.plan_total(c.plan)))
        self.assertEqual(support.read_json(self.dir / "run.json")["rounds"], 8)
        self.play(again, 8)
        self.assertEqual((again.finished, len(again.history)), (True, 8))

    def test_an_extension_file_of_another_run_in_the_folder_is_not_applied(self):
        self.coordinator().extend(rounds=8)
        other = self.coordinator(rounds=5)
        self.assertEqual((other.settings.rounds, other.plan[0]["rounds"]), (5, 5))

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


class ReplicaCarriesThePlan(unittest.TestCase):
    def test_a_replica_one_round_behind_hands_its_successor_the_extended_plan(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            c = support.build(root / "run", clock=Clock(), inner_steps=3, rounds=4)
            c.join("a")
            view = c.round_view(0, "a")
            c.submit(0, "a", view["generation"], 0, 10, view["steps"], support.constant_delta(c, .01), 1.)
            replica, shards = root / "replica", root / "shards"
            replica.mkdir()
            member = studio_diloco.Member(None, "run1", "n", 0, root / "member", threading.Event(), replica_dir=replica)
            member.holding = (c.next_round, 0)  # it replicated the round that is open now, before the extension
            for name, data in (("theta.npz", studio_diloco.encode_arrays(c.theta)), ("momentum.npz", studio_diloco.encode_arrays(c.momentum)),
                               ("diloco.json", (root / "run" / "diloco.json").read_bytes()), ("info.json", json.dumps(c.info()).encode()),
                               ("validation.npz", c.validation_bytes())):
                (replica / name).write_bytes(data)
            for shard in range(c.shard_count):
                studio_diloco.install_shard(shards, shard, c.shard_bytes(shard), c.tokenizer.vocab_size)
            c.extend(rounds=8, inner_steps=4)
            member._replica_plan(c.round_view(c.next_round, "a"))  # the next view the member gets says so
            taken = studio_diloco.Coordinator.from_replica("run1", root / "taken", replica, shards, 2, clock=Clock())
            self.assertEqual((taken.settings.rounds, taken.settings.inner_steps), (8, 4))
            self.assertEqual(taken.plan, c.plan)
            self.assertEqual(taken.info()["total_steps"], c.info()["total_steps"])

    def test_a_view_without_a_change_leaves_the_replica_alone(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            replica = Path(tmp) / "replica"
            replica.mkdir()
            member = studio_diloco.Member(None, "run1", "n", 0, Path(tmp) / "member", threading.Event(), replica_dir=replica)
            record = {"settings": {"rounds": 4, "inner_steps": 3}, "plan": PLAN}
            (replica / "diloco.json").write_text(json.dumps(record), encoding="utf-8")
            member._replica_plan({"plan": [*PLAN, {"round": 3, "q": .75, "h": 5, "rounds": 8}]})  # holding no round: nothing is kept
            self.assertEqual(json.loads((replica / "diloco.json").read_text(encoding="utf-8")), record)


class ProgressCarriesThePlan(unittest.TestCase):
    def test_a_member_training_a_round_hands_its_replica_an_extension_made_meanwhile(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            c = support.build(root / "run", clock=Clock(), inner_steps=3, rounds=4)
            c.join("a")
            view = c.round_view(0, "a")
            c.submit(0, "a", view["generation"], 0, 10, view["steps"], support.constant_delta(c, .01), 1.)
            view = c.round_view(1, "a")  # the member has this round's view, which says nothing of the extension to come
            replica = root / "replica"
            replica.mkdir()
            studio_diloco.COORDINATORS["progress-plan"] = c
            self.addCleanup(studio_diloco.COORDINATORS.pop, "progress-plan", None)
            with TestClient(support.app()) as client:
                member = studio_diloco.Member(support.Peer(client), "progress-plan", "a", 0, root / "member", threading.Event(), replica_dir=replica)
                member.term, member.report_round, member.info = c.term, 1, c.info()
                member.generation, member.holding = view["generation"], (c.next_round, 0)
                for name, data in (("diloco.json", (root / "run" / "diloco.json").read_bytes()), ("info.json", json.dumps(c.info()).encode())):
                    (replica / name).write_bytes(data)
                c.extend(rounds=8, inner_steps=4)  # while the member trains round 1
                self.assertEqual(json.loads((replica / "diloco.json").read_text(encoding="utf-8"))["plan"], PLAN)
                self.assertTrue(member.report())  # its next progress report
            record = json.loads((replica / "diloco.json").read_text(encoding="utf-8"))
            self.assertEqual((record["plan"], record["settings"]["rounds"], record["settings"]["inner_steps"]), (c.plan, 8, 4))


class FakeExtending(unittest.TestCase):
    def test_the_fake_coordinator_applies_a_new_h_two_rounds_ahead_as_the_real_one_does(self):
        clock = Clock()
        run = runs.Run("fake1", "label", 1, 0., {"rounds": 5, "inner_steps": 100}, "mac", [runs.Machine("mac", "local", device="mps", device_name="Apple M2")])
        fake = FakeCoordinator(run, {"rounds": 5, "inner_steps": 100, "delta_dtype": "float16"}, clock=clock, unit=1.)
        fake.start()
        self.assertEqual(fake.extend(inner_steps=50), {"rounds": 5, "inner_steps": 50, "from_round": 2})
        clock.now = 1000.
        self.assertEqual([row["step"] for row in fake.snapshot()["history"]], [100, 200, 250, 300, 350])  # rounds 0 and 1 kept their 100 steps
        self.assertTrue(fake.finished)

    def test_the_fake_coordinator_refuses_what_the_real_one_refuses(self):
        clock = Clock()
        run = runs.Run("fake1", "label", 1, 0., {"rounds": 3, "inner_steps": 100}, "mac", [runs.Machine("mac", "local", device="mps", device_name="Apple M2")])
        fake = FakeCoordinator(run, {"rounds": 3, "inner_steps": 100, "delta_dtype": "float16"}, clock=clock, unit=1.)
        fake.start()
        clock.now = 13.  # two rounds are closed, so the third is open and no round is left that another H could reach
        with self.assertRaisesRegex(ValueError, "no round is left"):
            fake.extend(inner_steps=50)
        self.assertEqual(fake.extend(rounds=5, inner_steps=50)["from_round"], 3)


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
            seen.append((member.round, position, member.total_steps))
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
        schedule = Cosine(CosineConfig(warmup=support.CONFIG["warmup"]))
        rates = [schedule.rate(position, total, 1.) for _, position, total in seen]
        after = rates[support.CONFIG["warmup"] + 1:]  # past the warmup
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(after, after[1:])), rates)  # the rate never steps back up
        self.assertLess(max(a - b for a, b in zip(rates, rates[1:])), .15, rates)  # nor does it jump down at the extension
        self.assertAlmostEqual(rates[-1], .1)  # and it ends at the floor


if __name__ == "__main__":
    unittest.main()

"""The stop rules, plateau pivots and best checkpoint inside a DiLoCo coordinator (#487): tiny runs on an injected clock whose validation
losses are scripted, so each rule is reached in a few rounds."""
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


class Rounds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "run"
        self.clock = Clock()

    def coordinator(self, losses=None, **changes):
        c = support.build(self.dir, clock=self.clock, **{"inner_steps": 2, "rounds": 12, **changes})
        self.script(c, losses)
        return c

    @staticmethod
    def script(c, losses):
        """Make the validation loss of round n `losses[n]`; it also records theta as each round ended (what a round's best.npz must hold)."""
        c.seen = {}
        if losses is None:
            return
        queue = iter(losses)

        def evaluate(theta):
            c.seen[c.next_round] = {name: value.copy() for name, value in theta.items()}
            return next(queue)
        c.evaluate = evaluate

    def play(self, c, rounds, seconds=1., node="a", value=.01):
        """Run `rounds` rounds with one member (or until the run ends)."""
        c.join(node)
        for _ in range(rounds):
            if c.finished:
                break
            number = c.next_round
            view = c.round_view(number, node)
            self.assertEqual(view["state"], "open", view)
            self.clock.now += seconds
            c.submit(number, node, view["generation"], 0, 10, 2, support.constant_delta(c, value), seconds)

    def record(self):
        return support.read_json(self.dir / "run.json")

    def events(self, c, kind):
        return [e for e in c.snapshot()["events"] if e["kind"] == kind]

    # -- stop rules
    def test_no_rule_set_runs_to_rounds(self):
        c = self.coordinator([5., 6., 7., 8., 9.], rounds=5)
        self.play(c, 9)
        self.assertEqual((c.next_round, c.finished, self.record()["status"]), (5, True, "completed"))
        self.assertNotIn("stop_rule", self.record())
        self.assertEqual(self.events(c, "finished")[-1]["text"], "The run finished after 5 rounds.")

    def test_patience_ends_the_run_with_its_reason(self):
        c = self.coordinator([5., 4., 4.5, 4.6, 4.7, 1.], patience_rounds=3)
        self.play(c, 9)
        self.assertEqual((c.next_round, c.finished), (5, True))
        record = self.record()
        self.assertEqual((record["status"], record["stop_rule"], record["best_round"], record["best_val_loss"]), ("completed", "patience", 1, 4.))
        self.assertIn("3 rounds in a row without a better validation loss", record["stop_reason"])
        self.assertEqual(self.events(c, "finished")[-1]["text"], record["stop_reason"])
        self.assertEqual(c.round_view(5, "a")["state"], "finished")

    def test_target_val_loss_ends_the_run_with_its_reason(self):
        c = self.coordinator([5., 4., 3., 2.], target_val_loss=3.5)
        self.play(c, 9)
        self.assertEqual(c.next_round, 3)
        self.assertEqual((self.record()["stop_rule"], self.record()["best_val_loss"]), ("target_val_loss", 3.))
        self.assertIn("target 3.5", self.record()["stop_reason"])

    def test_max_seconds_ends_the_run_after_that_much_active_time(self):
        c = self.coordinator([5., 4., 3., 2., 1.], max_seconds=25.)
        self.play(c, 9, seconds=10.)
        self.assertEqual((c.next_round, c.finished), (3, True))  # 10 s a round: the third round brings it past 25 s
        self.assertEqual(self.record()["stop_rule"], "max_seconds")
        self.assertGreaterEqual(c.rules.active_seconds, 25.)

    def test_time_below_min_members_does_not_count_against_max_seconds(self):
        c = self.coordinator([5., 4., 3.], max_seconds=25., min_members=2)
        c.join("a")
        self.clock.now += 1000.  # alone, the run waits: that is not active time
        c.round_view(0, "a")
        self.assertEqual(c.rules.active_seconds, 0.)
        self.assertIsNone(c.current)
        c.join("b")
        self.assertIsNotNone(c.current)
        self.clock.now += 10.
        for node in ("a", "b"):
            view = c.round_view(0, node)
            c.submit(0, node, view["generation"], 0, 10, 2, support.constant_delta(c), 1.)
        self.assertEqual((c.next_round, c.finished), (1, False))
        self.assertAlmostEqual(c.rules.active_seconds, 10.)

    def test_a_stop_rule_is_not_a_stop_by_the_owner(self):
        c = self.coordinator([5., 6.], patience_rounds=1)
        self.play(c, 9)
        self.assertEqual((c.finished, c.stopped), (True, False))
        self.assertEqual(c.final_bytes()[:2], b"PK")  # a rule-ended run still hands over its weights

    # -- plateau, pivots and the best weights
    def test_a_rising_loss_goes_back_to_the_best_theta_and_applies_the_pivots_in_order_then_ends(self):
        losses = [5., 4., 4.5, 4.6, 4.4, 4.3, 4.2, 4.1, 3.9]
        c = self.coordinator(losses, worse_limit=2, pivots=("halve_lr", "next_dataset"), rounds=20)
        c.add_source(c.train)  # a second source, so next_dataset has somewhere to go
        self.play(c, 3)
        best_theta = c.seen[1]  # round 1 (loss 4.) is the best so far
        self.assertEqual(c.rules.lr_scale, 1.)
        self.play(c, 1)  # round 3 is the second worse round in a row: back to round 1 and halve the learning rate
        self.assertEqual((c.next_round, c.rules.lr_scale, c.rules.source), (4, .5, 0))
        for name, value in c.theta.items():
            np.testing.assert_array_equal(value, best_theta[name])
        self.assertEqual(c.history[3]["pivot"], "halve_lr")
        pivot = self.events(c, "pivot")[0]
        self.assertIn("got worse 2 rounds in a row", pivot["text"])
        self.assertIn("best 4 at round 1", pivot["text"])
        self.assertIn("halved the learning rate", pivot["text"])
        self.assertEqual(c.round_view(4, "a")["lr_scale"], .5)  # the next round is opened with the halved rate
        self.play(c, 9)
        self.assertEqual([e["text"].split("and ")[-1] for e in self.events(c, "pivot")], ["halved the learning rate.", "switched to the next data source."])
        self.assertEqual((c.finished, self.record()["stop_rule"], c.rules.source, c.rules.lr_scale), (True, "pivots", 1, .5))
        self.assertEqual(c.next_round, 8)
        self.assertIn("no change left to try", self.record()["stop_reason"])

    def test_a_plateau_pivots_too(self):
        c = self.coordinator([5., 4., 3.995, 3.99, 3.99, 3.99, 3.99, 3.99], plateau_checks=3, plateau_min_improvement=.05, pivots=("halve_lr",), rounds=8)
        self.play(c, 9)
        self.assertEqual(self.events(c, "pivot")[0]["text"].count("stopped improving"), 1)
        self.assertEqual(c.rules.lr_scale, .5)

    def test_best_npz_is_rewritten_only_on_improvement_and_the_pivot_restores_it(self):
        c = self.coordinator([5., 4., 4.5, 3., 3.5], rounds=5)
        best = self.dir / "best.npz"
        stamps = []
        for expected in ([0], [1], [1], [3], [3]):
            self.play(c, 1)
            with np.load(best) as archive:
                stamps.append(int(archive["weight::" + next(iter(c.theta))].tobytes().__hash__() % 10 ** 9))
            self.assertEqual(c.rules.best_round, expected[0])
        self.assertEqual(len(set(stamps)), 3)  # three different files over five rounds: rounds 0, 1 and 3 improved
        with np.load(best) as archive:
            for name, value in c.seen[3].items():
                np.testing.assert_array_equal(archive["weight::" + name], value)  # the weights as round 3 ended, not round 4's
        self.assertNotEqual(self.last_hash(), self.best_hash())

    def last_hash(self):
        with np.load(self.dir / "last.npz") as archive:
            return hash(archive["weight::" + next(k[8:] for k in archive.files if k.startswith("weight::"))].tobytes())

    def best_hash(self):
        with np.load(self.dir / "best.npz") as archive:
            return hash(archive["weight::" + next(k[8:] for k in archive.files if k.startswith("weight::"))].tobytes())

    def test_the_final_weights_a_successor_serves_are_the_best_ones(self):
        c = self.coordinator([5., 3., 4.], rounds=3)
        self.play(c, 9)
        served = c.final_bytes()
        self.assertEqual(served, (self.dir / "best.npz").read_bytes())
        self.assertNotEqual(served, (self.dir / "last.npz").read_bytes())

    def test_keep_checkpoints_keeps_the_latest_rounds_only(self):
        c = self.coordinator([5., 4., 3., 2., 1.], rounds=5, keep_checkpoints=2)
        self.play(c, 3)
        self.assertEqual(sorted(p.name for p in self.dir.glob("round_*.npz")), ["round_2.npz", "round_3.npz"])
        self.play(c, 9)
        self.assertEqual(sorted(p.name for p in self.dir.glob("round_*.npz")), ["round_4.npz", "round_5.npz"])
        plain = Path(self.tmp.name) / "plain"
        c2 = support.build(plain, clock=self.clock, inner_steps=2, rounds=2)
        self.play(c2, 2)
        self.assertEqual(list(plain.glob("round_*.npz")), [])

    # -- persistence
    def test_the_rule_state_survives_a_coordinator_restart(self):
        losses = [5., 4., 4.5, 4.6, 4.4]
        c = self.coordinator(losses, worse_limit=2, pivots=("halve_lr",), patience_rounds=6, rounds=20)
        self.play(c, 5, seconds=7.)
        before = c.rules.to_json()
        self.assertEqual((before["lr_scale"], before["pivot_index"], before["best_round"], before["worse_streak"]), (.5, 1, 1, 1))
        self.assertAlmostEqual(before["active_seconds"], 28., delta=8)
        again = self.coordinator([4.4, 4.45, 4.5, 4.55, 4.6, 4.65, 4.7], worse_limit=2, pivots=("halve_lr",), patience_rounds=6, rounds=20)
        self.assertEqual(again.next_round, 5)
        self.assertEqual(again.rules.to_json(), before)
        again.join("a")
        self.assertEqual(again.round_view(5, "a")["lr_scale"], .5)
        self.play(again, 9)  # the worse streak (1) carries over: one more worse round is the limit, and no pivot is left
        self.assertEqual((again.finished, again.next_round, self.record()["stop_rule"]), (True, 6, "pivots"))

    def test_a_restart_after_the_run_ended_by_a_rule_stays_ended(self):
        c = self.coordinator([5., 6.], patience_rounds=1)
        self.play(c, 9)
        self.assertTrue(c.finished)
        again = self.coordinator()
        self.assertTrue(again.finished)
        self.assertEqual(again.rules.stop_rule, "patience")
        self.assertEqual(support.read_json(self.dir / "diloco.json")["rules"]["stop_rule"], "patience")

    def test_a_pivot_after_a_restart_still_restores_the_best_weights(self):
        c = self.coordinator([5., 4., 4.5], worse_limit=2, pivots=("halve_lr",), rounds=20)
        self.play(c, 3)
        best = c.seen[1]
        again = self.coordinator([4.6], worse_limit=2, pivots=("halve_lr",), rounds=20)
        self.play(again, 1)
        self.assertEqual(again.rules.lr_scale, .5)
        for name, value in again.theta.items():
            np.testing.assert_array_equal(value, best[name])

    # -- the next data source
    def test_next_dataset_deals_the_later_sources_shards(self):
        c = self.coordinator([5., 4., 4.5, 4.6, 4.4], worse_limit=2, pivots=("next_dataset",), rounds=20, shards=4)
        c.add_source(c.train)
        self.assertEqual(c.spans, [(0, 4), (4, 4)])
        self.play(c, 1)
        self.assertTrue(all(s < 4 for shards in c.current.deal.values() for s in shards) or c.current is None)
        self.play(c, 3)
        self.assertEqual(c.rules.source, 1)
        view = c.round_view(c.next_round, "a")
        self.assertTrue(view["shards"])
        self.assertTrue(all(4 <= s < 8 for s in view["shards"]), view["shards"])
        for shard in view["shards"]:
            self.assertGreater(len(c.shard_bytes(shard)), 0)
        self.assertEqual(support.read_json(self.dir / "diloco.json")["layout"], [4, 4])

    def test_a_pivot_with_nothing_left_to_move_to_ends_the_run(self):
        c = self.coordinator([5., 6., 7.], worse_limit=2, pivots=("next_dataset",))
        self.play(c, 9)
        self.assertEqual(self.record()["stop_rule"], "pivots")

    # -- a missing loss, the last round, halve_lr's outer step and resuming
    def test_without_validation_windows_a_patience_rule_never_fires(self):
        c = self.coordinator([None] * 5, patience_rounds=1, rounds=5)
        self.play(c, 9)
        self.assertEqual((c.next_round, c.finished, self.record()["status"]), (5, True, "completed"))
        self.assertNotIn("stop_rule", self.record())
        self.assertFalse((self.dir / "best.npz").exists())
        self.assertEqual(c.final_bytes(), (self.dir / "last.npz").read_bytes())  # no best to hand over: the last round's weights

    def test_a_missing_or_nan_loss_before_the_first_finite_one_does_not_pivot(self):
        c = self.coordinator([None, float("nan"), 5., 4.], worse_limit=1, pivots=("halve_lr",), rounds=4)
        self.play(c, 9)
        self.assertEqual((c.next_round, c.rules.lr_scale, self.events(c, "pivot")), (4, 1., []))
        self.assertEqual(c.rules.best_round, 3)

    def test_the_round_rounds_ends_anyway_skips_pivots_and_stop_rules(self):
        c = self.coordinator([5., 4., 6.], worse_limit=1, pivots=("halve_lr",), patience_rounds=2, rounds=3)
        self.play(c, 9)
        self.assertEqual((c.next_round, c.rules.lr_scale, self.events(c, "pivot")), (3, 1., []))
        self.assertNotIn("stop_rule", self.record())
        self.assertNotIn("pivot", c.history[-1])
        for name, value in c.theta.items():
            np.testing.assert_array_equal(value, c.seen[2][name])  # not restored to round 1's
        self.assertEqual(self.record()["best_round"], 1)  # and the Version still gets round 1's weights from best.npz

    def test_halve_lr_also_halves_the_outer_step(self):
        c = self.coordinator([5., 6., 7., 8.], worse_limit=1, pivots=("halve_lr",), rounds=4)
        self.play(c, 4)
        self.assertEqual(c.rules.lr_scale, .5)
        sizes = [row["outer_step_norm"] for row in c.history]
        self.assertAlmostEqual(sizes[2] / sizes[0], .5, places=5)  # round 1 pivoted back; round 2 stepped with half the outer lr
        self.assertEqual([row["lr"] for row in c.history], [1., 1., .5, .5][:len(c.history)])

    def test_resuming_after_a_next_dataset_pivot_needs_the_later_sources(self):
        build = studio_diloco.Coordinator.build
        c = build("run1", self.dir, support.CONFIG, support.EXAMPLES, support.settings(inner_steps=2, rounds=12, worse_limit=1, pivots=("next_dataset",)),
                  clock=self.clock, later=[support.EXAMPLES])
        self.script(c, [5., 6.])
        self.play(c, 2)
        self.assertEqual(c.rules.source, 1)
        with self.assertRaisesRegex(ValueError, "data source 2"):
            build("run1", self.dir, support.CONFIG, support.EXAMPLES, support.settings(inner_steps=2, rounds=12, worse_limit=1, pivots=("next_dataset",)),
                  clock=self.clock)
        again = build("run1", self.dir, support.CONFIG, support.EXAMPLES, support.settings(inner_steps=2, rounds=12, worse_limit=1, pivots=("next_dataset",)),
                      clock=self.clock, later=[support.EXAMPLES])
        self.assertEqual((again.rules.source, len(again.spans)), (1, 2))

    # -- the wire
    def test_the_member_scales_its_learning_rate_by_the_round_it_was_opened_with(self):
        member = studio_diloco.Member.__new__(studio_diloco.Member)
        member.entered = None
        self.assertEqual(member.rate_scale, 1.)
        member.entered = studio_diloco.Entered(3, {}, 1, 0, 10., [0], False, .25)
        self.assertEqual(member.rate_scale, .25)

    def test_the_snapshot_shows_the_rules(self):
        c = self.coordinator([5., 4.], rounds=4)
        self.play(c, 2)
        view = c.snapshot()["rules"]
        self.assertEqual((view["best_round"], view["best_loss"], view["lr_scale"]), (1, 4., 1.))


if __name__ == "__main__":
    unittest.main()

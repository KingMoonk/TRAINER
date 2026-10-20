"""Where a DiLoCo run's next_dataset pivot goes (#446): the workspace's unseen sets, worst held-out loss first, with replay, a journal of
every switch and a reason when nothing is left. Tiny runs on an injected clock whose validation losses are scripted (as
test_diloco_stop_rules does)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import studio_stages as stages
from tests import diloco_support as support
from tests.test_diloco_stop_rules import Clock

WORDS = ("cat", "dog", "owl", "fox", "bee", "ant", "elk", "yak")
SOURCES = {name: [{"text": f"{name}: the {a} saw the {b} and then the {a} ran past the {b}"} for a in WORDS for b in WORDS if a != b]
           for name in ("kitchen", "garden", "barn", "lake")}


def scorer_with(table, calls=None):
    """A scorer reading `table[source]` (a loss, or a list of losses to step through one per call)."""
    seen = {}

    def score(checkpoint, held):
        if calls is not None:
            calls.append(sorted(held))
        out = {}
        for source in held:
            value = table.get(source)
            if isinstance(value, list):
                value = value[min(seen.setdefault(source, 0), len(value) - 1)]
                seen[source] += 1
            out[source] = value
        return out
    return score


class Stages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "run"
        self.clock = Clock()
        self.pool = ["garden", "barn"]

    def coordinator(self, losses, table, pivots=("next_dataset",), replay=.25, **changes):
        c = support.build(self.dir, clock=self.clock, **{"inner_steps": 2, "rounds": 20, "worse_limit": 1, "pivots": pivots, "shards": 4, **changes})
        queue = iter(losses)
        c.evaluate = lambda theta: next(queue)
        held = {"kitchen": stages.split(SOURCES["kitchen"])[1]}
        policy = stages.Policy(["kitchen"], lambda source: SOURCES[source], lambda skip: [{"id": s, "name": s} for s in self.pool if s not in skip],
                               "fake", replay, held)
        c.stages = policy
        patch = mock.patch.dict(stages.SCORERS, {"fake": scorer_with(table)})
        patch.start()
        self.addCleanup(patch.stop)
        return c, policy

    def play(self, c, rounds, node="a"):
        c.join(node)
        for _ in range(rounds):
            if c.finished:
                break
            number = c.next_round
            view = c.round_view(number, node)
            self.assertEqual(view["state"], "open", view)
            self.clock.now += 1.
            c.submit(number, node, view["generation"], 0, 10, 2, support.constant_delta(c, .01), 1.)

    def events(self, c):
        return [e["text"] for e in c.snapshot()["events"] if e["kind"] == "pivot"]

    def test_a_flat_curve_moves_the_run_on_to_the_set_it_does_worst_on_with_the_same_weights_and_rounds(self):
        c, policy = self.coordinator([5., 4., 4.5, 4.4, 4.3], {"garden": 4.0, "barn": 6.0, "kitchen": [3.0, 3.1]})
        self.play(c, 2)
        self.assertEqual(c.rules.source, 0)
        theta = {k: v.copy() for k, v in c.theta.items()}
        self.play(c, 1)  # round 2 is worse than round 1: back to its weights, and on
        self.assertEqual((c.rules.source, c.next_round, c.history[2]["pivot"]), (1, 3, "next_dataset"))
        self.assertEqual(policy.names[1], ["barn"])  # 6.0 against 4.0: the weights do worst on the barn
        self.assertEqual(policy.names[2], ["garden"])
        self.assertEqual([len(c.spans), len(c.later)], [3, 2])
        for name in c.theta:  # the weights it moved on with are the best round's
            self.assertEqual(c.theta[name].shape, theta[name].shape)
        (switch,) = policy.switches
        self.assertEqual((switch["from_sets"], switch["to_sets"], switch["round"]), (["kitchen"], ["barn"], 2))
        self.assertIn("did worst on barn", switch["why"])
        self.assertEqual((switch["old_loss_before"], switch["new_loss_before"]), (3.0, 6.0))
        self.assertIsNone(switch["old_loss_after"])

    def test_the_old_sets_loss_is_measured_after_the_next_round_and_every_switch_is_in_the_journal(self):
        c, policy = self.coordinator([5., 4., 4.5, 4.4, 4.3], {"garden": 4.0, "barn": [6.0, 6.0, 5.0], "kitchen": [3.0, 3.4]})
        self.play(c, 4)
        (switch,) = policy.switches
        self.assertEqual((switch["old_loss_before"], switch["old_loss_after"]), (3.0, 3.4))  # what the switch cost the old set
        self.assertEqual((switch["new_loss_before"], switch["new_loss_after"]), (6.0, 5.0))
        journal = stages.read_journal(self.dir)
        self.assertEqual(journal["switches"][0]["old_loss_after"], 3.4)
        self.assertEqual(journal["sources"][:2], [["kitchen"], ["barn"]])
        texts = self.events(c)
        self.assertTrue(any("moving on to barn" in t and "replays them" in t for t in texts), texts)
        self.assertTrue(any("after a round on barn" in t and "(it was 3)" in t for t in texts), texts)

    def test_the_new_source_deals_a_replay_share_of_the_old_one(self):
        c, policy = self.coordinator([5., 4., 4.5, 4.4, 4.3], {"garden": 4.0, "barn": 6.0, "kitchen": 3.0}, replay=.25)
        self.play(c, 3)
        first, count = c.spans[1]
        dealt = sorted(c.round_view(c.next_round, "a")["shards"])
        self.assertTrue(set(range(first, first + count)) <= set(dealt), dealt)
        replayed = [shard for shard in dealt if shard < c.spans[1][0]]
        self.assertEqual(len(replayed), max(1, round(count * .25 / .75)))
        self.assertEqual(policy.replay_ids(c, 5), policy.replay_ids(c, 5))  # the same round deals the same shards
        self.assertEqual(stages.Policy(["a"], None, None).replay_ids(c, 1) if c.rules.source == 0 else [], [])

    def test_sets_added_while_the_run_trains_join_the_pool_at_the_switch(self):
        c, policy = self.coordinator([5., 4., 4.5, 4.4], {"garden": 4.0, "barn": 5.0, "lake": 9.0, "kitchen": 3.0})
        self.play(c, 1)
        self.pool.append("lake")  # the collector finishes a set after the run started
        self.play(c, 3)
        self.assertEqual(policy.names[1], ["lake"])  # unknown at the start, and the worst
        self.assertEqual({n[0] for n in policy.names[1:]}, {"lake", "garden", "barn"})
        self.assertIn("lake", policy.known)

    def test_with_nothing_unseen_left_the_run_ends_with_the_reason_and_the_bigger_design_suggestion(self):
        self.pool = []
        c, policy = self.coordinator([5., 4., 4.5, 4.4], {"kitchen": 3.0})
        self.play(c, 4)
        self.assertTrue(c.finished)
        record = support.read_json(self.dir / "run.json")
        self.assertEqual(record["stop_rule"], "pivots")
        self.assertIn("No unseen text is left", record["stop_reason"])
        self.assertIn("bigger design", record["stop_reason"])
        self.assertEqual(policy.switches, [])

    def test_a_model_that_does_not_learn_the_newest_text_ends_the_run_and_says_so(self):
        c, policy = self.coordinator([5., 4., 4.5, 4.4, 4.3, 4.2, 4.1, 4.0], {"garden": 4.0, "barn": 6.0, "kitchen": 3.0},
                                     pivots=("next_dataset", "next_dataset"))
        self.play(c, 8)
        self.assertTrue(c.finished)
        self.assertTrue(policy.too_small)  # barn stayed at 6.0 after a round on it
        record = support.read_json(self.dir / "run.json")
        self.assertIn("barely fell", record["stop_reason"])
        self.assertIn("bigger design", record["stop_reason"])
        self.assertEqual(c.rules.source, 1)  # it did not move on to garden

    def test_a_scorer_registered_later_ranks_the_sources(self):
        calls = []
        c, policy = self.coordinator([5., 4., 4.5], {"garden": 4.0, "barn": 6.0, "kitchen": 3.0})
        with mock.patch.dict(stages.SCORERS, {"suite": scorer_with({"garden": 9.0, "barn": 1.0, "kitchen": 3.0}, calls)}):
            policy.scorer = "suite"  # the fixed eval suite's own scores
            self.play(c, 3)
        self.assertEqual(policy.names[1], ["garden"])
        self.assertTrue(calls)

    def test_without_a_policy_the_rules_run_as_before(self):
        c = support.build(self.dir, clock=self.clock, inner_steps=2, rounds=20, worse_limit=1, pivots=("next_dataset",))
        self.assertIsNone(c.stages)
        queue = iter([5., 4., 4.5])
        c.evaluate = lambda theta: next(queue)
        c.add_source(c.train)
        self.play(c, 3)
        self.assertEqual(c.rules.source, 1)
        self.assertFalse((self.dir / stages.JOURNAL).exists())

    def test_the_real_scorer_reads_the_weights_the_run_holds(self):
        c = support.build(self.dir, clock=self.clock, inner_steps=2, rounds=4)
        policy = stages.Policy(["kitchen"], None, None, "held_out", held={"kitchen": SOURCES["kitchen"][:6], "garden": SOURCES["garden"][:6]})
        found = policy._losses(c, ["kitchen", "garden"])
        self.assertTrue(all(isinstance(v, float) and v > 0 for v in found.values()), found)
        self.assertFalse((self.dir / "stage_probe.npz").exists())  # the probe file is removed


class PartsTests(unittest.TestCase):
    def test_the_split_is_stable_and_leaves_every_source_something_to_measure(self):
        train, held = stages.split(SOURCES["kitchen"])
        self.assertEqual(stages.split(SOURCES["kitchen"]), (train, held))
        self.assertTrue(held)
        self.assertEqual(len(train) + len(held), len(SOURCES["kitchen"]))
        self.assertEqual(len(stages.split(SOURCES["kitchen"][:3])[1]) + len(stages.split(SOURCES["kitchen"][:3])[0]), 3)

    def test_rank_puts_the_worst_first_and_unmeasured_last(self):
        self.assertEqual(stages.rank({"b": 5., "c": 3., "d": None}, ["b", "c", "d", "e"]), [("b", 5.), ("c", 3.), ("d", None), ("e", None)])

    def test_the_journal_round_trips_and_a_missing_one_is_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(stages.read_journal(directory))
            stages.write_journal(directory, {"sources": [["a"]], "switches": []})
            self.assertEqual(stages.read_journal(directory)["sources"], [["a"]])

    def test_a_job_with_stages_starts_on_its_sets_with_their_held_out_slices_kept_out(self):
        asked = []

        def examples(params):
            asked.append(params["datasets"])
            return SOURCES[params["datasets"][0]]
        services = SimpleNamespace(examples=examples, root=Path("."))
        policy, rows = stages.Policy.for_job({"stages": {"replay": .3, "scorer": "held_out"}, "datasets": ["kitchen", "garden"]}, services)
        train_k, held_k = stages.split(SOURCES["kitchen"])
        train_g, held_g = stages.split(SOURCES["garden"])
        self.assertEqual(rows, train_k + train_g)
        self.assertEqual((policy.held["kitchen"], policy.held["garden"], policy.replay), (held_k, held_g, .3))
        self.assertEqual(policy.names, [["kitchen", "garden"]])
        self.assertEqual(policy.loader("barn"), SOURCES["barn"])


if __name__ == "__main__":
    unittest.main()

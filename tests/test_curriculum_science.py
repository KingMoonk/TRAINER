"""Curriculum that matches the science: reading text in pretraining, replay on, one learning-rate curve, weighted mixing."""
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import studio_curriculum
from studio_core import Workspace
from studio_curriculum import (DEFAULT_REPLAY, _schedule_window, _stage_config, _stage_rows, default_plan, mixed_plan, mixture_weights,
                               run_plan, validate_plan)
from studio_model import train
from studio_modules.schedule.cosine import Cosine, CosineConfig

QUIET = lambda event: None  # noqa: E731


def squad_row(index):
    passage = f"Passage number {index} is about the river town of Calder and its mill, built in {1800 + index}."
    return {"user": f"Answer using only the passage. If the passage does not contain the answer, say so.\n\nPassage:\n{passage}\n\nQuestion: When was the mill built, item {index}?",
            "assistant": str(1800 + index), "groupId": f"squad2:{index}"}


def workspace_of(directory, **sets):
    """A workspace with one catalog-linked dataset per keyword (catalog id -> rows)."""
    workspace = Workspace(directory)
    ids = {}
    for catalog_id, rows in sets.items():
        ids[catalog_id] = workspace.create_dataset(catalog_id, catalogId=catalog_id)
        workspace.add(ids[catalog_id], rows)
    return workspace, ids


def qa(prefix, count):
    return [{"user": f"{prefix} question {i}?", "assistant": f"{prefix} answer {i}."} for i in range(count)]


def language(count):
    return [{"text": f"Plain encyclopedia article number {i}. " + "The tide comes in twice a day. " * 3} for i in range(count)]


class KnowledgeInPretraining(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace, self.ids = workspace_of(
            self.directory.name, wikitext2=language(40), squad2=[squad_row(i) for i in range(40)],
            arc_easy=qa("Arc", 30), dolly15k=qa("Dolly", 30))

    def test_plain_text_and_reading_passages_are_in_the_first_stage_with_full_text_loss(self):
        plan = default_plan(self.workspace, context=1024)
        first, last = plan["stages"][0], plan["stages"][-1]
        self.assertEqual((first["kind"], first["objective"]), ("language", "all"))
        self.assertEqual(set(first["dataset_ids"]), {self.ids["wikitext2"], self.ids["squad2"]})
        self.assertEqual(first["passage_ids"], [self.ids["squad2"]])
        # the questions stay in the last, answer-only stage; the facts are not taught there
        self.assertEqual((last["kind"], last["objective"]), ("knowledge", "assistant"))
        self.assertEqual(set(last["dataset_ids"]), {self.ids["squad2"], self.ids["arc_easy"]})
        self.assertNotIn("passage_ids", last)

    def test_passages_are_plain_text_rows_not_questions(self):
        first = default_plan(self.workspace, context=1024)["stages"][0]
        rows = _stage_rows(self.workspace, first, 40, 1, QUIET, None)
        passages = [row for row in rows if row["datasetId"] == self.ids["squad2"]]
        self.assertTrue(passages)
        for row in passages:
            self.assertTrue(row["text"].startswith("Passage number"))
            self.assertFalse(row["messages"])  # plain text: trained on every token, whatever the objective
            self.assertNotIn("Question:", row["text"])
        self.assertEqual(len({row["text"] for row in passages}), len(passages))
        self.assertTrue(any(row["datasetId"] == self.ids["wikitext2"] for row in rows))
        self.assertLessEqual(len(passages), 10)  # at most PASSAGE_SHARE of the stage

    def test_short_context_keeps_the_passages_but_drops_the_long_questions(self):
        plan = default_plan(self.workspace, context=256)
        self.assertEqual(plan["stages"][0]["passage_ids"], [self.ids["squad2"]])
        self.assertNotIn(self.ids["squad2"], plan["stages"][-1]["dataset_ids"])
        self.assertTrue(any("passages join pretraining" in warning for warning in plan["warnings"]))
        self.assertEqual(validate_plan(self.workspace, plan), [])

    def test_a_held_out_passage_is_not_trained_on_in_any_stage(self):
        plan = default_plan(self.workspace, context=1024, replay=0)
        stages = [{**stage, "eval_records": 5} for stage in (plan["stages"][0], plan["stages"][-1])]
        trained = []

        def fake_train(cfg, examples, run_dir, *args, **kwargs):
            trained.extend(examples)
            return {"status": "completed"}
        with mock.patch.object(studio_curriculum, "train", fake_train):
            state = run_plan(self.workspace, {**plan, "stages": stages}, run_root=self.directory.name, state_dir=self.directory.name)
        held_out = [row for j in (1, 2) for row in studio_curriculum._read_jsonl(Path(state["state_dir"]) / f"{state['id']}_eval" / f"stage{j}.jsonl")]
        blocked = {key for row in held_out for key in studio_curriculum._row_keys(row)}
        self.assertEqual(len(held_out), 10)
        self.assertTrue(trained)
        self.assertFalse([row for row in trained if studio_curriculum._row_keys(row) & blocked])


class ReplayDefault(unittest.TestCase):
    def test_replay_is_on_unless_turned_off(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, _ = workspace_of(directory, dolly15k=qa("Dolly", 20))
            self.assertEqual(DEFAULT_REPLAY, .15)
            self.assertEqual(default_plan(workspace)["replay"], DEFAULT_REPLAY)
            self.assertEqual(default_plan(workspace, replay=0)["replay"], 0)
            validate_plan(workspace, default_plan(workspace))


class OneLearningRateCurve(unittest.TestCase):
    def plan(self, directory):
        workspace, _ = workspace_of(directory, wikitext2=language(40), dolly15k=qa("Dolly", 40), codealpaca20k=qa("Code", 40),
                                    gsm8k=qa("Math", 40))
        plan = default_plan(workspace)
        state = {"base_config": {}, "context": plan["context"], "schedule": plan["schedule"], "learning_rate": plan["learning_rate"],
                 "stages": plan["stages"]}
        return plan, state

    def test_stages_are_contiguous_slices_of_one_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, state = self.plan(directory)
            steps = [stage["steps"] for stage in plan["stages"]]
            self.assertEqual(len(steps), 4)
            windows = [_schedule_window(state, index) for index in range(len(steps))]
            self.assertEqual([w[0] for w in windows], [sum(steps[:i]) for i in range(len(steps))])
            self.assertEqual({w[1] for w in windows}, {sum(steps)})
            configs = [_stage_config(state, stage, index) for index, stage in enumerate(plan["stages"])]
            self.assertEqual({cfg["learning_rate"] for cfg in configs}, {plan["learning_rate"]})
            self.assertEqual(len({cfg["warmup"] for cfg in configs}), 1)

    def test_learning_rate_is_continuous_across_stage_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, state = self.plan(directory)
            cfg = _stage_config(state, plan["stages"][0], 0)
            schedule = Cosine(CosineConfig(warmup=cfg["warmup"]))
            rates = []
            for index, stage in enumerate(plan["stages"]):
                offset, total = _schedule_window(state, index)
                rates.append([schedule.rate(offset + n, total, cfg["learning_rate"]) for n in range(1, stage["steps"] + 1)])
            flat = [rate for stage_rates in rates for rate in stage_rates]
            peak = max(flat)
            self.assertAlmostEqual(peak, plan["learning_rate"])
            top = flat.index(peak)
            self.assertLessEqual(top, cfg["warmup"])  # the only climb is the one warmup, at the very start
            self.assertEqual(flat[top:], sorted(flat[top:], reverse=True))  # then one decay, never back up
            biggest_step = max(abs(a - b) for a, b in zip(flat, flat[1:]))
            for before, after in zip(rates, rates[1:]):
                self.assertLessEqual(abs(after[0] - before[-1]), biggest_step + 1e-12)
                self.assertGreater(after[0], .5 * peak * .1)  # a per-stage warmup would restart near zero
            fresh = Cosine(CosineConfig(warmup=cfg["warmup"])).rate(1, plan["stages"][1]["steps"], plan["learning_rate"])
            self.assertGreater(rates[1][0], 5 * fresh)

    def test_train_follows_its_slice_of_the_schedule(self):
        config = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2, steps=4,
                      learning_rate=.004, warmup=0, validation=0, eval_every=2, objective="assistant")
        examples = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog")] * 2
        schedule = Cosine(CosineConfig(warmup=0))
        for window, expected in (((10, 20), lambda step: schedule.rate(10 + step, 20, .004)), (None, lambda step: schedule.rate(step, 4, .004))):
            events = []
            with tempfile.TemporaryDirectory() as directory:
                train(config, examples, directory + "/run", events.append, schedule_window=window)
            rows = [e for e in events if e["type"] == "metric" and e["step"]]
            self.assertTrue(rows)
            for row in rows:
                self.assertAlmostEqual(row["lr"], expected(row["step"]), places=9)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            train(config, examples, directory + "/run", schedule_window=(10, 12))


class WeightedMixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace, self.ids = workspace_of(self.directory.name, dolly15k=qa("Dolly", 1600), codealpaca20k=qa("Code", 400),
                                                gsm8k=qa("Math", 100))

    def test_default_weights_are_square_root_of_size(self):
        weights = mixture_weights({"a": 1600, "b": 400, "c": 100})
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertAlmostEqual(weights["a"] / weights["b"], 2)  # sqrt(1600) / sqrt(400)
        self.assertAlmostEqual(weights["b"] / weights["c"], 2)
        self.assertAlmostEqual(mixture_weights({"a": 1600, "b": 400}, {"a": 3})["a"], 3 / (3 + 1 / 3))  # b keeps its square-root share
        with self.assertRaises(ValueError):
            mixture_weights({"a": 4}, {"a": 0})

    def test_mixed_plan_is_one_stage_with_all_datasets_and_weights(self):
        plan = mixed_plan(self.workspace, steps=40)
        self.assertEqual(len(plan["stages"]), 1)
        stage = plan["stages"][0]
        self.assertEqual(set(stage["dataset_ids"]), set(self.ids.values()))
        self.assertAlmostEqual(sum(stage["weights"].values()), 1)
        self.assertAlmostEqual(stage["weights"][self.ids["dolly15k"]] / stage["weights"][self.ids["gsm8k"]], 4)
        self.assertEqual(stage["balance"], "records")
        self.assertEqual(validate_plan(self.workspace, plan), [])
        custom = mixed_plan(self.workspace, steps=40, weights={self.ids["gsm8k"]: .5})["stages"][0]["weights"]
        self.assertAlmostEqual(custom[self.ids["gsm8k"]], .5 / (.5 + 6 / 7))

    def test_rows_are_sampled_in_proportion_to_the_weights(self):
        dolly, code, math = self.ids["dolly15k"], self.ids["codealpaca20k"], self.ids["gsm8k"]
        for weights, expected in ((None, {dolly: 4 / 7, code: 2 / 7, math: 1 / 7}), ({dolly: .2, code: .3, math: .5}, {dolly: .2, code: .3, math: .5})):
            stage = mixed_plan(self.workspace, steps=40, weights=weights)["stages"][0]
            rows = _stage_rows(self.workspace, stage, 200, 3, QUIET, None)
            counts = Counter(row["datasetId"] for row in rows)
            self.assertEqual(sum(counts.values()), 200)
            for key, share in expected.items():
                self.assertAlmostEqual(counts[key] / 200, share, delta=.02, msg=(weights, key))

    def test_a_run_trains_on_the_weighted_mixture(self):
        plan = mixed_plan(self.workspace, steps=40, replay=0)
        captured = {}

        def fake_train(cfg, examples, run_dir, *args, **kwargs):
            captured["examples"], captured["window"] = examples, kwargs.get("schedule_window")
            return {"status": "stopped"}
        with mock.patch.object(studio_curriculum, "train", fake_train):
            run_plan(self.workspace, plan, run_root=self.directory.name, state_dir=self.directory.name)
        counts = Counter(row["datasetId"] for row in captured["examples"])
        total = sum(counts.values())
        for key, share in plan["stages"][0]["weights"].items():
            self.assertAlmostEqual(counts[key] / total, share, delta=.06)
        self.assertEqual(captured["window"], (0, 40))


if __name__ == "__main__":
    unittest.main()

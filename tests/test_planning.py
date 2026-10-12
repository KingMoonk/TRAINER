"""Mixture budgets, deterministic sampling, and source-aware validation checks."""
import random
import tempfile
import threading
import unittest
from pathlib import Path

from studio_core import Workspace, fingerprint, normalize
from studio_model import DEFAULT_CONFIG, load_checkpoint, partition_examples, train
from studio_planning import build_training_set


def example(name, quality="Good", **metadata):
    row = normalize({"user": f"Question {name}", "assistant": f"Answer {name}", "quality": quality})
    row.update(id=f"row-{name}", **metadata)
    return row


class SampleWorkspace:
    def __init__(self, collections):
        self.collections = collections
        self.state = {"datasets": [{"id": key, "name": key} for key in collections]}
        self.draws = []

    def audit(self, dataset_id):
        records = self.collections[dataset_id]
        return {"records": len(records), "review": sum(row.get("quality") != "Good" for row in records),
                "characters": sum(len(row.get("user", "")) + len(row.get("assistant", "")) for row in records)}

    def sample(self, dataset_id, limit, seed, reviewed_only):
        self.draws.append((dataset_id, limit))
        rows = [row for row in self.collections[dataset_id] if not reviewed_only or row.get("quality") == "Good"]
        return random.Random(seed).sample(rows, min(limit, len(rows)))

    def examples(self, *args, **kwargs):
        raise AssertionError("Preparation must not read an entire dataset through examples().")


class MixtureChecks(unittest.TestCase):
    def test_weights_reproducibility_and_bounded_requests(self):
        workspace = SampleWorkspace({key: [example(f"{key}{index}") for index in range(100)] for key in ("a", "b")})
        first = build_training_set(workspace, ["a", "b"], 20, seed=17, weights={"a": 3, "b": 1})
        self.assertLessEqual(sum(count for _, count in workspace.draws), 40)
        self.assertEqual(first["report"]["sources"]["a"]["selected"], 15)
        self.assertEqual(first["report"]["sources"]["b"]["selected"], 5)
        second = build_training_set(workspace, ["a", "b"], 20, seed=17, weights={"a": 3, "b": 1})
        self.assertEqual([fingerprint(row) for row in first["examples"]], [fingerprint(row) for row in second["examples"]])
        self.assertEqual([row["id"] for row in first["examples"]], [row["id"] for row in second["examples"]])
        third = build_training_set(workspace, ["a", "b"], 20, seed=18, weights={"a": 3, "b": 1})
        self.assertNotEqual([row["id"] for row in first["examples"]], [row["id"] for row in third["examples"]])

    def test_short_source_quota_is_redistributed_and_zero_weight_excluded(self):
        workspace = SampleWorkspace({"tiny": [example("a"), example("b")], "large": [example(str(i)) for i in range(100)]})
        result = build_training_set(workspace, ["tiny", "large"], 10)
        self.assertEqual(result["report"]["sources"]["tiny"]["selected"], 2)
        self.assertEqual(result["report"]["sources"]["large"]["selected"], 8)
        workspace.draws.clear()
        result = build_training_set(workspace, ["tiny", "large"], 10, weights={"tiny": 0})
        self.assertEqual(result["report"]["sources"]["tiny"]["selected"], 0)
        self.assertEqual([key for key, _ in workspace.draws], ["large"])

    def test_cross_source_duplicates_preserve_lineage_and_source_metadata(self):
        first = example("base", source="public-source", provenance={"license": "CC0"})
        duplicate = dict(first, id="alias-base", groupId="imported-document")
        variant = example("variant", provenance={"source_ids": ["alias-base"]})
        workspace = SampleWorkspace({"a": [first], "b": [duplicate, variant]})
        result = build_training_set(workspace, ["a", "b"], 10)
        self.assertEqual(len(result["examples"]), 2)
        self.assertEqual(result["report"]["duplicates"], 1)
        kept = next(row for row in result["examples"] if row["id"] == first["id"])
        self.assertEqual(kept["provenance"]["license"], "CC0")
        self.assertIn("alias-base", kept["provenance"]["source_ids"])
        self.assertEqual(kept["source"], "public-source")
        training, validation, _ = partition_examples(result["examples"], .2, random.Random(42))
        self.assertEqual(len(training), 2)
        self.assertEqual(validation, [])
        self.assertNotIn("source_ids", first["provenance"])

    def test_review_filter_empty_invalid_and_cancel(self):
        workspace = SampleWorkspace({"a": [example("good"), example("review", quality="Review"), {"user": "missing", "quality": "Good"}]})
        result = build_training_set(workspace, ["a"], 20, reviewed_only=True)
        self.assertEqual([row["id"] for row in result["examples"]], ["row-good"])
        self.assertEqual(result["report"]["invalid"], 1)
        with self.assertRaisesRegex(ValueError, "No reviewed"):
            build_training_set(SampleWorkspace({"a": [{"user": "missing"}]}), ["a"], reviewed_only=True)
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_training_set(workspace, [])
        with self.assertRaisesRegex(ValueError, "weight"):
            build_training_set(workspace, ["a"], weights={"a": float("nan")})
        stop = threading.Event()
        def emit(event):
            stop.set()
        workspace.draws.clear()
        with self.assertRaises(InterruptedError):
            build_training_set(workspace, ["a"], stop=stop, emit=emit)
        self.assertEqual(workspace.draws, [])

    def test_workspace_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            first = workspace.create_dataset("Reviewed")
            second = workspace.create_dataset("Pending")
            workspace.add(first, [example(str(index)) for index in range(10)])
            workspace.add(second, [example(f"p{index}", quality="Review") for index in range(10)])
            result = build_training_set(workspace, [first, second], 7, reviewed_only=True)
            self.assertEqual(len(result["examples"]), 7)
            # "Pending" has no reviewed rows: it is kept with a warning instead of failing the plan.
            self.assertEqual(len(result["report"]["warnings"]), 1)
            self.assertTrue(result["report"]["sources"][second]["reviewed_fallback"])
            self.assertGreater(result["report"]["sources"][first]["selected"], 0)
            self.assertGreater(result["report"]["sources"][second]["selected"], 0)


def sized(name, characters, quality="Good"):
    row = normalize({"user": "q" + name, "assistant": "a" * characters, "quality": quality})
    row["id"] = "row-" + name
    return row


class PreparationChecks(unittest.TestCase):
    def test_calculator_markup_is_stripped_and_empty_answers_dropped(self):
        gsm = normalize({"user": "sum", "assistant": "48/2 = <<48/2=24>>24 clips\n#### 24", "quality": "Good"})
        empty = normalize({"user": "nothing", "assistant": "<<1+1=2>>", "quality": "Good"})
        workspace = SampleWorkspace({"a": [gsm, empty, example("plain")]})
        result = build_training_set(workspace, ["a"], 10)
        by_user = {row["user"]: row for row in result["examples"]}
        self.assertEqual(by_user["sum"]["assistant"], "48/2 = 24 clips\n#### 24")
        self.assertEqual(by_user["sum"]["messages"][1]["content"], "48/2 = 24 clips\n#### 24")
        self.assertNotIn("nothing", by_user)
        self.assertEqual(result["report"]["cleaning"], {"enabled": True, "annotated_rows": 1, "empty_answers_dropped": 1})
        raw = build_training_set(SampleWorkspace({"a": [gsm]}), ["a"], 10, clean=False)
        self.assertIn("<<", raw["examples"][0]["assistant"])

    def test_token_balance_equalizes_character_share(self):
        workspace = SampleWorkspace({"short": [sized(f"s{i}", 10) for i in range(500)],
                                     "long": [sized(f"l{i}", 100) for i in range(500)]})
        by_tokens = build_training_set(workspace, ["short", "long"], 220)["report"]
        by_records = build_training_set(workspace, ["short", "long"], 220, balance="records")["report"]
        self.assertEqual(by_records["sources"]["short"]["selected"], by_records["sources"]["long"]["selected"])
        shares = [info["character_share"] for info in by_tokens["sources"].values()]
        self.assertAlmostEqual(shares[0], .5, delta=.05)
        self.assertGreater(by_tokens["sources"]["short"]["selected"], 5 * by_tokens["sources"]["long"]["selected"])
        with self.assertRaisesRegex(ValueError, "Balance"):
            build_training_set(workspace, ["short"], 10, balance="words")

    def test_use_all_records_ignores_budget(self):
        workspace = SampleWorkspace({"a": [example(f"a{i}") for i in range(30)], "b": [example(f"b{i}") for i in range(12)]})
        result = build_training_set(workspace, ["a", "b"], 5, use_all=True)
        self.assertEqual(len(result["examples"]), 42)
        with self.assertRaisesRegex(ValueError, "1,000,000"):
            build_training_set(workspace, ["a"], 1000001)

    def test_reviewed_only_falls_back_per_source_with_counts(self):
        workspace = SampleWorkspace({"good": [example("g1"), example("g2"), example("p", quality="Review")],
                                     "pending": [example("x1", quality="Review"), example("x2", quality="Review")]})
        report = build_training_set(workspace, ["good", "pending"], 20, reviewed_only=True)["report"]
        self.assertEqual(report["sources"]["good"]["selected"], 2)
        self.assertEqual(report["sources"]["good"]["reviewed"], 2)
        self.assertEqual(report["sources"]["pending"]["selected"], 2)
        self.assertEqual(report["sources"]["pending"]["reviewed"], 0)
        self.assertIn("pending has no reviewed", report["warnings"][0])


class SplitChecks(unittest.TestCase):
    def test_original_and_transitive_synthetic_variants_stay_together(self):
        rows = [example("a"), example("b", provenance={"source_ids": ["row-a"]}),
                example("c", provenance={"source_ids": ["row-b"]}), example("independent")]
        training, validation, report = partition_examples(rows, .4, random.Random(4))
        self.assertEqual(report["groups"], 2)
        self.assertEqual({row["id"] for row in training}, {"row-a", "row-b", "row-c"})
        self.assertEqual({row["id"] for row in validation}, {"row-independent"})

    def test_shared_documents_and_single_group_have_no_leakage(self):
        rows = [example(str(index), groupId=f"doc-{index // 3}") for index in range(9)]
        training, validation, report = partition_examples(rows, .2, random.Random(11))
        self.assertTrue(validation)
        self.assertFalse({row["groupId"] for row in training} & {row["groupId"] for row in validation})
        self.assertEqual(report["groups"], 3)
        training, validation, report = partition_examples(rows[:3], .2, random.Random(11))
        self.assertEqual(len(training), 3)
        self.assertFalse(validation)
        self.assertIn("one source lineage", report["note"])

    def test_legacy_partition_is_unchanged(self):
        rows = [example(str(index), groupId="same") for index in range(10)]
        expected = list(range(10))
        random.Random(13).shuffle(expected)
        training, validation, report = partition_examples(rows, .2, random.Random(13), grouped=False)
        self.assertEqual([row["id"] for row in validation], [rows[index]["id"] for index in expected[:2]])
        self.assertEqual([row["id"] for row in training], [rows[index]["id"] for index in expected[2:]])
        self.assertEqual(report["strategy"], "rows-v1")

    def test_checkpoint_tracks_split_and_rejects_changed_lineage(self):
        rows = [example("a", groupId="one"), example("b", groupId="one"), example("c", groupId="two")]
        config = dict(DEFAULT_CONFIG, device="cpu", width=16, layers=1, heads=2, context=32,
                      expansion=1, batch_size=1, dropout=0, steps=2, validation=.3)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            record = train(config, rows, directory / "first")
            self.assertEqual(record["split"]["groups"], 2)
            checkpoint = load_checkpoint(directory / "first" / "last.npz")
            self.assertEqual(checkpoint["split_strategy"], "lineage-v1")
            changed = [dict(row, groupId="all-together") for row in rows]
            with self.assertRaisesRegex(ValueError, "source lineage"):
                train(dict(config, steps=3), changed, directory / "changed", resume=directory / "first" / "last.npz")


if __name__ == "__main__":
    unittest.main()

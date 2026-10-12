"""Substantive checks for target integrity, curation gates and bounded generation."""
import json
import re
import tempfile
import threading
import unittest
from unittest.mock import patch

from studio_core import Workspace, fingerprint, normalize
from studio_synthesis import FAMILIES, QualityGate, apply_draft_edit, generate_batch, summarize_batch, validate_config


class SynthesisChecks(unittest.TestCase):
    def test_weighted_curriculum_is_reproducible_and_covered(self):
        config = {"mode": "mixed", "count": 90, "seed": 13,
                  "families": FAMILIES, "difficulty": "curriculum"}
        first = generate_batch(config)
        second = generate_batch(config)
        self.assertEqual(len(first["rows"]), 90)
        self.assertEqual([fingerprint(row) for row in first["rows"]], [fingerprint(row) for row in second["rows"]])
        self.assertEqual(first["stats"]["coverage"], dict.fromkeys(FAMILIES, 10))
        self.assertEqual(first["stats"]["difficulty_coverage"], {"easy": 30, "medium": 30, "hard": 30})
        for row in first["rows"]:
            self.assertEqual(row["quality"], "Review")
            self.assertTrue(row["computedVerified"])
            self.assertEqual(row["qualityReport"]["status"], "needs_review")
            self.assertEqual(row["provenance"]["batch_id"], first["batch_id"])

    def test_computed_arithmetic_and_json_targets(self):
        result = generate_batch({"mode": "arithmetic", "count": 80, "difficulty": "hard"})
        for row in result["rows"]:
            derivation = row["provenance"]["derivation"]
            a, b = derivation["operands"]
            op = derivation["operation"]
            expected = {"+": a + b, "-": a - b, "*": a * b, "//": a // b, "%": a % b}[op]
            self.assertEqual(row["assistant"], str(expected))
        result = generate_batch({"mode": "structured", "count": 50, "difficulty": "hard", "quality": {"require_json": True}})
        self.assertEqual(len(result["rows"]), 50)
        for row in result["rows"]:
            derivation = row["provenance"]["derivation"]
            value = derivation["record"]
            for key in derivation["field"].split("."):
                value = value[key]
            self.assertEqual(json.loads(row["assistant"]), value)

    def test_correlated_facts_never_make_cartesian_false_pairs(self):
        config = {"mode": "templates", "count": 10, "prompt": "Capital of {country}?", "response": "{capital}",
                  "fact_rows": [{"country": "France", "capital": "Paris"}, {"country": "Japan", "capital": "Tokyo"}],
                  "variables": {"country": ["France", "Japan"], "capital": ["Paris", "Tokyo"]}}
        result = generate_batch(config)
        self.assertEqual({(row["user"], row["assistant"]) for row in result["rows"]}, {("Capital of France?", "Paris"), ("Capital of Japan?", "Tokyo")})
        self.assertEqual(result["stats"]["shortfall"], 8)
        self.assertEqual(result["stats"]["exhausted_families"], ["templates"])
        self.assertTrue(all(not row["computedVerified"] for row in result["rows"]))

    def test_grounding_exact_source_spans_and_answers(self):
        source = {"id": "corpus-1", "text": "Orion opened in 1998. It has 17 rooms.\nThe entrance is blue."}
        result = generate_batch({"mode": "grounded", "count": 12}, seeds=[source])
        self.assertGreaterEqual(len(result["rows"]), 3)
        for row in result["rows"]:
            detail = row["provenance"]["derivation"]
            self.assertEqual(row["assistant"], source["text"][detail["start"]:detail["end"]])
            self.assertEqual(row["provenance"]["source_ids"], ["corpus-1"])
            self.assertIn(source["text"], row["user"])
        seed = {"id": "chat-1", "messages": [{"role": "user", "content": "What is the code?"}, {"role": "assistant", "content": "The code is xYz-42."}]}
        quoted = generate_batch({"mode": "grounded", "count": 1}, seeds=[seed])["rows"][0]
        self.assertEqual(quoted["assistant"], "The code is xYz-42.")
        self.assertIn(quoted["assistant"], quoted["user"])
        self.assertEqual(quoted["provenance"]["source_ids"], ["chat-1"])

    def test_seed_variants_preserve_full_targets_and_source_ids(self):
        seed = {"id": "original-42", "messages": [{"role": "system", "content": "Reply precisely."},
                {"role": "user", "content": "One?"}, {"role": "assistant", "content": "1"},
                {"role": "user", "content": "Two?"}, {"role": "assistant", "content": "2"}]}
        result = generate_batch({"mode": "seeds", "count": 20}, seeds=[seed])
        self.assertEqual(len(result["rows"]), 8)
        for row in result["rows"]:
            self.assertEqual(row["messages"][0], seed["messages"][0])
            self.assertEqual(row["messages"][2:], seed["messages"][2:])
            self.assertEqual(row["provenance"]["source_ids"], ["original-42"])

    def test_exact_dedup_conflicts_and_numbers(self):
        gate = QualityGate(existing=[normalize({"user": " Add 2 and 3. ", "assistant": "5"})])
        self.assertEqual(gate.inspect(normalize({"user": "ADD   2 and 3.", "assistant": "5"})), "duplicate")
        self.assertEqual(gate.inspect(normalize({"user": "Add 2 and 3.", "assistant": "6"})), "prompt_conflict")
        self.assertIsNone(gate.inspect(normalize({"user": "Add 2 and 4.", "assistant": "6"})))

    def test_quality_gates_and_bounded_rejections(self):
        config = {"mode": "arithmetic", "count": 50, "quality": {"min_answer_chars": 100}, "attempt_multiplier": 3}
        result = generate_batch(config)
        self.assertEqual(result["stats"]["attempted"], 150)
        self.assertEqual(result["stats"]["accepted"], 0)
        self.assertEqual(result["stats"]["rejected"], 150)
        self.assertEqual(len(result["rejected"]), 100)
        gate = QualityGate({"require_json": True})
        self.assertEqual(gate.inspect(normalize({"user": "JSON?", "assistant": "NaN"})), "invalid_json")
        gate = QualityGate({"max_repeat_fraction": .8})
        self.assertEqual(gate.inspect(normalize({"user": "Say something", "assistant": "word " * 20})), "repetitive_answer")

    def test_near_duplicates_optional_and_bounded_index(self):
        source = normalize({"user": "Please summarize the following account of a small red boat arriving at the island today.", "assistant": "A red boat arrived."})
        candidate = normalize({"user": "Please summarize the following account of a small blue boat arriving at the island today.", "assistant": "A red boat arrived."})
        gate = QualityGate({"near_duplicate_threshold": .6}, existing=[source])
        self.assertEqual(gate.inspect(candidate), "near_duplicate")
        self.assertIsNone(QualityGate(existing=[source]).inspect(candidate))

    def test_cancellation_returns_partial_and_progress(self):
        stop = threading.Event()
        events = []
        def progress(event):
            events.append(event)
            if event["completed"]:
                stop.set()
        result = generate_batch({"mode": "arithmetic", "count": 10000}, stop=stop, emit=progress)
        self.assertTrue(result["stats"]["cancelled"])
        self.assertGreater(len(result["rows"]), 0)
        self.assertLess(len(result["rows"]), 10000)
        self.assertEqual(events[-1]["completed"], len(result["rows"]))
        stop.set()
        result = generate_batch({"mode": "model", "count": 2}, stop=stop)
        self.assertEqual(result["rows"], [])

    def test_local_model_receives_device_and_sampling_controls(self):
        class FakeModel:
            step = 17
            def __init__(self, path, seed, device):
                self.options = (path, seed, device)
            def generate(self, messages, **kwargs):
                self.controls = kwargs
                return "A generated answer."
        with patch("studio_model.LocalModel", FakeModel):
            result = generate_batch({"mode": "model", "count": 2, "checkpoint": "own.npz", "device": "cuda", "model_prompts": ["Question one?", "Question two?"]})
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(all(not row["computedVerified"] for row in result["rows"]))
        self.assertEqual(result["rows"][0]["provenance"]["derivation"]["checkpoint_step"], 17)

    def test_weighted_budgets_and_invalid_configs(self):
        result = generate_batch({"mode": "mixed", "count": 40, "families": ["arithmetic", "string_ops"], "weights": {"arithmetic": 3, "string_ops": 1}})
        self.assertEqual(result["stats"]["coverage"], {"arithmetic": 30, "string_ops": 10})
        for bad in ({"count": 100001}, {"count": 0}, {"count": 3.5}, {"quality": {"require_json": "false"}}, {"weights": {"arithmetic": float("nan")}}):
            with self.assertRaises(ValueError): validate_config(bad)
        with self.assertRaises(ValueError):
            generate_batch({"mode": "templates", "prompt": "{x.__class__}", "response": "hi", "variables": {"x": [1]}})


class RemixChecks(unittest.TestCase):
    def test_remix_reuses_source_targets(self):
        seeds = [{"user": f"Tom has {i} apples and buys 2. How many?", "assistant": f"{i}+2 = <<{i}+2={i + 2}>>{i + 2}\n#### {i + 2}"} for i in range(30)]
        seeds += [{"user": f"Who wrote book {i}?\n\nContext:\nBook {i} was written by Author {i}.", "assistant": f"Author {i}"} for i in range(30)]
        result = generate_batch({"mode": "remix", "count": 60, "seed": 3}, seeds=seeds)
        rows = result["rows"]
        self.assertEqual(len(rows), 60)
        by_family = {}
        for row in rows:
            by_family.setdefault(row["provenance"]["recipe"], []).append(row)
        self.assertTrue(set(by_family) <= {"final_answer", "clean_solution", "json_answer", "multi_turn", "context_first"})
        for row in by_family.get("final_answer", []):
            self.assertRegex(row["assistant"], r"^\d+$")
        for row in by_family.get("clean_solution", []):
            self.assertNotIn("<<", row["assistant"])
            self.assertRegex(row["assistant"], r"Answer: \d+$")
        for row in by_family.get("json_answer", []):
            self.assertTrue(json.loads(row["assistant"])["answer"].startswith("Author"))
        for row in by_family.get("multi_turn", []):
            self.assertGreaterEqual(len(row["messages"]), 4)


class RegressionChecks(unittest.TestCase):
    def test_model_mode_gives_several_drafts_per_prompt(self):
        """Bug (a): source prompts are also in `existing`; differing answers used to trip prompt_conflict."""
        class FakeModel:
            step = 3
            calls = 0
            def __init__(self, path, seed, device):
                pass
            def generate(self, messages, **kwargs):
                FakeModel.calls += 1
                return f"Generated answer number {FakeModel.calls}."
        seeds = [{"id": f"s{i}", "user": f"Question {i}?", "assistant": f"Human answer {i}."} for i in range(3)]
        with patch("studio_model.LocalModel", FakeModel):
            result = generate_batch({"mode": "model", "count": 9, "checkpoint": "own.npz"}, seeds=seeds, existing=seeds)
        self.assertEqual(len(result["rows"]), 9)
        self.assertEqual(result["stats"]["reasons"].get("prompt_conflict", 0), 0)
        self.assertEqual({tuple(row["provenance"]["source_ids"]) for row in result["rows"]}, {("s0",), ("s1",), ("s2",)})
        # exact duplicates are still removed
        class Constant(FakeModel):
            def generate(self, messages, **kwargs):
                return "Same every time."
        with patch("studio_model.LocalModel", Constant):
            result = generate_batch({"mode": "model", "count": 9, "checkpoint": "own.npz"}, seeds=seeds)
        self.assertEqual(len(result["rows"]), 3)

    def test_verification_flags_survive_normalize_and_keeping(self):
        """Bug (b): computedVerified/qualityReport were dropped when drafts were kept."""
        draft = generate_batch({"mode": "arithmetic", "count": 3})["rows"][0]
        kept = normalize(draft, "dataset", "synthetic")
        for key in ("computedVerified", "qualityReport", "provenance"):
            self.assertEqual(kept[key], draft[key])
        self.assertTrue(kept["computedVerified"])
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            dataset = workspace.create_dataset("kept")
            self.assertEqual(workspace.add(dataset, [draft], "synthetic")[0], 1)
            stored = workspace.examples(dataset)[0]
            self.assertTrue(stored["computedVerified"])
            self.assertEqual(stored["qualityReport"]["status"], "needs_review")
            self.assertEqual(stored["provenance"]["batch_id"], draft["provenance"]["batch_id"])
            workspace.mark_reviewed([stored["id"]])
            self.assertTrue(workspace.get_example(stored["id"])["computedVerified"])

    def test_edited_draft_keeps_provenance_and_reruns_gates(self):
        """Bug (c): editing rebuilt the row from scratch and skipped the gates."""
        rows = generate_batch({"mode": "arithmetic", "count": 4})["rows"]
        old, other = rows[0], rows[1]
        edit = normalize({"user": old["user"], "assistant": "42", "intent": old["intent"], "quality": "Good"})
        edit["id"] = old["id"]
        row, reason = apply_draft_edit(old, edit, others=rows)
        self.assertIsNone(reason)
        self.assertEqual((row["id"], row["assistant"], row["quality"]), (old["id"], "42", "Good"))
        self.assertEqual(row["provenance"]["batch_id"], old["provenance"]["batch_id"])
        self.assertTrue(row["provenance"]["edited"])
        self.assertFalse(row["computedVerified"])
        self.assertTrue(row["qualityReport"]["edited"])
        self.assertTrue(old["computedVerified"])                        # the original dict is untouched
        # unchanged text (only the review mark changed) keeps the computed flag
        same, reason = apply_draft_edit(old, {**old, "quality": "Good"}, others=rows)
        self.assertTrue(same["computedVerified"])
        self.assertIsNone(reason)
        # an edit that duplicates another draft, or empties the answer, is flagged by the gates
        clash = normalize({"user": other["user"], "assistant": other["assistant"]})
        clash["id"] = old["id"]
        self.assertEqual(apply_draft_edit(old, clash, others=rows)[1], "duplicate")
        conflict = normalize({"user": other["user"], "assistant": "definitely different"})
        conflict["id"] = old["id"]
        self.assertEqual(apply_draft_edit(old, conflict, others=rows)[1], "prompt_conflict")
        self.assertEqual(apply_draft_edit(old, edit, others=rows, quality={"max_answer_chars": 1})[1], "answer_too_long")

    def test_plain_words_summary(self):
        result = {"rows": [], "stats": {"requested": 200, "accepted": 186, "reasons": {"duplicate": 9, "answer_too_long": 5}}}
        self.assertEqual(summarize_batch(result), "Made 186 examples. Skipped 14: 9 duplicates, 5 too long. (You asked for 200.)")
        self.assertEqual(summarize_batch({"stats": {"requested": 1, "accepted": 1, "reasons": {}}}), "Made 1 example.")


if __name__ == "__main__":
    unittest.main()

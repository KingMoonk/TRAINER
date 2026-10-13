"""Tests for studio_curriculum bugs: _read_jsonl, _checkpoint."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import studio_curriculum
from studio_core import Workspace
from studio_curriculum import _checkpoint, _held_out, _prompt_key, _read_jsonl, _stage_config, default_plan


def workspace_with(directory, count=30):
    workspace = Workspace(directory)
    dataset = workspace.create_dataset("Questions")
    workspace.add(dataset, [{"user": f"Question {i}?", "assistant": f"Answer {i}."} for i in range(count)])
    return workspace, dataset


def stage(dataset, **extra):
    return {"name": "S", "kind": "instructions", "dataset_ids": [dataset], "steps": 5, "max_records": 100,
            "learning_rate": 1e-3, "eval_records": 5, "status": "pending", "run_dirs": [], **extra}


class ReadJsonlBug(unittest.TestCase):
    def test_read_jsonl_skips_bad_lines(self):
        """Bug 3: _read_jsonl raises on corrupt lines instead of skipping them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            # Write one good line, one bad JSON line, one good line
            path.write_text('{"a": 1}\n{"bad json\n{"b": 2}\n', encoding="utf-8")
            # Should return only the good lines, not raise
            result = _read_jsonl(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], {"a": 1})
            self.assertEqual(result[1], {"b": 2})

    def test_read_jsonl_skips_bad_utf8(self):
        """Bug 3: _read_jsonl should skip lines with bad UTF-8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            # Write one good line, one bad UTF-8 line, one good line (as bytes)
            path.write_bytes(b'{"a": 1}\n\xff\xfe\n{"b": 2}\n')
            # Should return only the good lines, not raise
            result = _read_jsonl(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], {"a": 1})
            self.assertEqual(result[1], {"b": 2})

    def test_read_jsonl_empty_file(self):
        """Empty file should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text('', encoding="utf-8")
            result = _read_jsonl(path)
            self.assertEqual(result, [])

    def test_read_jsonl_missing_file(self):
        """Missing file should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.jsonl"
            result = _read_jsonl(path)
            self.assertEqual(result, [])


class CheckpointBug(unittest.TestCase):
    def test_checkpoint_returns_best_if_exists(self):
        """If best.npz exists, return it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "best.npz").touch()
            result = _checkpoint(tmpdir)
            self.assertEqual(result, str(Path(tmpdir) / "best.npz"))

    def test_checkpoint_returns_last_if_best_missing(self):
        """If only last.npz exists, return it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "last.npz").touch()
            result = _checkpoint(tmpdir)
            self.assertEqual(result, str(Path(tmpdir) / "last.npz"))

    def test_checkpoint_returns_best_over_last(self):
        """If both exist, return best."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "best.npz").touch()
            Path(tmpdir, "last.npz").touch()
            result = _checkpoint(tmpdir)
            self.assertEqual(result, str(Path(tmpdir) / "best.npz"))

    def test_checkpoint_returns_none_if_neither_exists(self):
        """Bug 4: if neither file exists, return None (this caller must handle)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _checkpoint(tmpdir)
            self.assertIsNone(result)


class HeldOutTests(unittest.TestCase):
    def test_small_stages_get_disjoint_held_out_sets_across_shared_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, dataset = workspace_with(directory)
            state = {"stages": [stage(dataset), stage(dataset)]}
            eval_dir = Path(directory) / "eval"
            registry = _held_out(workspace, state, eval_dir, 42, lambda e: None, None)
            first, second = (_read_jsonl(eval_dir / f"stage{j}.jsonl") for j in (1, 2))
            self.assertEqual((len(first), len(second)), (5, 5))
            self.assertFalse({_prompt_key(r) for r in first} & {_prompt_key(r) for r in second})
            self.assertEqual(len(registry), 10)

    def test_default_plan_skips_validation_splits_and_reports_unreadable_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, dataset = workspace_with(directory)
            workspace.state["datasets"].append({"id": "catalog_tinystories_valid", "name": "Valid", "recordCount": 9})
            broken = workspace.create_dataset("Broken")
            real = workspace.count
            with mock.patch.object(workspace, "count", side_effect=lambda key, *a: (_ for _ in ()).throw(OSError("gone")) if key == broken else real(key, *a)):
                plan = default_plan(workspace)
            self.assertEqual([s["dataset_ids"] for s in plan["stages"]], [[dataset]])
            self.assertTrue(any("Valid" in w for w in plan["warnings"]))
            self.assertTrue(any("Broken" in w for w in plan["warnings"]))

    def test_stage_config_rejects_legacy_spec_without_embedding(self):
        with self.assertRaises(ValueError):
            _stage_config({"base_config": {"architecture": {"blocks": []}}, "context": 32}, stage("d"), 0)

    def test_resume_uses_newest_last_npz_across_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, dataset = workspace_with(directory)
            first, second = Path(directory) / "a1", Path(directory) / "a2"
            first.mkdir()
            second.mkdir()
            (first / "last.npz").write_bytes(b"x")
            (first / "dataset.jsonl").write_text(json.dumps({"user": "Question 1?", "assistant": "Answer 1."}) + "\n")
            state = {"id": "p", "name": "P", "status": "stopped", "context": 32, "replay": 0.0, "base_config": {},
                     "run_root": directory, "state_dir": directory, "eval_table": [], "warnings": [],
                     "stages": [stage(dataset, status="stopped", run_dir=str(second), run_dirs=[str(first), str(second)])]}
            seen = {}

            def fake_train(cfg, examples, run_dir, *args, resume=None, init_from=None):
                seen["resume"] = resume
                return {"status": "stopped"}
            with mock.patch.object(studio_curriculum, "train", fake_train):
                studio_curriculum._execute(workspace, state, lambda e: None, threading.Event(), threading.Event())
            self.assertEqual(seen["resume"], str(first / "last.npz"))


if __name__ == "__main__":
    unittest.main()

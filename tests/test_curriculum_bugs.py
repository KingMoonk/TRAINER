"""Tests for studio_curriculum bugs: _read_jsonl, _checkpoint."""
import tempfile
import unittest
from pathlib import Path

from studio_curriculum import _read_jsonl, _checkpoint


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


if __name__ == "__main__":
    unittest.main()

"""Gate test for S10 fixture workspace: idempotence, timing, file structure."""
import json
import tempfile
import time
import unittest
from pathlib import Path

from tests.fixtures.ia_workspace import build


class TestIAFixture(unittest.TestCase):
    """Test that the fixture workspace builds correctly and meets requirements."""

    def test_build_creates_workspace_in_time(self):
        """Build workspace and verify timing is under 20 seconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            start = time.time()
            ws = build(tmpdir)
            elapsed = time.time() - start

            self.assertLess(elapsed, 20, f"Build took {elapsed:.1f}s, must be < 20s on CPU")
            self.assertEqual(ws.root, tmpdir)

    def test_build_creates_required_files(self):
        """Verify all required files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            build(tmpdir)

            # Check workspace structure
            self.assertTrue((tmpdir / "data" / "workspace.json").exists(), "workspace.json missing")
            self.assertTrue((tmpdir / "data" / "versions.json").exists(), "versions.json missing")
            self.assertTrue((tmpdir / "data" / "queue.json").exists(), "queue.json missing")
            self.assertTrue((tmpdir / "data" / "heldout.json").exists(), "heldout.json missing")
            self.assertTrue((tmpdir / "runs" / "v1").exists(), "v1 run directory missing")
            self.assertTrue((tmpdir / "runs" / "v2").exists(), "v2 run directory missing")

    def test_versions_have_checkpoints(self):
        """Verify both versions have trained checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            build(tmpdir)

            # Check that both versions have last.npz and best.npz
            self.assertTrue((tmpdir / "runs" / "v1" / "last.npz").exists(), "v1/last.npz missing")
            self.assertTrue((tmpdir / "runs" / "v1" / "best.npz").exists(), "v1/best.npz missing")
            self.assertTrue((tmpdir / "runs" / "v2" / "last.npz").exists(), "v2/last.npz missing")
            self.assertTrue((tmpdir / "runs" / "v2" / "best.npz").exists(), "v2/best.npz missing")

    def test_versions_json_schema(self):
        """Verify versions.json has correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            build(tmpdir)

            versions = json.loads((tmpdir / "data" / "versions.json").read_text(encoding="utf-8"))
            self.assertIsInstance(versions, list)
            self.assertEqual(len(versions), 2)

            for v in versions:
                self.assertIn("number", v)
                self.assertIn("change", v)
                self.assertIn("parent", v)
                self.assertIn("spec", v)
                self.assertIn("created_at", v)
                self.assertIn("runs", v)
                self.assertIn("best_checkpoint", v)
                self.assertIn("status", v)
                self.assertIn("scorecard", v)
                self.assertIn("verdict", v)

            # Check version details
            self.assertEqual(versions[0]["number"], 1)
            self.assertIsNone(versions[0]["parent"])
            self.assertEqual(versions[1]["number"], 2)
            self.assertEqual(versions[1]["parent"], 1)

    def test_queue_json_schema(self):
        """Verify queue.json has correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            build(tmpdir)

            queue = json.loads((tmpdir / "data" / "queue.json").read_text(encoding="utf-8"))
            self.assertIn("now", queue)
            self.assertIn("up_next", queue)
            self.assertIn("plan", queue)
            self.assertIsNone(queue["now"])
            self.assertIsInstance(queue["up_next"], list)
            self.assertEqual(len(queue["up_next"]), 0)

    def test_heldout_json_schema(self):
        """Verify heldout.json exists and contains documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            build(tmpdir)

            heldout = json.loads((tmpdir / "data" / "heldout.json").read_text(encoding="utf-8"))
            self.assertIsInstance(heldout, list)
            self.assertGreaterEqual(len(heldout), 5, "Must have at least 5 held-out documents")

    def test_build_is_idempotent(self):
        """Verify build() returns existing workspace without retraining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # First build
            build(tmpdir)
            v1_mtime = (tmpdir / "runs" / "v1" / "last.npz").stat().st_mtime
            v2_mtime = (tmpdir / "runs" / "v2" / "last.npz").stat().st_mtime

            # Wait a bit to see time difference
            import time
            time.sleep(0.1)

            # Second build should not retrain
            build(tmpdir)
            v1_mtime2 = (tmpdir / "runs" / "v1" / "last.npz").stat().st_mtime
            v2_mtime2 = (tmpdir / "runs" / "v2" / "last.npz").stat().st_mtime

            # Mtimes should be identical (no retraining)
            self.assertEqual(v1_mtime, v1_mtime2, "v1 was retrained on second build")
            self.assertEqual(v2_mtime, v2_mtime2, "v2 was retrained on second build")

    def test_workspace_has_datasets(self):
        """Verify workspace contains expected datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            ws = build(tmpdir)

            dataset_names = {d["name"] for d in ws.state["datasets"]}
            self.assertIn("text", dataset_names)
            self.assertIn("conversations", dataset_names)
            self.assertIn("problems", dataset_names)

    def test_config_is_correct(self):
        """Verify version configs have correct parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            build(tmpdir)

            versions = json.loads((tmpdir / "data" / "versions.json").read_text(encoding="utf-8"))

            # V1 config
            cfg1 = versions[0]["spec"]
            self.assertEqual(cfg1["width"], 16)
            self.assertEqual(cfg1["layers"], 1)
            self.assertEqual(cfg1["context"], 32)
            self.assertEqual(cfg1["steps"], 30)
            self.assertEqual(cfg1["device"], "cpu")

            # V2 config
            cfg2 = versions[1]["spec"]
            self.assertEqual(cfg2["width"], 16)
            self.assertEqual(cfg2["layers"], 3)
            self.assertEqual(cfg2["context"], 32)
            self.assertEqual(cfg2["steps"], 10)


if __name__ == "__main__":
    unittest.main()

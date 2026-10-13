import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import studio_cleanup
from studio_core import Workspace


def make(root):
    ws = Workspace(root)
    ws.add(ws.create_dataset("kept"), ["one document"])
    for name, age_days in (("old", 40), ("new", 1)):
        run = Path(root) / "runs" / name
        run.mkdir(parents=True)
        (run / "best.npz").write_bytes(b"x" * 100)
        stamp = time.time() - age_days * 86400
        os.utime(run / "best.npz", (stamp, stamp))
    versions = [{"number": n, "runs": [f"runs/{name}"], "best_checkpoint": f"runs/{name}/best.npz", "status": "trained"}
                for n, name in ((1, "old"), (2, "new"))]
    (Path(root) / "data" / "versions.json").write_text(json.dumps({"versions": versions}))
    return ws


class Cleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = make(self.root)

    def versions(self):
        return json.loads((self.root / "data" / "versions.json").read_text())["versions"]

    def test_older_than_keeps_recent_runs_and_fixes_versions(self):
        self.assertEqual(studio_cleanup.clean(self.ws, days=30), (1, 100))
        self.assertEqual(sorted(p.name for p in (self.root / "runs").iterdir()), ["new"])
        old, new = self.versions()
        self.assertEqual((old["runs"], old["best_checkpoint"], old["status"]), ([], None, "saved"))
        self.assertEqual((new["runs"], new["status"]), (["runs/new"], "trained"))

    def test_all_history_keeps_data_unless_asked(self):
        studio_cleanup.clean(self.ws)
        self.assertEqual(list((self.root / "runs").iterdir()), [])
        self.assertFalse((self.root / "data" / "versions.json").exists())
        self.assertEqual(len(Workspace(self.root).state["datasets"]), 1)
        studio_cleanup.clean(self.ws, data=True)
        self.assertEqual(Workspace(self.root).state["datasets"], [])

    def test_plan_matches_what_clean_deletes(self):
        paths = studio_cleanup.plan(self.root, days=30)
        self.assertEqual((len(paths), studio_cleanup.size(paths)), studio_cleanup.clean(self.ws, days=30))


if __name__ == "__main__":
    unittest.main()

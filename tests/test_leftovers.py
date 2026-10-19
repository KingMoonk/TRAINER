"""studio_cleanup.leftovers: interrupted attempts, failed runs and orphan dataset snapshots are listed with their sizes; protected
folders never are. Window-free."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import studio_cleanup
import studio_dataset_snapshot as snapshots
from studio_core import Workspace


def job(id, status, **params):
    return {"id": id, "label": f"Job {id}", "status": status, "params": params}


class Leftovers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.runs = self.root / "runs"
        Workspace(self.root).save()

    def folder(self, name, weights=100, snapshot=None):
        path = self.runs / name
        path.mkdir(parents=True)
        (path / "last.npz").write_bytes(b"w" * weights)
        if snapshot is not None:
            stored, _ = snapshots.store(snapshot, self.runs / snapshots.FOLDER)
            snapshots.link(stored, path / "dataset.jsonl")
        return path

    def ids(self, jobs):
        return {row["id"]: row for row in studio_cleanup.leftovers(self.root, jobs)}

    def test_interrupted_attempts_failed_runs_and_orphan_snapshots_are_listed_with_sizes(self):
        data = [{"text": "x" * 500}]
        self.folder("queue_a-before-restart", 100, data)
        self.folder("queue_a-before-restart-2", 200, data)
        self.folder("queue_b", 300)
        stored, _ = snapshots.store([{"text": "nobody uses this"}], self.runs / snapshots.FOLDER)
        found = self.ids([job("a", "running"), job("b", "failed")])
        self.assertEqual(sorted((r["kind"], r["id"]) for r in found.values()),
                         [("failed", "runs/queue_b"), ("interrupted", "runs/queue_a-before-restart"),
                          ("interrupted", "runs/queue_a-before-restart-2"), ("snapshot", f"runs/.snapshots/{stored.name}")])
        shared = found["runs/queue_a-before-restart"]
        self.assertEqual((shared["bytes"], shared["shared"]), (100, (self.runs / "queue_a-before-restart" / "dataset.jsonl").stat().st_size))  # the snapshot stays: another folder links it
        self.assertEqual(found["runs/queue_b"]["bytes"], 300)
        self.assertEqual(found[f"runs/.snapshots/{stored.name}"]["bytes"], stored.stat().st_size)
        self.assertEqual(found["runs/queue_b"]["belongs_to"], "Job b")

    def test_a_snapshot_a_folder_links_is_not_an_orphan_and_becomes_one_when_the_last_folder_goes(self):
        self.folder("queue_a-before-restart", 10, [{"text": "y"}])
        self.assertFalse([r for r in self.ids([]).values() if r["kind"] == "snapshot"])
        self.assertEqual(studio_cleanup.remove_leftovers(self.root, ["runs/queue_a-before-restart"], [])[0], 1)
        orphans = [r for r in self.ids([]).values() if r["kind"] == "snapshot"]
        self.assertEqual(len(orphans), 1)

    def test_a_snapshot_is_freed_only_when_every_link_to_it_goes(self):
        data = [{"text": "z" * 1000}]
        first = self.folder("queue_a-before-restart", 10, data)
        second = self.folder("queue_a-before-restart-2", 10, data)
        both = studio_cleanup.size([first, second])
        self.assertEqual(studio_cleanup.size([first]), 10)
        self.assertEqual(both, 20)  # the store still names the snapshot
        (stored, size, _), = snapshots.listing(self.runs / snapshots.FOLDER)
        self.assertEqual(studio_cleanup.size([first, second, stored]), 20 + size)

    def test_what_a_running_or_waiting_job_uses_is_never_listed(self):
        self.folder("queue_run")
        aside = self.folder("queue_wait-before-restart")
        self.folder("queue_wait-before-restart-2")
        self.folder("queue_init-before-restart")
        self.folder("queue_gone-before-restart")
        jobs = [job("run", "running"), job("wait", "waiting", resume=str(aside / "last.npz")),
                job("later", "held", init_from=str(self.runs / "queue_init-before-restart" / "best.npz"))]
        self.assertEqual(sorted(self.ids(jobs)), ["runs/queue_gone-before-restart", "runs/queue_wait-before-restart-2"])

    def test_a_relative_path_and_a_failed_job_a_version_keeps_are_protected(self):
        self.folder("queue_kept")
        self.folder("queue_dropped")
        (self.root / "data").mkdir(exist_ok=True)
        (self.root / "data" / "versions.json").write_text(json.dumps({"versions": [
            {"number": 1, "runs": ["runs/queue_kept"], "best_checkpoint": "runs/queue_kept/last.npz", "status": "trained"}]}))
        self.assertEqual(sorted(self.ids([job("kept", "failed"), job("dropped", "failed")])), ["runs/queue_dropped"])

    def test_a_finished_job_and_unrelated_folders_are_left_alone(self):
        self.folder("queue_done")
        self.folder("compare_x")
        self.folder("imported_gpt2")
        self.assertEqual(self.ids([job("done", "done"), job("x", "failed")]), {})

    def test_remove_deletes_only_what_is_listed(self):
        self.folder("queue_a-before-restart")
        self.folder("queue_run")
        self.folder("queue_c-before-restart", 50)
        jobs = [job("run", "running"), job("c", "waiting", resume=str(self.runs / "queue_c-before-restart" / "last.npz"))]
        done = studio_cleanup.remove_leftovers(self.root, ["runs/queue_a-before-restart", "runs/queue_run", "runs/queue_c-before-restart",
                                                           "runs/nothing", "../outside", "data/versions.json"], jobs)
        self.assertEqual(done, (1, 100))
        self.assertEqual(sorted(p.name for p in self.runs.iterdir()), ["queue_c-before-restart", "queue_run"])

    def test_an_unfinished_snapshot_write_is_listed_once_it_is_old(self):
        store = self.runs / snapshots.FOLDER
        store.mkdir(parents=True)
        fresh, old = store / ".tmp-1-a", store / ".tmp-2-b"
        for path in (fresh, old):
            path.write_bytes(b"half")
        stamp = time.time() - 2 * studio_cleanup.UNFINISHED
        os.utime(old, (stamp, stamp))
        self.assertEqual(list(self.ids([])), [f"runs/.snapshots/{old.name}"])

    def test_the_age_based_clean_judges_snapshots_one_by_one(self):
        old_stamp = time.time() - 40 * 86400
        used, _ = snapshots.store([{"text": "used"}], self.runs / snapshots.FOLDER)
        orphan, _ = snapshots.store([{"text": "orphan"}], self.runs / snapshots.FOLDER)
        run = self.folder("queue_recent")
        snapshots.link(used, run / "dataset.jsonl")
        for path in (used, orphan):
            os.utime(path, (old_stamp, old_stamp))
        self.assertEqual([p.name for p in studio_cleanup.plan(self.root, days=30)], [orphan.name])


class ThroughServices(unittest.TestCase):
    def test_the_queue_decides_what_is_protected_and_the_remove_runs_under_its_lock(self):
        from studio_ia.services import WorkspaceServices
        from studio_queue import Job
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            services = WorkspaceServices(Workspace(root))
            self.addCleanup(services.close)
            failed = Job(kind="pretrain", label="Version 2 from scratch", status="failed", id="f1")
            waiting = Job(kind="pretrain", label="Version 3", status="held", id="w1", params={"resume": str(root / "runs" / "queue_w1-before-restart" / "last.npz")})
            services.runner.jobs.extend([failed, waiting])
            for name in ("queue_f1", "queue_w1-before-restart", "queue_old-before-restart"):
                (root / "runs" / name).mkdir(parents=True)
                (root / "runs" / name / "last.npz").write_bytes(b"x" * 10)
            rows = {r["id"]: r for r in services.leftovers()}
            self.assertEqual(sorted(rows), ["runs/queue_f1", "runs/queue_old-before-restart"])
            self.assertEqual(rows["runs/queue_f1"]["belongs_to"], "Version 2 from scratch")
            self.assertEqual(services.remove_leftovers(list(rows) + ["runs/queue_w1-before-restart"]), (2, 20))
            self.assertEqual([p.name for p in (root / "runs").iterdir()], ["queue_w1-before-restart"])


if __name__ == "__main__":
    unittest.main()

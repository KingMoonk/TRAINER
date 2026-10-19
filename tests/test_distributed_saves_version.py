"""A distributed run that finishes saves its coordinator's final weights on the version it trained; one that is stopped saves nothing.
After a failover the weights come from the successor coordinator the where-file names, over its /collective routes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import studio_diloco
import studio_distributed as runs
import studio_queue
from studio_core import Workspace
from studio_ia.services import WorkspaceServices
from studio_spec import default_spec
from tests import diloco_support as support


class RunSavesVersion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(runs.RUNS.clear)
        self.svc = WorkspaceServices(Workspace(Path(self.tmp.name) / "ws"))
        self.addCleanup(self.svc.close)
        self.version = self.svc.save_version(default_spec(), "first design")
        self.out = Path(self.tmp.name) / "queue_run1"
        self.coordinator = support.build(self.out, run="run1", rounds=2, inner_steps=2, outer_lr=1.)
        self.job = studio_queue.Job(id="run1", kind="pretrain_distributed", label="Version 1 across 2 machines", status="running", params={
            "version": self.version.number, "output_dir": str(self.out),
            "machines": [{"node": "mac", "address": "local"}, {"node": "box", "address": "box:8787"}]})
        self.svc._to(self.version.number, "queued")
        self.svc.runner.jobs.append(self.job)
        runs.open_run(self.job)
        self.svc._record_job("job", {"job": "run1", "status": "running"})

    def train(self, rounds):
        for node in ("mac", "box"):
            self.coordinator.join(node)
        for number in range(rounds):
            for node in list(self.coordinator.current.members):  # whoever the open round waits for (a late joiner starts the next one)
                self.coordinator.submit(number, node, self.coordinator.current.generation, 0, 10, 2, support.constant_delta(self.coordinator), 1., 3.)

    def end(self, status):
        self.job.status = status
        self.svc._record_job("job", {"job": "run1", "status": status})
        return self.svc.store.get(self.version.number)

    def test_the_queued_version_is_training_while_the_run_goes(self):
        self.assertEqual(self.svc.store.get(1)["status"], "training")

    def test_a_finished_run_leaves_a_trained_version_holding_the_final_weights(self):
        self.train(2)
        self.assertTrue(self.coordinator.finished)
        row = self.end("done")
        self.assertEqual((row["status"], row["runs"], len(self.svc.store.versions())), ("trained", [str(self.out)], 1))
        self.assertEqual(Path(row["best_checkpoint"]), self.out / "last.npz")
        with np.load(row["best_checkpoint"], allow_pickle=False) as archive:
            for name, value in self.coordinator.theta.items():
                np.testing.assert_array_equal(archive["weight::" + name], value)
        self.assertIn("distributed run run1 on mac, box", row["change"])
        self.assertEqual(runs.RUNS["run1"].saved, 1)
        self.assertEqual(runs.view(runs.RUNS["run1"], self.coordinator, self.job)["saved_version"], 1)

    def test_a_stopped_run_saves_nothing_even_after_a_round(self):
        self.train(1)
        self.coordinator.stop()
        row = self.end("done")  # the local member ends normally once it is told the run is over
        self.assertEqual((row["status"], row["runs"], row["best_checkpoint"]), ("saved", [], None))
        self.assertIsNone(runs.RUNS["run1"].saved)

    def test_a_failed_run_saves_nothing(self):
        self.train(2)
        row = self.end("failed")
        self.assertEqual((row["status"], row["runs"], row["best_checkpoint"]), ("saved", [], None))


class FinishedAfterFailover(unittest.TestCase):
    """The owner's coordinator stepped down after a round (term 1); a successor on box took over at term 2. The owner's member followed
    it and its job ends done once the run is over there."""
    URL = "https://box.tail.ts.net"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(runs.RUNS.clear)
        self.svc = WorkspaceServices(Workspace(Path(self.tmp.name) / "ws"))
        self.addCleanup(self.svc.close)
        self.version = self.svc.save_version(default_spec(), "first design")
        self.out = Path(self.tmp.name) / "queue_run1"
        first = support.build(self.out, run="run1", rounds=2, inner_steps=2, outer_lr=1.)
        self.train(first, 1)
        first.defer(2)  # it steps down: its rounds are discarded
        self.successor = support.build(Path(self.tmp.name) / "box" / "run1_term2", run="run1", rounds=2, inner_steps=2, outer_lr=1.)
        self.successor.term = 2
        self.job = studio_queue.Job(id="run1", kind="pretrain_distributed", label="Version 1 across 2 machines", status="running", params={
            "version": self.version.number, "output_dir": str(self.out),
            "machines": [{"node": "mac", "address": "local"}, {"node": "box", "address": "box:8787"}]})
        self.svc._to(self.version.number, "queued")
        self.svc.runner.jobs.append(self.job)
        runs.open_run(self.job)
        self.svc._record_job("job", {"job": "run1", "status": "running"})
        where = studio_diloco.where_file(self.svc.root, "run1")
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(json.dumps({"node": "box", "term": 2, "url": self.URL}), encoding="utf-8")
        app = support.app()
        app.state.coordinators = {"run1": self.successor}
        self.box = self.enterContext(TestClient(app))
        self.asked = []
        self.enterContext(mock.patch("studio_peer.PeerClient", self.client_for))

    def client_for(self, url, **options):
        self.asked.append(url)
        return support.Peer(self.box)

    @staticmethod
    def train(coordinator, rounds):
        for node in ("mac", "box"):
            coordinator.join(node)
        for number in range(rounds):
            for node in list(coordinator.current.members):
                coordinator.submit(number, node, coordinator.current.generation, 0, 10, 2, support.constant_delta(coordinator), 1., 3.)

    def end(self, status):
        self.job.status = status
        self.svc._record_job("job", {"job": "run1", "status": status})
        return self.svc.store.get(self.version.number)

    def test_the_owner_saves_the_successor_s_final_weights(self):
        self.train(self.successor, 2)
        self.assertTrue(self.successor.finished)
        row = self.end("done")
        self.assertEqual(self.asked, [self.URL])
        self.assertEqual((row["status"], row["runs"]), ("trained", [str(self.out)]))
        self.assertEqual(Path(row["best_checkpoint"]), self.out / "last_term2.npz")
        with np.load(row["best_checkpoint"], allow_pickle=False) as archive:
            for name, value in self.successor.theta.items():
                np.testing.assert_array_equal(archive["weight::" + name], value)
        self.assertIn("distributed run run1 on mac, box", row["change"])
        self.assertEqual(runs.RUNS["run1"].saved, 1)

    def test_nothing_is_saved_while_the_successor_has_rounds_left(self):
        self.train(self.successor, 1)
        row = self.end("done")
        self.assertEqual((row["status"], row["runs"], row["best_checkpoint"]), ("saved", [], None))
        self.assertFalse((self.out / "last_term2.npz").exists())

    def test_nothing_is_saved_when_the_successor_was_stopped(self):
        self.train(self.successor, 1)
        self.successor.stop()
        row = self.end("done")
        self.assertEqual((row["status"], row["runs"], row["best_checkpoint"]), ("saved", [], None))

    def test_a_failed_owner_job_still_saves_the_run_that_finished_under_the_successor(self):
        self.train(self.successor, 2)
        row = self.end("failed")
        self.assertEqual(self.asked, [self.URL])
        self.assertEqual((row["status"], row["runs"]), ("trained", [str(self.out)]))
        self.assertEqual(Path(row["best_checkpoint"]), self.out / "last_term2.npz")
        self.assertEqual(runs.RUNS["run1"].saved, 1)

    def test_a_stopped_owner_job_still_saves_the_run_that_finished_under_the_successor(self):
        self.train(self.successor, 2)
        row = self.end("stopped")
        self.assertEqual((row["status"], row["runs"]), ("trained", [str(self.out)]))

    def test_a_run_the_person_stopped_saves_nothing_and_asks_no_one(self):
        self.train(self.successor, 2)
        runs.RUNS["run1"].stopped = True
        row = self.end("stopped")
        self.assertEqual((row["status"], row["runs"], row["best_checkpoint"], self.asked), ("saved", [], None, []))

    def test_a_failed_owner_job_saves_nothing_while_the_successor_has_rounds_left(self):
        self.train(self.successor, 1)
        row = self.end("failed")
        self.assertEqual((row["status"], row["runs"], row["best_checkpoint"]), ("saved", [], None))

if __name__ == "__main__":
    unittest.main()

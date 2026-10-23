"""The Phase 6 done line, end to end and window-free (issue #162, docs/cluster.md): a pretraining job queued on node A is placed on
node B, trained there for real (a tiny model on the CPU), published to a shared FolderStore, pulled back onto A, and A then
benchmarks the pulled weights with the real suite code. Two Runners with their own folders, clocks and worker processes, joined by
a fake network (tests/fake_cluster.py) that calls the other's Cluster the way the HTTP routes do. The two real machines are the
manual half of the issue."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import studio_bench
import studio_scheduler as sched
from studio_publish import FolderStore, sha256_of
from studio_queue import Job, Runner
from studio_remote import Cluster
from tests.fake_cluster import Clock, Net, refuse
from tests.test_bench import pieces
from tests.test_publish import EXAMPLES, TINY
from tests.test_queue_cluster import ANY, own_node, payload, wait


def pretrain(job, context, emit, stop, pause):
    """The real trainer on the tiny config: what the worker would run, minus the memory plan."""
    import studio_model
    out = Path(job.params["output_dir"])
    studio_model.train(job.params["config"], job.params["examples"], out, emit=emit, stop=stop, pause=pause)


class Line(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.a_root, self.b_root, self.shared = base / "a", base / "b", base / "shared"
        self.shared.mkdir()
        self.net, self.a_clock, self.b_clock = Net(), Clock(), Clock()
        self.net.handlers.update({"gpu-box": self.runner_side, "owner-a": self.owner_side})
        (self.a_root / "data").mkdir(parents=True)
        (self.a_root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": ["gpu-box"]}}), encoding="utf-8")
        self.a = Runner(self.a_root / "data" / "queue.json", {}, tick=.01, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: FolderStore(self.shared, origin="mac"), own=own_node, clock=self.a_clock,
            address="owner-a", threads=False))
        self.b = Runner(self.b_root / "data" / "queue.json", {"pretrain": pretrain}, tick=.01, prepare=self.prepare, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: FolderStore(self.shared, origin="gpu-box"), own=own_node, clock=self.b_clock,
            threads=False))
        for runner in (self.a, self.b):
            runner.set_plan(ANY)
            self.addCleanup(runner.shutdown)

    def prepare(self, job):
        job.params["output_dir"] = str(self.b_root / "runs" / f"queue_{job.id}")
        return {"root": str(self.b_root), "run_dir": str(self.b_root / "runs")}

    def owner_side(self, method, path, body):
        _, _, job_id, what = path.split("/")
        try:
            if what == "renew":
                lease = self.a.cluster.renewed(job_id, body)
                return {"epoch": lease.epoch, "duration_s": lease.duration_s}
            self.a.cluster.finished(job_id, body)
        except sched.StaleEpoch as exc:
            raise refuse(409, str(exc)) from None

    def runner_side(self, method, path, body):
        if path == "node":
            return payload(busy=self.b.running is not None)
        try:
            if path == "cluster/jobs":
                self.b.cluster.accept(body["job"], sched.Lease.from_dict(body["lease"]), body["owner_address"], body["fetch"])
            else:
                self.b.cluster.stop_guest(path.split("/")[2], body["epoch"])
        except (sched.StaleEpoch, sched.Refused) as exc:
            raise refuse(409, str(exc)) from None

    def test_queued_on_a_trained_on_b_published_pulled_on_a_and_benchmarked_on_a(self):
        # 1. queued on A, a CPU node: a pretrain job needs CUDA, so only B can take it
        job = self.a.enqueue(Job("pretrain", "Train", params={"config": dict(TINY), "examples": EXAMPLES, "version": 1}))
        self.assertEqual(job.status, "waiting")
        # 2. placed on B, which trains it in its own worker process
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertEqual((job.status, job.node, job.epoch), ("running", "gpu-box", 1))
        self.assertTrue(wait(lambda: job.status == "done", 120), f"the job did not finish: {job.status} {job.error}")
        self.assertTrue(wait(lambda: job.params.get("output_dir") and not job.params.get("landing")), "the weights were not brought to A")
        ran = next(j for j in self.b.jobs if j.owner)
        self.assertEqual(ran.status, "done")
        trained = Path(ran.params["output_dir"]) / "best.npz"
        if not trained.is_file():
            trained = trained.with_name("last.npz")
        self.assertTrue(trained.is_file())
        self.assertEqual(list((self.a_root / "runs").rglob("last.npz")), [])  # nothing was trained on A
        # 3. published to the store by B, and pulled onto A where a local run would have left it
        remote = job.params["remote_result"]
        self.assertEqual(remote["node"], "gpu-box")
        self.assertTrue(remote["key"].startswith("gpu-box/"), remote)
        self.assertIn(remote["key"], [e["key"] for e in FolderStore(self.shared).list()])
        landed = Path(job.params["output_dir"]) / "best.npz"
        self.assertTrue(landed.is_file())
        self.assertTrue(str(landed).startswith(str(self.a_root)), landed)
        self.assertNotIn("landing", job.params)
        again = Path(FolderStore(self.shared).pull(remote["key"], self.a_root / "again"))
        self.assertEqual(sha256_of(again), sha256_of(landed))  # what A holds is what the store holds
        with np.load(landed, allow_pickle=False) as pulled, np.load(trained, allow_pickle=False) as original:
            names = [n for n in original.files if n.startswith("weight::")]
            self.assertTrue(names)
            for name in names:  # the store keeps half precision; the pull widens it again
                self.assertEqual(pulled[name].dtype, np.float32)
                np.testing.assert_allclose(pulled[name], original[name], rtol=2e-3, atol=1e-6)
            self.assertEqual([n for n in pulled.files if n.startswith("adam_")], [])
        # 4. a benchmark on A against the pulled weights
        suite = studio_bench.build_suite(**pieces(8))
        used = []

        def run(version, suite, tokens=None, **kwargs):
            used.append(studio_bench._facts(version)[1])
            return studio_bench.run_suite(version, suite, tokens, **kwargs)

        result = studio_bench.benchmark(self.a_root, (1, landed), suite, run=run)
        self.assertEqual(used, [landed])  # scored from the pulled file, on A
        self.assertEqual((result["suite"], result["version"]), (suite["id"], 1))
        self.assertGreater(result["bpc"], 0)
        self.assertIsNotNone(result["few_shot"])
        self.assertIsNotNone(result["problems"])
        self.assertEqual(studio_bench.cached(self.a_root, (1, landed), suite)["bpc"], result["bpc"])  # kept under A's workspace
        self.assertFalse((self.b_root / "data" / "bench").exists())  # and not on B

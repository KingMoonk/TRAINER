"""The queue as a cluster member (studio_queue.Runner + studio_remote.Cluster): a job is dispatched to a peer under a lease, the peer's
progress reaches this node's events, a stale epoch is refused, a lost node's job is placed again from its last checkpoint, and the
peer self-fences. Peers are fakes; the two-nodes-in-one-process tests run real worker processes (no window) over the real version store
(a folder), with each node's own clock so one can be "asleep" while the other moves on.
"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import studio_cluster as sc
import studio_publish
import studio_remote
import studio_scheduler as sched
from studio_peer import PeerError
from studio_publish import NO_STORE, FolderStore, S3Store, StoreLimitError, StoreUsageError, _Store
from studio_queue import Job, Plan, Runner
from studio_remote import Cluster
from tests.fake_cluster import Clock, Net, refuse
from tests.test_publish import CONFIG, StubS3

ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)
GB = 2 ** 30


def wait(condition, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(.01)
    return False


def write_checkpoint(path, step):
    """The smallest file studio_publish accepts as a checkpoint."""
    meta = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "step": step}
    np.savez(path, metadata=np.array(json.dumps(meta)), **{"weight::w": np.arange(4, dtype=np.float32)})  # (the step is in the bytes, so the key)


def trainish(job, context, emit, stop, pause):
    """A pretrain stand-in: logs what it was given, saves last.npz at step 3, waits there while 'hold' exists, then runs to the end."""
    root, out = Path(context.root), Path(job.params["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    steps = job.params["config"]["steps"]
    (root / "ran.json").write_text(json.dumps({"steps": steps, "init_from": job.params.get("init_from"),
                                               "examples": len(job.params.get("examples") or []), "version": job.params.get("version")}))
    for step in range(1, steps + 1):
        emit({"type": "metric", "step": step, "total": steps, "loss": 1.})
        if step == 3:
            write_checkpoint(out / "last.npz", step)
            while (root / "hold").exists() and not stop.is_set():
                time.sleep(.005)
        if stop.is_set():
            write_checkpoint(out / "last.npz", step)
            return
        time.sleep(.002)
    write_checkpoint(out / "best.npz", steps)


def payload(**changes):
    return {"hostname": "gpu-box", "version": "1", "device": "cuda", "device_name": "RTX", "memory_bytes": 24 * GB, "busy": False,
            "dtypes": ["float32", "bfloat16"], **changes}


def own_node():
    return sc.NodeInfo(address=sc.LOCAL, hostname="laptop", device="cpu", memory_bytes=16 * GB, dtypes=sc.DTYPES["cpu"])


class FakeStore:
    """Keeps what was published; `pull` writes a checkpoint at the step recorded for the key."""

    def __init__(self, root):
        self.root, self.published, self.pulled = Path(root), [], []
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, version, path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"There is no checkpoint at {path}.")
        key = f"fake/v{version['number']}-{len(self.published)}"
        self.published.append((key, version, Path(path)))
        return key

    def pull(self, key, dest):
        self.pulled.append(key)
        Path(dest).mkdir(parents=True, exist_ok=True)
        target = Path(dest) / (key.replace("/", "-") + ".npz")
        write_checkpoint(target, 0)
        return target

    def list(self):
        return []


class Peer:
    """A fake peer: answers `GET /node` and records what is dispatched to it."""

    def __init__(self, net, address="gpu-box", **node):
        self.net, self.address, self.node = net, address, node
        self.dispatched, self.stopped, self.refuse = [], [], None
        net.handlers[address] = self.handle

    def handle(self, method, path, body):
        if path == "node":
            return payload(hostname=self.address, **self.node)
        if path == "cluster/jobs":
            if self.refuse:
                raise refuse(409, self.refuse)
            self.dispatched.append(body)
            return {"job_id": body["lease"]["job_id"], "epoch": body["lease"]["epoch"]}
        if path.endswith("/stop"):
            self.stopped.append(body)
            return None
        raise AssertionError(f"unexpected {method} {path}")


class OwnerSide(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock, self.net = Clock(), Net()
        self.peer = Peer(self.net)
        self.store = FakeStore(self.root / "store")
        self.events = []
        self.nodes = ["gpu-box"]
        self.runner = self.make_runner()

    def write_config(self):
        (self.root / "data").mkdir(exist_ok=True)
        (self.root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": self.nodes}}), encoding="utf-8")

    def make_runner(self, store="default"):
        self.write_config()
        store = self.store if store == "default" else store
        runner = Runner(self.root / "data" / "queue.json", {}, tick=.01, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: store, own=own_node, clock=self.clock, address="owner-a", threads=False))
        runner.set_plan(ANY)
        runner.subscribe(lambda kind, event: self.events.append((kind, event)))
        self.addCleanup(runner.shutdown)
        return runner

    def job(self, **params):
        base = {"config": {"steps": 1000, "batch_size": 8}, "examples": [{"text": "hello"}], "version": 3, "max_batch_size": 64}
        return Job("pretrain", "Train", params={**base, **params})

    def place(self):
        self.runner.cluster.poll_once()
        self.runner._place_waiting()

    def sent(self, kind):
        return [e for k, e in self.events if k == kind]

    def test_a_waiting_pretrain_job_is_dispatched_to_the_free_cuda_peer(self):
        job = self.runner.enqueue(self.job())
        self.place()
        self.assertEqual((job.status, job.node, job.epoch), ("running", "gpu-box", 1))
        self.assertEqual(len(self.peer.dispatched), 1)
        self.assertIn({"job": job.id, "status": "running"}, self.sent("job"))
        self.assertEqual(self.sent("cluster")[-1]["state"], "dispatched")
        self.assertEqual(self.runner.state().job, None)

    def test_the_dispatch_carries_the_lease_the_data_and_nothing_that_belongs_to_this_machine(self):
        job = self.runner.enqueue(self.job(output_dir="C:/here/run", pinned=None, user_text="x", previous_best=2))
        self.place()
        body = self.peer.dispatched[0]
        self.assertEqual(body["lease"], {"job_id": job.id, "node": "gpu-box", "epoch": 1, "duration_s": sched.LEASE_S})
        self.assertEqual(body["owner_address"], "owner-a")
        spec = body["job"]
        self.assertEqual((spec["id"], spec["kind"]), (job.id, "pretrain"))
        self.assertEqual(spec["params"]["examples"], [{"text": "hello"}])
        self.assertEqual(spec["params"]["config"]["steps"], 1000)
        for local in ("version", "output_dir", "max_batch_size", "user_text", "previous_best", "pinned", "cluster"):
            self.assertNotIn(local, spec["params"])
        self.assertEqual(body["fetch"], {})

    def test_a_version_the_job_starts_from_is_published_and_handed_over_as_a_key(self):
        start = self.root / "start.npz"
        write_checkpoint(start, 10)
        self.runner.enqueue(self.job(mode="catch_up", init_from=str(start)))
        self.place()
        body = self.peer.dispatched[0]
        self.assertEqual(body["fetch"], {"init_from": self.store.published[0][0]})
        self.assertNotIn("init_from", body["job"]["params"])
        self.assertEqual(self.store.published[0][2], start)

    def test_a_job_that_cannot_be_packaged_waits_with_the_reason(self):
        job = self.runner.enqueue(self.job(mode="catch_up", init_from=str(self.root / "gone.npz")))
        self.place()
        self.assertEqual(job.status, "waiting")
        self.assertIn("There is no checkpoint", self.runner.cluster.waiting[job.id])
        self.assertEqual(self.peer.dispatched, [])

    def test_no_peers_listed_means_everything_stays_here(self):
        self.nodes = []
        self.runner = self.make_runner()
        job = self.runner.enqueue(self.job())
        self.place()
        self.assertEqual(self.net.calls, [])
        self.assertEqual((job.status, self.runner._next_local()), ("waiting", job))

    def test_a_busy_peer_gets_nothing_and_the_job_runs_here(self):
        self.peer.node["busy"] = True
        job = self.runner.enqueue(self.job())
        self.place()
        self.assertEqual((job.status, self.runner._next_local()), ("waiting", job))

    def test_a_paused_queue_dispatches_nothing(self):
        self.runner.enqueue(self.job())
        self.runner.pause()
        self.place()
        self.assertEqual(self.peer.dispatched, [])

    def test_a_job_other_than_pretraining_stays_here(self):
        job = self.runner.enqueue(Job("compare", "Compare", params={"experiment": {}}))
        self.place()
        self.assertEqual((job.status, self.peer.dispatched), ("waiting", []))
        self.assertEqual(self.runner._next_local(), job)

    def test_renewals_put_the_peers_progress_into_this_nodes_events(self):
        job = self.runner.enqueue(self.job())
        self.place()
        events = [{"type": "metric", "step": 5, "total": 1000, "loss": 2.}, {"type": "viz_snapshot", "step": 5, "path": "/elsewhere"}]
        lease = self.runner.cluster.renewed(job.id, {"epoch": 1, "state": "running", "events": events})
        self.assertEqual((lease.epoch, lease.node), (1, "gpu-box"))
        self.assertEqual(self.sent("train"), [{"job": job.id, "type": "metric", "step": 5, "total": 1000, "loss": 2.}])
        self.assertEqual(self.sent("cluster")[-1]["state"], "running")

    def test_a_renewal_with_another_epoch_is_refused(self):
        job = self.runner.enqueue(self.job())
        self.place()
        with self.assertRaises(sched.StaleEpoch):
            self.runner.cluster.renewed(job.id, {"epoch": 2, "events": [{"type": "metric", "step": 1}]})
        self.assertEqual(self.sent("train"), [])
        with self.assertRaises(sched.StaleEpoch):
            self.runner.cluster.renewed("nobody", {"epoch": 1})

    def test_a_finished_job_records_the_version_it_published(self):
        job = self.runner.enqueue(self.job())
        self.place()
        with self.assertRaises(sched.StaleEpoch):
            self.runner.cluster.finished(job.id, {"epoch": 9, "status": "done", "key": "k"})
        self.assertEqual(job.status, "running")
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "done", "key": "gpu-box/v0-abc", "step": 1000})
        self.assertEqual(job.status, "done")
        self.assertEqual(job.params["remote_result"], {"key": "gpu-box/v0-abc", "node": "gpu-box", "step": 1000})
        self.assertEqual(self.sent("job")[-1], {"job": job.id, "status": "done", "error": ""})
        with self.assertRaises(sched.StaleEpoch):  # and a second report is refused
            self.runner.cluster.finished(job.id, {"epoch": 1, "status": "done", "key": "k"})

    def test_a_failed_job_keeps_the_peers_sentence(self):
        job = self.runner.enqueue(self.job())
        self.place()
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "failed", "error": "The dataset is empty."})
        self.assertEqual((job.status, job.error), ("failed", "The dataset is empty."))

    def interrupted(self, step=None):
        job = self.runner.enqueue(self.job())
        self.place()
        if step:
            self.runner.cluster.renewed(job.id, {"epoch": 1, "checkpoint": {"key": "gpu-box/v0-aaa", "step": step}})
        self.net.down.add("gpu-box")
        cluster = self.runner.cluster
        self.clock.advance(31)
        cluster.tick()
        self.assertEqual(self.sent("cluster")[-1]["state"], "lapsed")
        self.assertEqual(job.status, "running")  # not yet: the runner may only be slow to renew
        self.clock.advance(30)
        cluster.tick()
        return job

    def test_a_lost_node_is_waited_out_then_its_job_goes_back_to_waiting_from_its_checkpoint(self):
        job = self.interrupted(step=400)
        self.assertEqual((job.status, job.node), ("waiting", ""))
        self.assertEqual(self.sent("cluster")[-1]["state"], "interrupted")
        self.assertEqual((job.params["mode"], job.params["fetch"], job.params["config"]["steps"]),
                         ("catch_up", {"init_from": "gpu-box/v0-aaa"}, 600))
        self.assertIn("Interrupted on gpu-box at step 400", job.note)
        self.assertIn("optimizer starts again from zero", job.note)
        with self.assertRaises(sched.StaleEpoch):  # the node's old copy, if it comes back, is refused
            self.runner.cluster.renewed(job.id, {"epoch": 1})

    def test_the_job_is_placed_again_when_the_node_returns_at_the_next_epoch_resuming_from_the_key(self):
        job = self.interrupted(step=400)
        self.net.down.clear()
        self.place()
        self.assertEqual((job.status, job.node, job.epoch), ("running", "gpu-box", 2))
        body = self.peer.dispatched[-1]
        self.assertEqual((body["lease"]["epoch"], body["fetch"], body["job"]["params"]["config"]["steps"]),
                         (2, {"init_from": "gpu-box/v0-aaa"}, 600))
        self.assertEqual(body["job"]["params"]["mode"], "catch_up")
        self.runner.cluster.renewed(job.id, {"epoch": 2, "events": [{"type": "metric", "step": 10, "total": 600}]})
        self.assertEqual(self.sent("train")[-1]["step"], 410)  # progress goes on from 400, it does not start over
        self.assertEqual(self.sent("train")[-1]["total"], 1000)

    def test_another_node_takes_the_job_when_the_first_stays_away(self):
        other = Peer(self.net, "other-box")
        self.nodes = ["gpu-box", "other-box"]
        self.runner = self.make_runner()
        job = self.runner.enqueue(self.job())
        self.place()
        self.assertEqual(job.node, "gpu-box")
        self.runner.cluster.renewed(job.id, {"epoch": 1, "checkpoint": {"key": "gpu-box/v0-aaa", "step": 100}})
        self.net.down.add("gpu-box")
        for _ in range(7):  # polls keep failing for a minute
            self.clock.advance(10)
            self.runner.cluster.poll_once()
            self.runner.cluster.tick()
        self.assertEqual(job.status, "waiting")
        self.runner._place_waiting()
        self.assertEqual((job.status, job.node, job.epoch), ("running", "other-box", 2))
        self.assertEqual(other.dispatched[0]["fetch"], {"init_from": "gpu-box/v0-aaa"})

    def test_without_a_checkpoint_it_starts_again_and_says_so(self):
        job = self.interrupted()
        self.assertEqual(job.status, "waiting")
        self.assertNotIn("fetch", job.params)
        self.assertEqual(job.params["config"]["steps"], 1000)
        self.assertIn("before its first checkpoint", job.note)

    def test_a_job_whose_last_checkpoint_was_its_last_step_is_done_not_run_again(self):
        job = self.interrupted(step=1000)
        self.assertEqual(job.status, "done")
        self.assertEqual(job.params["remote_result"]["key"], "gpu-box/v0-aaa")

    # ---- the checkpoints a job leaves in the bucket ------------------------------------------------------------------
    def use_bucket(self):
        """A bucket (the fake S3 client) as the store; returns the store. Its `client.log` says every call the store made."""
        self.client = StubS3()
        self.bucket = S3Store(CONFIG, client=self.client, origin="gpu-box")
        self.runner = self.make_runner(store=self.bucket)
        return self.bucket

    def report_checkpoint(self, job, step, epoch=1):
        """The runner publishes a checkpoint of `job` and tells the owner on its next renewal; returns the key."""
        path = self.root / f"checkpoint-{job.id}-{epoch}-{step}.npz"
        write_checkpoint(path, step)
        key = self.bucket.publish(studio_publish.checkpoint_version(job.id, step), path)
        self.runner.cluster.renewed(job.id, {"epoch": epoch, "checkpoint": {"key": key, "step": step}})
        return key

    def publish_result(self, job, step=1000):
        path = self.root / f"result-{job.id}.npz"
        write_checkpoint(path, step)
        return self.bucket.publish({"number": 0, "name": f"job:{job.id}", "change": f"result at step {step}"}, path)

    def deleted(self):
        return [c[2].rsplit("/", 1)[-1] for c in self.client.log if c[0] == "delete_object"]

    def held(self, *keys):
        return all(self.bucket.exists(key) for key in keys)

    def lose_the_runner(self, job):
        """The runner stops answering; the owner waits two leases and puts the job back to waiting."""
        self.net.down.add("gpu-box")
        for _ in range(7):
            self.clock.advance(10)
            self.runner.cluster.poll_once()
            self.runner.cluster.tick()
        self.assertEqual(job.status, "waiting")

    def test_a_job_that_finishes_on_another_node_after_its_runner_was_lost_leaves_nothing_but_its_result(self):
        Peer(self.net, "other-box")
        self.nodes = ["gpu-box", "other-box"]
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        self.assertEqual(job.node, "gpu-box")
        old = self.report_checkpoint(job, 100)
        self.lose_the_runner(job)
        self.runner.cluster.tick()
        self.assertTrue(self.held(old), "the job waits to be placed again: it resumes from this checkpoint")
        self.assertEqual(self.deleted(), [])
        self.runner._place_waiting()
        self.assertEqual((job.status, job.node, job.epoch), ("running", "other-box", 2))
        self.runner.cluster.tick()
        self.assertTrue(self.held(old))  # the new runner has only just pulled it
        newer = self.report_checkpoint(job, 200, epoch=2)
        self.runner.cluster.tick()
        self.assertFalse(self.held(old))  # a newer checkpoint is what a resume would use: this one never will be
        self.assertTrue(self.held(newer))
        result = self.publish_result(job)
        self.runner.cluster.finished(job.id, {"epoch": 2, "status": "done", "key": result, "step": 800})
        self.runner.cluster.tick()
        self.assertEqual(job.status, "done")
        self.assertEqual([e["key"] for e in self.bucket.list()], [result])
        self.assertEqual(job.params["cluster"]["checkpoints"], [])

    def test_a_cancelled_job_loses_its_checkpoints(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        first, second = self.report_checkpoint(job, 100), self.report_checkpoint(job, 200)
        self.runner.cluster.tick()
        self.assertEqual((self.held(first), self.held(second)), (False, True))
        self.runner.remove(job.id)
        self.assertEqual(job.status, "stopped")
        self.runner.cluster.tick()
        self.assertEqual(self.bucket.list(), [])
        self.assertEqual(job.params["cluster"]["checkpoints"], [])

    def test_a_job_that_failed_for_good_loses_its_checkpoints(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        key = self.report_checkpoint(job, 100)
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "failed", "error": "The dataset is empty."})
        self.runner.cluster.tick()
        self.assertFalse(self.held(key))

    def test_an_interrupted_job_waiting_to_be_placed_keeps_the_checkpoint_it_resumes_from(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        key = self.report_checkpoint(job, 400)
        self.lose_the_runner(job)
        for _ in range(3):
            self.clock.advance(120)
            self.runner.cluster.tick()
        self.assertEqual(job.params["fetch"], {"init_from": key})
        self.assertTrue(self.held(key))
        self.assertEqual((self.deleted(), job.params["cluster"]["checkpoints"]), ([], [key]))

    def test_a_waiting_job_taken_out_of_the_queue_loses_its_checkpoints_too(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        key = self.report_checkpoint(job, 400)
        self.lose_the_runner(job)
        self.runner.remove(job.id)
        self.assertNotIn(job, self.runner.jobs)
        self.runner.cluster.tick()
        self.assertFalse(self.held(key))
        self.assertEqual(self.runner.cluster._removed, {})

    def test_a_job_with_no_checkpoint_costs_the_bucket_nothing_to_look_at(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "failed", "error": "x"})
        self.runner.cluster.tick()
        self.assertEqual(self.client.log, [])

    def test_a_delete_that_fails_is_logged_kept_and_tried_again_later(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        key = self.report_checkpoint(job, 100)
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "failed", "error": "x"})
        with mock.patch.object(S3Store, "_delete", side_effect=OSError("share went away")) as delete, \
                self.assertLogs("studio_remote", "WARNING") as logged:
            self.runner.cluster.tick()  # (the job's own end is not held up by it)
            self.assertTrue(any("Could not delete checkpoint" in line for line in logged.output))
            self.assertEqual(job.status, "failed")
            self.runner.cluster.tick()
            self.assertEqual(delete.call_count, 1)  # not again straight away
        self.assertTrue(self.held(key))
        self.assertEqual(job.params["cluster"]["checkpoints"], [key])
        self.clock.advance(studio_remote.SWEEP_RETRY_S + 1)
        self.runner.cluster.tick()
        self.assertFalse(self.held(key))
        self.assertEqual(job.params["cluster"]["checkpoints"], [])

    def test_the_checkpoints_still_to_delete_survive_a_restart_of_the_app(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        key = self.report_checkpoint(job, 100)
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "done", "key": self.publish_result(job), "step": 1})
        self.runner.shutdown()
        self.runner = self.make_runner(store=self.bucket)
        self.runner.cluster.tick()
        self.assertFalse(self.held(key))

    def test_only_a_key_the_runner_reported_for_that_job_and_nobody_needs_is_deleted(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        other = self.runner.enqueue(self.job())
        self.place()
        self.runner._place_waiting()
        live = self.report_checkpoint(other, 50)  # the other job's, and it is still running
        mine = self.report_checkpoint(job, 100)
        stray = self.bucket.publish(studio_publish.checkpoint_version(job.id, 7), self.make_checkpoint(7))  # published, never reported
        theirs = self.bucket.publish({"number": 4, "name": "Version 4", "change": "a version"}, self.make_checkpoint(8))
        shared = self.bucket.publish(studio_publish.checkpoint_version("someone", 9), self.make_checkpoint(9))
        result = self.publish_result(job)
        with self.runner.lock:  # records that name keys that are not this job's checkpoints
            job.params["cluster"]["checkpoints"] += [theirs, shared, result, live]
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "done", "key": result, "step": 5})
        with self.assertLogs("studio_remote", "WARNING") as logged:
            self.runner.cluster.tick()
        self.assertEqual(sum("Not deleting" in line for line in logged.output), 2)  # a version, and another job's checkpoint
        self.assertFalse(self.held(mine))
        self.assertTrue(self.held(stray, theirs, shared, result, live))  # a version, another job's, the result, a live job's
        self.assertEqual(sorted(self.deleted()), sorted(os.path.basename(mine) + e for e in (".json", ".npz")))

    def make_checkpoint(self, step):
        path = self.root / f"made-{step}.npz"
        write_checkpoint(path, step)
        return path

    def test_deleting_never_uploads(self):
        self.use_bucket()
        job = self.runner.enqueue(self.job())
        self.place()
        self.report_checkpoint(job, 100)
        self.report_checkpoint(job, 200)
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "failed", "error": "x"})
        before = len(self.client.log)
        self.runner.cluster.tick()
        calls = {c[0] for c in self.client.log[before:]}
        self.assertTrue(calls <= {"delete_object", "get_object", "head_object"}, calls)
        self.assertIn("delete_object", calls)

    def test_a_pinned_job_waits_for_its_node_and_is_never_moved_to_another(self):
        Peer(self.net, "other-box")
        self.nodes = ["gpu-box", "other-box"]
        self.runner = self.make_runner()
        self.net.down.add("gpu-box")
        job = self.runner.enqueue(self.job(pinned="gpu-box"))
        self.place()
        self.assertEqual(job.status, "waiting")
        self.assertIsNone(self.runner._next_local())
        self.assertIn("pinned to gpu-box, which is not answering", self.runner.cluster.waiting[job.id])
        self.net.down.clear()
        self.place()
        self.assertEqual((job.status, job.node), ("running", "gpu-box"))

    def test_a_job_pinned_to_this_machine_never_leaves_it(self):
        job = self.runner.enqueue(self.job(pinned="laptop"))
        self.place()
        self.assertEqual((job.status, self.peer.dispatched), ("waiting", []))
        self.assertEqual(self.runner._next_local(), job)

    def test_a_waiting_job_does_not_hold_back_the_ones_behind_it(self):
        self.net.down.add("gpu-box")
        pinned = self.runner.enqueue(self.job(pinned="gpu-box"))
        free = self.runner.enqueue(Job("compare", "Compare"))
        self.place()
        self.assertEqual(self.runner._next_local(), free)
        self.assertEqual(pinned.status, "waiting")

    def test_without_a_version_store_nothing_leaves_and_a_pinned_job_says_what_to_set_up(self):
        self.runner = self.make_runner(store=None)
        job = self.runner.enqueue(self.job(pinned="gpu-box"))
        free = self.runner.enqueue(self.job())
        self.place()
        self.assertEqual(self.peer.dispatched, [])
        self.assertIn("version store", self.runner.cluster.waiting[job.id])
        self.assertEqual(self.runner._next_local(), free)
        self.assertIn("TRAINER_STORE_FOLDER", self.runner.cluster.store_problem)

    def use_workspace_store(self, environ=None):
        """The cluster finds its store the way the routes do: the workspace's folder, then the environment (no store is handed in)."""
        self.runner.cluster._store_factory = self.runner.cluster.workspace_store
        self.runner.cluster.environ = environ or {}

    def write_store_folder(self, folder):
        (self.root / "data" / "workspace.json").write_text(
            json.dumps({"studioConfig": {"nodes": self.nodes, "store": {"folder": str(folder)}}}), encoding="utf-8")

    def test_with_neither_the_workspace_folder_nor_the_environment_the_refusal_is_the_routes_sentence(self):
        self.use_workspace_store()
        job = self.runner.enqueue(self.job())
        self.runner.cluster.poll_once()
        self.assertFalse(self.runner.cluster.dispatch(job, "gpu-box"))  # placement would not send it here; dispatch refuses regardless
        self.assertEqual(self.peer.dispatched, [])
        self.assertEqual(self.runner.cluster.store_problem, NO_STORE)
        self.assertIn(NO_STORE, self.runner.cluster.waiting[job.id])
        self.assertEqual(job.status, "waiting")

    def test_the_workspace_folder_is_the_store_before_the_environment_one(self):
        self.write_store_folder(self.root / "workspace-store")
        self.use_workspace_store({"TRAINER_STORE_FOLDER": str(self.root / "env-store")})
        self.assertEqual(self.runner.cluster.store().root, self.root / "workspace-store")

    def test_a_folder_set_in_the_workspace_while_the_app_runs_is_used_without_a_restart(self):
        self.use_workspace_store()
        self.assertIsNone(self.runner.cluster.store())
        self.assertEqual(self.runner.cluster.store_problem, NO_STORE)
        self.write_store_folder(self.root / "shared")
        os.utime(self.root / "data" / "workspace.json", ns=(2 * 10 ** 9, 2 * 10 ** 9))  # the file changed: a coarse clock may not show it
        self.assertEqual(self.runner.cluster.store().root, self.root / "shared")
        self.assertEqual(self.runner.cluster.store_problem, "")

    def test_a_peer_that_refuses_leaves_the_job_waiting_with_its_sentence_and_is_not_asked_again_at_once(self):
        self.peer.refuse = "This machine has no version store."
        job = self.runner.enqueue(self.job())
        self.place()
        self.assertEqual(job.status, "waiting")
        self.assertIn("This machine has no version store.", self.runner.cluster.waiting[job.id])
        self.assertEqual(self.sent("cluster")[-1]["state"], "refused")
        self.assertEqual(self.runner._next_local(), job)  # it may run here meanwhile
        self.peer.refuse = None
        self.place()
        self.assertEqual(self.peer.dispatched, [])
        self.clock.advance(sched.REFUSAL_COOLDOWN + 1)
        self.place()
        self.assertEqual(len(self.peer.dispatched), 1)

    def test_a_peer_that_does_not_answer_the_dispatch_may_still_be_running_it(self):
        def silent(method, path, body):
            if path == "node":
                return payload()
            raise PeerError("did not answer")
        self.net.handlers["gpu-box"] = silent
        job = self.runner.enqueue(self.job())
        self.place()
        self.assertEqual((job.status, job.node), ("running", "gpu-box"))
        self.assertEqual(self.sent("cluster")[-1]["state"], "unconfirmed")

    def test_a_restart_places_a_remote_job_again_at_the_next_epoch(self):
        job = self.runner.enqueue(self.job())
        self.place()
        again = self.make_runner()
        restored = again.jobs[0]
        self.assertEqual((restored.status, restored.node, restored.epoch), ("waiting", "", 1))
        again.cluster.poll_once()
        again._place_waiting()
        self.assertEqual((restored.epoch, restored.node), (2, "gpu-box"))
        self.assertEqual(self.peer.dispatched[-1]["lease"]["epoch"], 2)
        self.assertEqual(job.id, restored.id)

    def test_removing_a_job_that_runs_on_a_peer_stops_it_there(self):
        job = self.runner.enqueue(self.job())
        self.place()
        self.runner.remove(job.id)
        self.assertEqual(self.peer.stopped, [{"epoch": 1}])
        self.assertEqual(job.status, "stopped")
        self.assertIn(job, self.runner.jobs)
        with self.assertRaises(sched.StaleEpoch):
            self.runner.cluster.renewed(job.id, {"epoch": 1})

    def test_closing_the_app_stops_the_peers_copy_and_the_job_waits_for_the_next_start(self):
        job = self.runner.enqueue(self.job())
        self.place()
        self.runner.shutdown()
        self.assertEqual(self.peer.stopped, [{"epoch": 1}])
        self.assertEqual((job.status, job.node), ("waiting", ""))

    def test_the_cluster_view_lists_the_nodes_the_jobs_and_the_waits(self):
        job = self.runner.enqueue(self.job())
        waiting = self.runner.enqueue(self.job(pinned="ghost"))
        self.place()
        self.runner.cluster.renewed(job.id, {"epoch": 1, "checkpoint": {"key": "k", "step": 7}})
        self.assertEqual(self.runner.cluster.health.state("gpu-box"), sched.UP)
        rows = {r["id"]: r for r in self.runner.cluster.jobs_view()}
        self.assertEqual((rows[job.id]["role"], rows[job.id]["node"], rows[job.id]["lease"], rows[job.id]["checkpoint_step"]),
                         ("owner", "gpu-box", "held", 7))
        self.assertIn("ghost", rows[waiting.id]["waiting_reason"])

    def test_a_node_listed_in_a_broken_config_is_a_plain_note_not_a_crash(self):
        (self.root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": "gpu-box"}}), encoding="utf-8")
        self.assertEqual(self.runner.cluster.peers(), [])
        self.assertIn("studioConfig.nodes", self.runner.cluster.config_problem)

    def test_the_failure_detector_decides_when_a_node_stops_getting_work(self):
        self.runner.cluster.poll_once()
        self.net.down.add("gpu-box")
        self.clock.advance(5)
        self.runner.cluster.poll_once()
        self.assertEqual(self.runner.cluster.health.state("gpu-box"), sched.SUSPECT)
        self.clock.advance(30)
        self.runner.cluster.poll_once()
        self.assertEqual(self.runner.cluster.health.state("gpu-box"), sched.DOWN)
        self.assertIn("did not answer", self.runner.cluster.nodes()[1].error)


class TwoNodes(unittest.TestCase):
    """An owner and a runner, each a Runner with its own folder, clock and worker processes, joined through a fake network that calls
    the other's Cluster the way the routes do, and sharing a real folder as the version store."""

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
            r, client_for=self.net.client_for, store=lambda: FolderStore(self.shared), own=own_node, clock=self.a_clock,
            address="owner-a", threads=False))
        self.b = Runner(self.b_root / "data" / "queue.json", {"pretrain": trainish}, tick=.01, prepare=self.prepare, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: FolderStore(self.shared, origin="gpu-box"), own=own_node, clock=self.b_clock,
            threads=False))
        for runner in (self.a, self.b):
            runner.set_plan(ANY)
            self.addCleanup(runner.shutdown)
        self.a_events = []
        self.a.subscribe(lambda kind, event: self.a_events.append((kind, event)))

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

    def job(self, steps, **params):
        return Job("pretrain", "Train", params={"config": {"steps": steps}, "examples": [{"text": "a"}, {"text": "b"}], "version": 4, **params})

    def b_job(self):
        return next((j for j in self.b.jobs if j.owner), None)

    def checkpointed(self):
        job = self.b_job()
        return bool(job and job.params.get("output_dir") and (Path(job.params["output_dir"]) / "last.npz").exists())

    def ran(self):
        return json.loads((self.b_root / "ran.json").read_text())

    def beat(self, seconds=10.):
        """A third of a lease passes on the runner and everything it owes the owner is sent."""
        self.b_clock.advance(seconds)
        self.b.cluster.tick()

    def test_a_job_runs_on_the_peer_and_comes_back_as_a_published_version(self):
        start = self.a_root / "start.npz"
        write_checkpoint(start, 50)
        job = self.a.enqueue(self.job(8, mode="catch_up", init_from=str(start)))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertEqual((job.status, job.node, job.epoch), ("running", "gpu-box", 1))
        self.assertTrue(wait(lambda: self.b_job() and self.b_job().status == "done"), "the job did not finish on the peer")
        self.assertTrue(wait(lambda: job.status == "done"), "the owner was not told")
        ran = self.ran()
        self.assertEqual((ran["steps"], ran["examples"], ran["version"]), (8, 2, None))
        self.assertTrue(Path(ran["init_from"]).is_file())  # the start was pulled from the store, the owner's path was not used
        self.assertNotIn(str(self.a_root), ran["init_from"])
        key = job.params["remote_result"]["key"]
        self.assertTrue(key.startswith("gpu-box/"), key)
        self.assertEqual(job.params["remote_result"]["step"], 8)
        self.assertIn(key, [e["key"] for e in FolderStore(self.shared).list()])  # in the store, for the owner to list and pull
        self.assertEqual(self.a.cluster.owner.leases.get(job.id), None)

    def test_progress_checkpoints_and_the_end_reach_the_owner(self):
        (self.b_root).mkdir(parents=True, exist_ok=True)
        (self.b_root / "hold").touch()
        job = self.a.enqueue(self.job(1000))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertTrue(wait(lambda: self.checkpointed()))
        run = self.b.cluster.guests[job.id]
        self.b.cluster.tick()  # publishes the checkpoint
        self.assertTrue(wait(lambda: run.checkpoint is not None), "the checkpoint was not published")
        self.beat()
        steps = [e["step"] for k, e in self.a_events if k == "train" and e.get("job") == job.id and e.get("type") == "metric"]
        self.assertTrue(steps and max(steps) >= 3, steps)
        notes = job.params["cluster"]
        self.assertEqual(notes["checkpoint"]["step"], 3)
        self.assertTrue(notes["checkpoint"]["key"].startswith("gpu-box/"))
        rows = {r["id"]: r for r in self.b.cluster.jobs_view()}
        self.assertEqual((rows[job.id]["role"], rows[job.id]["node"], rows[job.id]["epoch"]), ("runner", "owner-a", 1))
        (self.b_root / "hold").unlink()
        self.assertTrue(wait(lambda: job.status == "done"))

    def test_a_checkpoint_the_store_refuses_is_reported_to_the_owner_and_the_job_keeps_training(self):
        (self.b_root).mkdir(parents=True, exist_ok=True)
        (self.b_root / "hold").touch()
        job = self.a.enqueue(self.job(1000))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertTrue(wait(lambda: self.checkpointed()))
        run = self.b.cluster.guests[job.id]
        full = StoreLimitError("Not uploaded: this upload needs 1.00 MB and the bucket already holds 10.00 GB, which would make 10.00 GB, "
                               "over the 10.00 GB limit (the R2 free tier).")
        with mock.patch.object(_Store, "_admit", side_effect=full) as admit:
            self.b.cluster.tick()
            self.assertTrue(wait(lambda: run.refusal and not run.publishing), "the refusal was not recorded")
            self.b.cluster.tick()
            self.b.cluster.tick()  # the same checkpoint is not offered again: it would be refused again, a listing of the bucket each time
            self.assertEqual(admit.call_count, 1)
        self.assertIsNone(run.checkpoint)
        self.assertEqual(self.b_job().status, "running")  # training goes on; only the sharing of this checkpoint is lost
        self.assertIn("Checkpoint at step 3 not shared: Not uploaded", run.refusal)
        self.assertIn(run.refusal, {r["id"]: r for r in self.b.cluster.jobs_view()}[job.id]["waiting_reason"])
        self.beat()
        self.assertIn(run.refusal, job.note)  # the owner's job says it, where the queue page shows it
        self.assertIn(run.refusal, job.params["cluster"]["problem"])
        self.assertIn(("cluster", {"job": job.id, "state": "checkpoint_refused", "node": "gpu-box", "detail": run.refusal}), self.a_events)
        self.beat()
        self.assertEqual(job.note.count("not shared"), 1)  # said once
        run.refused_mtime = 0.  # room again (versions deleted): the next attempt goes through and the problem clears
        self.b.cluster.tick()
        self.assertTrue(wait(lambda: run.checkpoint is not None), "the checkpoint was not published once there was room")
        self.assertEqual(run.refusal, "")
        self.beat()
        self.assertNotIn("problem", job.params["cluster"])
        (self.b_root / "hold").unlink()
        self.assertTrue(wait(lambda: job.status == "done"))

    # ---- what the runner leaves in the store ------------------------------------------------------------------------
    def hold_at_checkpoint(self, steps=1000):
        """Start a job on the runner and wait at its first checkpoint (step 3, held there). Returns (job, run, the store)."""
        (self.b_root).mkdir(parents=True, exist_ok=True)
        (self.b_root / "hold").touch()
        job = self.a.enqueue(self.job(steps))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertTrue(wait(lambda: self.checkpointed()))
        return job, self.b.cluster.guests[job.id], FolderStore(self.shared, origin="gpu-box")

    def publish_checkpoint(self, run, step=None):
        """Let the runner publish the newest last.npz (a newer one, at `step`, when given); returns its key."""
        last = Path(self.b_job().params["output_dir"]) / "last.npz"
        if step is not None:
            write_checkpoint(last, step)
            os.utime(last, (time.time() + 100 * step, time.time() + 100 * step))
        before = run.checkpoint
        self.b.cluster.tick()
        self.assertTrue(wait(lambda: run.checkpoint is not before and not run.publishing), "the checkpoint was not published")
        return run.checkpoint["key"]

    def test_a_new_checkpoint_replaces_the_previous_one_once_the_owner_has_the_new_one(self):
        job, run, store = self.hold_at_checkpoint()
        first = self.publish_checkpoint(run)
        self.beat()
        self.assertEqual(job.params["cluster"]["checkpoint"]["key"], first)
        second = self.publish_checkpoint(run, step=4)
        self.assertNotEqual(first, second)
        self.assertTrue(store.exists(first) and store.exists(second))  # the owner could still be about to resume from the first
        self.beat()  # the renewal that tells the owner about the second
        self.assertEqual(job.params["cluster"]["checkpoint"]["key"], second)
        self.assertTrue(wait(lambda: not store.exists(first)), "the previous checkpoint was not deleted")
        self.assertTrue(store.exists(second))
        self.assertEqual(run.stored, [second])
        self.assertEqual(sorted(e["key"] for e in store.list()), [second])  # the weights and the listing both went
        (self.b_root / "hold").unlink()
        self.assertTrue(wait(lambda: job.status == "done"))

    def test_the_result_deletes_every_checkpoint_of_the_job_and_stays_itself(self):
        job, run, store = self.hold_at_checkpoint(steps=8)
        first = self.publish_checkpoint(run)
        self.beat()
        second = self.publish_checkpoint(run, step=4)
        self.assertTrue(store.exists(first) and store.exists(second))  # (nothing acknowledged the second yet)
        (self.b_root / "hold").unlink()
        self.assertTrue(wait(lambda: job.status == "done"), "the owner was not told")
        result = job.params["remote_result"]["key"]
        self.assertTrue(wait(lambda: not store.exists(first) and not store.exists(second)), "the checkpoints were not deleted")
        self.assertTrue(store.exists(result))
        self.assertEqual([e["key"] for e in store.list()], [result])

    def test_a_result_with_the_bytes_of_a_checkpoint_is_not_deleted_with_it(self):
        job, run, store = self.hold_at_checkpoint(steps=3)  # best.npz at step 3 is last.npz at step 3
        checkpoint = self.publish_checkpoint(run)
        (self.b_root / "hold").unlink()
        self.assertTrue(wait(lambda: job.status == "done"))
        self.assertEqual(job.params["remote_result"]["key"], checkpoint)
        self.assertTrue(wait(lambda: run.ended))
        time.sleep(.2)
        self.assertTrue(store.exists(checkpoint))

    def test_only_what_the_run_put_in_the_store_is_ever_deleted(self):
        job, run, store = self.hold_at_checkpoint(steps=8)
        last = Path(self.b_job().params["output_dir"]) / "last.npz"
        theirs = store.publish({"number": 0, "name": "someone else's"}, last)  # the same bytes were in the store before the run got there
        key = self.publish_checkpoint(run)
        self.assertEqual(key, theirs)
        self.assertEqual(run.stored, [])  # not created by this run: not its to delete
        self.beat()
        later = self.publish_checkpoint(run, step=4)
        self.beat()
        self.assertEqual(run.stored, [later])
        (self.b_root / "hold").unlink()
        self.assertTrue(wait(lambda: job.status == "done"))
        self.assertTrue(wait(lambda: not store.exists(later)))
        self.assertTrue(store.exists(theirs))

    def test_a_delete_that_fails_is_logged_and_the_job_still_finishes(self):
        job, run, store = self.hold_at_checkpoint(steps=8)
        first = self.publish_checkpoint(run)
        self.beat()
        with mock.patch.object(_Store, "delete", side_effect=OSError("share went away")), self.assertLogs("studio_remote", "WARNING") as logged:
            second = self.publish_checkpoint(run, step=4)
            self.beat()
            self.assertTrue(wait(lambda: any("Could not delete" in line for line in logged.output)))
            self.assertTrue(wait(lambda: run.stored == [first, second]), run.stored)  # kept, to be tried again
        (self.b_root / "hold").unlink()
        self.assertTrue(wait(lambda: job.status == "done"))
        self.assertTrue(wait(lambda: not store.exists(first) and not store.exists(second)))  # the result's cleanup took them

    def test_a_result_the_store_refuses_is_retried_once_after_the_jobs_checkpoints_are_deleted(self):
        job, run, store = self.hold_at_checkpoint(steps=8)
        checkpoint = self.publish_checkpoint(run)
        real = _Store.publish

        def full_until_room(self_, version, path):
            if store.exists(checkpoint):
                raise StoreLimitError("Not uploaded: this upload needs 1.00 MB and the bucket already holds 9.50 GB.")
            return real(self_, version, path)

        with mock.patch.object(_Store, "publish", autospec=True, side_effect=full_until_room):
            (self.b_root / "hold").unlink()
            self.assertTrue(wait(lambda: job.status == "done"), job.error)
        self.assertFalse(store.exists(checkpoint))
        self.assertIn(job.params["remote_result"]["key"], [e["key"] for e in store.list()])

    def test_a_result_the_store_still_refuses_fails_the_job_but_the_weights_stay_and_the_message_says_where(self):
        job, run, store = self.hold_at_checkpoint(steps=8)
        self.publish_checkpoint(run)
        sentence = "Not uploaded: this upload needs 1.00 MB and the bucket already holds 9.50 GB, which would make 9.50 GB."
        with mock.patch.object(_Store, "publish", side_effect=StoreLimitError(sentence)):
            (self.b_root / "hold").unlink()
            self.assertTrue(wait(lambda: job.status == "failed"), job.status)
        best = Path(self.b_job().params["output_dir"]) / "best.npz"
        self.assertTrue(best.is_file())  # the finished weights are still on the runner
        self.assertTrue(job.error.startswith(sentence), job.error)
        self.assertIn(f"The finished weights are kept on the machine that ran it, at {best}.", job.error)
        self.assertNotIn("remote_result", job.params)

    def test_a_result_refused_because_the_size_could_not_be_read_is_not_retried_and_keeps_the_checkpoints(self):
        job, run, store = self.hold_at_checkpoint(steps=8)
        checkpoint = self.publish_checkpoint(run)
        failure = StoreUsageError("Nothing was uploaded: the bucket's usage could not be read.")
        with mock.patch.object(_Store, "publish", side_effect=failure) as publish:
            (self.b_root / "hold").unlink()
            self.assertTrue(wait(lambda: job.status == "failed"))
        self.assertEqual(publish.call_count, 1)
        self.assertTrue(store.exists(checkpoint))  # deleting them would not have helped
        self.assertIn("kept on", job.error)

    def test_a_checkpoint_whose_size_check_failed_is_offered_again_later_and_the_owner_is_told_meanwhile(self):
        job, run, store = self.hold_at_checkpoint()
        unreadable = StoreUsageError("Nothing was uploaded: the bucket's usage could not be read.")
        with mock.patch.object(studio_remote, "USAGE_RETRY_S", 5.), mock.patch.object(_Store, "_admit", side_effect=unreadable) as admit:
            self.b.cluster.tick()
            self.assertTrue(wait(lambda: run.refusal and not run.publishing))
            self.b.cluster.tick()
            self.assertEqual(admit.call_count, 1)  # not straight away
            self.b_clock.advance(6.)
            self.b.cluster.tick()
            self.assertTrue(wait(lambda: admit.call_count == 2 and not run.publishing), "it was not offered again")
            self.assertEqual(run.refused_mtime, 0.)  # a failed read is not a refusal of this checkpoint
            self.assertIn("not shared", run.refusal)
            self.beat(6.)  # the owner hears of it on the renewal
            self.assertIn(run.refusal, job.note)
            self.b_clock.advance(6.)
        self.b.cluster.tick()
        self.assertTrue(wait(lambda: run.checkpoint is not None), "it was not published once the store could be read")
        self.assertEqual(run.refusal, "")
        (self.b_root / "hold").unlink()
        self.assertTrue(wait(lambda: job.status == "done"))

    def test_an_input_the_store_refuses_leaves_the_job_waiting_with_the_sentence_and_does_not_blame_the_peer(self):
        start = self.a_root / "start.npz"
        write_checkpoint(start, 50)
        job = self.a.enqueue(self.job(8, mode="catch_up", init_from=str(start)))
        self.a.cluster.poll_once()
        sentence = "Not uploaded: this upload needs 1.00 MB and the bucket already holds 9.50 GB, which would make 9.50 GB."
        with mock.patch.object(_Store, "publish", side_effect=StoreLimitError(sentence)):
            self.a._place_waiting()
        self.assertEqual(job.status, "waiting")
        self.assertEqual(self.a.cluster.waiting[job.id], sentence)  # as it is: no "gpu-box would not take it"
        self.assertEqual(self.a.cluster.owner.refused_by(job.id), set())  # no cooldown: the peer did nothing wrong
        self.assertIn(("cluster", {"job": job.id, "state": "store_refused", "node": "gpu-box", "detail": sentence}), self.a_events)
        self.a._place_waiting()  # room again: the same peer takes it at once
        self.assertEqual((job.status, job.node), ("running", "gpu-box"))
        self.assertTrue(wait(lambda: job.status == "done"))

    def test_when_the_owner_cannot_be_reached_the_runner_stops_its_job_by_itself_within_one_lease(self):
        (self.b_root).mkdir(parents=True, exist_ok=True)
        (self.b_root / "hold").touch()
        job = self.a.enqueue(self.job(1000))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertTrue(wait(lambda: self.b_job() and self.b_job().status == "running"))
        self.beat()  # a renewal gets through
        self.net.down.add("owner-a")
        for _ in range(2):
            self.beat()
            self.assertEqual(self.b_job().status, "running")  # 20 s since the last renewal was sent: still leased
        self.beat()
        self.assertTrue(wait(lambda: self.b_job().status == "stopped"), "the runner did not fence itself")
        self.assertEqual(self.b.cluster.guests[job.id].stopped_by, "fenced")
        self.assertEqual(job.status, "running")  # the owner has heard nothing: it will wait out two leases

    def test_a_lost_node_is_fenced_then_its_job_resumes_on_the_returning_node_from_the_published_checkpoint(self):
        (self.b_root).mkdir(parents=True, exist_ok=True)
        (self.b_root / "hold").touch()
        job = self.a.enqueue(self.job(100))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertTrue(wait(lambda: self.checkpointed()))
        run = self.b.cluster.guests[job.id]
        self.b.cluster.tick()
        self.assertTrue(wait(lambda: run.checkpoint is not None))
        self.beat()
        self.assertEqual(job.params["cluster"]["checkpoint"]["step"], 3)
        self.net.down.update({"owner-a", "gpu-box"})  # the cable is pulled
        for _ in range(3):
            self.beat()
        self.assertTrue(wait(lambda: self.b_job().status == "stopped"))
        self.a_clock.advance(31)
        self.a.cluster.tick()
        self.assertEqual(job.status, "running")
        self.a_clock.advance(30)
        self.a.cluster.tick()
        self.assertEqual(job.status, "waiting")  # interrupted at its checkpoint
        self.assertIn("step 3", job.note)
        self.assertEqual(job.params["config"]["steps"], 97)
        (self.b_root / "hold").unlink()
        self.net.down.clear()
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertEqual((job.status, job.node, job.epoch), ("running", "gpu-box", 2))
        self.assertTrue(wait(lambda: job.status == "done"), "the resumed job did not finish")
        ran = self.ran()
        self.assertEqual(ran["steps"], 97)
        self.assertTrue(Path(ran["init_from"]).is_file())
        self.assertEqual(job.params["remote_result"]["step"], 100)  # 3 steps before the loss + 97 after
        self.assertEqual(job.epoch, 2)
        self.assertIn("optimizer starts again from zero", job.note)

    def test_a_node_that_wakes_up_with_the_old_epoch_is_refused_and_stops_its_copy(self):
        (self.b_root).mkdir(parents=True, exist_ok=True)
        (self.b_root / "hold").touch()
        job = self.a.enqueue(self.job(1000))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertTrue(wait(lambda: self.b_job() and self.b_job().status == "running"))
        self.a_clock.advance(61)  # the owner heard nothing for two leases while the runner was asleep (its clock did not move)
        self.a.cluster.tick()
        self.assertEqual(job.status, "waiting")
        self.beat()  # the runner wakes and renews at epoch 1
        self.assertEqual(self.b.cluster.guests[job.id].stopped_by, "stale")
        self.assertTrue(wait(lambda: self.b_job().status == "stopped"))
        self.assertEqual(job.status, "waiting")  # and nothing it said reached the job

    def test_the_owner_cancelling_a_job_stops_the_copy_on_the_runner(self):
        (self.b_root).mkdir(parents=True, exist_ok=True)
        (self.b_root / "hold").touch()
        job = self.a.enqueue(self.job(1000))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertTrue(wait(lambda: self.b_job() and self.b_job().status == "running"))
        self.a.remove(job.id)
        self.assertTrue(wait(lambda: self.b_job().status == "stopped"))
        self.assertEqual(job.status, "stopped")

    def test_a_workspace_that_names_only_the_folder_dispatches_and_publishes_and_pulls_through_it(self):
        for root, studio_config in ((self.a_root, {"nodes": ["gpu-box"]}), (self.b_root, {})):
            (root / "data").mkdir(parents=True, exist_ok=True)
            studio_config["store"] = {"folder": str(self.shared)}
            (root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": studio_config}), encoding="utf-8")
        for runner in (self.a, self.b):  # no store handed in and no environment: each node finds the folder in its workspace
            runner.cluster._store_factory = runner.cluster.workspace_store
            runner.cluster.environ = {}
        start = self.a_root / "start.npz"
        write_checkpoint(start, 50)
        job = self.a.enqueue(self.job(8, mode="catch_up", init_from=str(start)))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertEqual((job.status, job.node, job.epoch), ("running", "gpu-box", 1))
        self.assertTrue(wait(lambda: self.b_job() and self.b_job().status == "done"), "the job did not finish on the peer")
        self.assertTrue(wait(lambda: job.status == "done"), "the owner was not told")
        ran = self.ran()
        self.assertTrue(Path(ran["init_from"]).is_file())  # pulled from the folder the workspace names
        self.assertNotIn(str(self.a_root), ran["init_from"])
        key = job.params["remote_result"]["key"]
        self.assertIn(key, [e["key"] for e in FolderStore(self.shared).list()])  # published there

    def test_a_runner_without_the_store_refuses_the_job_in_a_sentence(self):
        self.b.cluster._store_factory = lambda: None
        job = self.a.enqueue(self.job(5))
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertEqual(job.status, "waiting")
        self.assertIn("version store", self.a.cluster.waiting[job.id])


if __name__ == "__main__":
    unittest.main()

"""studio_api.routers.cluster: the node-to-node calls and the cluster views, over a Runner whose peers are fakes."""
import json
import tempfile
import unittest
from pathlib import Path

import studio_scheduler as sched
from studio_ia.services import FakeServices
from studio_queue import Job, Plan, Runner
from studio_remote import Cluster
from tests.fake_cluster import Clock, Net
from tests.sse_harness import ApiCase
from tests.test_queue_cluster import FakeStore, Peer, own_node, trainish, wait


class ClusterServices(FakeServices):
    """FakeServices with a real queue Runner, which is where a node keeps its cluster."""

    def __init__(self, runner):
        super().__init__()
        self.runner = runner
        runner.subscribe(self.emit)  # as WorkspaceServices does: the queue's events are the app's events


class ClusterRoutes(ApiCase):
    prefix = "/cluster"

    def make_services(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": ["gpu-box"]}}), encoding="utf-8")
        self.net, self.clock, self.store = Net(), Clock(), FakeStore(self.root / "store")
        self.peer = Peer(self.net)
        self.runner = Runner(self.root / "data" / "queue.json", {"pretrain": trainish}, tick=.01, prepare=self.prepare, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: self.store, own=own_node, clock=self.clock, address="here", threads=False))
        self.runner.set_plan(Plan(hours="any", idle_only=False, pause_on_battery=False))
        self.addCleanup(self.runner.shutdown)
        return ClusterServices(self.runner)

    def prepare(self, job):
        job.params["output_dir"] = str(self.root / "runs" / job.id)
        return {"root": str(self.root), "run_dir": str(self.root / "runs")}

    def lease(self, job_id, epoch=1, node="here"):
        return {"job_id": job_id, "node": node, "epoch": epoch, "duration_s": 30.}

    def dispatch(self, job_id, epoch=1, expect=200, **body):
        spec = {"id": job_id, "kind": "pretrain", "label": "Train", "params": {"config": {"steps": 5}, "examples": [{"text": "a"}]}}
        return self.call("POST", "/cluster/jobs", expect,
                         json={"job": spec, "lease": self.lease(job_id, epoch), "owner_address": "owner-x", **body})

    def owned_remote_job(self):
        job = self.runner.enqueue(Job("pretrain", "Mine", params={"config": {"steps": 100}, "examples": [{"text": "a"}]}))
        self.runner.cluster.poll_once()
        self.runner._place_waiting()
        self.assertEqual((job.status, job.node), ("running", "gpu-box"))
        return job

    def test_every_route_answers(self):
        self.assertEqual(self.call("GET", "/cluster/jobs"), [])

        job = self.owned_remote_job()
        rows = self.call("GET", "/cluster/jobs")
        self.assertEqual([(r["id"], r["role"], r["node"], r["epoch"], r["lease"]) for r in rows], [(job.id, "owner", "gpu-box", 1, "held")])

        answer = self.call("POST", "/cluster/jobs/{job_id}/renew", path={"job_id": job.id}, json={
            "epoch": 1, "state": "running", "events": [{"type": "metric", "step": 2, "total": 100}], "checkpoint": {"key": "k", "step": 2}})
        self.assertEqual(answer, {"epoch": 1, "duration_s": sched.LEASE_S})
        self.events.wait(lambda events: any(k == "train" and p.get("job") == job.id and p.get("step") == 2 for k, p in events),
                         "the runner's progress on /events")
        self.call("POST", "/cluster/jobs/{job_id}/renew", 409, path={"job_id": job.id}, json={"epoch": 2})

        self.call("POST", "/cluster/jobs/{job_id}/finished", 409, path={"job_id": job.id}, json={"epoch": 2, "status": "done"})
        self.call("POST", "/cluster/jobs/{job_id}/finished", 204, path={"job_id": job.id},
                  json={"epoch": 1, "status": "done", "key": "gpu-box/v0-abc", "step": 100})
        self.assertEqual((job.status, job.params["remote_result"]["key"]), ("done", "gpu-box/v0-abc"))
        self.assertEqual(self.call("GET", "/cluster/jobs"), [])

        self.assertEqual(self.dispatch("theirs"), {"job_id": "theirs", "epoch": 1})
        self.call("POST", "/cluster/jobs/{job_id}/stop", 409, path={"job_id": "theirs"}, json={"epoch": 7})
        self.call("POST", "/cluster/jobs/{job_id}/stop", 204, path={"job_id": "theirs"}, json={"epoch": 1})
        self.assertAllRoutesHit()

    def test_a_job_another_node_owns_runs_here_and_shows_as_a_runner_job(self):
        self.assertEqual(self.dispatch("theirs"), {"job_id": "theirs", "epoch": 1})
        self.assertTrue(wait(lambda: any(j.owner == "owner-x" for j in self.runner.jobs)))
        rows = self.call("GET", "/cluster/jobs")
        self.assertEqual([(r["id"], r["role"], r["node"], r["epoch"]) for r in rows], [("theirs", "runner", "owner-x", 1)])
        queued = next(j for j in self.runner.jobs if j.owner)
        self.assertEqual((queued.id, queued.epoch, queued.label), ("theirs-e1", 1, "Train (for owner-x)"))
        self.assertEqual(queued.params["examples"], [{"text": "a"}])
        self.assertTrue(self.runner.thread is not None)  # this node's own queue runs it

    def test_an_older_lease_than_the_one_held_is_refused_and_a_newer_one_replaces_it(self):
        self.dispatch("theirs", epoch=2)
        detail = self.call("POST", "/cluster/jobs", 409, json={
            "job": {"id": "theirs", "kind": "pretrain"}, "lease": self.lease("theirs", 1), "owner_address": "owner-x"})["detail"]
        self.assertIn("epoch 2", detail)
        self.assertEqual(self.dispatch("theirs", epoch=3), {"job_id": "theirs", "epoch": 3})

    def test_a_job_that_cannot_leave_its_owner_is_refused_with_a_sentence(self):
        detail = self.call("POST", "/cluster/jobs", 409, json={
            "job": {"id": "x", "kind": "compare"}, "lease": self.lease("x"), "owner_address": "owner-x"})["detail"]
        self.assertEqual(detail, "A compare job only runs on the machine that queued it.")

    def test_without_a_version_store_a_job_is_refused_with_what_to_set_up(self):
        self.runner.cluster._store_factory = lambda: None
        detail = self.dispatch("theirs", expect=409)["detail"]
        self.assertIn("TRAINER_STORE_FOLDER", detail)

    def test_the_lease_must_be_for_the_job(self):
        self.call("POST", "/cluster/jobs", 400, json={"job": {"id": "a", "kind": "pretrain"}, "lease": self.lease("b")})

    def test_the_address_the_request_came_from_is_the_owner_when_none_is_given(self):
        self.call("POST", "/cluster/jobs", 200, json={
            "job": {"id": "theirs", "kind": "pretrain", "params": {"config": {"steps": 5}, "examples": [{"text": "a"}]}},
            "lease": self.lease("theirs")})
        self.assertEqual(self.runner.cluster.guests["theirs"].owner_address, "testclient")

    def test_the_queue_lists_a_job_that_runs_on_a_peer_after_the_running_one(self):
        job = self.owned_remote_job()
        listed = self.call("GET", "/queue/jobs")
        self.assertEqual([(j["id"], j["status"], j["node"], j["epoch"]) for j in listed], [(job.id, "running", "gpu-box", 1)])

class NoQueue(ApiCase):
    prefix = "/cluster"

    def test_a_node_without_a_job_queue_says_so(self):
        self.assertIn("no job queue", self.call("GET", "/cluster/jobs", 409)["detail"])


if __name__ == "__main__":
    unittest.main()

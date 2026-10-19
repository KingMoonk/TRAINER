"""studio_api.routers.cluster: the node-to-node calls and the cluster views, over a Runner whose peers are fakes."""
import dataclasses
import inspect
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import studio_cluster
import studio_cluster_queue
import studio_peer
import studio_scheduler as sched
from fastapi.testclient import TestClient
from studio_api.app import create_app
from studio_api.hosts import ANY, HostNames
from studio_api.routers import cluster as cluster_routes
from studio_api.version import app_version
from studio_core import Workspace
from studio_ia.services import FakeServices, WorkspaceServices
from studio_peer import PeerError
from studio_queue import Job, Plan, Runner
from studio_remote import Cluster
from tests.fake_cluster import Clock, Net
from tests.sse_harness import ApiCase
from tests.test_queue_cluster import FakeStore, Peer, own_node, trainish, wait


GPU = {"available": True, "kind": "cuda", "name": "Test GPU", "memory_bytes": 8 * 2 ** 30, "reason": ""}
real_own_info = studio_cluster.own_info


def probed(**kwargs):
    """own_info with the GPU and memory probes answered, so the test never imports torch."""
    return real_own_info(**kwargs, status=GPU, memory=16 * 2 ** 30)


def node_payload(hostname, **changes):
    return {"hostname": hostname, "version": "1.2.3", "device": "mps", "device_name": "Apple GPU", "memory_bytes": 32 * 2 ** 30,
            "busy": False, "dtypes": ["float32", "float16"], **changes}


class FakePeers:
    """Stands in for studio_cluster.fetch_json: peers by the host in the /node URL. A name maps to a payload (online), an exception
    to raise (offline) or a threading.Event to wait on (a peer that never answers)."""

    def __init__(self, **peers):
        self.peers, self.calls, self.release = peers, [], threading.Event()

    def __call__(self, url, timeout):
        host = url.split("//")[1].split(":")[0].split("/")[0]
        self.calls.append((host, timeout))
        peer = self.peers[host]
        if peer == "stuck":
            self.release.wait(10)
            return node_payload(host)
        if isinstance(peer, Exception):
            raise peer
        return peer


class ClusterServices(FakeServices):
    """FakeServices with a real queue Runner, which is where a node keeps its cluster."""

    def __init__(self, runner):
        super().__init__()
        self.runner = runner
        runner.subscribe(self.emit)  # as WorkspaceServices does: the queue's events are the app's events

    def remove(self, job_id):
        self.runner.remove(job_id)

    def queue(self):
        """Up next is the Runner's waiting jobs, as WorkspaceServices reports them."""
        return dataclasses.replace(super().queue(), up_next=[j for j in self.runner.jobs if j.status in ("waiting", "held")])


class ClusterRoutes(ApiCase):
    prefix = "/cluster"

    def make_services(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
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

        self.peers = FakePeers(**{"gpu-box": node_payload("gpu-box")})
        with mock.patch.object(studio_cluster, "fetch_json", self.peers):
            self.assertEqual(self.call("PUT", "/cluster/nodes", json={"addresses": ["gpu-box"]}), {"addresses": ["gpu-box"]})
            self.assertEqual([n["address"] for n in self.call("GET", "/cluster/nodes")], ["local", "gpu-box"])
        self.services.set_setting("nodes", ["gpu-box"])
        with mock.patch.object(studio_cluster_queue, "fetch", lambda address, path, timeout: {"node": node_payload("gpu-box"), "queue/jobs": []}[path]):
            self.assertEqual([n["hostname"] for n in self.call("GET", "/cluster/merged")["nodes"]], ["laptop", "gpu-box"])
        self.assertEqual(self.call("GET", "/cluster/access"), {"allow_users": [], "allow_tags": ["tag:trainer"], "login": ""})
        self.call("PUT", "/cluster/access", json={"allow_users": ["me@example.com"], "allow_tags": ["tag:trainer"]})
        with mock.patch.object(studio_peer.PeerClient, "delete_json", return_value=None):
            self.call("POST", "/cluster/remote/{job_id}/stop", 204, path={"job_id": "x"}, json={"address": "gpu-box"})
        self.assertAllRoutesHit()

    def test_a_job_that_runs_on_a_peer_can_be_stopped_from_here(self):
        job = self.owned_remote_job()
        self.call("DELETE", "/queue/jobs/{job_id}", path={"job_id": job.id})
        self.assertTrue(wait(lambda: job.status == "stopped"), job.status)
        self.call("DELETE", "/queue/jobs/{job_id}", 404, path={"job_id": "nope"})

    def test_the_allow_lists_are_saved_beside_the_keys_the_page_never_sees(self):
        token = "t" * 40
        self.services.set_setting("cluster", {"funnelToken": token, "allowUsers": ["old@example.com"]})
        hosts = HostNames(binary=lambda: "tailscale", login=lambda: "me@example.com")
        hosts.refresh()
        self.app.state.hosts = hosts
        self.assertEqual(self.call("GET", "/cluster/access"),
                         {"allow_users": ["old@example.com"], "allow_tags": ["tag:trainer"], "login": "me@example.com"})
        saved = self.call("PUT", "/cluster/access", json={"allow_users": [" a@example.com ", "a@example.com", "b@example.com"],
                                                          "allow_tags": ["tag:trainer", "tag:gpu"]})
        self.assertEqual((saved["allow_users"], saved["allow_tags"]), (["a@example.com", "b@example.com"], ["tag:trainer", "tag:gpu"]))
        self.assertNotIn(token, json.dumps(saved))
        self.assertEqual(self.services.setting("cluster")["funnelToken"], token)  # kept, not echoed
        self.assertEqual(self.services.setting("cluster")["allowUsers"], ["a@example.com", "b@example.com"])

    def test_a_bad_allow_list_is_refused_with_a_sentence_and_nothing_is_saved(self):
        detail = self.call("PUT", "/cluster/access", 400, json={"allow_users": [], "allow_tags": ["trainer"]})["detail"]
        self.assertIn("tag:", detail)
        self.assertIn("allowUsers", self.call("PUT", "/cluster/access", 400, json={"allow_users": [" "], "allow_tags": []})["detail"])
        self.assertIsNone(self.services.setting("cluster"))
        self.services.set_setting("cluster", "nonsense")  # hand-edited since the API started
        self.assertIn("studioConfig.cluster", self.call("GET", "/cluster/access", 409)["detail"])

    def test_stopping_a_job_on_a_peer_asks_that_peer_and_only_a_listed_one(self):
        self.services.set_setting("nodes", ["gpu-box"])
        asked = []
        with mock.patch.object(studio_peer.PeerClient, "delete_json", lambda client, path: asked.append((client.address, path))):
            self.call("POST", "/cluster/remote/{job_id}/stop", 204, path={"job_id": "j1"}, json={"address": "gpu-box"})
            self.call("POST", "/cluster/remote/{job_id}/stop", 404, path={"job_id": "j1"}, json={"address": "stranger"})
        self.assertEqual(asked, [("gpu-box", "queue/jobs/j1")])
        refusal = PeerError("gpu-box answered 401 to DELETE /queue/jobs/j1: This node does not know the caller.", status=401)
        with mock.patch.object(studio_peer.PeerClient, "delete_json", side_effect=refusal):
            detail = self.call("POST", "/cluster/remote/{job_id}/stop", 502, path={"job_id": "j1"}, json={"address": "gpu-box"})["detail"]
        self.assertIn("answered 401", detail)

    def test_a_pin_decides_where_a_waiting_job_goes_and_the_merged_queue_says_so(self):
        job = self.runner.enqueue(Job("pretrain", "Mine", params={"config": {"steps": 100}, "examples": [{"text": "a"}]}))
        self.runner.cluster.poll_once()
        self.services.set_setting("nodes", ["gpu-box"])  # WorkspaceServices reads the same list from the workspace file
        peer_jobs = []
        asked = lambda address, path, timeout: {"node": node_payload("gpu-box"), "queue/jobs": peer_jobs}[path]  # noqa: E731

        def row():
            with mock.patch.object(studio_cluster_queue, "fetch", asked):
                return next(r for r in self.call("GET", "/cluster/merged")["jobs"] if r["id"] == job.id)

        def pin(pinned):
            self.call("POST", "/queue/jobs/{job_id}/pin", path={"job_id": job.id}, json={"pinned": pinned})
            self.runner._place_waiting()

        pin("ghost")  # nobody answers to it: it waits, and the row says why
        self.assertEqual((job.status, job.params["pinned"], row()["hostname"]), ("waiting", "ghost", "ghost"))
        self.assertIn("not one of this cluster's machines", row()["waiting_reason"])
        pin("gpu-box")  # the other machine: the job goes there
        self.assertEqual((job.status, job.node), ("running", "gpu-box"))
        self.assertEqual((row()["status"], row()["node"], row()["hostname"], row()["waiting_reason"]), ("running", "gpu-box", "gpu-box", ""))
        self.assertEqual(self.peer.dispatched[0]["job"]["id"], job.id)
        self.assertNotIn("pinned", self.peer.dispatched[0]["job"]["params"])  # a pin is this node's to keep

    def test_pinning_a_job_to_this_machine_keeps_it_here(self):
        job = self.runner.enqueue(Job("pretrain", "Mine", params={"config": {"steps": 100}, "examples": [{"text": "a"}]}))
        self.runner.cluster.poll_once()
        self.call("POST", "/queue/jobs/{job_id}/pin", path={"job_id": job.id}, json={"pinned": "laptop"})
        self.assertEqual(self.runner.cluster.place(job).where, "local")
        self.call("POST", "/queue/jobs/{job_id}/pin", path={"job_id": job.id}, json={"pinned": None})
        self.assertNotIn("pinned", job.params)

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
            "job": {"id": "x", "kind": "quicktest"}, "lease": self.lease("x"), "owner_address": "owner-x"})["detail"]
        self.assertEqual(detail, "A quicktest job only runs on the machine that queued it.")

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


class NodeList(ApiCase):
    """GET and PUT /cluster/nodes over FakeServices (no queue) and fake peers: one online, one offline, one that never answers."""
    prefix = "/cluster/nodes"

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
        self.peers = FakePeers(up=node_payload("up-box", busy=True), down=PeerError("down did not answer GET /node: Connection refused"),
                               slow="stuck")
        self.addCleanup(self.peers.release.set)
        self.enterContext(mock.patch.object(studio_cluster, "fetch_json", self.peers))

    def save(self, *addresses, expect=200):
        return self.call("PUT", "/cluster/nodes", expect, json={"addresses": list(addresses)})

    def test_this_node_alone_is_a_cluster_of_one(self):
        [me] = self.call("GET", "/cluster/nodes")
        self.assertEqual((me["address"], me["local"], me["online"], me["error"], me["busy"]), ("local", True, True, "", False))
        self.assertEqual((me["device"], me["device_name"], me["memory_bytes"], me["version"]), ("cuda", "Test GPU", 8 * 2 ** 30, app_version()))
        self.assertEqual(self.peers.calls, [])

    def test_only_this_machine_carries_its_tailnet_name(self):
        self.assertEqual(self.call("GET", "/cluster/nodes")[0]["tailnet_name"], "")  # no Tailscale names known (hosts=ANY)
        hosts = HostNames(tailscale=lambda: ("mini.tail1234.ts.net", "mini"), binary=lambda: "tailscale")
        hosts.refresh()
        self.app.state.hosts = hosts
        self.save("up")
        me, up = self.call("GET", "/cluster/nodes")
        self.assertEqual((me["tailnet_name"], up["tailnet_name"]), ("mini.tail1234.ts.net", ""))

    def test_all_three_peers_are_listed_and_the_slow_one_holds_up_nobody(self):
        self.save("up", "down", "slow")
        with mock.patch.object(studio_cluster, "POLL_TIMEOUT", .4):
            started = time.monotonic()
            rows = self.call("GET", "/cluster/nodes")
            took = time.monotonic() - started
        self.assertEqual([(r["address"], r["local"], r["online"]) for r in rows],
                         [("local", True, True), ("up", False, True), ("down", False, False), ("slow", False, False)])
        up, down, slow = rows[1:]
        self.assertEqual((up["hostname"], up["device"], up["memory_bytes"], up["busy"], up["version"], up["error"]),
                         ("up-box", "mps", 32 * 2 ** 30, True, "1.2.3", ""))
        self.assertEqual(down["error"], "down did not answer GET /node: Connection refused.")
        self.assertEqual(slow["error"], "slow did not answer within 0.4 seconds.")
        self.assertEqual(slow["hostname"], "")
        self.assertEqual(sorted(h for h, _ in self.peers.calls), ["down", "slow", "up"])  # all asked, none waiting for another
        self.assertEqual({t for _, t in self.peers.calls}, {.4})
        self.assertLess(took, 1.5)  # the stuck peer costs one timeout, not ten seconds

    def test_peers_are_asked_at_the_same_time(self):
        self.peers.peers = {f"p{i}": node_payload(f"p{i}") for i in range(4)}
        slow = self.peers.__call__
        self.enterContext(mock.patch.object(studio_cluster, "fetch_json", lambda url, timeout: (time.sleep(.3), slow(url, timeout))[1]))
        self.save(*self.peers.peers)
        started = time.monotonic()
        rows = self.call("GET", "/cluster/nodes")
        self.assertEqual([r["online"] for r in rows], [True] * 5)
        self.assertLess(time.monotonic() - started, .9)  # four in a row would be 1.2 s

    def test_the_routes_are_sync_so_the_event_loop_is_never_blocked(self):
        self.assertFalse(inspect.iscoroutinefunction(cluster_routes.nodes))
        self.assertFalse(inspect.iscoroutinefunction(cluster_routes.save_nodes))

    def test_a_saved_list_is_cleaned_and_read_back_in_order(self):
        self.assertEqual(self.save(" up ", "", "down", "up"), {"addresses": ["up", "down"]})
        self.assertEqual([r["address"] for r in self.call("GET", "/cluster/nodes")], ["local", "up", "down"])
        self.assertEqual(self.services.setting("nodes"), ["up", "down"])
        self.assertEqual(self.save(), {"addresses": []})
        self.assertEqual(len(self.call("GET", "/cluster/nodes")), 1)

    def test_a_bad_entry_is_refused_with_a_sentence_and_nothing_is_saved(self):
        self.save("up")
        for entry, why in (("ftp://box", "http://"), ("box:abc", "port"), ("a b", "spaces"), ("http://", "no machine name")):
            detail = self.save("up", entry, expect=400)["detail"]
            self.assertIn(f'"{entry}" is not a machine address', detail)
            self.assertIn(why, detail)
        self.assertEqual(self.services.setting("nodes"), ["up"])
        self.assertIn("up to 32", self.save(*[f"box{i}" for i in range(33)], expect=400)["detail"])

    def test_a_body_that_is_not_a_list_of_text_is_refused(self):
        self.call("PUT", "/cluster/nodes", 422, json={"addresses": "up"})
        self.call("PUT", "/cluster/nodes", 422, json={"addresses": [1]})

    def test_a_list_that_was_hand_edited_into_nonsense_says_what_to_fix(self):
        self.services.set_setting("nodes", "up")
        self.assertIn("studioConfig.nodes", self.call("GET", "/cluster/nodes", 400)["detail"])
        self.assertEqual(self.save("up"), {"addresses": ["up"]})  # saving over it repairs it
        self.assertEqual(len(self.call("GET", "/cluster/nodes")), 2)


class SavedList(unittest.TestCase):
    """The list lives in the workspace config: a new launch over the same folder reads it back."""

    def launch(self):
        svc = WorkspaceServices(Workspace(self.root))
        self.addCleanup(svc.close)
        app = create_app(svc, open=True, hosts=ANY)
        self.addCleanup(app.state.jobs.close)
        self.addCleanup(app.state.broadcaster.close)
        return TestClient(app)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
        self.enterContext(mock.patch.object(studio_cluster, "fetch_json", FakePeers(**{"gpu-box": node_payload("gpu-box")})))

    def test_it_survives_a_restart_and_is_the_list_the_cluster_reads(self):
        before = self.launch()
        self.assertEqual(len(before.get("/cluster/nodes").json()), 1)
        self.assertEqual(before.put("/cluster/nodes", json={"addresses": ["gpu-box", "http://100.64.0.7:8787"]}).status_code, 200)

        after = self.launch()  # a new process: a new Workspace read from disk
        rows = after.get("/cluster/nodes").json()
        self.assertEqual([r["address"] for r in rows[1:]], ["gpu-box", "http://100.64.0.7:8787"])
        self.assertEqual((rows[1]["online"], rows[1]["hostname"]), (True, "gpu-box"))
        state = json.loads((self.root / "data" / "workspace.json").read_text(encoding="utf-8"))
        self.assertEqual(studio_cluster.configured_nodes(state), ["gpu-box", "http://100.64.0.7:8787"])  # what the job scheduler polls


if __name__ == "__main__":
    unittest.main()

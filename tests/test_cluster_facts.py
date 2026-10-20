"""Every node's facts in the observation store, and any node's views answering from its own copy (studio_facts, GET /distributed/runs,
/cluster/nodes, /cluster/merged). Two in-process nodes over TestClient, a fake clock and a fake network; no sockets, no sleeping."""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_cluster
import studio_cluster_queue
import studio_distributed as runs
import studio_facts
import studio_state
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_peer import PeerError
from studio_queue import Runner
from studio_remote import Cluster
from tests.fake_cluster import Clock, Net
from tests.test_api_cluster import ClusterServices, FakePeers, node_payload, probed
from tests.test_queue_cluster import own_node

COMMANDS = ("/stop", "/pause", "/resume", "/restart", "/add")
MACHINES = [{"node": "A", "address": "local", "device": "mps", "device_name": "M3"},
            {"node": "B", "address": "B", "device": "cuda", "device_name": "RTX"}]


def snapshot(**changes):
    history = [{"round": 0, "step": 2, "contributed": ["A", "B"], "members": ["A", "B"], "dropped": [], "loss": 3., "val_loss": 3.5,
                "delta_bytes": 2000, "theta_bytes": 4000, "seconds": 5., "steps": {"A": 2, "B": 2}, "node_tokens": {"A": 100, "B": 50},
                "node_seconds": {"A": 2., "B": 5.}, "node_bytes": {"A": 1000, "B": 1000}, "delta_dtype": "float16"}]
    return {"state": "running", "round": 1, "rounds": 4, "generation": 1,
            "settings": {"rounds": 4, "inner_steps": 2, "outer_lr": .7, "outer_momentum": .9, "delta_dtype": "float16"},
            "nodes": {"A": {"state": "training", "missed": 0, "sent": True, "note": ""},
                      "B": {"state": "training", "missed": 0, "sent": False, "note": ""}}, "history": history, **changes}


def a_run(run_id="run1"):
    return runs.Run(id=run_id, label="Version 1 across 2 machines", version=1, started_at=100., settings={"rounds": 4, "inner_steps": 2},
                    owner="A", machines=runs.machines_of(MACHINES), final=snapshot(), term=2)


class Node:
    """One in-process node: a Runner with its Cluster (its own hostname and its own runs), and the app over it."""

    def __init__(self, case, name, peers, net, clock):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        case.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "data").mkdir()
        (root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": peers}}), encoding="utf-8")
        self.name, self.runs = name, []
        info = dataclasses.replace(own_node(), hostname=name)
        self.runner = Runner(root / "data" / "queue.json", {}, tick=.01, cluster=lambda r: Cluster(
            r, client_for=net.client_for, store=lambda: None, own=lambda: info, clock=clock, address=name, threads=False,
            hostname=name, local_runs=lambda: [(run, None, None) for run in self.runs]))
        case.addCleanup(self.runner.shutdown)
        self.cluster, self.state = self.runner.cluster, self.runner.cluster.state
        self.services = ClusterServices(self.runner)
        self.services.set_setting("nodes", peers)
        self.app = create_app(self.services, open=True, hosts=ANY)
        self.client = TestClient(self.app)
        case.addCleanup(self.app.state.jobs.close)
        case.addCleanup(self.app.state.broadcaster.close)

    def listed(self):
        return self.client.get("/distributed/runs").json()

    def facts(self):
        """What this node's replica holds, without the ages."""
        return sorted((e["origin"], e["kind"], e["key"], e["incarnation"], e["seq"], json.dumps(e["value"], sort_keys=True))
                      for e in self.state.entries())


class TwoNodes(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
        self.clock, self.net = Clock(), Net()
        self.a = Node(self, "A", ["B"], self.net, self.clock)
        self.b = Node(self, "B", ["A"], self.net, self.clock)
        for node in (self.a, self.b):
            self.net.handlers[node.name] = self.serve(node)

    def serve(self, node):
        def handler(method, path, body):
            if path == "node":
                return node_payload(node.name)
            if path == "state/sync":
                return node.state.handle_sync(json.loads(json.dumps(body)))
            if path.startswith("distributed/runs/") and path.endswith(COMMANDS):  # a command: the owner answers with the run it reads
                return node.client.get("/" + (path.split("/members/")[0] if "/members/" in path else path.rsplit("/", 1)[0])).json()
            raise PeerError(f"answered 404: {path}", status=404, detail=path)
        return handler

    def sync(self):
        self.a.cluster.poll_once()
        self.b.cluster.poll_once()

    def test_a_run_coordinated_on_a_appears_in_bs_list_within_one_sync_with_its_origin_and_age(self):
        self.a.runs.append(a_run())
        self.assertEqual(self.b.listed(), [])
        self.a.cluster.poll_once()  # A writes its facts and pushes them to B
        rows = self.b.listed()
        self.assertEqual([r["id"] for r in rows], ["run1"])
        self.assertEqual((rows[0]["origin"], rows[0]["coordinator"], rows[0]["term"], rows[0]["origin_down"]), ("A", "A", 2, False))
        self.assertEqual(rows[0]["age_s"], 0.)
        self.clock.advance(7)
        row = self.b.listed()[0]
        self.assertEqual(row["age_s"], 7.)
        self.assertEqual(row["round"], 1)
        self.assertEqual([m["node"] for m in row["members"]], ["A", "B"])
        self.assertEqual(self.b.client.get("/distributed/runs/run1").json()["origin"], "A")
        self.assertEqual(self.a.listed()[0]["age_s"], 0.)  # A reads its own run live
        self.assertEqual(self.a.listed()[0]["origin"], "A")

    def test_a_members_progress_on_b_shows_on_a(self):
        self.a.runs.append(a_run())
        self.b.cluster._on_event("train", {"type": "diloco_member", "run": "run1", "node": "B", "term": 2, "round": 1, "phase": "training",
                                           "step_in_round": 1, "round_steps": 2, "loss": 1.5, "tokens_per_second": 40., "shard_ids": ["s1", "s2"]})
        self.sync()
        self.clock.advance(3)
        member = next(m for m in self.a.listed()[0]["members"] if m["node"] == "B")
        self.assertEqual((member["phase"], member["step_in_round"], member["loss"], member["live_tokens_per_second"]), ("training", 1, 1.5, 40.))
        self.assertEqual(member["reported_ago"], 3.)
        self.assertEqual(self.a.state.get("B", "shards", "run1")["value"], {"ids": ["s1", "s2"]})

    def test_a_partition_keeps_as_last_facts_on_b_with_growing_age_and_a_marked_down(self):
        self.a.runs.append(a_run())
        self.sync()
        self.net.down.add("A")
        self.clock.advance(30)
        self.b.cluster.poll_once()
        row = self.b.listed()[0]
        self.assertEqual((row["origin"], row["round"], row["origin_down"]), ("A", 1, True))
        self.assertEqual(row["age_s"], 30.)
        self.clock.advance(60)
        self.assertEqual(self.b.listed()[0]["age_s"], 90.)
        self.net.down.discard("A")
        self.clock.advance(1000)
        self.sync()
        row = self.b.listed()[0]
        self.assertEqual((row["origin_down"], row["age_s"]), (False, 0.))

    def test_an_origin_not_heard_from_is_down_by_its_heartbeat_even_without_a_poll_of_it(self):
        self.a.runs.append(a_run())
        self.a.cluster.poll_once()
        self.assertFalse(self.b.listed()[0]["origin_down"])
        self.clock.advance(studio_facts.DOWN_AFTER + 1)
        self.assertTrue(self.b.listed()[0]["origin_down"])

    def test_a_heartbeat_keeps_a_quiet_node_from_being_pruned_and_a_silent_one_is_pruned(self):
        self.a.cluster.poll_once()
        for _ in range(5):  # nothing else changes on A, but each poll rewrites its heartbeat
            self.clock.advance(studio_state.PRUNE_AFTER / 2)
            self.a.cluster.poll_once()
        self.assertEqual(self.b.state.get("A", "heartbeat", "node")["value"]["beat"], 6)
        self.assertIn("A", studio_facts.origins(self.b.cluster))
        self.clock.advance(studio_state.PRUNE_AFTER + 1)  # now A says nothing at all
        self.assertIsNone(self.b.state.get("A", "heartbeat", "node"))
        self.assertNotIn("A", studio_facts.origins(self.b.cluster))

    def test_a_command_goes_to_the_owner_and_leaves_the_replica_unchanged(self):
        self.a.runs.append(a_run())
        self.sync()
        before, vector = self.b.facts(), self.b.state.vector()
        self.net.calls.clear()
        reply = self.b.client.post("/distributed/runs/run1/stop")
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual((reply.json()["id"], reply.json()["origin"]), ("run1", "A"))
        self.assertEqual([(a, m, p) for a, m, p, _, _ in self.net.calls], [("A", "POST", "distributed/runs/run1/stop")])
        self.assertEqual(self.b.facts(), before)
        self.assertEqual(self.b.state.vector(), vector)

    def test_pause_resume_and_restart_go_to_the_owner_the_same_way(self):
        self.a.runs.append(a_run())
        self.sync()
        before = self.b.facts()
        self.net.calls.clear()
        for verb, status in (("pause", 200), ("resume", 200), ("restart", 201)):
            reply = self.b.client.post(f"/distributed/runs/run1/{verb}")
            self.assertEqual((reply.status_code, reply.json()["origin"]), (status, "A"), reply.text)
        self.assertEqual([(a, m, p) for a, m, p, _, _ in self.net.calls],
                         [("A", "POST", f"distributed/runs/run1/{verb}") for verb in ("pause", "resume", "restart")])
        self.assertEqual(self.b.facts(), before)

    def test_a_join_from_b_goes_to_a_under_bs_name(self):
        self.a.runs.append(a_run())
        self.sync()
        self.net.calls.clear()
        reply = self.b.client.post("/distributed/runs/run1/join")
        self.assertEqual((reply.status_code, reply.json()["origin"]), (200, "A"), reply.text)
        self.assertEqual([(a, m, p) for a, m, p, _, _ in self.net.calls], [("A", "POST", "distributed/runs/run1/members/B/add")])
        self.assertEqual(self.a.client.post("/distributed/runs/run1/join").status_code, 409)  # A coordinates it
        self.assertEqual(self.b.client.post("/distributed/runs/nope/join").status_code, 404)

    def test_a_command_for_an_owner_that_does_not_answer_is_a_502_and_changes_nothing(self):
        self.a.runs.append(a_run())
        self.sync()
        before = self.b.facts()
        self.net.down.add("A")
        reply = self.b.client.post("/distributed/runs/run1/stop")
        self.assertEqual(reply.status_code, 502)
        self.assertEqual(self.b.facts(), before)

    def test_a_run_on_this_node_is_stopped_here_and_an_unknown_run_is_404(self):
        self.assertEqual(self.b.client.post("/distributed/runs/nope/stop").status_code, 404)
        self.assertEqual(self.b.client.get("/distributed/runs/nope").status_code, 404)

    def test_a_run_moved_to_a_later_term_on_another_node_is_listed_once_with_the_later_term(self):
        self.a.runs.append(a_run())
        later = a_run()
        later.term, later.owner = 3, "B"
        self.b.runs.append(later)
        self.sync()
        for node in (self.a, self.b):
            rows = node.listed()
            self.assertEqual([(r["id"], r["term"], r["origin"]) for r in rows], [("run1", 3, "B")])

    def test_what_a_node_writes_is_small_and_only_under_its_own_origin(self):
        self.a.runs.append(a_run())
        for i in range(300):
            self.a.cluster._on_event("cluster", {"type": "note", "i": i})
        self.sync()
        events = self.a.state.get("A", "events", "node")["value"]
        self.assertEqual((len(events), events[-1]["i"]), (studio_facts.EVENTS_KEPT, 299))
        self.assertEqual({e["origin"] for e in self.a.state.entries() if e["origin"] != "B"}, {"A"})
        self.assertEqual({e["origin"] for e in self.b.state.entries(kind="run")}, {"A"})
        summary = self.b.state.get("A", "run", "run1")["value"]
        self.assertNotIn("timeline", summary)
        self.assertNotIn("events", summary)
        kinds = {e["kind"] for e in self.a.state.entries("A")}
        self.assertTrue({"heartbeat", "node", "queue", "leases", "events", "runs", "run", "timeline", "run_events"} <= kinds)

    def test_the_summary_keeps_the_last_rounds_only(self):
        row = {"id": "r", "timeline": [1], "events": [2], "history": list(range(80)), "round": 80}
        summary = studio_facts.run_summary(row)
        self.assertEqual((summary["history"][0], len(summary["history"]), "timeline" in summary, "events" in summary),
                         (30, studio_facts.HISTORY_KEPT, False, False))
        self.assertEqual(studio_facts.TIMELINES_KEPT, 8)

    def test_the_node_list_and_the_merged_queue_carry_each_nodes_origin_and_age(self):
        self.sync()
        self.clock.advance(5)
        with mock.patch.object(studio_cluster, "fetch_json", FakePeers(B=node_payload("B"))):
            rows = {r["origin"]: r for r in self.a.client.get("/cluster/nodes").json()}
        self.assertEqual((rows["A"]["origin"], rows["A"]["age_s"], rows["A"]["origin_down"]), ("A", 0., False))
        self.assertEqual((rows["B"]["origin"], rows["B"]["age_s"]), ("B", 5.))
        with mock.patch.object(studio_cluster_queue, "fetch", lambda address, path, timeout: {"node": node_payload("B"), "queue/jobs": []}[path]):
            nodes = self.a.client.get("/cluster/merged").json()["nodes"]
        by_name = {n["hostname"]: n for n in nodes}
        self.assertEqual((by_name["A"]["origin"], by_name["A"]["age_s"]), ("A", 0.))
        self.assertEqual((by_name["B"]["origin"], by_name["B"]["age_s"], by_name["B"]["origin_down"]), ("B", 5., False))

    def test_the_merged_queue_keeps_a_down_nodes_last_queue_from_the_store(self):
        self.sync()
        self.b.state.put("queue", "jobs", [{"id": "j1", "kind": "pretrain", "label": "x", "status": "waiting", "node": "", "owner": "",
                                           "note": "", "params": {}}])
        self.a.cluster.poll_once()  # B's queue entry reaches A (B's own poll would rewrite it)
        self.net.down.add("B")
        self.clock.advance(40)
        self.a.cluster.poll_once()

        def refuse(address, path, timeout):
            raise PeerError("B did not answer: unreachable")
        with mock.patch.object(studio_cluster_queue, "fetch", refuse):
            merged = self.a.client.get("/cluster/merged").json()
        node = next(n for n in merged["nodes"] if n["hostname"] == "B")
        self.assertEqual((node["origin"], node["origin_down"], node["age_s"]), ("B", True, 40.))
        self.assertIn("j1", [j["id"] for j in merged["jobs"]])


class OneNode(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
        self.node = Node(self, "solo", [], Net(), Clock())

    def test_runs_still_list_with_no_peers_and_a_poll_still_writes_the_heartbeat(self):
        self.assertEqual(self.node.listed(), [])
        self.node.runs.append(a_run())
        self.node.cluster.poll_once()
        rows = self.node.listed()
        self.assertEqual([(r["id"], r["origin"], r["age_s"], r["origin_down"]) for r in rows], [("run1", "solo", 0., False)])
        self.assertEqual(self.node.state.get("solo", "heartbeat", "node")["value"], {"beat": 1})
        self.node.cluster.poll_once()
        self.assertEqual(self.node.state.get("solo", "heartbeat", "node")["value"], {"beat": 2})
        self.assertEqual(self.node.client.get("/distributed/runs/run1").json()["origin"], "solo")


if __name__ == "__main__":
    unittest.main()

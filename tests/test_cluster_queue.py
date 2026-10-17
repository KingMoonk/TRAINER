"""studio_cluster_queue (the merged queue) and GET /cluster/merged, POST /queue/jobs/{id}/pin over FakeServices and fake peers."""
import time
import unittest
from unittest import mock

import studio_cluster_queue as cq
from studio_cluster import NodeInfo
from studio_peer import PeerError
from studio_queue import Job
from tests.sse_harness import ApiCase
from tests.test_api_cluster import node_payload

OWN = NodeInfo(address="local", hostname="here-box")


class Net:
    """Stands in for studio_cluster_queue.fetch: peers by address; a name maps to {path: payload} (online) or an exception (offline)."""

    def __init__(self, **peers):
        self.peers, self.calls = peers, []

    def __call__(self, address, path, timeout):
        self.calls.append((address, path, timeout))
        peer = self.peers[address]
        if isinstance(peer, Exception):
            raise peer
        return peer[path]


def peer(name, *jobs, busy=False):
    return {"node": node_payload(name, busy=busy), "queue/jobs": list(jobs)}


def job(job_id, status="waiting", **extra):
    return {"id": job_id, "kind": "pretrain", "label": job_id.upper(), "status": status, "params": {}, **extra}


DOWN = PeerError("down did not answer GET /node: Connection refused")


class Merge(unittest.TestCase):
    def merge(self, net, jobs=(), waiting=None, addresses=None):
        peers = cq.Peers().survey(addresses or list(net.peers), net, timeout=1)
        return peers, {r.id: r for r in cq.merge(OWN, peers, jobs, waiting)}

    def test_a_job_here_says_where_it_runs_and_a_waiting_one_where_it_is_pinned(self):
        net = Net(gpu=peer("gpu-box"))
        rows = self.merge(net, [Job("pretrain", "now", status="running", id="a"), Job("pretrain", "away", status="running", id="b", node="gpu"),
                                Job("pretrain", "pin", id="c", params={"pinned": "GPU-box"}), Job("pretrain", "any", id="d"),
                                Job("pretrain", "lost", id="e", params={"pinned": "ghost"})], {"e": "It is pinned to ghost."})[1]
        self.assertEqual([(r.node, r.hostname, r.queued_on, r.queued_here) for r in (rows[k] for k in "abcde")],
                         [("local", "here-box", "here-box", True), ("gpu", "gpu-box", "here-box", True), ("gpu", "gpu-box", "here-box", True),
                          ("", "", "here-box", True), ("", "ghost", "here-box", True)])
        self.assertEqual((rows["c"].pinned, rows["d"].pinned, rows["e"].waiting_reason), ("GPU-box", "", "It is pinned to ghost."))
        self.assertFalse(any(r.stale for r in rows.values()))

    def test_a_peers_jobs_are_rows_on_that_peer(self):
        net = Net(gpu=peer("gpu-box", job("p1", "running"), job("p2", params={"pinned": "here-box"}), job("p3", "held")))
        rows = self.merge(net)[1]
        self.assertEqual([(r.id, r.status, r.node, r.hostname, r.queued_on, r.queued_here) for r in rows.values()],
                         [("p1", "running", "gpu", "gpu-box", "gpu-box", False), ("p2", "waiting", "local", "here-box", "gpu-box", False),
                          ("p3", "held", "", "", "gpu-box", False)])
        self.assertEqual(rows["p2"].pinned, "here-box")

    def test_machines_that_share_a_hostname_get_their_address_in_the_name_and_others_keep_the_bare_hostname(self):
        net = Net(a=peer("twin", job("pa", "running")), b=peer("TWIN", job("pb", params={"pinned": "a"})), c=peer("gpu-box", job("pc")))
        rows = self.merge(net, [Job("pretrain", "away", status="running", id="m", node="b"), Job("pretrain", "pin", id="n", params={"pinned": "a"}),
                                Job("pretrain", "free", id="o")])[1]
        self.assertEqual([(r.id, r.hostname, r.queued_on) for r in rows.values()],
                         [("m", "TWIN (b)", "here-box"), ("n", "twin (a)", "here-box"), ("o", "", "here-box"), ("pa", "twin (a)", "twin (a)"),
                          ("pb", "twin (a)", "TWIN (b)"), ("pc", "", "gpu-box")])
        self.assertEqual((rows["n"].node, rows["pb"].node), ("a", "a"))  # the address is still what tells them apart

    def test_a_peer_with_this_machines_hostname_is_told_apart_and_this_machine_keeps_its_bare_name(self):
        net = Net(a=peer("here-box", job("pa", "running")))
        rows = self.merge(net, [Job("pretrain", "now", status="running", id="m")])[1]
        self.assertEqual([(r.id, r.hostname, r.queued_on) for r in rows.values()],
                         [("m", "here-box", "here-box"), ("pa", "here-box (a)", "here-box (a)")])

    def test_a_pin_to_a_hostname_two_machines_share_names_neither(self):
        net = Net(a=peer("twin"), b=peer("TWIN"))
        rows = self.merge(net, [Job("pretrain", "typed", id="t", params={"pinned": "Twin"}), Job("pretrain", "addr", id="u", params={"pinned": "b"})])[1]
        self.assertEqual((rows["t"].node, rows["t"].hostname, rows["t"].pinned), ("", "Twin", "Twin"))
        self.assertEqual((rows["u"].node, rows["u"].hostname), ("b", "TWIN (b)"))

    def test_an_offline_peer_with_no_known_name_does_not_collide(self):
        net = Net(a=DOWN, b=DOWN, c=peer("gpu-box", job("pc", "running")))
        rows = self.merge(net)[1]
        self.assertEqual([(r.id, r.hostname, r.queued_on) for r in rows.values()], [("pc", "gpu-box", "gpu-box")])

    def test_a_job_a_peer_runs_for_someone_else_or_that_runs_here_is_not_listed_twice(self):
        net = Net(gpu=peer("gpu-box", job("mine", "running", node="here-box"), job("guest", "running", owner="other:8787"),
                           job("theirs", "running", node="third")), third=peer("third"))
        rows = self.merge(net)[1]
        self.assertEqual(sorted(rows), ["theirs"])  # "mine" is this machine's own row; "guest" is its owner's
        self.assertEqual((rows["theirs"].node, rows["theirs"].queued_on), ("third", "gpu-box"))

    def test_nonsense_in_a_peers_queue_is_skipped(self):
        net = Net(gpu=peer("gpu-box", "text", {"status": "running"}, job("ok"), job("done", "done")))
        self.assertEqual(list(self.merge(net)[1]), ["ok"])

    def test_a_peer_that_stops_answering_keeps_its_last_rows_marked_stale(self):
        net = Net(gpu=peer("gpu-box", job("p1", "running")))
        clock = [100.]
        peers = cq.Peers(clock=lambda: clock[0])
        [fresh] = peers.survey(["gpu"], net, 1)
        self.assertEqual((fresh.info.online, fresh.stale, fresh.seen_at, len(fresh.jobs)), (True, False, 100., 1))
        net.peers["gpu"], clock[0] = DOWN, 160.
        [gone] = peers.survey(["gpu"], net, 1)
        self.assertEqual((gone.info.online, gone.info.hostname, gone.stale, gone.seen_at, [j["id"] for j in gone.jobs]),
                         (False, "gpu-box", True, 100., ["p1"]))
        self.assertIn("did not answer", gone.info.error)
        self.assertEqual([(r.id, r.stale) for r in cq.merge(OWN, [gone], [])], [("p1", True)])
        net.peers["gpu"] = peer("gpu-box")  # back, with an empty queue: the old rows go
        [back] = peers.survey(["gpu"], net, 1)
        self.assertEqual((back.info.online, back.stale, back.jobs), (True, False, ()))

    def test_a_peer_that_never_answered_has_no_rows_and_one_removed_from_the_list_is_forgotten(self):
        net = Net(gpu=peer("gpu-box", job("p1")), down=DOWN)
        peers = cq.Peers()
        gpu, down = peers.survey(["gpu", "down"], net, 1)
        self.assertEqual((down.info.online, down.stale, down.jobs, down.seen_at), (False, True, (), 0.))
        self.assertEqual(len(gpu.jobs), 1)
        peers.survey(["down"], net, 1)
        net.peers["gpu"] = DOWN
        [again] = peers.survey(["gpu"], net, 1)
        self.assertEqual(again.jobs, ())  # it was dropped from the list, so its old rows are not remembered

    def test_a_node_body_that_is_not_a_node_is_offline_and_a_queue_that_is_not_a_list_keeps_the_last_one(self):
        net = Net(odd={"node": {"hello": 1}, "queue/jobs": []}, gpu=peer("gpu-box", job("p1")))
        peers = cq.Peers()
        odd, gpu = peers.survey(["odd", "gpu"], net, 1)
        self.assertFalse(odd.info.online)
        self.assertIn("not like a node", odd.info.error)
        net.peers["gpu"] = {"node": node_payload("gpu-box"), "queue/jobs": {"detail": "no"}}
        [gpu] = peers.survey(["gpu"], net, 1)
        self.assertEqual((gpu.info.online, gpu.stale, [j["id"] for j in gpu.jobs]), (True, True, ["p1"]))

    def test_a_peer_that_hangs_costs_one_timeout_and_the_others_are_not_held_up(self):
        def fetch(address, path, timeout):
            if address == "slow":
                time.sleep(3)
            return peer("gpu-box")[path]

        started = time.monotonic()
        slow, ok = cq.Peers().survey(["slow", "ok"], fetch, .3)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual((slow.info.online, ok.info.online), (False, True))
        self.assertIn("did not answer within 0.3 seconds", slow.info.error)


class QueueRoute(ApiCase):
    prefix = "/cluster/merged"

    def setUp(self):
        super().setUp()
        self.net = Net(gpu=peer("gpu-box", job("p1", "running")), down=DOWN)
        self.enterContext(mock.patch.object(cq, "fetch", self.net))
        self.enterContext(mock.patch.object(cq.socket, "gethostname", lambda: "here-box"))

    def test_a_cluster_of_one_has_no_nodes_and_no_rows_and_asks_nobody(self):
        self.call("POST", "/queue/jobs", 201, json={"kind": "pretrain", "label": "mine"})
        self.assertEqual(self.call("GET", "/cluster/merged"), {"nodes": [], "jobs": []})
        self.assertEqual(self.net.calls, [])

    def test_both_machines_jobs_come_back_each_with_its_node(self):
        self.call("PUT", "/cluster/nodes", json={"addresses": ["gpu"]})
        mine = self.call("POST", "/queue/jobs", 201, json={"kind": "pretrain", "label": "mine"})
        body = self.call("GET", "/cluster/merged")
        self.assertEqual([(n["address"], n["hostname"], n["pin"], n["local"], n["online"]) for n in body["nodes"]],
                         [("local", "here-box", "here-box", True, True), ("gpu", "gpu-box", "gpu", False, True)])
        self.assertEqual([(r["id"], r["status"], r["hostname"], r["queued_on"], r["queued_here"], r["stale"]) for r in body["jobs"]],
                         [(mine["id"], "waiting", "", "here-box", True, False), ("p1", "running", "gpu-box", "gpu-box", False, False)])

    def test_this_machines_pin_is_its_address_when_a_peer_shares_its_hostname(self):
        self.net.peers["gpu"]["node"] = node_payload("HERE-box")
        self.call("PUT", "/cluster/nodes", json={"addresses": ["gpu"]})
        body = self.call("GET", "/cluster/merged")
        self.assertEqual([(n["address"], n["pin"]) for n in body["nodes"]], [("local", "local"), ("gpu", "gpu")])

    def test_pinning_a_waiting_job_to_the_other_machine_changes_where_its_row_says_it_runs(self):
        self.call("PUT", "/cluster/nodes", json={"addresses": ["gpu"]})
        mine = self.call("POST", "/queue/jobs", 201, json={"kind": "pretrain", "label": "mine"})
        row = lambda: next(r for r in self.call("GET", "/cluster/merged")["jobs"] if r["id"] == mine["id"])  # noqa: E731
        self.assertEqual((row()["node"], row()["hostname"], row()["pinned"]), ("", "", ""))
        state = self.call("POST", "/queue/jobs/{job_id}/pin", path={"job_id": mine["id"]}, json={"pinned": "gpu-box"})
        self.assertEqual(state["up_next"][0]["params"]["pinned"], "gpu-box")
        self.assertEqual((row()["node"], row()["hostname"], row()["pinned"]), ("gpu", "gpu-box", "gpu-box"))
        self.assertIn("queue", self.emitted)
        self.call("POST", "/queue/jobs/{job_id}/pin", path={"job_id": mine["id"]}, json={"pinned": " here-box "})
        self.assertEqual((row()["node"], row()["hostname"]), ("local", "here-box"))
        self.call("POST", "/queue/jobs/{job_id}/pin", path={"job_id": mine["id"]}, json={"pinned": None})
        self.assertEqual((row()["node"], row()["pinned"]), ("", ""))
        self.assertNotIn("pinned", self.services.queue().up_next[0].params)

    def test_pinning_keeps_the_rest_of_the_jobs_params_and_an_empty_pin_clears_it(self):
        self.call("PUT", "/cluster/nodes", json={"addresses": ["gpu"]})
        mine = self.call("POST", "/queue/jobs", 201, json={"kind": "pretrain", "params": {"version": 3}})
        self.call("POST", "/queue/jobs/{job_id}/pin", path={"job_id": mine["id"]}, json={"pinned": "gpu"})
        self.assertEqual(self.services.queue().up_next[0].params, {"version": 3, "pinned": "gpu"})
        self.call("POST", "/queue/jobs/{job_id}/pin", path={"job_id": mine["id"]}, json={"pinned": ""})
        self.assertEqual(self.services.queue().up_next[0].params, {"version": 3})

    def test_a_job_that_is_not_waiting_cannot_be_pinned(self):
        self.call("POST", "/queue/jobs/{job_id}/pin", 404, path={"job_id": "nope"}, json={"pinned": "gpu"})
        self.call("POST", "/queue/jobs/{job_id}/pin", 422, path={"job_id": "nope"}, json={"pinned": "x" * 201})

    def test_a_peer_that_stops_answering_keeps_its_rows_greyed(self):
        self.call("PUT", "/cluster/nodes", json={"addresses": ["gpu"]})
        self.call("GET", "/cluster/merged")
        self.net.peers["gpu"] = DOWN
        body = self.call("GET", "/cluster/merged")
        gpu = body["nodes"][1]
        self.assertEqual((gpu["online"], gpu["hostname"], gpu["pin"], gpu["seen_at"] > 0), (False, "gpu-box", "gpu", True))
        self.assertIn("did not answer", gpu["error"])
        self.assertEqual([(r["id"], r["hostname"], r["stale"]) for r in body["jobs"]], [("p1", "gpu-box", True)])

    def test_the_route_is_sync_so_the_peers_are_asked_off_the_event_loop(self):
        import inspect

        from studio_api.routers import cluster
        self.assertFalse(inspect.iscoroutinefunction(cluster.merged_queue))


if __name__ == "__main__":
    unittest.main()

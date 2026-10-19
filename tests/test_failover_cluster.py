"""Coordinator failover end to end in one process (docs/distributed.md, "Terms and coordinator failover"): three nodes, each with its own app
(the real /collective routes over a TestClient), its own Cluster and observation store, and a member of the run training in a thread.
The member runs the worker's own code (studio_jobs.diloco_member), so after a failover it finds the new coordinator by the where-file
the app wrote, with studio_peer.PeerClient replaced by one that maps each node's URLs to its TestClient: a where-file naming the wrong
address fails the test. The network between the nodes is fake and cut by hand; the Cluster's clock is a fake one the test moves, so
the failure detector and the members' leases lapse without sleeping. Nothing opens a window or a socket."""
import dataclasses
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

import studio_diloco
import studio_distributed
import studio_failover
import studio_facts
import studio_jobs
from studio_peer import PeerError
from studio_queue import Runner
from studio_remote import Cluster
from tests import diloco_support as support
from tests.fake_cluster import Clock, Net
from tests.test_api_cluster import node_payload
from tests.test_queue_cluster import own_node

RUN = "run1"
ROUNDS = 40
CUT_AT = 3  # the cable to the coordinator is pulled when it opens this round
NAMES = ("n0", "n1", "n2")
PORTS = {"n0": 9100, "n1": 9101, "n2": 9102}  # not studio_peer.DEFAULT_PORT: the where-file must name the port the API listens on


def peer_url(name):
    """How the peers reach `name` (its tailnet name, where `tailscale serve` listens): not the bare name the workspace lists."""
    return f"https://{name}.tail.ts.net"


def local_url(name):
    return f"http://127.0.0.1:{PORTS[name]}"


class Cable(support.Peer):
    """A member's line to the first coordinator: it works until `cut` is set, which happens the first time `trigger()` is true."""

    def __init__(self, client, cut, trigger):
        super().__init__(client)
        self.cut, self.trigger = cut, trigger

    def _line(self, path):
        if not self.cut.is_set() and self.trigger():
            self.cut.set()
        if self.cut.is_set():
            raise PeerError(f"unreachable /{path}")

    def get_json(self, path, params=None, timeout=None):
        self._line(path)
        return super().get_json(path, params, timeout)

    def post_json(self, path, body=None, params=None, timeout=None):
        self._line(path)
        return super().post_json(path, body, params, timeout)

    def get_bytes(self, path, params=None, timeout=None):
        self._line(path)
        return super().get_bytes(path, params, timeout)

    def post_bytes(self, path, data, params=None, timeout=None):
        self._line(path)
        return super().post_bytes(path, data, params, timeout)


class Node:
    """One in-process node: a Runner with its Cluster, the app serving /collective from its own coordinator registry, and a member."""

    def __init__(self, fleet, name):
        self.fleet, self.name, self.registry = fleet, name, {}
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        fleet.case.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "data").mkdir()
        peers = [other for other in NAMES if other != name]
        (self.root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": peers}}), encoding="utf-8")
        info = dataclasses.replace(own_node(), hostname=name)
        self.net = fleet.nets[name]
        self.runner = Runner(self.root / "data" / "queue.json", {}, tick=.01, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: None, own=lambda: info, clock=fleet.clock, address=peer_url(name), threads=False,
            hostname=name, coordinators=self.registry))
        fleet.case.addCleanup(self.runner.shutdown)
        self.cluster = self.runner.cluster
        self.cluster.serve_at(PORTS[name])
        app = support.app()
        app.state.coordinators = self.registry
        self.client = fleet.case.enterContext(TestClient(app))
        self.stop, self.thread, self.error, self.notes = threading.Event(), None, None, []
        fleet.handlers[name] = self.serve

    def serve(self, method, path, body):
        if path == "node":
            return node_payload(self.name)
        if path == "state/sync":
            return self.cluster.state.handle_sync(json.loads(json.dumps(body)))
        raise PeerError(f"answered 404: {path}", status=404, detail=path)

    def where(self):
        """The where-file the app on this node wrote for the run, or None."""
        path = studio_diloco.where_file(self.root, RUN)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def replica(self):
        return dict(self.cluster.replicas.get(RUN) or {})

    def start_member(self, url):
        def emit(event):
            if event.get("type") == "info" and not event.get("text", "").startswith(("Round", "Holding")):
                self.notes.append(event.get("text"))
            if event.get("type") in ("diloco_member", "diloco_replica"):
                self.cluster._on_event("train", event)

        job = SimpleNamespace(id=f"member_{self.name}", params={"coordinator_url": url, "run": RUN, "node": self.name, "epoch": 0,
                                                              "output_dir": str(self.root / "out"),
                                                              "member_options": {"wait": .5, "sleep": lambda seconds: time.sleep(.05)}})
        context = SimpleNamespace(root=self.root, run_dir=self.root / "runs")

        def work():
            try:
                studio_jobs.diloco_member(job, context, emit, self.stop, threading.Event())
            except Exception as exc:  # noqa: BLE001 - reported by the test
                self.error = exc
        self.thread = threading.Thread(target=work, name=f"member {self.name}", daemon=True)
        self.thread.start()


class Fleet:
    """Three nodes whose coordinator, n0, runs the run until its cable is pulled."""

    def __init__(self, case, handoff="", **settings):
        self.case, self.clock = case, Clock()
        self.handlers, self.nets = {}, {name: Net() for name in NAMES}
        for net in self.nets.values():
            net.handlers = self.handlers  # one set of handlers, one `down` set per caller: a partition can be between two nodes only
        self.nodes = {name: Node(self, name) for name in NAMES}
        self.cut = threading.Event()
        self.line = Cable(self.nodes["n0"].client, self.cut, lambda: not handoff and self.first.next_round >= CUT_AT and self.ready("n1", "n2"))
        self.urls = {url(name): name for name in NAMES for url in (peer_url, local_url)}
        case.enterContext(mock.patch("studio_peer.PeerClient", self.client_for))  # the worker's PeerClient, in studio_jobs and file_locator
        studio_distributed.RUNS.clear()
        case.addCleanup(studio_distributed.RUNS.clear)
        studio_distributed.RUNS[RUN] = studio_distributed.Run(RUN, "Version 1 across 3 machines", 1, 100., {"rounds": ROUNDS}, "n0", handoff=handoff)
        self.first = support.build(self.nodes["n0"].root / "run", RUN, inner_steps=2, rounds=ROUNDS, outer_lr=.7, outer_momentum=.9,
                                   min_members=1, max_members=3, replicas=3, delta_dtype="float32", **settings)
        self.first.prefer = (handoff,) if handoff else ()  # (studio_diloco.prepare)
        self.nodes["n0"].registry[RUN] = self.first
        case.addCleanup(self.stop)

    def ready(self, *names):
        return all(self.nodes[n].replica().get("ready") and self.nodes[n].replica().get("round", 0) >= 1 for n in names)

    def client_for(self, url, **options):
        """studio_peer.PeerClient for the workers: n0's URLs go through the cable to the first coordinator, the others to their node."""
        name = self.urls[url]  # KeyError: a URL no node serves at
        return self.line if name == "n0" else support.Peer(self.nodes[name].client)

    def start_members(self):
        for name, node in self.nodes.items():
            node.start_member(local_url("n0") if name == "n0" else peer_url("n0"))
            time.sleep(.2)  # joins in order: n0 has rank 0

    def sync(self, *names, seconds=0.):
        """One poll of each named node (the live ones by default), after moving the fake clock on."""
        self.clock.advance(seconds)
        for name in names or [n for n in NAMES if n not in self.dead]:
            self.nodes[name].cluster.poll_once()

    dead = ()

    def wait_for(self, what, condition, limit=180., names=(), seconds=.2):
        """Poll until `condition()`, moving the fake clock a little each time. Fails with `what` when it takes longer than `limit`."""
        end = time.monotonic() + limit
        while time.monotonic() < end:
            self.sync(*names, seconds=seconds)
            if condition():
                return
            time.sleep(.05)
        state = {n: (RUN in nd.registry and (nd.registry[RUN].term, nd.registry[RUN].superseded, nd.registry[RUN].finished, nd.registry[RUN].next_round),
                     nd.cluster.claims.get(RUN), nd.replica(), nd.thread.is_alive() if nd.thread else None, repr(nd.error), nd.notes[-6:]) for n, nd in self.nodes.items()}
        self.case.fail(f"Timed out waiting for {what}: {state}")

    def pull_the_cable(self):
        """n0 and its member are gone for good (until `heal`)."""
        self.dead = ("n0",)
        for net in self.nets.values():
            net.down.add("n0")
        self.nodes["n0"].stop.set()

    def lapse(self):
        """Let the coordinator's silence outlast the failure detector and then the members' leases, polling the live nodes in between so
        their own facts stay fresh, as they are while the app runs."""
        for seconds in (studio_facts.DOWN_AFTER + 1, 6., 6.):
            self.sync(seconds=seconds)
            self.sync()

    def live(self):
        """The coordinators of the run that are serving (not stepped down)."""
        return {name: node.registry[RUN] for name, node in self.nodes.items() if RUN in node.registry and not node.registry[RUN].superseded}

    def stop(self):
        for node in self.nodes.values():
            node.stop.set()
        for node in self.nodes.values():
            if node.thread is not None:
                node.thread.join(30)


class Failover(unittest.TestCase):
    def setUp(self):
        self.fleet = Fleet(self)
        self.addCleanup(studio_diloco.COORDINATORS.clear)

    def expected_successor(self):
        held = {name: self.fleet.nodes[name].replica() for name in ("n1", "n2")}
        self.assertTrue(all(found.get("held") for found in held.values()), held)
        best = min(held.items(), key=lambda item: (-item[1]["round"], item[1]["rank"]))
        return best[0], best[1]["round"]

    def test_the_lowest_ranked_replica_takes_over_at_term_2_the_run_converges_and_the_returning_coordinator_steps_down(self):
        fleet = self.fleet
        fleet.start_members()
        fleet.wait_for("the cable to be pulled and two whole replicas", fleet.cut.is_set, names=())
        self.assertFalse(fleet.first.finished)
        self.assertGreaterEqual(fleet.first.next_round, CUT_AT)
        fleet.pull_the_cable()
        chosen, last_round = self.expected_successor()
        before = {n: fleet.nodes[n].replica()["round"] for n in ("n1", "n2")}

        fleet.lapse()
        fleet.wait_for("a coordinator of term 2", lambda: any(c.term == 2 for c in fleet.live().values()))

        (winner, coordinator), = [(n, c) for n, c in fleet.live().items() if n != "n0"]
        self.assertEqual(winner, chosen)  # the highest round held, ties to the lowest rank
        self.assertEqual(coordinator.term, 2)
        claim = fleet.nodes[winner].cluster.claims[RUN]
        self.assertEqual((claim["term"], claim["round"]), (2, last_round))
        self.assertGreaterEqual(last_round, max(before.values()) - 1)
        # the other node saw the claim in its store and follows it
        other = "n2" if winner == "n1" else "n1"
        fleet.wait_for("the other node to hold the claim", lambda: (fleet.nodes[other].cluster.state.get(winner, "claim", RUN) or {}).get("value", {}).get("term") == 2)
        # the where-files the members read: the successor's own API port on its node, its published address on the other
        self.assertEqual(fleet.nodes[winner].where(), {"node": winner, "term": 2, "url": local_url(winner)})
        fleet.wait_for("the other node's where-file", lambda: fleet.nodes[other].where() is not None)
        self.assertEqual(fleet.nodes[other].where(), {"node": winner, "term": 2, "url": peer_url(winner)})

        fleet.wait_for("the run to finish under term 2", lambda: coordinator.finished and not fleet.nodes["n1"].thread.is_alive() and not fleet.nodes["n2"].thread.is_alive())
        self.assertEqual([n.error for n in (fleet.nodes["n1"], fleet.nodes["n2"])], [None, None])
        rounds = [row["round"] for row in coordinator.history]
        self.assertEqual(rounds, list(range(ROUNDS)))  # continued from the last completed round: none repeated, none skipped
        self.assertLess(coordinator.history[-1]["val_loss"], coordinator.history[0]["val_loss"])  # and kept converging
        self.assertEqual(coordinator.snapshot()["round"], ROUNDS)

        # the old coordinator comes back with its coordinator still in its registry: its first poll shows the higher claim
        old = fleet.first
        self.assertEqual(old.term, 1)
        self.assertFalse(old.superseded)
        for net in fleet.nets.values():
            net.down.discard("n0")
        fleet.dead = ()
        fleet.nodes["n0"].cluster.poll_once()
        self.assertNotIn(RUN, fleet.nodes["n0"].registry)
        self.assertGreaterEqual(old.superseded, 2)
        self.assertEqual(old.theta, {})  # its rounds are discarded
        self.assertEqual(fleet.nodes["n0"].cluster.claims.get(RUN), {"released": True})
        self.assertEqual(list(fleet.live()), [winner])  # one coordinator of the run, one theta
        row = fleet.nodes["n0"].cluster.state.get(winner, "run", RUN)["value"]
        self.assertEqual(row["term"], 2)


class Handoff(unittest.TestCase):
    """A run started with another machine chosen to coordinate it: n0 starts it, n1 is chosen, and n0 stays up throughout."""

    def setUp(self):
        self.fleet = Fleet(self, handoff="n1")
        self.addCleanup(studio_diloco.COORDINATORS.clear)

    def test_the_chosen_node_takes_the_run_over_while_the_starter_is_up_and_the_members_follow_it(self):
        fleet = self.fleet
        fleet.start_members()
        self.assertEqual(studio_diloco.replica_nodes(["n0", "n1", "n2"], 1, fleet.first.prefer), ["n1"])
        fleet.wait_for("n1 to coordinate at term 2", lambda: RUN in fleet.nodes["n1"].registry and fleet.nodes["n1"].registry[RUN].term == 2)
        self.assertEqual(fleet.nodes["n1"].cluster.claims[RUN]["term"], 2)
        self.assertGreaterEqual(fleet.nodes["n1"].cluster.claims[RUN]["round"], 1)  # from a round it held, never from nothing
        self.assertFalse(fleet.first.finished)  # n0 was up: nobody waited for it to be lost
        fleet.wait_for("n0 to step down", lambda: RUN not in fleet.nodes["n0"].registry)
        self.assertGreaterEqual(fleet.first.superseded, 2)
        self.assertEqual(fleet.nodes["n0"].cluster.claims[RUN], {"released": True})
        fleet.wait_for("n0's where-file", lambda: fleet.nodes["n0"].where() is not None)
        self.assertEqual(fleet.nodes["n0"].where(), {"node": "n1", "term": 2, "url": peer_url("n1")})  # the starter's member finds the chosen node
        self.assertEqual(fleet.nodes["n1"].where(), {"node": "n1", "term": 2, "url": local_url("n1")})
        coordinator = fleet.nodes["n1"].registry[RUN]
        fleet.wait_for("the run to finish under n1", lambda: coordinator.finished and not any(n.thread.is_alive() for n in fleet.nodes.values()))
        self.assertEqual([n.error for n in fleet.nodes.values()], [None, None, None])
        self.assertEqual([row["round"] for row in coordinator.history], list(range(ROUNDS)))
        self.assertEqual(list(fleet.live()), ["n1"])  # one coordinator, and the starter's member trained in it to the end
        self.assertIn("n0", {node for row in coordinator.history[-3:] for node in row["contributed"]})


class Conflict(unittest.TestCase):
    def setUp(self):
        self.fleet = Fleet(self)
        self.addCleanup(studio_diloco.COORDINATORS.clear)

    def test_two_claims_of_the_same_term_settle_on_the_higher_one_and_the_loser_s_rounds_are_gone(self):
        fleet = self.fleet
        fleet.start_members()
        fleet.wait_for("the cable to be pulled and two whole replicas", fleet.cut.is_set)
        fleet.pull_the_cable()
        # n1 and n2 lose each other as well: each sees itself as the only live replica and claims term 2
        fleet.nets["n1"].down.add("n2")
        fleet.nets["n2"].down.add("n1")
        fleet.lapse()
        fleet.wait_for("both to claim", lambda: all(fleet.nodes[n].registry.get(RUN) is not None and fleet.nodes[n].registry[RUN].term == 2 for n in ("n1", "n2")),
                       names=("n1", "n2"))
        claims = {n: fleet.nodes[n].cluster.failover.host.claims[RUN] for n in ("n1", "n2")}
        a, b = claims["n1"], claims["n2"]
        win, lose = (a, b) if studio_failover.beats(a, b) else (b, a)
        self.assertEqual((a.term, b.term), (2, 2))
        winner, loser = fleet.nodes[win.node].registry[RUN], fleet.nodes[lose.node].registry[RUN]
        self.assertIsNot(winner, loser)
        fleet.wait_for("each side to close a round of its own", lambda: winner.next_round > win.round and loser.next_round > lose.round,
                       names=("n1", "n2"))
        loser_rounds = len(loser.history)
        self.assertGreater(loser_rounds, lose.round)  # the rounds the loser will lose

        fleet.nets["n1"].down.discard("n2")  # the partition heals
        fleet.nets["n2"].down.discard("n1")
        fleet.wait_for("the loser to step down", lambda: RUN not in fleet.nodes[lose.node].registry, names=("n1", "n2"))
        self.assertEqual([n for n in fleet.live() if n != "n0"], [win.node])  # (n0 is still cut off)
        self.assertGreaterEqual(loser.superseded, 2)
        self.assertEqual(loser.theta, {})  # one theta remains
        self.assertTrue(winner.theta)
        self.assertEqual(fleet.nodes[lose.node].cluster.claims[RUN], {"released": True})

        fleet.wait_for("the run to finish on the winner", lambda: winner.finished and not fleet.nodes["n1"].thread.is_alive()
                       and not fleet.nodes["n2"].thread.is_alive(), names=("n1", "n2"))
        self.assertEqual([n.error for n in (fleet.nodes["n1"], fleet.nodes["n2"])], [None, None])
        self.assertEqual([row["round"] for row in winner.history], list(range(ROUNDS)))  # the loser's rounds are not in it


if __name__ == "__main__":
    unittest.main()

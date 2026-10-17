"""studio_state: the observation store's merge, sync, restart and pruning rules over fake clocks and fake peers, and its route."""
import json
import random
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import studio_cluster
import studio_state as state
from studio_api.routers.state import router
from studio_ia.services import FakeServices
from studio_peer import PeerError
from studio_queue import Runner
from studio_remote import Cluster
from tests.fake_cluster import Clock, Net
from tests.sse_harness import ApiCase
from tests.test_api_cluster import ClusterServices, node_payload, probed
from tests.test_queue_cluster import own_node


class Mesh:
    """Replicas on one fake clock, joined by links that can be cut, lose calls and deliver them twice."""

    def __init__(self, names, seed=0, **options):
        self.clock, self.rng = Clock(), random.Random(seed)
        self.replicas = {n: state.Replica(n, 1, self.clock, **options) for n in names}
        self.cut, self.loss, self.dup = set(), 0., 0.

    def client(self, source, target):
        mesh = self

        class Link:
            def post_json(self, path, body=None, params=None, timeout=None):
                if frozenset((source, target)) in mesh.cut or mesh.rng.random() < mesh.loss:
                    raise PeerError(f"{target} did not answer POST /{path}: unreachable")
                body = json.loads(json.dumps(body))  # the wire
                reply = mesh.replicas[target].handle_sync(body)
                if mesh.rng.random() < mesh.dup:
                    mesh.replicas[target].handle_sync(body)  # delivered twice
                return json.loads(json.dumps(reply))
        return Link()

    def sync(self, source, target):
        try:
            state.sync_peer(self.replicas[source], self.client(source, target))
        except PeerError:
            pass

    def settle(self, rounds=10):
        """Heal every link and sync every pair until nothing changes."""
        self.cut, self.loss, self.dup = set(), 0., 0.
        names = list(self.replicas)
        for _ in range(rounds):
            before = [r.contents() for r in self.replicas.values()]
            for a in names:
                for b in names:
                    if a != b:
                        self.sync(a, b)
            if before == [r.contents() for r in self.replicas.values()]:
                return
        self.fail("did not settle")

    def same(self):
        views = [r.contents() for r in self.replicas.values()]
        return all(v == views[0] for v in views)


def wire(origin, kind, key, inc, seq, value, age=0.):
    return {"origin": origin, "kind": kind, "key": key, "incarnation": inc, "seq": seq, "value": value, "age": age}


class Convergence(unittest.TestCase):
    def test_random_order_duplication_loss_and_partitions_converge_after_the_links_heal(self):
        names = ["a", "b", "c", "d", "e"]
        for seed in range(40):
            with self.subTest(seed=seed):
                mesh, rng = Mesh(names, seed), random.Random(seed)
                mesh.loss, mesh.dup = .3, .3
                expected = {}  # (origin, kind, key) -> the value of the latest write
                for step in range(120):
                    action = rng.random()
                    if action < .35:
                        origin = rng.choice(names)
                        kind, key, value = rng.choice(["node", "run"]), rng.choice("xyz"), rng.randrange(1000)
                        mesh.replicas[origin].put(kind, key, {"n": value})
                        expected[origin, kind, key] = {"n": value}
                    elif action < .45:
                        if rng.random() < .3:
                            mesh.cut = set()
                        else:
                            mesh.cut.add(frozenset(rng.sample(names, 2)))
                    elif action < .5:  # a restart: a new incarnation, and its first write replaces everything the old one left
                        origin = rng.choice(names)
                        old = mesh.replicas[origin]
                        mesh.replicas[origin] = state.Replica(origin, old.incarnation + 1, mesh.clock)
                        for k in [k for k in expected if k[0] == origin]:
                            del expected[k]
                        mesh.replicas[origin].put("node", "boot", {"n": step})
                        expected[origin, "node", "boot"] = {"n": step}
                    else:
                        a, b = rng.sample(names, 2)
                        mesh.sync(a, b)
                    mesh.clock.advance(rng.random())
                mesh.settle()
                self.assertTrue(mesh.same())
                held = {(r["origin"], r["kind"], r["key"]): r["value"] for r in mesh.replicas["a"].entries()}
                self.assertEqual(held, expected)

    def test_merge_in_any_order_and_any_number_of_times_gives_the_same_state(self):
        rows = [wire("o", "k", "x", 1, 1, 1), wire("o", "k", "x", 1, 2, 2), wire("o", "k", "y", 1, 3, 3), wire("p", "k", "x", 1, 1, 4),
                wire("o", "k", "x", 2, 1, 5), wire("p", "k", "z", 1, 2, 6)]
        views = set()
        for seed in range(50):
            rng = random.Random(seed)
            replica = state.Replica("me", 1, Clock())
            for _ in range(3):
                shuffled = rows[:]
                rng.shuffle(shuffled)
                replica.merge(shuffled[:rng.randint(1, len(rows))])
            replica.merge(rows)
            replica.merge(rows)
            views.add(json.dumps(replica.contents()))
        self.assertEqual(len(views), 1)

    def test_a_fact_reaches_a_node_that_cannot_reach_its_origin_through_a_third(self):
        mesh = Mesh(["a", "b", "c"])
        mesh.cut.add(frozenset(("a", "c")))
        mesh.replicas["a"].put("node", "self", {"device": "cuda"})
        mesh.sync("c", "a")
        self.assertEqual(mesh.replicas["c"].entries(), [])
        mesh.sync("b", "a")
        mesh.clock.advance(5)
        mesh.sync("c", "b")
        got = mesh.replicas["c"].get("a", "node", "self")
        self.assertEqual((got["value"], got["age"]), ({"device": "cuda"}, 5.))  # the age includes the relay delay
        self.assertTrue(mesh.replicas["c"].vector() == mesh.replicas["a"].vector())

    def test_an_empty_vector_gets_the_whole_state_and_a_current_one_nothing(self):
        replica = state.Replica("a", 1, Clock())
        replica.put("node", "self", 1)
        replica.put("run", "r1", 2)
        self.assertEqual(len(replica.delta({})), 2)
        self.assertEqual(replica.delta(replica.vector()), [])
        self.assertEqual([r["key"] for r in replica.delta({"a": [1, 1]})], ["r1"])

    def test_sync_pushes_back_what_the_peer_lacks(self):
        mesh = Mesh(["a", "b"])
        mesh.replicas["a"].put("node", "self", "from a")
        mesh.replicas["b"].put("node", "self", "from b")
        mesh.sync("a", "b")  # one call from a: it pulls b's entry and pushes its own
        self.assertTrue(mesh.same())
        self.assertEqual(len(mesh.replicas["b"].entries()), 2)


class Versions(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.replica = state.Replica("me", 1, self.clock)

    def test_the_higher_version_wins_and_a_lower_one_is_ignored(self):
        self.assertEqual(self.replica.merge([wire("o", "k", "x", 1, 5, "new")]), 1)
        self.assertEqual(self.replica.merge([wire("o", "k", "x", 1, 4, "old")]), 0)
        self.assertEqual(self.replica.get("o", "k", "x")["value"], "new")

    def test_a_newer_incarnation_replaces_all_of_the_origins_older_entries(self):
        self.replica.merge([wire("o", "k", "x", 1, 1, "x1"), wire("o", "k", "y", 1, 2, "y1")])
        self.replica.merge([wire("o", "k", "x", 2, 1, "x2")])
        self.assertEqual([(r["key"], r["value"], r["incarnation"]) for r in self.replica.entries("o")], [("x", "x2", 2)])
        self.assertEqual(self.replica.vector(), {"o": [2, 1]})

    def test_a_relayed_entry_of_an_old_incarnation_is_refused(self):
        self.replica.merge([wire("o", "k", "x", 2, 1, "new")])
        self.assertEqual(self.replica.merge([wire("o", "k", "y", 1, 9, "stale")]), 0)
        self.assertIsNone(self.replica.get("o", "k", "y"))

    def test_only_the_origin_writes_its_entries(self):
        self.replica.put("k", "x", "mine")
        self.assertEqual(self.replica.merge([wire("me", "k", "x", 9, 9, "forged"), wire("me", "k", "y", 1, 9, "forged")]), 0)
        self.assertEqual([(r["key"], r["value"]) for r in self.replica.entries()], [("x", "mine")])

    def test_a_restart_replaces_the_stale_entries_the_others_hold(self):
        mesh = Mesh(["a", "b"])
        mesh.replicas["a"].put("run", "old", 1)
        mesh.replicas["a"].put("node", "self", 1)
        mesh.sync("b", "a")
        self.assertEqual(len(mesh.replicas["b"].entries()), 2)
        mesh.replicas["a"] = state.Replica("a", 2, mesh.clock)  # restarted: seq starts over, the incarnation is higher
        mesh.replicas["a"].put("node", "self", 2)
        mesh.sync("b", "a")
        self.assertEqual([(r["key"], r["value"], r["incarnation"]) for r in mesh.replicas["b"].entries()], [("self", 2, 2)])
        mesh.sync("a", "b")  # b offers nothing of a's old incarnation back
        self.assertEqual([r["key"] for r in mesh.replicas["a"].entries()], ["self"])

    def test_malformed_rows_are_skipped_and_the_rest_taken(self):
        rows = [None, {}, wire("o", "k", "a", "x", 1, 1), wire("o", "k", "b", 1, 0, 1), wire("o", "k", "c", 1, 1, 1, age=float("nan")),
                wire("o", "k", "d", 1, 1, "fine")]
        self.assertEqual(self.replica.merge(rows), 1)
        self.assertEqual(self.replica.get("o", "k", "d")["value"], "fine")

    def test_a_value_that_is_not_json_is_refused_when_written(self):
        with self.assertRaises(TypeError):
            self.replica.put("k", "x", object())


class Ages(unittest.TestCase):
    def test_an_age_grows_on_the_local_clock_from_the_age_it_arrived_with(self):
        clock = Clock()
        replica = state.Replica("me", 1, clock)
        replica.merge([wire("o", "k", "x", 1, 1, 1, age=5.)])
        clock.advance(10)
        self.assertEqual(replica.get("o", "k", "x")["age"], 15.)
        self.assertEqual(replica.delta({})[0]["age"], 15.)

    def test_the_same_write_seen_again_by_a_shorter_way_has_the_smaller_age(self):
        clock = Clock()
        replica = state.Replica("me", 1, clock)
        replica.merge([wire("o", "k", "x", 1, 1, 1, age=9.)])
        replica.merge([wire("o", "k", "x", 1, 1, 1, age=2.)])
        replica.merge([wire("o", "k", "x", 1, 1, 1, age=30.)])
        self.assertEqual(replica.get("o", "k", "x")["age"], 2.)

    def test_a_node_reads_its_own_entry_as_fresh_as_when_it_wrote_it(self):
        clock = Clock()
        replica = state.Replica("me", 1, clock)
        replica.put("k", "x", 1)
        clock.advance(3)
        self.assertEqual(replica.get("me", "k", "x")["age"], 3.)


class Pruning(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.replica = state.Replica("me", 1, self.clock, prune_after=100, tombstone_for=1000)
        self.replica.put("node", "self", "mine")

    def test_an_origin_silent_for_prune_after_is_pruned_and_this_nodes_own_entries_are_not(self):
        self.replica.merge([wire("o", "k", "x", 1, 1, 1)])
        self.clock.advance(99)
        self.assertEqual(len(self.replica.entries()), 2)
        self.clock.advance(1)
        self.assertEqual([r["origin"] for r in self.replica.entries()], ["me"])
        self.assertEqual(list(self.replica.vector()), ["me"])

    def test_a_newest_entry_keeps_the_origins_older_ones(self):
        self.replica.merge([wire("o", "k", "x", 1, 1, 1)])
        self.clock.advance(90)
        self.replica.merge([wire("o", "k", "y", 1, 2, 2)])
        self.clock.advance(50)
        self.assertEqual(len(self.replica.entries("o")), 2)

    def test_the_tombstone_refuses_a_relayed_old_incarnation_but_accepts_a_returning_node(self):
        self.replica.merge([wire("o", "k", "x", 3, 1, 1)])
        self.clock.advance(100)
        self.assertEqual(self.replica.entries("o"), [])
        self.assertEqual(self.replica.merge([wire("o", "k", "x", 3, 1, 1), wire("o", "k", "z", 2, 7, 1)]), 0)  # relayed back, old
        self.assertEqual(self.replica.entries("o"), [])
        self.assertEqual(self.replica.merge([wire("o", "k", "x", 4, 1, "back")]), 1)  # the node returned with a new incarnation
        self.assertEqual(self.replica.get("o", "k", "x")["value"], "back")

    def test_a_relayed_entry_that_is_already_old_enough_is_pruned_on_arrival(self):
        self.replica.merge([wire("o", "k", "x", 1, 1, 1, age=500.)])
        self.assertEqual(self.replica.entries("o"), [])
        self.assertEqual(self.replica.merge([wire("o", "k", "x", 1, 1, 1, age=500.)]), 0)

    def test_a_tombstone_is_dropped_after_its_time(self):
        self.replica.merge([wire("o", "k", "x", 1, 1, 1)])
        self.clock.advance(100)
        self.replica.prune()
        self.assertEqual(self.replica.merge([wire("o", "k", "x", 1, 1, 1)]), 0)
        self.clock.advance(1000)
        self.assertEqual(self.replica.merge([wire("o", "k", "x", 1, 1, 1)]), 1)  # (a fresh sighting: the origin is back in the cluster)

    def test_replicas_prune_a_departed_node_and_stay_identical(self):
        mesh = Mesh(["a", "b", "c"], prune_after=100)
        mesh.replicas["c"].put("node", "self", 1)
        mesh.settle()
        for _ in range(4):  # c leaves; a and b keep syncing and writing
            mesh.clock.advance(30)
            mesh.replicas["a"].put("node", "self", mesh.clock())
            mesh.sync("a", "b")
            mesh.sync("b", "a")
        self.assertEqual(mesh.replicas["a"].contents(), mesh.replicas["b"].contents())  # (c is gone: its own copy is no part of the cluster)
        self.assertEqual({r["origin"] for r in mesh.replicas["a"].entries()}, {"a"})
        mesh.sync("b", "a")
        self.assertEqual({r["origin"] for r in mesh.replicas["b"].entries()}, {"a"})


class Incarnation(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "data" / state.INCARNATION_FILE

    def test_it_counts_up_at_every_start_and_survives_in_the_file(self):
        first = state.next_incarnation(self.path, now=lambda: 1000.)
        self.assertEqual(first, 1001)  # no file: from the clock, above any counter an earlier folder held
        self.assertEqual([state.next_incarnation(self.path), state.next_incarnation(self.path)], [1002, 1003])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"incarnation": 1003})

    def test_a_damaged_file_starts_again_from_the_clock(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(state.next_incarnation(self.path, now=lambda: 5000.), 5001)

    def test_a_folder_that_cannot_be_written_still_gives_a_number(self):
        blocker = self.path.parent.parent / "file"
        blocker.write_text("x", encoding="utf-8")
        self.assertEqual(state.next_incarnation(blocker / "data" / state.INCARNATION_FILE, now=lambda: 7.), 8)


class Routes(ApiCase):
    """POST /state/sync over a Runner whose Cluster has fake peers, and the poll that calls it."""
    prefix = "/state"

    def make_services(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": ["gpu-box"]}}), encoding="utf-8")
        self.net, self.clock = Net(), Clock()
        self.runner = Runner(self.root / "data" / "queue.json", {}, tick=.01, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: None, own=own_node, clock=self.clock, address="here", threads=False))
        self.addCleanup(self.runner.shutdown)
        self.state = self.runner.cluster.state
        return ClusterServices(self.runner)

    def test_every_route_answers(self):
        self.state.put("node", "self", {"device": "mps"})
        reply = self.call("POST", "/state/sync", json={"vector": {}, "entries": []})
        self.assertEqual([(e["origin"], e["kind"], e["key"], e["value"]) for e in reply["entries"]],
                         [(self.state.origin, "node", "self", {"device": "mps"})])
        self.assertEqual(reply["vector"], {self.state.origin: [self.state.incarnation, 1]})
        self.assertEqual(self.call("POST", "/state/sync", json={"vector": reply["vector"]})["entries"], [])

        pushed = {"origin": "far", "kind": "run", "key": "r1", "incarnation": 4, "seq": 2, "age": 3.0, "value": [1, 2]}
        reply = self.call("POST", "/state/sync", json={"vector": {}, "entries": [pushed]})
        self.assertEqual(self.state.get("far", "run", "r1")["value"], [1, 2])
        self.assertEqual(reply["vector"]["far"], [4, 2])
        self.assertEqual(len(reply["entries"]), 2)

        self.call("POST", "/state/sync", 422, json={"vector": {}, "entries": [{**pushed, "seq": 0}]})
        self.call("POST", "/state/sync", 422, json={"vector": {"far": [1]}})
        self.assertIn(("POST", "/state/sync"), self.hit)
        self.assertEqual({(m, p) for r in router.routes for m in r.methods for p in [r.path]}, self.hit)

    def test_a_route_over_a_node_without_a_queue_is_a_conflict(self):
        client = self.client
        self.app.state.services = FakeServices()
        self.assertEqual(client.post("/state/sync", json={}).status_code, 409)

    def test_the_peer_poll_syncs_the_store_both_ways_and_a_peer_without_the_route_stays_online(self):
        peer = state.Replica("gpu-box", 3, self.clock)
        peer.put("node", "self", {"device": "cuda"})
        self.state.put("node", "self", {"device": "mps"})

        def handler(method, path, body):
            return node_payload("gpu-box") if path == "node" else peer.handle_sync(json.loads(json.dumps(body)))
        self.net.handlers["gpu-box"] = handler
        with unittest.mock.patch.object(studio_cluster, "own_info", probed):
            self.runner.cluster.poll_once()
        self.assertEqual(self.state.get("gpu-box", "node", "self")["value"], {"device": "cuda"})
        self.assertEqual(peer.get(self.state.origin, "node", "self")["value"], {"device": "mps"})
        self.assertEqual([p for m, p, b in self.net.to("gpu-box")], ["node", "state/sync", "state/sync"])
        self.assertTrue(self.runner.cluster.nodes()[1].online)

        def old_peer(method, path, body):
            if path == "node":
                return node_payload("gpu-box")
            raise PeerError("answered 404: Not Found", status=404, detail="Not Found")
        self.net.handlers["gpu-box"] = old_peer
        self.runner.cluster.poll_once()
        self.assertTrue(self.runner.cluster.nodes()[1].online)

    def test_the_incarnation_is_read_from_the_data_folder_and_counts_up_at_every_start(self):
        first = self.state.incarnation
        again = Cluster(self.runner, client_for=self.net.client_for, store=lambda: None, own=own_node, clock=self.clock, address="here",
                        threads=False)
        self.assertEqual(again.state.incarnation, first + 1)
        self.assertEqual(json.loads((self.root / "data" / state.INCARNATION_FILE).read_text(encoding="utf-8")), {"incarnation": first + 1})


if __name__ == "__main__":
    unittest.main()

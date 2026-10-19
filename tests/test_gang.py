"""Gang placement of a distributed run (studio_scheduler.decide_gang and Gangs, studio_remote.Cluster): a run takes `min_members` to
`max_members` free machines at once or none, two runs never hold parts of each other's machines, and a member slot whose lease lapses
is freed and offered to a machine that is free then. Fake nodes, fake peers and fake clocks."""
import json
import tempfile
import unittest
from pathlib import Path

import studio_cluster as sc
import studio_distributed as runs
import studio_diloco
import studio_scheduler as sched
from studio_queue import Job, Plan, Runner
from studio_remote import Cluster
from tests import diloco_support as support
from tests.fake_cluster import Clock, Net
from tests.test_queue_cluster import GB, Peer, own_node

A = sc.LOCAL  # this machine
ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)


def node(address, busy=False, online=True):
    return sc.NodeInfo(address=address, hostname=address, device="cpu", memory_bytes=16 * GB, busy=busy, online=online, dtypes=sc.DTYPES["cpu"])


def up(*addresses):
    health = sched.Health(Clock())
    for address in addresses:
        health.observe(address, True)
    return health


class Placement(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.gangs = sched.Gangs(self.clock, lease_s=30.)
        self.health = up("b", "c")

    def three(self, **busy):
        return [node(A), *(node(x, busy=busy.get(x, False)) for x in ("b", "c"))]

    def test_three_free_nodes_give_three_leases_in_one_placement(self):
        gang = self.gangs.place("r1", self.three(), self.health, 2, 3)
        self.assertEqual(gang.nodes, (A, "b", "c"))
        held = self.gangs.held("r1")
        self.assertEqual({a: lease.epoch for a, lease in held.items()}, {A: 1, "b": 1, "c": 1})
        self.assertEqual({lease.duration_s for lease in held.values()}, {30.})

    def test_it_takes_no_more_than_max_members(self):
        self.assertEqual(self.gangs.place("r1", self.three(), self.health, 1, 2).nodes, (A, "b"))

    def test_one_free_node_gives_none_and_the_run_waits_with_a_reason(self):
        gang = self.gangs.place("r1", self.three(b=True, c=True), self.health, 2, 3)
        self.assertEqual(gang.nodes, ())
        self.assertIn("needs 2 machines free at once; 1 of the 3 chosen is free (local)", gang.reason)
        self.assertEqual(self.gangs.held("r1"), {})  # no partial set of leases

    def test_two_nodes_free_meet_min_two_and_the_run_gets_both(self):
        self.assertEqual(self.gangs.place("r1", self.three(c=True), self.health, 2, 3).nodes, (A, "b"))

    def test_a_peer_that_is_down_or_offline_is_not_free(self):
        nodes = [node(A), node("b", online=False), node("c")]
        self.assertEqual(self.gangs.place("r1", nodes, self.health, 1, 3).nodes, (A, "c"))
        health = sched.Health(Clock())  # nobody has answered yet
        self.assertEqual(sched.decide_gang(self.three(), health, 1, 3).nodes, (A,))

    def test_two_runs_and_three_nodes_one_is_placed_whole_and_the_other_waits(self):
        first = self.gangs.place("r1", self.three(), self.health, 2, 2)
        second = self.gangs.place("r2", self.three(), self.health, 2, 2)
        self.assertEqual(first.nodes, (A, "b"))
        self.assertEqual(second.nodes, ())
        self.assertEqual(self.gangs.held("r2"), {})
        self.assertEqual(set(self.gangs.held("r1")), {A, "b"})

    def test_a_run_that_waits_gets_its_machines_when_the_other_ends(self):
        self.gangs.place("r1", self.three(), self.health, 2, 2)
        self.assertEqual(self.gangs.place("r2", self.three(), self.health, 2, 2).nodes, ())
        self.gangs.end("r1")
        self.assertEqual(self.gangs.place("r2", self.three(), self.health, 2, 2).nodes, (A, "b"))

    def test_asking_again_returns_the_slots_the_run_holds(self):
        first = self.gangs.place("r1", self.three(), self.health, 1, 2)
        again = self.gangs.place("r1", self.three(b=True), self.health, 1, 2)
        self.assertEqual(again, first)
        self.assertEqual(self.gangs.held("r1")["b"].epoch, 1)


class Lapse(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.gangs = sched.Gangs(self.clock, lease_s=30.)
        self.health = up("b", "c", "d")
        self.nodes = [node(A), node("b"), node("c"), node("d")]
        self.gangs.place("r1", self.nodes, self.health, 2, 3)  # a, b, c

    def test_a_slot_that_is_heard_does_not_lapse_and_one_that_is_silent_does(self):
        self.clock.advance(20)
        self.gangs.heard("r1", "b")
        self.gangs.heard("r1", "c")
        self.clock.advance(20)
        self.assertEqual(self.gangs.lapsed(), [])  # 20 s since the renewals; a is this machine, which never lapses
        self.clock.advance(15)
        self.gangs.heard("r1", "c")
        self.assertEqual(self.gangs.lapsed(), [("r1", "b")])

    def test_a_slot_never_heard_from_gets_longer_to_join(self):
        self.clock.advance(31)
        self.assertEqual(self.gangs.lapsed(), [])
        self.clock.advance(sched.JOIN_GRACE * 30.)
        self.assertEqual(set(self.gangs.lapsed()), {("r1", "b"), ("r1", "c")})

    def test_a_dropped_slot_is_freed_and_a_newly_free_node_joins_with_a_new_lease(self):
        self.gangs.drop("r1", "b")
        self.assertEqual(set(self.gangs.held("r1")), {A, "c"})
        self.assertEqual(self.gangs.fill("r1", self.nodes, self.health, 3), ["d"])  # b was just lost; d is free
        self.assertEqual(set(self.gangs.held("r1")), {A, "c", "d"})
        self.assertEqual(self.gangs.held("r1")["d"].epoch, 1)

    def test_the_lost_node_is_not_taken_straight_back_but_is_after_two_leases(self):
        self.gangs.drop("r1", "b")
        self.assertEqual(self.gangs.fill("r1", [node(A), node("b"), node("c")], self.health, 3), [])
        self.clock.advance(61)
        self.assertEqual(self.gangs.fill("r1", [node(A), node("b"), node("c")], self.health, 3), ["b"])
        self.assertEqual(self.gangs.held("r1")["b"].epoch, 2)  # a new lease: the earlier one is fenced

    def test_fill_never_goes_past_max_members(self):
        self.assertEqual(self.gangs.fill("r1", self.nodes, self.health, 3), [])

    def test_another_runs_machines_are_not_offered(self):
        self.gangs.drop("r1", "c")
        self.gangs.place("r2", [node("d")], self.health, 1, 1)
        self.assertEqual(self.gangs.fill("r1", self.nodes, self.health, 3), [])  # d is r2's, c was just lost


class CoordinatorLose(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()

    def test_a_lost_member_leaves_at_the_round_boundary_and_a_newcomer_joins_the_round_after(self):
        c = support.build(Path(self.tmp.name), clock=self.clock, inner_steps=2, rounds=5, min_members=2, max_members=2)
        c.join("a")
        c.join("b")
        self.assertEqual(c.round_view(0, "a")["members"], 2)
        c.lose("b")
        c.join("c")  # the freed slot goes to a node that is free now
        self.assertEqual(c.snapshot()["nodes"]["c"]["state"], "joining")  # not in this round
        self.assertEqual(c.snapshot()["expected"], 1)  # round 0 no longer waits for b
        view = c.round_view(0, "a")
        c.submit(0, "a", view["generation"], 0, 10, 2, support.constant_delta(c), 1.)
        self.assertEqual(c.round_view(1, "a")["members"], 2)  # a and c
        nodes = c.snapshot()["nodes"]
        self.assertEqual((nodes["b"]["state"], nodes["c"]["state"]), ("left", "training"))
        self.assertTrue(any("lost its lease" in event["text"] for event in c.snapshot()["events"]))

    def test_losing_a_node_that_is_not_a_member_does_nothing(self):
        c = support.build(Path(self.tmp.name), clock=self.clock, inner_steps=2)
        c.join("a")
        c.lose("ghost")
        self.assertEqual(c.snapshot()["nodes"]["a"]["state"], "training")


class OwnerCluster(unittest.TestCase):
    """The Cluster that owns a distributed run: places it as a gang, renews slots from what the coordinator hears, frees a lapsed slot
    and invites the machine that took its place."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock, self.net = Clock(), Net()
        self.addresses = ["b", "c", "d"]
        self.peers = {a: Peer(self.net, a, device="cpu", dtypes=["float32"]) for a in self.addresses}
        for address in self.addresses:
            self.net.handlers[address] = self.handler(address)
        (self.root / "data").mkdir()
        (self.root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": self.addresses}}), encoding="utf-8")
        self.runner = Runner(self.root / "data" / "queue.json", {}, tick=.01, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: None, own=own_node, clock=self.clock, address="owner", threads=False))
        runner = self.runner
        runner.set_plan(ANY)
        self.addCleanup(runner.shutdown)
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.invited = []

    def handler(self, address):
        peer = self.peers[address]

        def handle(method, path, body):
            if path == "queue/jobs" and method == "POST":
                self.invited.append((address, body))
                return {"id": f"member-{address}"}
            if path.startswith("queue/jobs/") and method == "DELETE":
                self.invited.append((address, "deleted"))
                return None
            return peer.handle(method, path, body)
        return handle

    def machines(self, *addresses):
        return [{"node": "laptop", "address": sc.LOCAL, "device": "cpu"}, *({"node": a, "address": a, "device": "cpu"} for a in addresses)]

    def job(self, *addresses, **params):
        return self.runner.enqueue(Job("pretrain_distributed", "Distributed", params={
            "machines": self.machines(*addresses), "peer_url": "https://laptop.example", "owner": "laptop", **params}))

    def poll(self):
        self.runner.cluster.poll_once()

    def test_a_run_with_enough_free_machines_is_placed_with_a_slot_on_each(self):
        job = self.job("b", "c", min_members=2, max_members=3)
        self.poll()
        self.assertEqual(self.runner.cluster.place(job).where, "local")
        self.assertEqual(job.params["gang"], [sc.LOCAL, "b", "c"])
        self.assertNotIn(job.id, self.runner.cluster.waiting)

    def test_a_run_waits_when_too_few_machines_are_free(self):
        self.peers["b"].node["busy"] = True
        self.peers["c"].node["busy"] = True
        job = self.job("b", "c", min_members=2)
        self.poll()
        placement = self.runner.cluster.place(job)
        self.assertEqual(placement.where, "wait")
        self.assertIn("needs 2 machines free at once", self.runner.cluster.waiting[job.id])
        self.assertNotIn("gang", job.params)
        self.peers["b"].node["busy"] = False
        self.poll()
        self.assertEqual(self.runner.cluster.place(job).where, "local")
        self.assertEqual(job.params["gang"], [sc.LOCAL, "b"])

    def test_a_second_run_waits_while_the_first_holds_this_machine(self):
        first, second = self.job("b", "c", min_members=2), self.job("b", "c", min_members=2)
        self.poll()
        self.assertEqual(self.runner.cluster.place(first).where, "local")
        self.assertEqual(self.runner.cluster.place(second).where, "wait")
        self.assertNotIn("gang", second.params)

    def test_invite_only_goes_to_the_machines_with_a_slot(self):
        job = self.job("b", "c", max_members=2)
        self.poll()
        self.runner.cluster.place(job)
        runs.invite(job, self.net.client_for)
        self.assertEqual([a for a, _ in self.invited], ["b"])
        rows = {m.node: m for m in runs.run_of(job).machines}
        self.assertEqual((rows["b"].state, rows["c"].state), ("invited", "unplaced"))
        view = runs.member_rows(runs.run_of(job), None, "")
        self.assertEqual([(r["node"], r["state"], r["can_stop"]) for r in view], [("laptop", "invited", False), ("b", "invited", True), ("c", "waiting", False)])
        self.assertEqual(view[2]["note"], runs.UNPLACED)

    def test_the_run_is_named_for_the_machines_that_took_a_slot(self):
        job = self.job("b", "c", max_members=2, version=3)
        job.label = "Version 3 across 3 machines"
        self.poll()
        self.runner.cluster.place(job)
        runs.invite(job, self.net.client_for)
        self.assertEqual((runs.run_of(job).label, job.label), ("Version 3 across 2 machines",) * 2)

    def test_the_name_is_kept_when_every_chosen_machine_took_a_slot(self):
        job = self.job("b", "c", max_members=3, version=3)
        job.label = "Version 3 across 3 machines"
        self.poll()
        self.runner.cluster.place(job)
        runs.invite(job, self.net.client_for)
        self.assertEqual(runs.run_of(job).label, "Version 3 across 3 machines")

    def test_a_lapsed_member_slot_is_freed_and_a_newly_free_node_is_invited(self):
        job = self.job("b", "c", "d", min_members=2, max_members=3)
        self.poll()
        cluster = self.runner.cluster
        self.peers["d"].node["busy"] = True  # d is busy at first, so b and c take the other slots
        self.poll()
        cluster.place(job)
        self.assertEqual(job.params["gang"], [sc.LOCAL, "b", "c"])
        job.status = "running"
        coordinator = support.build(self.root / "run", run=job.id, clock=Clock(), inner_steps=2, rounds=5, min_members=1, max_members=3)
        studio_diloco.register(job.id, coordinator)
        runs.invite(job, self.net.client_for)
        coordinator.join("c")  # c's member joined; b's never does
        self.invited.clear()
        self.peers["d"].node["busy"] = False
        self.poll()
        cluster.tick()
        self.assertEqual(set(cluster.gangs.held(job.id)), {sc.LOCAL, "b", "c"})
        self.clock.advance(sched.LEASE_S * (sched.JOIN_GRACE + 1))
        self.poll()
        cluster.tick()  # b lapses: freed (its member job leaves b's queue)
        self.assertIn(("b", "deleted"), self.invited)
        self.assertEqual({m.node: m.state for m in runs.run_of(job).machines}["b"], "unplaced")
        cluster.tick()  # the next tick offers the open slot to d
        self.assertEqual(set(cluster.gangs.held(job.id)), {sc.LOCAL, "c", "d"})
        self.assertEqual([(a, body["params"]["node"]) for a, body in self.invited if isinstance(body, dict)], [("d", "d")])
        self.assertEqual(self.runner.cluster.gangs.held(job.id)["d"].epoch, 1)

    def test_slots_are_freed_when_the_run_ends(self):
        job = self.job("b", min_members=2)
        self.poll()
        self.runner.cluster.place(job)
        self.assertTrue(self.runner.cluster.gangs.held(job.id))
        job.status = "done"
        self.runner.cluster.tick()
        self.assertEqual(self.runner.cluster.gangs.held(job.id), {})


if __name__ == "__main__":
    unittest.main()

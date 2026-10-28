"""A machine that comes online while a distributed run goes joins it (studio_remote.Cluster.offer): the owner invites a peer that fits
while the run is under `max_members` and its `auto_join` switch is on, the member joins at the next round boundary sized to its own
memory, and a peer that cannot join (another app version, busy, too little memory) is not invited and the run's row for it says why.
Fake peers, a fake clock and a scripted coordinator; no GPU, no window, no network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import studio_cluster as sc
import studio_diloco
import studio_distributed as runs
import studio_fit
from studio_api.routers import distributed as router
from studio_ia import pretraining_plan as plan
from studio_queue import Job, Plan, Runner
from studio_remote import Cluster
from studio_spec import default_spec
from tests import diloco_support as support
from tests.fake_cluster import Clock, Net
from tests.test_queue_cluster import GB, Peer

ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)
BUILD = "1.0.0"


def own_node():
    return sc.NodeInfo(address=sc.LOCAL, hostname="laptop", version=BUILD, device="cpu", memory_bytes=16 * GB, dtypes=sc.DTYPES["cpu"])


def profile(vram, free):
    return studio_fit.Profile(device="cuda", gpu_name=f"{vram} GB", vram_total=int(vram * GB), vram_free=int(free * GB), ram_total=64 * GB,
                              ram_free=48 * GB, disk_free=500 * GB, cpu_cores=16).to_dict()


class LateJoin(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch("studio_memory.training_bytes", lambda cfg, vocab=None, offload=False: (GB, GB)))  # 1 GB + 1 GB a row
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock, self.net = Clock(), Net()
        self.peers = {a: Peer(self.net, a, version=BUILD, device="cuda", dtypes=["float32"], profile=profile(80, 78)) for a in ("b", "late")}
        self.invited = []
        for address, peer in self.peers.items():
            self.net.handlers[address] = self.handler(peer, address)
        self.net.down.add("late")  # it is not on yet
        (self.root / "data").mkdir()
        (self.root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": list(self.peers)}}), encoding="utf-8")
        self.runner = Runner(self.root / "data" / "queue.json", {}, tick=.01, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: None, own=own_node, clock=self.clock, address="owner", threads=False))
        self.runner.set_plan(ANY)
        self.addCleanup(self.runner.shutdown)
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.cluster = self.runner.cluster

    def handler(self, peer, address):
        def handle(method, path, body):
            if path == "queue/jobs" and method == "POST":
                self.invited.append((address, body))
                return {"id": f"member-{address}"}
            return peer.handle(method, path, body)
        return handle

    def start(self, max_members=3, **params):
        """A running run chosen over this machine and b, with b's member joined and a round open."""
        config = plan.pretrain_config(default_spec(device="cpu"), 6, router.ROWS, router.HOLD)
        machines = [{"node": "laptop", "address": sc.LOCAL, "device": "cpu"}, {"node": "b", "address": "b", "device": "cuda"}]
        job = self.runner.enqueue(Job("pretrain_distributed", "Distributed", params={
            "machines": machines, "peer_url": "https://laptop.example", "owner": "laptop", "config": config, "min_members": 1,
            "max_members": max_members, "settings": {"rounds": 5, "inner_steps": 2, "min_members": 1, "max_members": max_members}, **params}))
        self.cluster.poll_once()
        self.cluster.place(job)
        job.status = "running"
        self.coordinator = support.build(self.root / "run", run=job.id, clock=self.clock, inner_steps=2, rounds=5, min_members=1, max_members=max_members)
        studio_diloco.register(job.id, self.coordinator)
        runs.invite(job, self.net.client_for)
        self.coordinator.join("laptop")
        self.coordinator.join("b")
        self.cluster.tick()
        self.invited.clear()
        return job

    def come_online(self):
        self.net.down.discard("late")
        self.clock.advance(600)  # a peer that did not answer is asked again after a while
        self.cluster.poll_once()
        self.cluster.tick()

    def row(self, job, node):
        return next(r for r in runs.member_rows(runs.run_of(job), self.coordinator.snapshot(), "") if r["node"] == node)

    def test_a_peer_that_comes_online_mid_run_is_invited_and_joins_at_the_next_round_boundary(self):
        job = self.start()
        self.cluster.tick()
        self.assertEqual(self.invited, [])  # nobody else is on yet
        self.come_online()
        self.assertEqual([a for a, _ in self.invited], ["late"])
        params = self.invited[0][1]["params"]
        self.assertEqual((params["node"], params["run"]), ("late", job.id))
        self.assertEqual(set(params["fit"]), {"batch_size", "accumulation", "offload_optimizer"})  # sized like the first members
        self.assertEqual(set(self.cluster.gangs.held(job.id)), {sc.LOCAL, "b", "late"})
        self.coordinator.join("late", epoch=1)  # its member job starts and joins
        snap = self.coordinator.snapshot()
        self.assertEqual((snap["round"], snap["nodes"]["late"]["state"], snap["nodes"]["late"]["joined_round"]), (0, "joining", 1))
        self.assertIsNone(self.row(job, "b")["joined_round"])  # a machine the run started with is not a late joiner
        self.assertIn("joined and starts with round 2", [e for e in snap["events"] if e["node"] == "late"][0]["text"])
        view = self.row(job, "late")
        self.assertEqual((view["state"], view["joined_round"]), ("joining", 1))

    def test_a_peer_on_another_build_is_not_invited_and_the_run_says_why(self):
        self.peers["late"].node["version"] = "0.9.0"
        job = self.start()
        self.come_online()
        self.assertEqual(self.invited, [])
        self.assertEqual(set(self.cluster.gangs.held(job.id)), {sc.LOCAL, "b"})
        row = self.row(job, "late")
        self.assertEqual(row["state"], "waiting")
        self.assertIn("runs app version 0.9.0; this computer runs 1.0.0", row["reason"])

    def test_a_peer_without_the_memory_is_not_invited_and_the_run_says_why(self):
        self.peers["late"].node["profile"] = profile(1, .2)
        job = self.start()
        self.come_online()
        self.assertEqual(self.invited, [])
        row = self.row(job, "late")
        self.assertEqual(row["state"], "waiting")
        self.assertIn("late cannot take part", row["reason"])

    def test_a_busy_peer_is_not_invited_until_it_is_free(self):
        self.peers["late"].node["busy"] = True
        job = self.start()
        self.come_online()
        self.assertEqual(self.invited, [])
        self.assertIn("busy with another job", self.row(job, "late")["reason"])
        self.peers["late"].node["busy"] = False
        self.clock.advance(600)
        self.cluster.poll_once()
        self.cluster.tick()
        self.assertEqual([a for a, _ in self.invited], ["late"])
        self.assertEqual(self.row(job, "late")["state"], "invited")

    def test_a_run_at_max_members_invites_no_one(self):
        job = self.start(max_members=2)
        self.come_online()
        self.assertEqual(self.invited, [])
        self.assertEqual(set(self.cluster.gangs.held(job.id)), {sc.LOCAL, "b"})

    def test_a_run_with_the_switch_off_invites_no_one_but_still_asks_a_machine_that_asks_to_join(self):
        job = self.start(auto_join=False)
        self.come_online()
        self.assertEqual(self.invited, [])
        runs.choose(job, "late", "late")  # the person on it asked to join
        self.cluster.tick()
        self.assertEqual([a for a, _ in self.invited], ["late"])

    def test_the_switch_is_turned_off_and_on_while_the_run_goes(self):
        job = self.start()
        self.assertTrue(runs.view(runs.run_of(job), self.coordinator, job)["auto_join"])
        runs.set_auto_join(job, False)
        self.assertFalse(runs.view(runs.run_of(job), self.coordinator, job)["auto_join"])
        self.come_online()
        self.assertEqual(self.invited, [])
        runs.set_auto_join(job, True)
        self.cluster.tick()
        self.assertEqual([a for a, _ in self.invited], ["late"])

    def test_a_machine_the_owner_stopped_is_not_invited_again(self):
        job = self.start()
        self.coordinator.banned.add("late")
        self.come_online()
        self.assertEqual(self.invited, [])
        self.assertIn("stopped by the run's owner", self.row(job, "late")["reason"])


if __name__ == "__main__":
    unittest.main()

"""A distributed run survives its coordinator restarting (#479): the owner's queue resumes it from the last completed round or ends it with a
reason, places it again against the machines chosen at launch, and invites the members again; a member whose coordinator vanished ends
saying so. The restart is a new Runner over the same workspace; the peers are fakes and the clocks are by hand."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

import studio_cluster as sc
import studio_diloco
import studio_distributed as runs
from studio_peer import PeerError
from studio_queue import Job, Plan, Runner
from studio_remote import Cluster
from tests import diloco_support as support
from tests.fake_cluster import Clock, Net
from tests.test_queue_cluster import Peer, own_node

ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)
MACHINES = [{"node": "laptop", "address": sc.LOCAL, "device": "cpu"}, {"node": "b", "address": "b", "device": "cpu"}]


class Services:
    def examples(self, params):
        return support.EXAMPLES


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "workspace.json").write_text(json.dumps({"studioConfig": {"nodes": ["b"]}}), encoding="utf-8")
        self.out = self.root / "runs" / "queue_run1"
        self.net, self.clock, self.calls = Net(), Clock(), []
        self.peer = Peer(self.net, "b", device="cpu", dtypes=["float32"])
        self.net.handlers["b"] = self.handle
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.addCleanup(runs.unload, self.root)

    def handle(self, method, path, body):
        if path.startswith("queue/jobs"):
            self.calls.append((method, path, body))
            return {"id": "member-new"} if method == "POST" else None
        return self.peer.handle(method, path, body)

    def runner(self):
        runner = Runner(self.root / "data" / "queue.json", {}, tick=.01, cluster=lambda r: Cluster(
            r, client_for=self.net.client_for, store=lambda: None, own=own_node, clock=self.clock, address="owner", threads=False))
        runner.set_plan(ANY)
        self.addCleanup(runner.shutdown)
        return runner

    def first_life(self, rounds_done=2):
        """The owner before the restart: the job running, its coordinator `rounds_done` rounds in, the member on b invited. Returns the coordinator."""
        runs.load(self.root)
        runner = self.runner()
        params = {"config": support.CONFIG, "examples": support.EXAMPLES, "machines": MACHINES, "peer_url": "https://laptop.example", "owner": "laptop",
                  "rounds": 4, "inner_steps": 3, "outer_lr": 1., "outer_momentum": .5, "shards": 4, "delta_dtype": "float32", "min_members": 2,
                  "gang": [sc.LOCAL], "output_dir": str(self.out), "version": 1}
        job = runner.enqueue(Job("pretrain_distributed", "Version 1 across 2 machines", id="run1", params=params))
        runner.jobs[0].status = "running"
        coordinator = support.build(self.out, run="run1", rounds=4, inner_steps=3, outer_lr=1., outer_momentum=.5)
        studio_diloco.register("run1", coordinator)
        for node in ("laptop", "b"):
            coordinator.join(node)
        for _ in range(rounds_done):
            number = coordinator.current.number
            for node in list(coordinator.current.members):
                coordinator.submit(number, node, coordinator.current.generation, 0, 10, 3, support.constant_delta(coordinator), 1., 3.)
        run = runs.open_run(job)
        run.machines[1].peer_job, run.machines[1].state = "member-old", "invited"
        runs.remember("run1", coordinator)
        runner._save()
        return coordinator

    def restart(self):
        """A new process: the coordinators and records in memory are gone, the workspace is not."""
        runs.RUNS.clear()
        studio_diloco.COORDINATORS.clear()
        runs.load(self.root)
        return self.runner()

    def test_it_resumes_at_the_next_round_with_the_same_outer_state_and_invites_the_peer_again(self):
        before = self.first_life(rounds_done=2)
        theta, momentum = {k: v.copy() for k, v in before.theta.items()}, {k: v.copy() for k, v in before.momentum.items()}
        runner = self.restart()
        (job,) = runner.jobs
        self.assertEqual((job.status, job.params["resuming"]), ("waiting", {"round": 2}))
        self.assertEqual(runs.view(runs.RUNS["run1"], None, job)["state"], "queued")
        self.assertIn("Resuming from round 2", runs.view(runs.RUNS["run1"], None, job)["note"])
        runner.cluster.poll_once()
        self.assertEqual(runner.cluster.place(job).where, "local")
        self.assertEqual([c[0] for c in self.calls], ["DELETE"])  # the old coordinator's member is taken off b's queue first
        self.assertTrue(self.calls[0][1].endswith("member-old"))
        coordinator = studio_diloco.prepare(job, Services(), self.out, 8787, client_for=self.net.client_for)
        self.assertEqual(coordinator.next_round, 2)
        for name in theta:
            self.assertTrue(np.array_equal(coordinator.theta[name], theta[name]))
            self.assertTrue(np.array_equal(coordinator.momentum[name], momentum[name]))
        invites = [c for c in self.calls if c[0] == "POST"]
        self.assertEqual(len(invites), 1)
        self.assertEqual((invites[0][2]["kind"], invites[0][2]["params"]["run"], invites[0][2]["params"]["node"]), ("diloco_member", "run1", "b"))
        for node in ("laptop", "b"):
            coordinator.join(node)
        self.assertEqual(coordinator.current.number, 2)  # round 3 of 4 is the next one to run
        self.assertNotIn("resuming", job.params)
        self.assertFalse(runs.RUNS["run1"].restarted)
        self.assertEqual(runs.view(runs.RUNS["run1"], coordinator, job)["round"], 2)

    def test_placement_after_a_resume_uses_the_machines_chosen_at_launch_not_the_saved_gang(self):
        self.first_life()
        runner = self.restart()
        (job,) = runner.jobs
        self.assertNotIn("gang", job.params)  # the saved gang, [local] alone, is not trusted
        runner.cluster.poll_once()
        self.assertEqual(runner.cluster.place(job).where, "local")
        self.assertEqual(job.params["gang"], [sc.LOCAL, "b"])
        self.assertEqual([m.address for m in runs.RUNS["run1"].machines], [sc.LOCAL, "b"])

    def test_a_run_whose_round_state_is_missing_ends_with_the_reason_and_the_tab_says_so(self):
        self.first_life(rounds_done=2)
        (self.out / "diloco.json").unlink()
        runner = self.restart()
        (job,) = runner.jobs
        reason = "Ended: this machine restarted at round 2; outer state for that round was not saved."
        self.assertEqual((job.status, job.error), ("failed", reason))
        self.assertEqual(runner.ended_on_load, [job])
        row = runs.view(runs.RUNS["run1"], None, job)
        self.assertEqual((row["state"], row["note"], row["can_stop"]), ("failed", reason, False))

    def test_an_unreadable_state_file_ends_the_run_too(self):
        self.first_life()
        (self.out / "diloco.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(self.restart().jobs[0].status, "failed")

    def test_a_run_that_finished_no_round_starts_again(self):
        self.first_life(rounds_done=0)
        job = self.restart().jobs[0]
        self.assertEqual((job.status, job.params["resuming"]), ("waiting", {"round": 0}))

    def test_a_folder_that_no_longer_matches_ends_the_run_saying_it_could_not_resume(self):
        self.first_life()
        job = self.restart().jobs[0]
        job.params["examples"] = support.EXAMPLES[:30]
        with self.assertRaisesRegex(ValueError, "Ended: this machine restarted at round 2, and the run could not be resumed"):
            studio_diloco.prepare(job, Services(), self.out, 8787, client_for=self.net.client_for)


class MemberOfAVanishedCoordinator(unittest.TestCase):
    def member(self, error):
        clock = Clock()

        class Client:
            def get_json(self, *args, **kwargs):
                raise error

        return studio_diloco.Member(Client(), "run1", "b", 0, "shards", threading.Event(), clock=clock, patience=10., sleep=clock.advance), Client()

    def test_it_ends_with_a_reason_once_the_coordinator_has_been_silent_for_its_patience(self):
        member, client = self.member(PeerError("laptop did not answer: unreachable"))
        with self.assertRaisesRegex(PeerError, "coordinator of run run1 did not answer for 10 seconds, so this member left. It joins again"):
            member._call(lambda: client.get_json("/x"))

    def test_it_ends_at_once_when_the_coordinator_answers_that_it_no_longer_has_the_run(self):
        detail = "No distributed run 'run1' on this node."
        member, client = self.member(PeerError(f"404: {detail}", status=404, detail=detail))
        with self.assertRaisesRegex(PeerError, "no longer has run run1"):
            member._call(lambda: client.get_json("/x"))


if __name__ == "__main__":
    unittest.main()

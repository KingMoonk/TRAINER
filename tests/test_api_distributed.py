"""The Training tab's backend: a coordinator's snapshot and controls (studio_diloco), the invitations and the read model
(studio_distributed), and the routes over FakeServices (studio_api.routers.distributed)."""
import socket
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import uvicorn
from fastapi.testclient import TestClient

import studio_cluster
import studio_diloco
import studio_distributed as runs
from studio_api import app as api_app
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices, Job
from studio_peer import PeerError
from studio_spec import default_spec
from tests import diloco_support as support
from tests.fake_cluster import Clock


def delta(coordinator):
    return support.constant_delta(coordinator)


class CoordinatorControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def submit(self, coordinator, node, number=0, generation=1, **fields):
        coordinator.submit(number, node, generation, 0, fields.get("tokens", 10), 2, delta(coordinator), fields.get("seconds", 2.), 3.0, 1234)

    def test_the_snapshot_says_who_is_in_and_what_each_sent(self):
        coordinator = support.build(self.dir, inner_steps=2, rounds=3, outer_lr=1.)
        self.assertEqual(coordinator.snapshot()["state"], "waiting")
        coordinator.join("a")
        coordinator.join("b")  # the round is open: b joins at its end
        snap = coordinator.snapshot()
        self.assertEqual((snap["state"], snap["round"], snap["rounds"]), ("running", 0, 3))
        self.assertEqual((snap["nodes"]["a"]["state"], snap["nodes"]["b"]["state"]), ("training", "joining"))
        self.submit(coordinator, "a")
        snap = coordinator.snapshot()
        self.assertEqual(snap["round"], 1)
        row = snap["history"][0]
        self.assertEqual((row["node_tokens"], row["node_seconds"], row["node_bytes"]), ({"a": 10}, {"a": 2.}, {"a": 1234}))
        self.assertGreater(row["theta_bytes"], 0)
        self.assertEqual(snap["nodes"]["b"]["state"], "training")

    def test_a_member_that_misses_its_rounds_is_behind_then_dropped_with_a_reason(self):
        clock = Clock()
        coordinator = support.build(self.dir, clock=clock, inner_steps=2, rounds=5, min_members=2, missed_limit=2, outer_lr=1.)
        coordinator.join("a")
        coordinator.join("b")
        self.submit(coordinator, "a")
        clock.advance(2000)
        snap = coordinator.snapshot()
        self.assertEqual((snap["round"], snap["history"][0]["dropped"]), (1, ["b"]))
        self.assertEqual((snap["nodes"]["b"]["state"], snap["nodes"]["b"]["missed"]), ("behind", 1))
        self.assertIn("Missed 1 round in a row", snap["nodes"]["b"]["note"])
        self.submit(coordinator, "a", number=1)
        clock.advance(2000)
        snap = coordinator.snapshot()
        self.assertEqual(snap["nodes"]["b"]["state"], "dropped")
        self.assertIn("dropped it", snap["nodes"]["b"]["note"])
        self.assertEqual(snap["nodes"]["a"]["state"], "training")

    def test_dropping_a_member_keeps_it_out_and_the_others_go_on(self):
        coordinator = support.build(self.dir, inner_steps=2, rounds=4, min_members=2, outer_lr=1.)
        coordinator.join("a")
        coordinator.join("b")
        coordinator.drop("b")
        stopped = coordinator.snapshot()["nodes"]["b"]
        self.assertEqual({k: stopped[k] for k in ("state", "missed", "sent", "note")}, {"state": "stopped", "missed": 0, "sent": False, "note": "Stopped by you."})
        self.assertRaises(studio_diloco.Stale, coordinator.join, "b")
        self.assertRaises(studio_diloco.Stale, coordinator.drop, "nobody")

    def test_stopping_ends_the_run_for_every_member_and_records_it(self):
        coordinator = support.build(self.dir, inner_steps=2, rounds=4, outer_lr=1.)
        coordinator.join("a")
        coordinator.stop()
        self.assertEqual(coordinator.snapshot()["state"], "stopped")
        self.assertEqual(coordinator.round_view(0, "a")["state"], "finished")
        self.assertRaises(studio_diloco.Stale, coordinator.join, "b")
        self.assertEqual(support.read_json(self.dir / "run.json")["status"], "stopped")


class FakeClient:
    """A PeerClient's calls to one machine, recorded; `fail` is the PeerError (or any exception) it answers with."""

    calls = []

    def __init__(self, address, fail=None, **options):
        self.address, self.fail = address, fail

    def post_json(self, path, body=None, params=None, timeout=None):
        FakeClient.calls.append((self.address, "POST", path, body))
        if self.fail:
            raise self.fail
        return {"id": f"job-{self.address}"}

    def delete_json(self, path, params=None, timeout=None):
        FakeClient.calls.append((self.address, "DELETE", path, None))


def clients(**failing):
    return lambda address, **options: FakeClient(address, failing.get(address), **options)


def owner_job(machines, **params):
    return types.SimpleNamespace(id="run1", label="Version 1 across 3 machines", created_at=100., status="running", error="", kind="pretrain_distributed",
                                 params={"version": 1, "settings": {"rounds": 4, "inner_steps": 2}, "machines": machines, "owner": "mac",
                                         "peer_url": "mac.tail.ts.net:8787", **params})


MACHINES = [{"node": "mac", "address": "local", "device": "mps", "device_name": "M3"},
            {"node": "box", "address": "box", "device": "cuda", "device_name": "RTX"},
            {"node": "old", "address": "old", "device": "cpu", "device_name": "", "error": "old did not answer within 2 seconds."}]


class Invitations(unittest.TestCase):
    def setUp(self):
        FakeClient.calls = []
        self.addCleanup(runs.RUNS.clear)

    def test_each_reachable_machine_is_asked_to_join_and_an_offline_one_is_not_called(self):
        job = owner_job(MACHINES)
        runs.invite(job, clients())
        self.assertEqual(FakeClient.calls, [("box", "POST", "/queue/jobs", {
            "kind": "diloco_member", "label": "Version 1 across 3 machines (member)",
            "params": {"coordinator_url": "mac.tail.ts.net:8787", "run": "run1", "node": "box", "epoch": 0}})])
        rows = {m.node: m for m in runs.run_of(job).machines}
        self.assertEqual((rows["box"].state, rows["box"].peer_job), ("invited", "job-box"))
        self.assertEqual((rows["old"].state, rows["old"].reason), ("unreachable", "old did not answer within 2 seconds."))

    def test_the_owner_s_member_options_go_to_the_invited_members_too(self):
        runs.invite(owner_job(MACHINES, member_options={"overlap": True}), clients())
        self.assertEqual(FakeClient.calls[0][3]["params"]["member_options"], {"overlap": True})

    def test_a_machine_that_answers_401_says_what_to_change_and_the_others_still_join(self):
        job = owner_job([*MACHINES[:2], {"node": "pc", "address": "pc", "device": "cuda"}])
        refuse = PeerError("pc answered 401", status=401, detail="not allowed")
        runs.invite(job, clients(pc=refuse))
        rows = {m.node: m for m in runs.run_of(job).machines}
        self.assertEqual(rows["box"].state, "invited")
        self.assertEqual(rows["pc"].state, "unreachable")
        self.assertIn("allowed users", rows["pc"].reason)
        self.assertIn("not allowed", rows["pc"].reason)

    def test_a_machine_that_cannot_be_reached_is_told_in_a_sentence(self):
        job = owner_job(MACHINES[:2])
        runs.invite(job, clients(box=ConnectionRefusedError()))
        machine = runs.run_of(job).machines[1]
        self.assertEqual(machine.state, "unreachable")
        self.assertIn("refused the connection", machine.reason)

    def test_a_loopback_address_is_no_use_to_another_machine(self):
        job = owner_job(MACHINES[:2], peer_url="http://127.0.0.1:8787")
        runs.invite(job, clients())
        self.assertEqual(FakeClient.calls, [])
        self.assertIn("TRAINER_NODE_ADDRESS", runs.run_of(job).machines[1].reason)

    def test_two_machines_with_one_hostname_get_distinct_names(self):
        rows = [{"node": "mac", "address": "local"}, {"node": "box", "address": "a"}, {"node": "box", "address": "b"}]
        self.assertEqual([m.node for m in runs.machines_of(rows)], ["mac", "box", "box (b)"])


class ReadModel(unittest.TestCase):
    def setUp(self):
        self.addCleanup(runs.RUNS.clear)
        self.job = owner_job(MACHINES)
        self.run = runs.run_of(self.job)
        FakeClient.calls = []

    def snap(self, **changes):
        history = [{"round": 0, "step": 2, "contributed": ["mac", "box"], "members": ["mac", "box"], "dropped": [], "loss": 3., "val_loss": 3.5,
                    "delta_bytes": 2000, "theta_bytes": 4000, "seconds": 5., "steps": {"mac": 2, "box": 2}, "node_tokens": {"mac": 100, "box": 50},
                    "node_seconds": {"mac": 2., "box": 5.}, "node_bytes": {"mac": 1000, "box": 1000}, "delta_dtype": "float16"}]
        return {"state": "running", "round": 1, "rounds": 4, "generation": 1, "settings": {"rounds": 4, "inner_steps": 2, "outer_lr": .7,
                                                                                         "outer_momentum": .9, "delta_dtype": "float16"},
                "nodes": {"mac": {"state": "training", "missed": 0, "sent": True, "note": ""},
                          "box": {"state": "behind", "missed": 1, "sent": False, "note": "Missed 1 round in a row."}}, "history": history, **changes}

    def coordinator(self, snap):
        return types.SimpleNamespace(snapshot=lambda: snap)

    def test_a_running_run_reads_with_a_row_for_every_machine_and_a_reason_for_the_one_that_is_out(self):
        view = runs.view(self.run, self.coordinator(self.snap()), self.job)
        self.assertEqual((view["state"], view["round"], view["rounds"], view["can_stop"]), ("running", 1, 4, True))
        rows = {m["node"]: m for m in view["members"]}
        self.assertEqual(rows["mac"]["state"], "training")
        self.assertEqual((rows["mac"]["rounds_done"], rows["mac"]["steps"], rows["mac"]["tokens_per_second"], rows["mac"]["last_delta_bytes"]),
                         (1, 2, 50., 1000))
        self.assertEqual((rows["box"]["state"], rows["box"]["missed"], rows["box"]["can_stop"], rows["box"]["tokens_per_second"]), ("behind", 1, True, 10.))
        self.assertEqual((rows["old"]["state"], rows["old"]["note"]), ("unreachable", "old did not answer within 2 seconds."))
        self.assertFalse(rows["mac"]["can_stop"])  # this computer coordinates: stop the run instead
        self.assertFalse(rows["old"]["can_stop"])
        self.assertEqual(view["history"][0]["bytes_up"], 2000)
        self.assertEqual((view["history"][0]["bytes_down"], view["history"][0]["val_loss"]), (4000, 3.5))

    def test_a_run_that_has_not_started_is_queued_or_starting_and_its_machines_are_invited(self):
        self.job.status = "waiting"
        view = runs.view(self.run, None, self.job)
        self.assertEqual((view["state"], view["rounds"], view["can_stop"]), ("queued", 4, True))
        self.assertEqual([m["state"] for m in view["members"]], ["invited", "invited", "unreachable"])
        self.job.status = "running"
        self.assertEqual(runs.view(self.run, None, self.job)["state"], "starting")

    def test_a_failed_job_says_why(self):
        self.job.status, self.job.error = "failed", "The dataset is empty."
        view = runs.view(self.run, None, self.job)
        self.assertEqual((view["state"], view["note"], view["can_stop"]), ("failed", "The dataset is empty.", False))

    def test_an_ended_run_has_no_live_members_and_keeps_what_it_showed(self):
        snap = self.snap(state="finished", round=4)
        view = runs.view(self.run, self.coordinator(snap), None)
        self.assertEqual((view["state"], view["can_stop"]), ("finished", False))
        self.assertEqual({m["node"]: m["state"] for m in view["members"]}, {"mac": "finished", "box": "finished", "old": "unreachable"})
        runs.archive("run1", self.coordinator(snap))
        again = runs.view(self.run, None, None)
        self.assertEqual((again["state"], len(again["history"])), ("finished", 1))

    def ended_rows(self, state, nodes, events=()):
        snap = self.snap(state=state, round=4, nodes=nodes, events=list(events))
        return {m["node"]: m for m in runs.view(self.run, self.coordinator(snap), None)["members"]}

    def test_an_ended_run_keeps_why_a_machine_never_joined(self):
        box = self.run.machines[1]
        box.state, box.reason = "unplaced", runs.UNPLACED
        rows = self.ended_rows("finished", {"mac": {"state": "training", "missed": 0, "sent": False, "note": ""}})
        self.assertEqual((rows["mac"]["state"], rows["mac"]["note"]), ("finished", ""))
        self.assertEqual((rows["box"]["state"], rows["box"]["note"], rows["box"]["can_stop"]), ("never_joined", runs.UNPLACED, False))
        self.assertEqual((rows["old"]["state"], rows["old"]["note"]), ("unreachable", "old did not answer within 2 seconds."))
        self.assertEqual(self.ended_rows("stopped", {})["box"]["state"], "never_joined")

    def test_an_invited_machine_that_was_never_heard_from_never_joined_when_the_run_ends(self):
        rows = self.ended_rows("finished", {"mac": {"state": "training", "missed": 0, "sent": False, "note": ""}})
        self.assertEqual((rows["box"]["state"], rows["box"]["note"]), ("never_joined", runs.NEVER_JOINED))
        self.assertEqual(rows["mac"]["state"], "finished")  # this computer's own member is never "never joined"
        self.assertEqual(self.ended_rows("finished", {})["mac"]["state"], "finished")

    def test_a_member_that_left_once_the_run_finished_reads_as_finished_but_one_that_left_earlier_still_left(self):
        left = {"state": "left", "missed": 0, "sent": False, "note": "Left the run."}
        events = [{"seq": 1, "kind": "left", "node": "box", "round": 1}, {"seq": 2, "kind": "finished", "node": "", "round": 3},
                  {"seq": 3, "kind": "left", "node": "mac", "round": 4}]
        rows = self.ended_rows("finished", {"mac": left, "box": left}, events)
        self.assertEqual((rows["mac"]["state"], rows["mac"]["note"]), ("finished", ""))
        self.assertEqual((rows["box"]["state"], rows["box"]["note"]), ("left", "Left the run."))
        stopped = self.ended_rows("stopped", {"mac": left, "box": left}, events)
        self.assertEqual(stopped["mac"]["state"], "left")

    def test_stopping_a_run_that_waits_in_the_queue_ends_it(self):
        runs.stop_run(self.run, None, clients())
        view = runs.view(self.run, None, None)
        self.assertEqual(view["state"], "stopped")
        self.assertEqual(FakeClient.calls, [])  # no member job had been placed


class Routes(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices(step_seconds=.005)
        self.services.save_version(default_spec(), "first")
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.addCleanup(self.services.wait_idle, 10)

    def start(self, **body):
        return self.client.post("/distributed/runs", json={"dataset_ids": ["Your files"], "rounds": 3, "inner_steps": 2, **body})

    def wait_state(self, run_id, *states, timeout=10.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            run = self.client.get(f"/distributed/runs/{run_id}").json()
            if run["state"] in states:
                return run
            time.sleep(.01)
        self.fail(f"run {run_id} is {run['state']}, not {states}")

    def test_the_form_starts_from_the_engines_defaults(self):
        self.assertEqual(self.client.get("/distributed/defaults").json(),
                         {"rounds": 10, "inner_steps": 100, "outer_lr": .7, "outer_momentum": .9, "delta_dtype": "float16"})

    def test_a_start_is_checked_with_a_sentence(self):
        self.assertEqual(self.start(dataset_ids=["nope"]).json()["detail"], "No text set nope.")
        self.assertEqual(self.start(machines=["ghost"]).json()["detail"], "ghost is not on the Machines tab.")
        self.assertEqual(self.start(outer_momentum=1.5).status_code, 422)
        self.assertEqual(self.client.post("/distributed/runs", json={"dataset_ids": []}).status_code, 422)
        self.assertEqual(self.start(version=9).status_code, 404)
        self.assertEqual(self.client.get("/distributed/runs/none").status_code, 404)

    def test_the_most_and_least_machines_a_run_needs_are_checked_and_kept_on_the_job(self):
        self.services.set_setting("nodes", ["127.0.0.1:9"])
        too_many = self.start(machines=["127.0.0.1:9"], min_members=3)
        self.assertEqual((too_many.status_code, too_many.json()["detail"]), (400, "The run needs 3 machines at once, but only 2 are chosen, this computer included."))
        self.assertEqual(self.start(min_members=3, max_members=2).status_code, 400)
        run_id = self.start(machines=["127.0.0.1:9"], min_members=2, max_members=2, rounds=2000).json()["id"]
        job = next(j for j in [self.services.queue().job, *self.services.queue().up_next] if j is not None and j.id == run_id)
        self.assertEqual((job.params["min_members"], job.params["max_members"]), (2, 2))
        self.client.post(f"/distributed/runs/{run_id}/stop")

    def test_an_overlapped_start_puts_the_option_on_the_job_for_the_owner_and_the_invited_members(self):
        plain = self.start(rounds=2000).json()["id"]
        overlapped = self.start(rounds=2000, overlap=True).json()["id"]
        jobs = {j.id: j for j in [self.services.queue().job, *self.services.queue().up_next] if j is not None}
        self.assertNotIn("member_options", jobs[plain].params)
        self.assertEqual(jobs[overlapped].params["member_options"], {"overlap": True})  # studio_jobs._member (owner) and invite (members) read it
        self.assertEqual(self.start(overlap="sometimes").status_code, 422)
        for run_id in (plain, overlapped):
            self.client.post(f"/distributed/runs/{run_id}/stop")

    def test_the_demo_run_is_live_with_a_cuda_and_an_mps_member_and_one_can_be_stopped(self):
        run = self.services.demo_run()
        runs_now = self.client.get("/distributed/runs").json()
        demo = next(r for r in runs_now if r["id"] == run.id)
        self.assertEqual(demo["state"], "running")
        self.assertEqual({m["node"]: m["device"] for m in demo["members"]}, {"mac-mini": "mps", "gpu-box": "cuda"})
        self.assertIn(demo["phase"], ("next_round", "training", "collecting"))
        self.assertTrue(all(m["phase"] for m in demo["members"]))
        self.assertEqual(len(demo["timeline"]), 1)
        self.assertEqual({lane["node"] for lane in demo["timeline"][0]["lanes"]}, {"mac-mini", "gpu-box"})
        self.assertEqual([e["kind"] for e in demo["events"]], ["joined", "joined"])
        stopped = self.client.post(f"/distributed/runs/{run.id}/members/gpu-box/stop")
        self.assertEqual(stopped.status_code, 200, stopped.text)
        self.assertEqual({m["node"]: m["state"] for m in stopped.json()["members"]}["gpu-box"], "stopped")
        self.assertEqual(self.client.post(f"/distributed/runs/{run.id}/stop").json()["state"], "stopped")

    def test_a_run_on_this_computer_and_a_machine_that_is_down_runs_and_finishes(self):
        self.services.set_setting("nodes", ["127.0.0.1:9"])
        started = self.start(machines=["127.0.0.1:9"])
        self.assertEqual(started.status_code, 201, started.text)
        run = self.wait_state(started.json()["id"], "finished")
        self.assertEqual((run["round"], run["rounds"], run["inner_steps"], run["version"]), (3, 3, 2, 1))
        rows = {m["address"]: m for m in run["members"]}
        self.assertEqual(rows["local"]["state"], "finished")
        self.assertEqual(rows["127.0.0.1:9"]["state"], "unreachable")
        self.assertTrue(rows["127.0.0.1:9"]["note"])
        self.assertEqual([r["round"] for r in run["history"]], [0, 1, 2])
        self.assertEqual([r["id"] for r in self.client.get("/distributed/runs").json()], [run["id"]])
        end = time.monotonic() + 10.
        while time.monotonic() < end and self.client.get(f"/distributed/runs/{run['id']}").json()["saved_version"] is None:
            time.sleep(.01)  # the job notes the saved version just after the coordinator reads finished
        self.assertEqual(self.client.get(f"/distributed/runs/{run['id']}").json()["saved_version"], 1)

    def test_a_run_can_be_stopped_and_one_member_taken_out(self):
        run_id = self.start(rounds=2000).json()["id"]
        self.wait_state(run_id, "running")
        local = next(m for m in self.client.get(f"/distributed/runs/{run_id}").json()["members"] if m["local"])
        self.assertEqual(self.client.post(f"/distributed/runs/{run_id}/members/{local['node']}/stop").status_code, 409)
        self.assertEqual(self.client.post(f"/distributed/runs/{run_id}/members/nobody/stop").status_code, 409)
        stopped = self.client.post(f"/distributed/runs/{run_id}/stop")
        self.assertEqual((stopped.status_code, stopped.json()["state"], stopped.json()["can_stop"]), (200, "stopped", False))
        stopped_run = self.wait_state(run_id, "stopped")
        self.assertEqual((stopped_run["members"][0]["state"], stopped_run["saved_version"]), ("stopped", None))

    def test_a_run_waiting_in_the_queue_can_be_stopped_before_it_runs(self):
        self.services.pause()
        run_id = self.start().json()["id"]
        self.assertEqual(self.client.get(f"/distributed/runs/{run_id}").json()["state"], "queued")
        self.assertEqual(self.client.post(f"/distributed/runs/{run_id}/stop").json()["state"], "stopped")
        self.assertEqual(self.client.get("/queue").json()["up_next"], [])


class AllMachines(unittest.TestCase):
    """The queue's "All machines (DiLoCo)": a waiting pretrain job becomes a run on every listed machine that answers on this version."""

    PEERS = {"box": studio_cluster.NodeInfo(address="box", hostname="box", version="1.0.0", device="cuda", device_name="RTX"),
             "old": studio_cluster.offline("old", "old did not answer within 2 seconds."),
             "new": studio_cluster.NodeInfo(address="new", hostname="newpc", version="2.0.0", device="cuda")}

    def setUp(self):
        self.services = FakeServices(step_seconds=.005)
        self.services.save_version(default_spec(), "first")
        self.services.pause()  # the jobs stay in Up next to be read; no member is invited
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.services.set_setting("nodes", list(self.PEERS))
        self.enterContext(mock.patch("studio_api.routers.distributed.app_version", return_value="1.0.0"))
        self.surveyed = []

        def survey(state, own, *args, **kwargs):
            self.surveyed.append(state["studioConfig"]["nodes"])
            return [own, *(self.PEERS[a] for a in state["studioConfig"]["nodes"])]
        self.enterContext(mock.patch.object(studio_cluster, "survey", survey))
        for label in ("before", "pretrain", "after"):
            self.services.enqueue(Job(id=label, kind="pretrain" if label == "pretrain" else "quicktest", label=label,
                                      params={"version": 1, "datasets": ["Your files"]} if label == "pretrain" else {}))

    def start(self, **body):
        defaults = self.client.get("/distributed/defaults").json()
        return self.client.post("/distributed/runs", json={"dataset_ids": ["Your files"], "all_machines": True, "replaces": "pretrain",
                                                           **{k: defaults[k] for k in ("rounds", "inner_steps", "outer_lr", "outer_momentum")},
                                                           **body})

    def test_the_pretrain_job_becomes_a_run_in_its_place_on_every_machine_that_answers(self):
        started = self.start()
        self.assertEqual(started.status_code, 201, started.text)
        run = started.json()
        self.assertEqual(self.surveyed, [list(self.PEERS)])  # every listed machine was asked
        waiting = self.services.queue().up_next
        self.assertEqual([j.id for j in waiting], ["before", run["id"], "after"])
        job = waiting[1]
        self.assertEqual((job.kind, job.params["version"], job.params["datasets"]), ("pretrain_distributed", 1, ["Your files"]))
        self.assertEqual((job.params["rounds"], job.params["inner_steps"]), (studio_diloco.Settings().rounds, studio_diloco.Settings().inner_steps))
        self.assertEqual([m["address"] for m in job.params["machines"]], ["local", "box"])
        self.assertEqual(job.params["left_out"], [
            {"node": "old", "address": "old", "reason": "old did not answer within 2 seconds."},
            {"node": "newpc", "address": "new", "reason": "newpc runs app version 2.0.0; this computer runs 1.0.0."}])
        self.assertEqual([m["address"] for m in run["members"]], ["local", "box"])
        self.assertEqual(run["state"], "queued")

    def test_when_no_other_machine_can_join_the_pretrain_job_stays_and_the_reply_says_why(self):
        self.services.set_setting("nodes", ["old", "new"])
        refused = self.start()
        self.assertEqual(refused.status_code, 409)
        self.assertIn("No other machine can join", refused.json()["detail"])
        self.assertIn("old did not answer", refused.json()["detail"])
        self.assertEqual([j.id for j in self.services.queue().up_next], ["before", "pretrain", "after"])

    def test_only_a_waiting_pretrain_job_can_be_replaced(self):
        self.assertEqual(self.start(replaces="before").status_code, 409)
        self.assertEqual(self.start(replaces="gone").status_code, 404)
        self.assertEqual([j.id for j in self.services.queue().up_next], ["before", "pretrain", "after"])

    def test_the_run_takes_the_pretrain_jobs_version_and_text_over_the_body(self):
        self.services.save_version(default_spec(), "second")
        run = self.start(version=2, dataset_ids=["nope"]).json()
        self.assertEqual(run["version"], 1)


class CoordinatorChoice(unittest.TestCase):
    """Choosing which machine coordinates a run: validated at the start, kept on the job, and nothing changes without it."""

    PEERS = {"box": studio_cluster.NodeInfo(address="box", hostname="box", version="1.0.0", device="cuda", device_name="RTX"),
             "off": studio_cluster.offline("off", "off did not answer within 2 seconds."),
             "new": studio_cluster.NodeInfo(address="new", hostname="newpc", version="2.0.0", device="cuda"),
             "spare": studio_cluster.NodeInfo(address="spare", hostname="spare", version="1.0.0", device="mps")}

    def setUp(self):
        self.services = FakeServices(step_seconds=.005)
        self.services.save_version(default_spec(), "first")
        self.services.pause()
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.services.set_setting("nodes", list(self.PEERS))
        self.enterContext(mock.patch("studio_api.routers.distributed.app_version", return_value="1.0.0"))
        self.enterContext(mock.patch.object(studio_cluster, "survey", lambda state, own, *a, **k: [own, *(self.PEERS[x] for x in state["studioConfig"]["nodes"])]))

    def start(self, **body):
        return self.client.post("/distributed/runs", json={"dataset_ids": ["Your files"], "rounds": 3, "inner_steps": 2, **body})

    def job(self, run):
        return next(j for j in self.services.queue().up_next if j.id == run["id"])

    def test_a_chosen_machine_that_is_invited_and_answers_is_kept_on_the_job_and_named_on_the_run(self):
        started = self.start(machines=["box", "spare"], coordinator="box")
        self.assertEqual(started.status_code, 201, started.text)
        self.assertEqual(self.job(started.json()).params["coordinator"], "box")
        self.assertEqual(started.json()["handoff"], "box")
        self.assertEqual(self.client.get(f"/distributed/runs/{started.json()['id']}").json()["handoff"], "box")

    def test_all_machines_lets_any_listed_machine_that_answers_coordinate(self):
        started = self.start(all_machines=True, coordinator="spare")
        self.assertEqual((started.status_code, started.json()["handoff"]), (201, "spare"))

    def test_without_a_choice_this_computer_coordinates_as_before(self):
        started = self.start(machines=["box"])
        self.assertEqual(started.status_code, 201, started.text)
        self.assertNotIn("coordinator", self.job(started.json()).params)
        self.assertEqual(started.json()["handoff"], "")
        self.assertEqual(self.start(machines=["box"], coordinator="").json()["handoff"], "")

    def test_a_machine_that_is_not_invited_cannot_coordinate(self):
        refused = self.start(machines=["box"], coordinator="spare")
        self.assertEqual((refused.status_code, refused.json()["detail"]),
                         (400, "spare cannot coordinate the run: it is not one of the machines chosen for it."))
        self.assertEqual(self.start(machines=["box"], coordinator="ghost").status_code, 400)
        self.assertEqual(self.services.queue().up_next, [])  # nothing was queued

    def test_a_machine_that_does_not_answer_or_runs_another_version_cannot_coordinate(self):
        down = self.start(machines=["box", "off"], coordinator="off")
        self.assertEqual((down.status_code, down.json()["detail"]), (409, "off cannot coordinate the run: off did not answer within 2 seconds."))
        old = self.start(machines=["box", "new"], coordinator="new")
        self.assertEqual((old.status_code, old.json()["detail"]),
                         (409, "newpc cannot coordinate the run: it runs app version 2.0.0; this computer runs 1.0.0."))
        every = self.start(all_machines=True, coordinator="off")
        self.assertEqual(every.status_code, 409)  # also when the rest of the machines would have been enough
        self.assertEqual(self.services.queue().up_next, [])

    def test_the_run_remembers_the_node_the_chosen_machine_joins_under(self):
        job = Job(id="run9", created_at=1., kind="pretrain_distributed", label="x", params={
            "machines": [MACHINES[0], {"node": "peer", "address": "box"}], "coordinator": "box", "owner": "me"})
        self.addCleanup(runs.RUNS.clear)
        self.assertEqual(runs.run_of(job).handoff, "peer")
        plain = Job(id="run8", created_at=1., kind="pretrain_distributed", label="x", params={"machines": [MACHINES[0]], "owner": "me"})
        self.assertEqual(runs.run_of(plain).handoff, "")


class Invited(unittest.TestCase):
    """The invitation over real HTTP: a second app (a FakeServices) takes the member job on its own queue."""

    def test_the_chosen_machine_gets_a_member_job_naming_this_run(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        peer = FakeServices()
        peer.pause()  # the job stays in Up next to be read
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            server = uvicorn.Server(uvicorn.Config(create_app(peer, open=True, hosts=ANY), host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.addCleanup(setattr, server, "should_exit", True)
        self.addCleanup(runs.RUNS.clear)
        end = time.monotonic() + 10
        while not server.started and time.monotonic() < end:
            time.sleep(.01)
        address = f"127.0.0.1:{port}"
        job = owner_job([MACHINES[0], {"node": "peer", "address": address, "device": "cuda"}], peer_url=f"127.0.0.1:{port + 1}")
        runs.invite(job)
        member = runs.run_of(job).machines[1]
        self.assertEqual(member.state, "invited")
        queued = peer.queue().up_next[0]
        self.assertEqual((queued.kind, queued.id), ("diloco_member", member.peer_job))
        self.assertEqual(queued.params, {"coordinator_url": f"127.0.0.1:{port + 1}", "run": "run1", "node": "peer", "epoch": 0})
        runs.release_member(runs.run_of(job), "peer")
        self.assertEqual(peer.queue().up_next, [])


class AddressForPeersTest(unittest.TestCase):
    def request(self, tailnet):
        from types import SimpleNamespace
        return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(hosts=SimpleNamespace(tailnet_name=tailnet))),
                               url=SimpleNamespace(port=8787), base_url="http://127.0.0.1:8787/")

    def test_peers_reach_the_coordinator_through_tailscale_serve(self):
        from studio_api.routers.distributed import _address_for_peers
        with mock.patch.dict("os.environ", {"TRAINER_NODE_ADDRESS": ""}):
            self.assertEqual(_address_for_peers(self.request("mac.tail.ts.net")), "https://mac.tail.ts.net")
            self.assertEqual(_address_for_peers(self.request("")), "http://127.0.0.1:8787")
        with mock.patch.dict("os.environ", {"TRAINER_NODE_ADDRESS": "https://other.tail.ts.net"}):
            self.assertEqual(_address_for_peers(self.request("mac.tail.ts.net")), "https://other.tail.ts.net")


if __name__ == "__main__":
    unittest.main()

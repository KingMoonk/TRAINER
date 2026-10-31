"""The Pretraining page pools the machines that are ready by default: POST /pretraining/pool lists them, /start queues a distributed
run with them, "this computer only" and the lack of any eligible machine keep a solo job, and Auto Run's pretrain phase does the same."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

import studio_cluster
import studio_distributed as runs
import studio_diloco
from studio_api.routers.pretraining import auto_run_pretrain
from studio_ia.services import FakeServices, Job
from studio_spec import default_spec

from tests.sse_harness import ApiCase

BODY = {"dataset_ids": ["Your files"], "mode": "from_scratch"}
APP_VERSIONS = ("studio_api.routers.distributed.app_version", "studio_api.version.app_version")


class PoolTests(ApiCase):
    prefix = "/pretraining"
    PEERS = {"box": studio_cluster.NodeInfo(address="box", hostname="box", version="1.0.0", device="cuda"),
             "old": studio_cluster.offline("old", "old did not answer within 2 seconds."),
             "new": studio_cluster.NodeInfo(address="new", hostname="newpc", version="2.0.0", device="cuda"),
             "busy": studio_cluster.NodeInfo(address="busy", hostname="busypc", version="1.0.0", device="cuda", busy=True)}

    def setUp(self):
        super().setUp()
        self.services.save_version(default_spec(), "first")
        self.services.pause()
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        for name in APP_VERSIONS:
            self.enterContext(mock.patch(name, return_value="1.0.0"))
        self.enterContext(mock.patch.object(studio_cluster, "survey", lambda state, own, *a, **k: [own, *(self.PEERS[x] for x in state["studioConfig"]["nodes"])]))

    def nodes(self, *names):
        self.services.set_setting("nodes", list(names))

    def test_a_ready_machine_is_listed_and_the_others_say_why_they_are_left_out(self):
        self.nodes("box", "old", "new", "busy")
        pool = self.call("POST", "/pretraining/pool", json=BODY)
        self.assertEqual((pool["pooled"], [m["node"] for m in pool["machines"]]), (True, ["box"]))
        self.assertEqual({row["node"]: row["reason"] for row in pool["left_out"]}, {
            "old": "old did not answer within 2 seconds.",
            "newpc": "newpc runs app version 2.0.0; this computer runs 1.0.0.",
            "busypc": "busypc is busy with another job."})

    def test_start_with_one_eligible_peer_queues_a_distributed_run_with_both_machines(self):
        self.nodes("box")
        started = self.call("POST", "/pretraining/start", 202, json=BODY)
        job = self.services.queue().up_next[-1]
        self.assertEqual((job.kind, started["id"], started["run_id"], started["machines"]), ("pretrain_distributed", job.id, job.id, ["box"]))
        self.assertEqual([m["address"] for m in job.params["machines"]], ["local", "box"])
        self.assertEqual(job.params["datasets"], ["Your files"])
        self.assertEqual(self.client.get(f"/distributed/runs/{job.id}").json()["id"], job.id)

    def test_start_with_ready_and_busy_peer_includes_both_for_auto_join(self):
        self.nodes("box", "busy")
        self.call("POST", "/pretraining/start", 202, json=BODY)
        job = self.services.queue().up_next[-1]
        self.assertEqual(job.kind, "pretrain_distributed")
        self.assertEqual([m["address"] for m in job.params["machines"]], ["local", "box", "busy"])

    def test_with_no_eligible_peer_start_queues_the_solo_job_and_says_why(self):
        self.nodes("old", "new")
        pool = self.call("POST", "/pretraining/pool", json=BODY)
        self.assertFalse(pool["pooled"])
        self.assertIn("old did not answer", pool["reason"])
        self.assertEqual(len(pool["left_out"]), 2)
        started = self.call("POST", "/pretraining/start", 202, json=BODY)
        self.assertEqual(self.services.queue().up_next[-1].kind, "pretrain")
        self.assertEqual((started["run_id"], started["note"]), (None, pool["reason"]))

    def test_with_no_machines_listed_the_run_stays_on_this_computer(self):
        pool = self.call("POST", "/pretraining/pool", json=BODY)
        self.assertEqual((pool["pooled"], pool["reason"]), (False, "No other machine is on the Machines list."))
        self.call("POST", "/pretraining/start", 202, json=BODY)
        self.assertEqual(self.services.queue().up_next[-1].kind, "pretrain")

    def while_sizing(self):
        """Start with the size of the text unknown: the placeholder job waits (in second place) until `release` is set."""
        known, preview, release = {}, self.services.pretrain_preview, threading.Event()

        def slow(ids, rows=16, progress=None):
            self.assertTrue(release.wait(10))
            return known.setdefault(tuple(ids), preview(ids, rows))
        self.services.cached_preview = lambda ids, rows=16: known.get(tuple(ids))
        self.services.pretrain_preview = slow
        return release

    def until(self, check, what):
        end = time.monotonic() + 10
        while not check():
            self.assertLess(time.monotonic(), end, f"waited for {what}")
            time.sleep(0.02)

    def test_a_run_added_while_the_size_is_worked_out_is_pooled_in_the_placeholders_place(self):
        self.nodes("box")
        release = self.while_sizing()
        pool = self.call("POST", "/pretraining/pool", json=BODY)
        self.assertEqual((pool["pooled"], pool["pending"]), (False, True))
        self.assertIn("Once it is, this run joins the machines that are ready", pool["reason"])
        self.services.enqueue(Job(id="first", kind="other", label="first", created_at=0))
        started = self.call("POST", "/pretraining/start", 202, json=BODY)
        placeholder = next(j for j in self.services.queue().up_next if j.id == started["id"])
        self.assertEqual(placeholder.params, {"sizing": True})
        release.set()
        self.until(lambda: all(j.id != started["id"] for j in self.services.queue().up_next), "the placeholder to be replaced")
        up_next = self.services.queue().up_next
        self.assertEqual([j.kind for j in up_next], ["other", "pretrain_distributed"])
        self.assertEqual([m["address"] for m in up_next[1].params["machines"]], ["local", "box"])
        self.assertEqual(self.client.get(f"/distributed/runs/{up_next[1].id}").json()["id"], up_next[1].id)

    def test_a_run_added_while_the_size_is_worked_out_stays_solo_without_an_eligible_machine(self):
        self.nodes("old")
        release = self.while_sizing()
        started = self.call("POST", "/pretraining/start", 202, json=BODY)
        release.set()
        queued = next(j for j in self.services.queue().up_next if j.id == started["id"])
        self.until(lambda: "config" in queued.params, "the run's steps")
        self.assertEqual(queued.kind, "pretrain")

    def test_a_placeholder_that_cannot_be_pooled_ends_solo_and_its_note_names_the_refusal(self):
        self.nodes("box")
        release = self.while_sizing()
        started = self.call("POST", "/pretraining/start", 202, json=BODY)
        queued = next(j for j in self.services.queue().up_next if j.id == started["id"])
        refusal = HTTPException(409, "box cannot hold its share")
        with mock.patch("studio_api.routers.distributed.build_run", side_effect=refusal):
            release.set()
            self.until(lambda: "config" in queued.params, "the run's steps")
        self.assertEqual(queued.kind, "pretrain")
        self.assertIn("box cannot hold its share", queued.note)
        self.assertEqual(self.call("GET", "/queue")["up_next"][0]["note"], queued.note)

    def test_a_placeholder_with_no_eligible_machine_says_why_it_trains_alone(self):
        self.nodes("old")
        release = self.while_sizing()
        started = self.call("POST", "/pretraining/start", 202, json=BODY)
        release.set()
        queued = next(j for j in self.services.queue().up_next if j.id == started["id"])
        self.until(lambda: "config" in queued.params, "the run's steps")
        self.assertIn("old did not answer", queued.note)

    def test_this_computer_only_keeps_the_placeholder_solo(self):
        self.nodes("box")
        release = self.while_sizing()
        started = self.call("POST", "/pretraining/start", 202, json={**BODY, "solo": True})
        release.set()
        queued = next(j for j in self.services.queue().up_next if j.id == started["id"])
        self.until(lambda: "config" in queued.params, "the run's steps")
        self.assertEqual(queued.kind, "pretrain")

    def test_this_computer_only_forces_a_solo_job(self):
        self.nodes("box")
        pool = self.call("POST", "/pretraining/pool", json={**BODY, "solo": True})
        self.assertEqual((pool["pooled"], pool["machines"]), (False, []))
        self.call("POST", "/pretraining/start", 202, json={**BODY, "solo": True})
        self.assertEqual(self.services.queue().up_next[-1].kind, "pretrain")

    def test_rows_and_held_back_share_chosen_on_the_page_reach_the_pooled_run(self):
        self.nodes("box")
        started = self.call("POST", "/pretraining/start", 202, json={**BODY, "rows": 32, "hold": 0.1})
        job = self.services.queue().up_next[-1]
        self.assertEqual((started["run_id"], job.params["heldout_fraction"], job.params["config"]["batch_size"], job.params["config"]["validation"]),
                         (job.id, 0.1, 32, 0.1))

    def test_a_chat_mix_reaches_the_pooled_run(self):
        self.nodes("box")
        sets = [{"id": "talk", "name": "Talk", "count": 9}]
        with mock.patch.object(self.services, "conversation_sets", create=True, return_value=sets):
            self.call("POST", "/pretraining/start", 202, json={**BODY, "chat_share": 0.1})
        job = self.services.queue().up_next[-1]
        self.assertEqual((job.kind, job.params["chat_share"], job.params["chat_datasets"]), ("pretrain_distributed", 0.1, ["talk"]))
        self.assertIn("10% chat", job.label)

    def test_the_stop_when_flat_rule_ends_a_pooled_run_too(self):
        self.nodes("box")
        self.call("POST", "/pretraining/start", 202, json={**BODY, "plateau": True})
        job = self.services.queue().up_next[-1]
        rule = self.call("POST", "/pretraining/plan", json={**BODY, "plateau": True})["plateau"]
        self.assertGreaterEqual(job.params["plateau_checks"], 2)
        self.assertEqual(job.params["plateau_min_improvement"], rule["min_gain"])
        self.assertEqual(job.params["settings"]["plateau_checks"], job.params["plateau_checks"])

    def test_without_the_rule_the_pooled_run_has_no_plateau_window(self):
        self.nodes("box")
        self.call("POST", "/pretraining/start", 202, json={**BODY, "plateau": False})
        self.assertNotIn("plateau_checks", self.services.queue().up_next[-1].params)

    def trained(self, **spec_changes):
        folder = Path(tempfile.mkdtemp())
        (folder / "best.npz").write_bytes(b"x")
        first = self.services.versions()[0]
        first.best_checkpoint = str(folder / "best.npz")
        second = self.services.save_version({**first.spec, **spec_changes}, "second")
        return first, second

    def test_a_catch_up_with_a_ready_machine_starts_the_run_from_the_chosen_versions_best_weights(self):
        self.nodes("box")
        first, second = self.trained()
        started = self.call("POST", "/pretraining/start", 202, json={**BODY, "mode": "catch_up", "version": second.number, "from_version": first.number})
        job = self.services.queue().up_next[-1]
        self.assertEqual((job.kind, started["run_id"], job.params["mode"], job.params["restart_from"]), ("pretrain_distributed", job.id, "catch_up", first.best_checkpoint))
        self.assertIn(f"continuing version {first.number}", job.label)

    def test_a_catch_up_from_another_design_stays_on_this_computer_and_says_why(self):
        self.nodes("box")
        first, second = self.trained(dims={**default_spec()["dims"], "n_layer": 3})
        body = {**BODY, "mode": "catch_up", "version": second.number, "from_version": first.number}
        pool = self.call("POST", "/pretraining/pool", json=body)
        self.assertEqual(pool["pooled"], False)
        self.assertIn("another design", pool["reason"])
        self.call("POST", "/pretraining/start", 202, json=body)
        self.assertEqual(self.services.queue().up_next[-1].kind, "pretrain")

    def test_a_machine_that_fails_the_start_checks_after_the_survey_leaves_a_solo_job(self):
        self.nodes("box")
        with mock.patch("studio_api.routers.distributed.build_run", side_effect=HTTPException(409, "box went away")):
            started = self.call("POST", "/pretraining/start", 202, json=BODY)
        self.assertEqual(self.services.queue().up_next[-1].kind, "pretrain")
        self.assertIn("box went away", started["note"])


class AutoRunPoolTests(unittest.TestCase):
    SETTINGS = {"hours": 1, "passes": 1, "mode": "from_scratch"}

    def setUp(self):
        self.services = FakeServices()
        self.services.save_version(default_spec(), "first")
        self.services.set_setting("nodes", ["box"])
        self.addCleanup(runs.RUNS.clear)
        for name in APP_VERSIONS:
            self.enterContext(mock.patch(name, return_value="1.0.0"))
        peer = studio_cluster.NodeInfo(address="box", hostname="box", version="1.0.0", device="cuda")
        self.enterContext(mock.patch.object(studio_cluster, "survey", lambda state, own, *a, **k: [own, peer]))

    def test_the_pretrain_phase_pools_the_ready_machines_like_the_start_route(self):
        job = auto_run_pretrain(self.services, self.SETTINGS, 1, peer_url=lambda: "https://me.ts.net")
        self.assertEqual(job.kind, "pretrain_distributed")
        self.assertEqual(([m["address"] for m in job.params["machines"]], job.params["peer_url"]), (["local", "box"], "https://me.ts.net"))

    def test_this_computer_only_and_no_address_give_the_solo_job(self):
        self.assertEqual(auto_run_pretrain(self.services, {**self.SETTINGS, "solo": True}, 1, peer_url=lambda: "https://me.ts.net").kind, "pretrain")
        self.assertEqual(auto_run_pretrain(self.services, self.SETTINGS, 1, peer_url=lambda: "").kind, "pretrain")
        self.assertEqual(auto_run_pretrain(self.services, self.SETTINGS, 1).kind, "pretrain")


if __name__ == "__main__":
    unittest.main()

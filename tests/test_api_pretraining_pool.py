"""The Pretraining page pools the machines that are ready by default: POST /pretraining/pool lists them, /start queues a distributed
run with them, "this computer only" and the lack of any eligible machine keep a solo job, and Auto Run's pretrain phase does the same."""
import unittest
from unittest import mock

from fastapi import HTTPException

import studio_cluster
import studio_distributed as runs
import studio_diloco
from studio_api.routers.pretraining import auto_run_pretrain
from studio_ia.services import FakeServices
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

    def test_this_computer_only_forces_a_solo_job(self):
        self.nodes("box")
        pool = self.call("POST", "/pretraining/pool", json={**BODY, "solo": True})
        self.assertEqual((pool["pooled"], pool["machines"]), (False, []))
        self.call("POST", "/pretraining/start", 202, json={**BODY, "solo": True})
        self.assertEqual(self.services.queue().up_next[-1].kind, "pretrain")

    def test_choices_a_pooled_run_cannot_honour_keep_it_on_this_computer(self):
        self.nodes("box")
        rows = self.call("POST", "/pretraining/pool", json={**BODY, "rows": 32})
        self.assertEqual(rows["pooled"], False)
        self.assertIn("16 rows per step", rows["reason"])

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

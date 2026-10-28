"""studio_api.routers.auto_run over FakeServices: make, change, start, pause, resume, stop, skip and continue an Auto Run; no runner, no window.
Jobs are ended by hand with FakeServices.finish_job."""
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices, Version

GOOD = {"ok": True, "checkpoint": "ck-good", "checks": {}}
BAD = {"ok": False, "checkpoint": "ck-kept", "verdict": "It got worse."}


def plan_of(kinds=("sft", "rlvr"), version=1, hours=10, **policy):
    phases = [{"kind": k, "enabled": True, "settings": {}, "on_pass": "continue", "on_fail": "pause", **policy} for k in kinds]
    return {"id": "run1", "name": "Night run", "version": version, "budget": {"hours": hours}, "phases": phases}


class Base(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.services._versions.append(Version(1, "v1", "first", status="trained", best_checkpoint="ck0"))
        self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))

    def make(self, plan=None):
        r = self.client.post("/auto-runs", json={"plan": plan or plan_of()})
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def go(self, plan=None):
        self.make(plan)
        r = self.client.post("/auto-runs/run1/start")
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def go_started(self):
        r = self.client.post("/auto-runs/run1/start")
        self.assertEqual(r.status_code, 200, r.text)

    def job(self):
        return self.services.queue().up_next[0].id if self.services.queue().up_next else self.services.queue().job.id

    def end(self, job_id, status="done", result=None):
        self.services.finish_job(job_id, status, result)


class AutoRunApiTest(Base):
    def test_nothing_yet(self):
        self.assertEqual(self.client.get("/auto-runs").json(), [])
        self.assertEqual(self.client.get("/auto-runs/nope").status_code, 404)
        self.assertEqual(self.client.post("/auto-runs/nope/start").status_code, 404)

    def test_defaults_for_a_version(self):
        plan = self.client.get("/auto-runs/defaults", params={"version": 3}).json()
        self.assertEqual((plan["version"], plan["id"]), (3, ""))
        self.assertEqual([p["kind"] for p in plan["phases"]][:2], ["data", "pretrain"])
        self.assertFalse(next(p for p in plan["phases"] if p["kind"] == "midtrain")["enabled"])
        self.assertEqual(self.client.get("/auto-runs/defaults").status_code, 422)

    def test_a_new_run_is_a_draft_and_listed(self):
        body = self.make()
        self.assertEqual((body["state"]["status"], body["state"]["current"], body["progress"]), ("draft", None, None))
        self.assertEqual(body["plan"]["name"], "Night run")
        rows = self.client.get("/auto-runs").json()
        self.assertEqual([(r["id"], r["status"], r["version"]) for r in rows], [("run1", "draft", 1)])

    def test_a_run_without_an_id_is_given_one(self):
        body = self.make({**plan_of(), "id": ""})
        self.assertTrue(body["id"].startswith("ar-"))

    def test_a_bad_plan_is_422_with_every_sentence(self):
        bad = plan_of(("rlvr", "sft"))
        bad["name"] = ""
        r = self.client.post("/auto-runs", json={"plan": bad})
        self.assertEqual(r.status_code, 422)
        self.assertIn("needs a name", r.json()["detail"])
        self.assertIn("out of order", r.json()["detail"])
        none_on = plan_of()
        for phase in none_on["phases"]:
            phase["enabled"] = False
        self.assertIn("At least one phase", self.client.post("/auto-runs", json={"plan": none_on}).json()["detail"])
        self.assertEqual(self.client.get("/auto-runs").json(), [])

    def test_a_duplicate_id_is_422(self):
        self.make()
        r = self.client.post("/auto-runs", json={"plan": plan_of()})
        self.assertEqual(r.status_code, 422)
        self.assertIn("already an Auto Run", r.json()["detail"])

    def test_start_queues_the_first_phase_and_progress_names_its_job(self):
        body = self.go()
        self.assertEqual((body["state"]["status"], body["state"]["current"]), ("running", "sft"))
        job = body["progress"]
        self.assertEqual((job["id"], job["kind"], job["status"]), (self.job(), "post_round", "waiting"))
        self.assertIsNone(job["step"])
        self.assertEqual(self.client.get("/auto-runs/run1").json()["progress"]["id"], job["id"])
        self.assertIn("start", [line["event"] for line in body["journal"]])

    def test_a_finished_phase_moves_on_and_the_run_completes(self):
        self.go()
        self.end(self.job(), result=GOOD)
        body = self.client.get("/auto-runs/run1").json()
        self.assertEqual((body["state"]["current"], body["state"]["phases"][0]["status"]), ("rlvr", "passed"))
        self.end(self.job(), result=GOOD)
        body = self.client.get("/auto-runs/run1").json()
        self.assertEqual(body["state"]["status"], "done")
        self.assertIsNone(body["progress"])

    def test_start_twice_is_409(self):
        self.go()
        r = self.client.post("/auto-runs/run1/start")
        self.assertEqual(r.status_code, 409)
        self.assertIn("already started", r.json()["detail"])

    def test_pause_and_resume(self):
        self.make()
        self.assertEqual(self.client.post("/auto-runs/run1/pause").status_code, 409)     # a draft cannot be paused
        self.assertEqual(self.client.post("/auto-runs/run1/resume").status_code, 409)
        self.go_started()
        self.assertEqual(self.client.post("/auto-runs/run1/pause").json()["state"]["status"], "paused")   # (its job only waited: held)
        self.assertEqual(self.services.find_job(self.job()).status, "held")
        r = self.client.post("/auto-runs/run1/resume").json()
        self.assertEqual(r["state"]["status"], "running")
        self.assertEqual(r["progress"]["status"], "waiting")

    def test_stop_removes_the_job_and_keeps_the_run(self):
        self.go()
        body = self.client.post("/auto-runs/run1/stop", json=None).json()
        self.assertEqual(body["state"]["status"], "stopped")
        self.assertEqual(self.services.queue().up_next, [])
        self.assertEqual(self.client.post("/auto-runs/run1/stop").json()["state"]["status"], "stopped")   # stopping again changes nothing

    def test_a_failed_phase_pauses_and_continue_chooses(self):
        self.go()
        self.end(self.job(), result=BAD)
        state = self.client.get("/auto-runs/run1").json()["state"]
        self.assertEqual((state["status"], state["current"], state["phases"][0]["status"]), ("paused", "sft", "failed"))
        self.assertEqual(self.client.post("/auto-runs/run1/continue", json={"choice": "nope"}).status_code, 422)
        body = self.client.post("/auto-runs/run1/continue", json={"choice": "next"}).json()
        self.assertEqual((body["state"]["status"], body["state"]["current"], body["state"]["phases"][0]["status"]), ("running", "rlvr", "skipped"))
        self.assertEqual(self.client.post("/auto-runs/run1/continue", json={"choice": "next"}).status_code, 409)   # nothing waits now

    def test_continue_retry_and_stop(self):
        self.go()
        self.end(self.job(), result=BAD)
        body = self.client.post("/auto-runs/run1/continue", json={"choice": "retry"}).json()
        self.assertEqual((body["state"]["status"], body["state"]["phases"][0]["attempt"]), ("running", 2))
        self.end(self.job(), result=BAD)
        body = self.client.post("/auto-runs/run1/continue", json={"choice": "stop"}).json()
        self.assertEqual(body["state"]["status"], "stopped")

    def test_continue_adopt_without_a_checkpoint_to_keep_is_409(self):
        self.go()
        self.end(self.job(), result=BAD)
        r = self.client.post("/auto-runs/run1/continue", json={"choice": "adopt"})
        self.assertEqual(r.status_code, 409)

    def test_skip_a_failed_phase_then_resume(self):
        self.go()
        self.end(self.job(), result=BAD)
        body = self.client.post("/auto-runs/run1/skip").json()
        self.assertEqual((body["state"]["status"], body["state"]["phases"][0]["status"], body["state"]["current"]), ("paused", "skipped", "rlvr"))
        self.assertEqual(self.client.post("/auto-runs/run1/resume").json()["state"]["status"], "running")

    def test_skip_with_nothing_to_skip_is_409(self):
        self.go()
        self.assertEqual(self.client.post("/auto-runs/run1/skip").status_code, 409)     # the sft phase is running

    def test_a_second_run_on_the_same_version_cannot_start(self):
        self.go()
        self.make({**plan_of(), "id": "run2"})
        r = self.client.post("/auto-runs/run2/start")
        self.assertEqual(r.status_code, 409)
        self.assertIn("already running", r.json()["detail"])

    def test_a_phase_that_cannot_be_set_up_blocks_the_run(self):
        self.services._versions[0].best_checkpoint = ""
        body = self.go()
        self.assertEqual(body["state"]["status"], "waiting_user")
        self.assertIn("no trained weights", body["state"]["reason"])


class PlanChangeTest(Base):
    def put(self, plan):
        return self.client.put("/auto-runs/run1/plan", json={"plan": plan})

    def test_a_draft_takes_any_valid_plan(self):
        self.make()
        r = self.put({**plan_of(("sft", "rlvr", "dpo"), hours=5), "name": "Renamed"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual((body["plan"]["name"], body["plan"]["budget"]["hours"]), ("Renamed", 5))
        self.assertEqual([p["kind"] for p in body["state"]["phases"]], ["sft", "rlvr", "dpo"])
        self.assertEqual(body["state"]["status"], "draft")
        self.assertEqual(self.client.get("/auto-runs/run1").json()["plan"]["name"], "Renamed")

    def test_a_bad_plan_is_422_and_changes_nothing(self):
        self.make()
        r = self.put({**plan_of(), "name": ""})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.client.get("/auto-runs/run1").json()["plan"]["name"], "Night run")
        self.assertEqual(self.put({**plan_of(), "id": "other"}).status_code, 422)

    def test_a_running_stopped_or_unknown_run_is_409_or_404(self):
        self.assertEqual(self.client.put("/auto-runs/nope/plan", json={"plan": plan_of()}).status_code, 404)
        self.go()
        self.assertEqual(self.put(plan_of()).status_code, 409)
        self.client.post("/auto-runs/run1/stop")
        self.assertEqual(self.put(plan_of()).status_code, 409)

    def paused_after_first(self):
        self.go(plan_of(("sft", "rlvr", "dpo"), on_pass="continue"))
        self.end(self.job(), result=GOOD)           # sft passed, rlvr queued
        self.client.post("/auto-runs/run1/pause")   # rlvr only waits: held, parked
        self.assertEqual(self.client.get("/auto-runs/run1").json()["state"]["status"], "paused")

    def test_a_paused_run_changes_the_phases_after_the_current_one(self):
        self.paused_after_first()
        plan = plan_of(("sft", "rlvr", "dpo"))
        plan["phases"][2].update(enabled=False)
        r = self.put(plan)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["state"]["phases"][2]["status"], "skipped")
        plan["phases"][2].update(enabled=True, settings={"steps": 5})
        r = self.put(plan)
        self.assertEqual((r.json()["state"]["phases"][2]["status"], r.json()["plan"]["phases"][2]["settings"]), ("pending", {"steps": 5}))
        self.assertIn("plan_changed", [line["event"] for line in r.json()["journal"]])

    def test_a_paused_run_keeps_what_has_passed_or_has_a_job(self):
        self.paused_after_first()
        before = plan_of(("sft", "rlvr", "dpo"))
        passed = [dict(p) for p in before["phases"]]
        passed[0]["settings"] = {"steps": 1}
        r = self.put({**before, "phases": passed})
        self.assertEqual(r.status_code, 422)
        self.assertIn("sft phase can no longer be changed", r.json()["detail"])
        queued = [dict(p) for p in before["phases"]]
        queued[1]["on_fail"] = "stop"
        self.assertEqual(self.put({**before, "phases": queued}).status_code, 422)
        self.assertEqual(self.put({**before, "phases": before["phases"][:2]}).status_code, 422)     # no phase may go
        self.assertEqual(self.put({**before, "budget": {"hours": 99}}).status_code, 422)
        self.assertEqual(self.put({**before, "version": 2}).status_code, 422)


if __name__ == "__main__":
    unittest.main()

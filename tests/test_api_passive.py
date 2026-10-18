"""studio_api.routers.passive over FakeServices: every route answers. The passive methods emit no events, which is checked."""
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from tests.api_tap import EventTap

SESSION = {"id": "p1", "status": "running", "created": "2026-10-01T22:00:00", "stop": "8 hours", "trainer": "Running",
           "watchdog": "Running", "restarts": 2, "heard": 4.5, "error": ""}


class PassiveApiTest(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.services.passive_sessions, self.services.passive_signin = [dict(SESSION)], True
        self.app = create_app(self.services, open=True, hosts=ANY)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def test_sessions_and_signin(self):
        self.assertEqual(self.client.get("/passive").json(), {"sessions": [SESSION], "signin": True})

    def test_never_used_is_empty(self):
        client = TestClient(create_app(FakeServices(), open=True, hosts=ANY))
        self.assertEqual(client.get("/passive").json(), {"sessions": [], "signin": False})

    def test_a_stopped_session_has_no_heartbeat(self):
        self.services.passive_sessions = [{**SESSION, "status": "stopped", "trainer": "Stopped", "watchdog": "Stopped", "heard": None}]
        self.assertIsNone(self.client.get("/passive").json()["sessions"][0]["heard"])

    def test_stop_returns_the_status_afterwards(self):
        stopped = self.client.post("/passive/p1/stop")
        self.assertEqual((stopped.status_code, stopped.json()), (200, {"status": "stopped"}))
        self.assertEqual(self.client.get("/passive").json()["sessions"][0]["status"], "stopped")

    def test_stopping_an_unknown_session_is_a_404(self):
        self.assertEqual(self.client.post("/passive/nope/stop").status_code, 404)

    def test_keep_going_brings_back_a_dead_session(self):
        self.services.passive_sessions = [{**SESSION, "trainer": "Stopped", "watchdog": "Stopped", "heard": None}]
        kept = self.client.post("/passive/p1/keep-going")
        self.assertEqual((kept.status_code, kept.json()), (200, {"status": "running"}))
        self.assertEqual((self.services.passive_kept, self.client.get("/passive").json()["sessions"][0]["watchdog"]), (["p1"], "Running"))

    def test_keep_going_unknown_is_404_and_an_ended_session_is_409(self):
        self.assertEqual(self.client.post("/passive/nope/keep-going").status_code, 404)
        self.client.post("/passive/p1/stop")
        self.assertEqual(self.client.post("/passive/p1/keep-going").status_code, 409)

    def test_pause_and_resume_a_session(self):
        paused = self.client.post("/passive/p1/pause")
        self.assertEqual((paused.status_code, paused.json()), (200, {"status": "paused"}))
        resumed = self.client.post("/passive/p1/resume")
        self.assertEqual((resumed.status_code, resumed.json()), (200, {"status": "running"}))
        self.assertEqual(self.services.passive_commands, [("p1", "pause"), ("p1", "resume")])

    def test_pause_is_refused_for_an_ended_session_and_unknown_is_404(self):
        self.client.post("/passive/p1/stop")
        self.assertEqual(self.client.post("/passive/p1/pause").status_code, 409)
        self.services.passive_sessions[0]["status"] = "finished"
        self.assertEqual(self.client.post("/passive/p1/resume").status_code, 409)
        for route in ("pause", "resume"):
            self.assertEqual(self.client.post(f"/passive/nope/{route}").status_code, 404)

    def test_start_creates_a_session_from_the_form_and_lists_it(self):
        self.client.post("/passive/p1/stop")
        started = self.client.post("/passive", json={"duration": 2, "unit": "days", "patience_rounds": 0, "target_error": 1.5, "teacher": False,
                                                     "memory_fraction": 0.25, "keep_awake": False, "pause_on_battery": False, "sources": ["a"]})
        self.assertEqual((started.status_code, started.json()["status"]), (201, "queued"))
        merged, version = self.services.passive_started[0]
        self.assertEqual((merged["stop"]["duration"], merged["stop"]["unit"], merged["stop"]["patience_rounds"], merged["stop"]["target_error"]), (2, "days", 0, 1.5))
        self.assertEqual((merged["teacher"], merged["limits"]["memory_fraction"], merged["limits"]["keep_awake"], merged["data"]["sources"], version),
                         ({"polish": False, "lessons": False}, 0.25, False, ["a"], None))
        newest = self.client.get("/passive").json()["sessions"][0]
        self.assertEqual((newest["id"], newest["status"]), (started.json()["id"], "queued"))

    def test_start_with_the_defaults(self):
        self.client.post("/passive/p1/stop")
        self.assertEqual(self.client.post("/passive", json={}).status_code, 201)
        self.assertEqual(self.services.passive_started[0][0]["stop"]["duration"], 8)

    def test_a_bad_plan_is_a_400_with_a_sentence_and_a_second_active_session_a_409(self):
        self.client.post("/passive/p1/stop")
        bad = self.client.post("/passive", json={"duration": 0})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("time limit", bad.json()["detail"])
        self.assertEqual(self.client.post("/passive", json={"version": 99}).status_code, 400)
        self.assertEqual(self.client.post("/passive", json={}).status_code, 201)
        busy = self.client.post("/passive", json={})
        self.assertEqual(busy.status_code, 409)
        self.assertIn("already running", busy.json()["detail"])

    def test_resume_puts_a_stopped_or_failed_session_back_in_the_queue(self):
        self.client.post("/passive/p1/stop")
        resumed = self.client.post("/passive/p1/resume")
        self.assertEqual((resumed.status_code, resumed.json()), (200, {"status": "queued"}))
        self.assertEqual(self.client.get("/passive").json()["sessions"][0]["status"], "queued")
        self.assertEqual(self.client.post("/passive/p1/resume").status_code, 200)  # queued: resume is the ordinary command
        self.services.passive_sessions[0]["status"] = "failed"
        self.assertEqual(self.client.post("/passive/p1/resume").json(), {"status": "queued"})

    def test_resume_is_refused_while_another_session_is_active(self):
        self.services.passive_sessions = [{**SESSION, "id": "p0", "status": "failed"}, dict(SESSION)]
        self.assertEqual(self.client.post("/passive/p0/resume").status_code, 409)

    def test_removing_the_signin_watchdog(self):
        self.assertEqual(self.client.delete("/passive/signin").status_code, 204)
        self.assertFalse(self.client.get("/passive").json()["signin"])

    def test_none_of_it_emits_events(self):
        self.client.get("/passive")
        self.client.post("/passive/p1/keep-going")
        self.client.post("/passive/p1/pause")
        self.client.post("/passive/p1/resume")
        self.client.post("/passive/p1/stop")
        self.client.post("/passive/p1/resume")
        self.client.post("/passive/p1/stop")
        self.client.post("/passive", json={})
        self.client.delete("/passive/signin")
        self.assertEqual([event for event in self.tap.events if event[0] != "status"], [])  # initial shell snapshot is independent


if __name__ == "__main__":
    unittest.main()

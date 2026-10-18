"""studio_api.routers.passive over FakeServices: every route answers. The passive methods emit no events, which is checked."""
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from tests.api_tap import EventTap

SESSION = {"id": "p1", "status": "running", "created": "2026-10-01T22:00:00", "stop": "8 hours", "trainer": "Running",
           "watchdog": "Running", "restarts": 2, "heard": 4.5, "error": "", "checkpoints": 0, "version": None}


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
        for route in ("pause", "resume"):
            self.assertEqual(self.client.post(f"/passive/p1/{route}").status_code, 409)
            self.assertEqual(self.client.post(f"/passive/nope/{route}").status_code, 404)

    def test_save_makes_the_best_checkpoint_a_version_and_saving_again_gives_the_same_one(self):
        self.services.passive_sessions = [{**SESSION, "status": "finished", "checkpoints": 3}]
        first = self.client.post("/passive/p1/save")
        self.assertEqual((first.status_code, first.json()), (200, {"version": 1, "created": True}))
        again = self.client.post("/passive/p1/save")
        self.assertEqual(again.json(), {"version": 1, "created": False})
        self.assertEqual([(v.number, v.status) for v in self.services.versions()], [(1, "trained")])
        self.assertEqual(self.client.get("/passive").json()["sessions"][0]["version"], 1)
        self.tap.wait("versions")  # the versions list changed

    def test_save_is_refused_without_a_checkpoint_and_404_for_an_unknown_session(self):
        self.assertEqual(self.client.post("/passive/p1/save").status_code, 409)  # SESSION has saved no checkpoint
        self.assertEqual(self.client.post("/passive/nope/save").status_code, 404)

    def test_removing_the_signin_watchdog(self):
        self.assertEqual(self.client.delete("/passive/signin").status_code, 204)
        self.assertFalse(self.client.get("/passive").json()["signin"])

    def test_none_of_it_emits_events(self):
        self.client.get("/passive")
        self.client.post("/passive/p1/keep-going")
        self.client.post("/passive/p1/pause")
        self.client.post("/passive/p1/resume")
        self.client.post("/passive/p1/stop")
        self.client.delete("/passive/signin")
        self.assertEqual([event for event in self.tap.events if event[0] != "status"], [])  # initial shell snapshot is independent


if __name__ == "__main__":
    unittest.main()

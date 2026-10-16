"""studio_api.routers.passive over FakeServices: every route answers. The passive methods emit no events, which is checked."""
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_ia.services import FakeServices
from tests.api_tap import EventTap

SESSION = {"id": "p1", "status": "running", "created": "2026-10-01T22:00:00", "stop": "8 hours", "trainer": "Running",
           "watchdog": "Running", "restarts": 2, "heard": 4.5, "error": ""}


class PassiveApiTest(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.services.passive_sessions, self.services.passive_signin = [dict(SESSION)], True
        self.app = create_app(self.services, open=True)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def test_sessions_and_signin(self):
        self.assertEqual(self.client.get("/passive").json(), {"sessions": [SESSION], "signin": True})

    def test_never_used_is_empty(self):
        client = TestClient(create_app(FakeServices(), open=True))
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

    def test_removing_the_signin_watchdog(self):
        self.assertEqual(self.client.delete("/passive/signin").status_code, 204)
        self.assertFalse(self.client.get("/passive").json()["signin"])

    def test_none_of_it_emits_events(self):
        self.client.get("/passive")
        self.client.post("/passive/p1/stop")
        self.client.delete("/passive/signin")
        self.assertEqual(self.tap.events, [])


if __name__ == "__main__":
    unittest.main()

"""Issue #567: the busy registry (studio_ia.busy) over GET /busy and /events, and a slow handler not starving a fast request."""
import threading
import time
import unittest

from fastapi.testclient import TestClient

from studio_api import app as app_module
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.busy import Busy
from studio_ia.services import FakeServices
from tests.sse_harness import ApiCase, Capture


class Registry(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.busy = Busy(lambda kind, payload: self.events.append((kind, payload)))

    def test_start_progress_and_end_are_published_as_the_whole_list(self):
        with self.busy("Loading", total=2) as task:
            task.step()
            self.assertEqual(self.busy.items(), [{"id": task.id, "label": "Loading", "done": 1, "total": 2}])
            task.step()
        self.assertEqual({kind for kind, _ in self.events}, {"busy"})
        shown = [[(i["label"], i["done"], i["total"]) for i in payload["items"]] for _, payload in self.events]
        self.assertEqual(shown, [[("Loading", 0, 2)], [("Loading", 1, 2)], [("Loading", 2, 2)], []])
        self.assertEqual(self.busy.items(), [])

    def test_work_without_steps_is_indeterminate(self):
        with self.busy("Reading runs") as task:
            self.assertIsNone(task.view()["total"])
            self.assertIsNone(self.events[0][1]["items"][0]["total"])

    def test_the_task_ends_when_the_work_fails(self):
        with self.assertRaises(ValueError):
            with self.busy("Loading"):
                raise ValueError("no")
        self.assertEqual(self.events[-1][1], {"items": []})

    def test_several_at_once_are_listed_oldest_first_and_end_independently(self):
        with self.busy("A") as a:
            with self.busy("B"):
                self.assertEqual([i["label"] for i in self.busy.items()], ["A", "B"])
            self.assertEqual([i["id"] for i in self.busy.items()], [a.id])

    def test_a_burst_of_steps_is_throttled_but_the_last_one_is_not(self):
        with self.busy("Many", total=1000) as task:
            before = len(self.events)
            for _ in range(999):
                task.step()
            self.assertLess(len(self.events) - before, 20)
            task.step()
            self.assertEqual(self.events[-1][1]["items"][0]["done"], 1000)

    def test_label_and_total_can_be_updated_as_the_work_learns_them(self):
        with self.busy("Reading") as task:
            task.set(1, total=4, label="Counting")
            self.assertEqual(task.view(), {"id": task.id, "label": "Counting", "done": 1, "total": 4})


class BusyApi(ApiCase):
    prefix = "/busy"

    def test_get_busy_lists_what_is_running_and_is_empty_when_idle(self):
        self.assertEqual(self.call("GET", "/busy"), {"items": []})
        with self.services.busy("Loading the model for chat", total=2) as task:
            task.step()
            body = self.call("GET", "/busy")
            self.assertEqual(body, {"items": [{"id": task.id, "label": "Loading the model for chat", "done": 1, "total": 2}]})
        self.assertEqual(self.call("GET", "/busy"), {"items": []})
        self.assertAllRoutesHit()

    def test_events_carry_the_start_the_progress_and_the_end(self):
        with self.services.busy("Reading and counting your text", total=3) as task:
            self.events.wait(lambda events: any(k == "busy" and p["items"] for k, p in events), "the busy start")
            task.step()
            self.events.wait(lambda events: any(k == "busy" and p["items"] and p["items"][0]["done"] == 1 for k, p in events), "progress")
        self.events.wait(lambda events: [p for k, p in events if k == "busy"][-1] == {"items": []}, "the busy end")

    def test_a_client_that_connects_mid_work_is_told_what_is_running(self):
        with self.services.busy("Loading", total=None):
            late = Capture(self.app)
            self.addCleanup(late.close)
            late.wait(lambda events: any(k == "busy" and p["items"] for k, p in events), "the busy snapshot")
            self.assertEqual(late.of("busy")[0]["items"][0]["label"], "Loading")


class OffTheRequestPath(unittest.TestCase):
    def test_slow_handlers_do_not_block_a_fast_request(self):
        """Twice anyio's old 40-thread default of slow plain `def` routes in flight, and a fast one still answers at once."""
        app = create_app(FakeServices(), open=True, hosts=ANY)
        release = threading.Event()

        @app.get("/slow")
        def slow():
            release.wait(5)
            return {}

        @app.get("/fast")
        def fast():
            return {"ok": True}

        with TestClient(app) as client:
            pending = [threading.Thread(target=client.get, args=("/slow",)) for _ in range(app_module.THREADS - 8)]
            for thread in pending:
                thread.start()
            time.sleep(.3)
            started = time.monotonic()
            self.assertEqual(client.get("/fast").json(), {"ok": True})
            elapsed = time.monotonic() - started
            release.set()
            for thread in pending:
                thread.join(10)
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()

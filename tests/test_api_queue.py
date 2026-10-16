import dataclasses
import threading
import unittest

from studio_ia.services import FakeServices
from tests.sse_harness import TIMEOUT, ApiCase


def ids(state):
    return [j["id"] for j in state["up_next"]]


class QueueRouterTests(ApiCase):
    prefix = "/queue"

    def make_services(self):
        """A fake whose job never advances (its first step waits for the test to end), so POST /queue/start leaves the queue as it was."""
        release = threading.Event()
        self.addCleanup(release.set)
        return FakeServices(sleep=lambda seconds: release.wait(TIMEOUT))

    def add(self, label):
        return self.call("POST", "/queue/jobs", 201, json={"kind": "pretrain", "label": label, "params": {"version": 1}})

    def test_every_route_answers(self):
        state = self.call("GET", "/queue")
        self.assertIsNone(state["job"])
        self.assertEqual(state["up_next"], [])
        self.assertFalse(state["paused"])
        self.assertIsNone(state["gate"]["reason"])

        a, b, c = self.add("a"), self.add("b"), self.add("c")
        self.assertEqual((a["label"], a["status"], a["params"]), ("a", "waiting", {"version": 1}))
        listed = self.call("GET", "/queue/jobs")
        self.assertEqual([j["id"] for j in listed], [a["id"], b["id"], c["id"]])

        state = self.call("POST", "/queue/jobs/{job_id}/move_up", path={"job_id": c["id"]})
        self.assertEqual(ids(state), [a["id"], c["id"], b["id"]])
        state = self.call("POST", "/queue/jobs/{job_id}/move", path={"job_id": b["id"]}, json={"index": 0})
        self.assertEqual(ids(state), [b["id"], a["id"], c["id"]])

        state = self.call("POST", "/queue/jobs/{job_id}/hold", path={"job_id": a["id"]})
        self.assertEqual({j["id"]: j["status"] for j in state["up_next"]}[a["id"]], "held")
        state = self.call("POST", "/queue/jobs/{job_id}/hold", path={"job_id": a["id"]}, json={"held": False})
        self.assertEqual({j["id"]: j["status"] for j in state["up_next"]}[a["id"]], "waiting")

        self.assertTrue(self.call("GET", "/queue/outlook")["outlook"])
        self.assertTrue(self.call("GET", "/queue/jobs/{job_id}/outlook", path={"job_id": a["id"]})["outlook"])

        self.assertTrue(self.call("POST", "/queue/pause")["paused"])
        self.assertFalse(self.call("POST", "/queue/resume")["paused"])
        self.assertEqual(self.call("POST", "/queue/checkpoint_now"), {"started": False})  # nothing is running
        self.assertEqual(self.call("POST", "/queue/stop_after_cooldown")["stopping"], False)
        started = self.call("POST", "/queue/start")  # the fake runs the first job: it leaves Up next and becomes the running job
        self.assertEqual((started["job"]["id"], ids(started)), (b["id"], [a["id"], c["id"]]))

        plan = self.call("GET", "/queue/plan")
        self.assertEqual(plan["hours"], "any")
        saved = self.call("PUT", "/queue/plan", json={**plan, "hours": "nights", "memory": "half"})
        self.assertEqual((saved["hours"], saved["memory"]), ("nights", "half"))
        self.assertEqual(self.call("GET", "/queue/plan"), saved)

        step = self.call("GET", "/queue/next_step")
        self.assertEqual(set(step), {"route", "title", "line"})

        state = self.call("DELETE", "/queue/jobs/{job_id}", path={"job_id": b["id"]})
        self.assertEqual(ids(state), [a["id"], c["id"]])
        self.assertAllRoutesHit()

    def test_a_running_job_and_its_gate_come_through(self):
        from datetime import datetime

        from studio_ia.services import Job, QueueState
        from studio_queue import Gate

        running = Job("r1", "pretrain", "the big one", status="running", eta_s=90.0, created_at=1.0)
        self.services.set_queue(QueueState(job=running, step=40, total=100, phase="steady", lr=3e-4, eta_s=60.0,
                                           blocked="outside hours", gate=Gate("outside hours", datetime(2030, 1, 1, 22), True)))
        state = self.call("GET", "/queue")
        self.assertEqual((state["job"]["id"], state["job"]["status"], state["step"], state["phase"]), ("r1", "running", 40, "steady"))
        self.assertEqual(state["gate"], {"reason": "outside hours", "resumes_at": "2030-01-01T22:00:00", "supported": True})
        self.assertEqual([j["id"] for j in self.call("GET", "/queue/jobs")], ["r1"])
        self.assertEqual(self.call("GET", "/queue/jobs/{job_id}/outlook", path={"job_id": "r1"})["outlook"], "Running now")
        self.assertEqual(self.call("DELETE", "/queue/jobs/{job_id}", path={"job_id": "r1"}), self.call("GET", "/queue"))

    def test_bad_requests_say_what_is_wrong(self):
        self.call("POST", "/queue/jobs", 422, json={"label": "no kind"})
        self.call("PUT", "/queue/plan", 422, json={"hours": "whenever"})
        self.call("PUT", "/queue/plan", 422, json={"start_hour": 25})
        self.call("POST", "/queue/jobs/{job_id}/move", 422, path={"job_id": "x"}, json={"index": -1})
        for method, path in (("POST", "/queue/jobs/nope/move_up"), ("POST", "/queue/jobs/nope/hold"), ("DELETE", "/queue/jobs/nope")):
            self.assertEqual(self.client.request(method, path, json={"index": 0} if "move" in path else None).status_code, 404, path)
        self.assertEqual(self.client.post("/queue/jobs/nope/move", json={"index": 0}).status_code, 404)
        self.assertEqual(self.call("GET", "/queue/jobs/{job_id}/outlook", path={"job_id": "nope"})["outlook"], "")

    def test_a_runner_that_refuses_a_kind_is_a_400(self):
        def refuse(job):
            raise ValueError(f"Unknown job kind: {job.kind}")
        self.services.enqueue = refuse
        response = self.client.post("/queue/jobs", json={"kind": "nonsense"})
        self.assertEqual((response.status_code, response.json()["detail"]), (400, "Unknown job kind: nonsense"))

    def test_every_event_a_queue_route_emits_reaches_events(self):
        a, b = self.add("a"), self.add("b")
        self.call("POST", "/queue/jobs/{job_id}/move_up", path={"job_id": b["id"]})
        self.call("POST", "/queue/jobs/{job_id}/move", path={"job_id": a["id"]}, json={"index": 1})
        self.call("POST", "/queue/jobs/{job_id}/hold", path={"job_id": a["id"]})
        self.call("POST", "/queue/pause")
        self.call("POST", "/queue/resume")
        self.call("POST", "/queue/checkpoint_now")
        self.call("PUT", "/queue/plan", json={"hours": "set", "start_hour": 21, "end_hour": 6})
        self.call("DELETE", "/queue/jobs/{job_id}", path={"job_id": a["id"]})
        self.assertDelivered()
        queue_events = self.events.of("queue")
        self.assertEqual(len(queue_events), self.emitted.count("queue"))
        self.assertEqual([j["id"] for j in queue_events[0]["up_next"]], [a["id"]])  # the payload is the QueueState, as JSON
        self.assertEqual(queue_events[-1]["plan"]["hours"], "set")
        self.assertEqual([j["id"] for j in queue_events[-1]["up_next"]], [b["id"]])


class ModelsFollowTheDataclasses(unittest.TestCase):
    def test_no_field_is_dropped(self):
        import studio_queue
        from studio_api.routers import queue as q
        from studio_ia import services

        def names(cls):
            return {f.name for f in dataclasses.fields(cls)}

        self.assertEqual(set(q.PlanModel.model_fields), names(studio_queue.Plan))
        self.assertEqual(set(q.GateModel.model_fields), names(studio_queue.Gate))
        self.assertEqual(set(q.QueueStateModel.model_fields), names(services.QueueState))
        self.assertEqual(set(q.QueueStateModel.model_fields), names(studio_queue.QueueState))
        self.assertEqual(set(q.JobModel.model_fields), names(studio_queue.Job))
        self.assertLessEqual(names(services.Job), set(q.JobModel.model_fields))
        self.assertEqual(q.PlanModel().model_dump(), dataclasses.asdict(studio_queue.Plan()))


if __name__ == "__main__":
    unittest.main()

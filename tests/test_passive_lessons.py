"""Reviewing a passive session's lessons: dropping one keeps it out of the next round's training, and the routes list and change them."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices, WorkspaceServices
from studio_passive import PassiveStore, merge_plan
from tests.test_passive import Rig

LESSON = {"messages": [{"role": "user", "content": "say yes"}], "chosen": "yes, indeed", "rejected": "y"}


class Store(unittest.TestCase):
    def test_a_dropped_lesson_is_left_out_of_the_next_rounds_training_and_a_kept_one_comes_back(self):
        services = SimpleNamespace(root=None)
        seen = {}

        def teacher(n, checkpoint):
            if n == 2:  # between round 2 and round 3: the person drops the lesson round 1's teacher gave
                lesson = rig.store.lessons(rig.id)[0]
                seen["dropped"] = WorkspaceServices.set_passive_lesson(services, rig.id, lesson["id"], "dropped")
            if n == 3:
                lesson = rig.store.lessons(rig.id)[0]
                seen["kept"] = WorkspaceServices.set_passive_lesson(services, rig.id, lesson["id"], "pending")
            return {"checkpoint": checkpoint, "lessons": [LESSON] if n == 1 else []}
        rig = Rig(self, [3.0, 2.0, 1.0, 0.8, 0.4], stop={"target_error": 0.5}, plateau={"checks": 9}, teacher=teacher)
        services.root = rig.root
        rig.set_teacher()
        rig.run()
        taught = [sum(1 for e in call["examples"] if e.get("messages")) for call in rig.calls]
        self.assertEqual(taught, [0, 1, 0, 1, 1])  # round 2 sees it, round 3 (after the drop) does not, rounds 4 and 5 (after the keep) do
        self.assertEqual((seen["dropped"]["status"], seen["kept"]["status"]), ("dropped", "pending"))

    def test_set_lesson_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PassiveStore(tmp)
            sid = store.create(merge_plan({}))
            store.set_lessons(sid, [{"id": "a", "status": "pending"}, {"id": "b", "status": "left_out"}])
            self.assertEqual(store.set_lesson_status(sid, "a", "dropped")["status"], "dropped")
            self.assertEqual(store.set_lesson_status(sid, "b", "pending")["status"], "pending")  # keeping brings back one a step back left out
            self.assertEqual([r["status"] for r in store.lessons(sid)], ["dropped", "pending"])
            self.assertIsNone(store.set_lesson_status(sid, "nope", "dropped"))
            for bad in ("left_out", "deleted", ""):
                with self.subTest(bad), self.assertRaises(ValueError):
                    store.set_lesson_status(sid, "a", bad)


class Services(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.services = SimpleNamespace(root=Path(tmp.name))
        self.store = PassiveStore(self.services.root)
        self.id = self.store.create(merge_plan({}))
        self.store.set_lessons(self.id, [
            {"id": "l1", "round": 1, "checkpoint": 1, "prompt": "say yes", "reply": "y", "answer": "yes, indeed", "messages": [{"role": "user", "content": "say yes"}],
             "status": "pending", "at": "2026-10-01T22:00:00"},
            {"id": "l2", "round": 2, "checkpoint": 2, "prompt": "say no", "reply": "n", "answer": "no, thanks", "messages": [], "status": "left_out", "at": "2026-10-01T23:00:00"}])

    def test_lessons_are_listed_without_the_raw_messages(self):
        lessons = WorkspaceServices.passive_lessons(self.services, self.id)
        self.assertEqual([(r["id"], r["status"], r["prompt"], r["answer"]) for r in lessons], [("l1", "pending", "say yes", "yes, indeed"), ("l2", "left_out", "say no", "no, thanks")])
        self.assertNotIn("messages", lessons[0])

    def test_unknown_session_or_lesson_and_a_workspace_never_used(self):
        self.assertIsNone(WorkspaceServices.passive_lessons(self.services, "nope"))
        self.assertIsNone(WorkspaceServices.set_passive_lesson(self.services, "nope", "l1", "dropped"))
        self.assertIsNone(WorkspaceServices.set_passive_lesson(self.services, self.id, "nope", "dropped"))
        with tempfile.TemporaryDirectory() as empty:
            fresh = SimpleNamespace(root=Path(empty))
            self.assertIsNone(WorkspaceServices.passive_lessons(fresh, "x"))
            self.assertIsNone(WorkspaceServices.set_passive_lesson(fresh, "x", "l", "dropped"))
            self.assertFalse((Path(empty) / "data" / "passive.sqlite").exists())

    def test_dropping_writes_the_file_the_trainer_reads(self):
        WorkspaceServices.set_passive_lesson(self.services, self.id, "l1", "dropped")
        self.assertEqual([r["status"] for r in self.store.lessons(self.id)], ["dropped", "left_out"])


class Route(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.services.passive_sessions = [{"id": "p1", "status": "running", "created": "2026-10-01T22:00:00", "stop": "8 hours", "trainer": "Running",
                                           "watchdog": "Running", "restarts": 0, "heard": 1.0, "error": ""}]
        self.services.passive_lesson_rows = {"p1": [{"id": "l1", "round": 1, "checkpoint": 1, "prompt": "say yes", "reply": "y", "answer": "yes, indeed", "status": "pending", "at": None}]}
        self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))

    def test_list_and_change(self):
        self.assertEqual(self.client.get("/passive/p1/lessons").json()[0]["answer"], "yes, indeed")
        dropped = self.client.patch("/passive/p1/lessons/l1", json={"status": "dropped"})
        self.assertEqual((dropped.status_code, dropped.json()["status"]), (200, "dropped"))
        self.assertEqual(self.client.get("/passive/p1/lessons").json()[0]["status"], "dropped")
        self.assertEqual(self.client.patch("/passive/p1/lessons/l1", json={"status": "pending"}).json()["status"], "pending")

    def test_unknown_session_or_lesson_is_404_and_a_bad_status_is_422(self):
        self.assertEqual(self.client.get("/passive/nope/lessons").status_code, 404)
        self.assertEqual(self.client.patch("/passive/nope/lessons/l1", json={"status": "dropped"}).status_code, 404)
        self.assertEqual(self.client.patch("/passive/p1/lessons/nope", json={"status": "dropped"}).status_code, 404)
        self.assertEqual(self.client.patch("/passive/p1/lessons/l1", json={"status": "left_out"}).status_code, 422)
        self.assertEqual(self.client.patch("/passive/p1/lessons/l1", json={}).status_code, 422)


if __name__ == "__main__":
    unittest.main()

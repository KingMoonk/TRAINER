"""The passive session detail: progress and journal as a test trainer wrote them (Rig runs the real loop with fakes), and the route."""
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices, WorkspaceServices
from studio_passive import PassiveStore, merge_plan, progress_summary
from tests.test_passive import Rig


def services_for(root):
    return SimpleNamespace(root=root)


class Detail(unittest.TestCase):
    def setUp(self):
        self.rig = Rig(self, [3.0, 2.0, 2.5, 1.0], stop={"target_error": 1.5}, plateau={"checks": 9}, worse={"limit": 5, "pivots": ["halve_lr"]})
        self.rig.run()
        self.services = services_for(self.rig.root)

    def detail(self, session_id=None, **kw):
        return WorkspaceServices.passive_detail(self.services, session_id or self.rig.id, **kw)

    def test_state_and_journal_of_a_session_a_trainer_wrote(self):
        found = self.detail()
        self.assertEqual(found["session"]["id"], self.rig.id)
        self.assertEqual((found["session"]["status"], found["plan"]["stop"]), ("finished", "Stops after 1 day, or after test error reaches 1.5."))
        progress = found["progress"]
        self.assertEqual((progress["phase"], progress["round"], progress["best"], progress["best_error"], progress["stop_reason"]), ("idle", 4, 4, 1.0, "target"))
        self.assertEqual([(c["n"], c["error"], c["best"]) for c in progress["checkpoints"]], [(1, 3.0, False), (2, 2.0, False), (3, 2.5, False), (4, 1.0, True)])
        self.assertEqual((progress["limit_seconds"], progress["current_error"]), (86400.0, 1.0))
        journal = found["journal"]
        types = [e["type"] for e in journal["events"]]
        self.assertEqual((types[0], types[-1], journal["total"], journal["first"]), ("start", "stop", len(types), 0))
        self.assertEqual([e["index"] for e in journal["events"]], list(range(len(types))))
        self.assertEqual(types.count("checkpoint"), 4)

    def test_the_journal_is_paged_from_the_newest_back(self):
        everything = self.detail()["journal"]["events"]
        newest = self.detail(limit=3)["journal"]
        self.assertEqual((newest["events"], newest["first"], newest["total"]), (everything[-3:], len(everything) - 3, len(everything)))
        older = self.detail(limit=3, before=newest["first"])["journal"]
        self.assertEqual((older["events"], older["first"]), (everything[-6:-3], len(everything) - 6))
        first = self.detail(limit=100, before=2)["journal"]
        self.assertEqual((first["events"], first["first"]), (everything[:2], 0))
        self.assertEqual(self.detail(limit=5, before=0)["journal"]["events"], [])

    def test_rollback_and_restart_reasons_are_in_the_journal(self):
        self.rig.store.append(self.rig.id, {"type": "rollback", "from": 3, "to": 2, "pivot": "halve_lr", "discarded": [3], "error": 2.5})
        self.rig.store.append(self.rig.id, {"type": "restart", "reason": "silent", "pid": 7})
        events = self.detail(limit=2)["journal"]["events"]
        self.assertEqual([(e["type"], e.get("pivot"), e.get("reason")) for e in events], [("rollback", "halve_lr", None), ("restart", None, "silent")])

    def test_unknown_session_and_never_used_workspace_are_none(self):
        self.assertIsNone(self.detail("nope"))
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(WorkspaceServices.passive_detail(services_for(Path(empty)), "x"))
            self.assertFalse((Path(empty) / "data" / "passive.sqlite").exists())

    def test_a_session_that_has_not_started_has_an_empty_progress(self):
        store = PassiveStore(self.rig.root)
        sid = store.create(merge_plan({}))  # the first one has finished, so this is allowed
        progress = self.detail(sid)["progress"]
        self.assertEqual((progress["phase"], progress["round"], progress["best"], progress["best_error"], progress["checkpoints"]), ("", 0, None, None, []))
        self.assertEqual(self.detail(sid)["journal"], {"events": [], "total": 0, "first": 0})

    def test_progress_summary_marks_the_polish_and_the_discard(self):
        state = {"best": 1, "current": 2, "checkpoints": [{"n": 1, "error": 1.0, "teacher": {"polished": True, "win_rate": 0.7, "lessons": 4}},
                                                          {"n": 2, "error": 2.0, "discarded": True, "pivot": "halve_lr", "teacher": None}]}
        first, second = progress_summary(state, merge_plan({}))["checkpoints"]
        self.assertEqual((first["polished"], first["win_rate"], first["lessons"], first["best"]), (True, 0.7, 4, True))
        self.assertEqual((second["discarded"], second["pivot"], second["polished"], second["best"]), (True, "halve_lr", False, False))


class Route(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.services.passive_sessions = [{"id": "p1", "status": "running", "created": "2026-10-01T22:00:00", "stop": "8 hours", "trainer": "Running",
                                           "watchdog": "Running", "restarts": 0, "heard": 1.0, "error": ""}]
        self.services.passive_details = {"p1": {"state": {"round": 2, "phase": "train", "best": 1, "current": 1, "checkpoints": [{"n": 1, "error": 2.5}]},
                                                "journal": [{"type": "start", "at": "t0"}, {"type": "round", "round": 1}, {"type": "checkpoint", "n": 1, "error": 2.5}]}}
        self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))

    def test_detail_answers_with_progress_and_a_page_of_events(self):
        body = self.client.get("/passive/p1", params={"limit": 2}).json()
        self.assertEqual((body["session"]["id"], body["progress"]["phase"], body["progress"]["best_error"]), ("p1", "train", 2.5))
        self.assertEqual(body["progress"]["checkpoints"][0]["error"], 2.5)
        self.assertEqual(([e["type"] for e in body["journal"]["events"]], body["journal"]["first"], body["journal"]["total"]), (["round", "checkpoint"], 1, 3))
        older = self.client.get("/passive/p1", params={"limit": 2, "before": 1}).json()["journal"]
        self.assertEqual(([e["type"] for e in older["events"]], older["first"]), (["start"], 0))
        self.assertEqual(older["events"][0]["at"], "t0")

    def test_unknown_is_404_and_a_bad_page_is_422(self):
        self.assertEqual(self.client.get("/passive/nope").status_code, 404)
        self.assertEqual(self.client.get("/passive/p1", params={"limit": 0}).status_code, 422)


if __name__ == "__main__":
    unittest.main()

"""studio_api.routers.quicktest over FakeServices: stage a change, queue the test, read the result, save or discard the idea."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import studio_quicktest
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from studio_spec import default_spec, validate_spec


def changed():
    spec = copy.deepcopy(validate_spec(default_spec()))
    spec["dims"]["n_layer"] += 1
    return validate_spec(spec)


class QuickTestApiTest(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))

    def stage(self):
        self.assertEqual(self.client.put("/quicktest/draft", json={"spec": changed()}).status_code, 204)

    def test_nothing_staged(self):
        body = self.client.get("/quicktest").json()
        self.assertIsNone(body["draft"])
        self.assertIsNone(body["last"])
        self.assertEqual((body["current"]["title"], body["next_number"]), ("Current version", 1))
        self.assertIn("look-back heads", body["current"]["description"])

    def test_a_staged_change_is_described_with_its_cost(self):
        self.stage()
        draft = self.client.get("/quicktest").json()["draft"]
        self.assertTrue(draft["changes"])
        self.assertIn("blocks", draft["changes"][0])
        self.assertIn("bigger", draft["cost"])

    def test_a_spec_that_does_not_validate_is_rejected(self):
        bad = copy.deepcopy(changed())
        bad["dims"]["n_layer"] = -1
        self.assertEqual(self.client.put("/quicktest/draft", json={"spec": bad}).status_code, 400)
        self.assertIsNone(self.services.draft_spec())

    def test_run_queues_a_quicktest_job_from_the_staged_change(self):
        self.stage()
        started = self.client.post("/quicktest/run", json={"dataset": "problems", "minutes": 1, "shrink": True, "guess": "Lower"})
        self.assertEqual(started.status_code, 202)
        job = self.services.queue().up_next[-1]
        self.assertEqual((started.json()["job_id"], job.kind, job.label), (job.id, "quicktest", "Quick test · 1 min each"))
        self.assertEqual({k: job.params[k] for k in ("dataset", "minutes", "shrink", "guess")},
                         {"dataset": "problems", "minutes": 1, "shrink": True, "guess": "Lower"})
        self.assertEqual(job.params["draft_spec"], self.services.draft_spec())
        self.assertTrue(started.json()["outlook"])
        self.assertEqual(started.json()["started_at"], job.created_at)

    def test_run_without_a_staged_change_is_a_400(self):
        r = self.client.post("/quicktest/run", json={"guess": "Same"})
        self.assertEqual((r.status_code, r.json()["detail"]), (400, "Stage a change in the designer first."))
        self.assertEqual(self.services.queue().up_next, [])

    def test_run_needs_no_guess_but_a_known_choice(self):
        self.stage()
        self.assertEqual(self.client.post("/quicktest/run", json={}).status_code, 202)
        for body in ({"guess": "Maybe"}, {"guess": "Same", "minutes": 2}, {"guess": "Same", "dataset": "x"}):
            self.assertEqual(self.client.post("/quicktest/run", json=body).status_code, 422, body)

    def test_a_slice_of_your_data_needs_text(self):
        self.stage()
        self.services._rows()["text"].clear()
        r = self.client.post("/quicktest/run", json={"dataset": "slice", "guess": "Same"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("no plain text", r.json()["detail"])

    def test_save_makes_the_next_version_and_clears_the_stage(self):
        self.stage()
        saved = self.client.post("/quicktest/save")
        self.assertEqual((saved.status_code, saved.json()["number"], saved.json()["status"]), (201, 1, "saved"))
        self.assertIn("blocks", saved.json()["change"])
        self.assertIsNone(self.services.draft_spec())
        self.assertEqual(self.client.get("/quicktest").json()["next_number"], 2)
        self.assertEqual(self.client.post("/quicktest/save").status_code, 409)

    def test_discard_clears_the_stage(self):
        self.stage()
        self.assertEqual(self.client.delete("/quicktest/draft").status_code, 204)
        self.assertIsNone(self.client.get("/quicktest").json()["draft"])

    def test_last_result_is_served_only_if_newer_than_since(self):
        result = {"minutes": 3, "shrink": False, "dataset": "stories", "verdict": "lower", "delta": -.2, "threshold": .05,
                  "references": {"uniform": 5., "unigram": 3.}, "learned_context": True,
                  "checks": [{"id": "finite", "ok": True, "detail": "ok"}, {"id": "error_falls", "ok": None, "detail": "not run"}],
                  "current": {"steps": 10, "bpc": 3., "curve": [[1, 4.], [10, 3.]], "sample": "a"},
                  "draft": {"steps": 9, "bpc": 2.8, "curve": [[1, 4.], [9, 2.8]], "sample": "b"}, "finished_at": 1000.}
        with tempfile.TemporaryDirectory() as root:
            self.services.root = root
            path = Path(root).joinpath(*studio_quicktest.LAST)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
            last = self.client.get("/quicktest").json()["last"]
            self.assertEqual((last["verdict"], last["draft"]["curve"][1], last["checks"][1]["ok"]), ("lower", [9, 2.8], None))
            self.assertIsNone(self.client.get("/quicktest", params={"since": 2000}).json()["last"])


if __name__ == "__main__":
    unittest.main()

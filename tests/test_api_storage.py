"""studio_api.routers.storage over FakeServices: every route answers and its events reach /events."""
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from tests.api_tap import EventTap


class SizedServices(FakeServices):
    """FakeServices with something to clean; clean_storage announces the queue as WorkspaceServices does."""

    def storage(self, days=None, data=False):
        return (2, 3000) if days is None else (1, 1000)

    def clean_storage(self, days=None, data=False):
        super().clean_storage(days, data)
        self.emit("queue", {})
        return self.storage(days, data)


class StorageApiTest(unittest.TestCase):
    def setUp(self):
        self.services = SizedServices()
        self.app = create_app(self.services, open=True, hosts=ANY)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def test_storage_says_what_a_clean_would_delete(self):
        self.assertEqual(self.client.get("/storage").json(), {"items": 2, "bytes": 3000})
        self.assertEqual(self.client.get("/storage", params={"days": 30, "data": "false"}).json(), {"items": 1, "bytes": 1000})

    def test_clean_passes_its_arguments_and_announces_the_queue(self):
        done = self.client.post("/storage/clean", json={"days": 7})
        self.assertEqual((done.status_code, done.json()), (200, {"items": 1, "bytes": 1000}))
        self.assertEqual(self.services.cleaned, (7, False))
        self.client.post("/storage/clean", json={"data": True})
        self.assertEqual(self.services.cleaned, (None, True))
        self.tap.wait("queue", 2)

    def test_free_space(self):
        self.assertEqual(self.client.get("/storage/free").json(), {"bytes": 120 * 10 ** 9})

    def test_the_fake_workspace_has_nothing_to_clean(self):
        client = TestClient(create_app(FakeServices(), open=True, hosts=ANY))
        self.assertEqual(client.get("/storage").json(), {"items": 0, "bytes": 0})
        self.assertEqual(client.post("/storage/clean", json={}).json(), {"items": 0, "bytes": 0})

    def test_a_running_job_blocks_the_clean(self):
        def busy(days=None, data=False):
            raise RuntimeError("Stop the running job first.")
        self.services.clean_storage = busy
        refused = self.client.post("/storage/clean", json={})
        self.assertEqual((refused.status_code, refused.json()["detail"]), (409, "Stop the running job first."))

    def test_exports_list_trained_versions_with_the_reason_gguf_is_unavailable(self):
        import studio_spec
        word = studio_spec.default_spec()
        word["tokenizer"] = {"id": "word"}
        for spec in (studio_spec.default_spec(), word, studio_spec.default_spec()):
            self.services.save_version({"architecture": spec}, "")
        for v in self.services._versions[:2]:
            v.best_checkpoint = "best.npz"
        rows = self.client.get("/storage/exports").json()
        self.assertEqual([r["number"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["gguf_problem"], "")
        self.assertTrue(rows[1]["gguf_problem"].startswith("The"))

    def test_exports_are_empty_before_anything_trains(self):
        self.assertEqual(self.client.get("/storage/exports").json(), [])

    def test_negative_days_are_refused(self):
        self.assertEqual(self.client.get("/storage", params={"days": -1}).status_code, 422)
        self.assertEqual(self.client.post("/storage/clean", json={"days": -1}).status_code, 422)


if __name__ == "__main__":
    unittest.main()

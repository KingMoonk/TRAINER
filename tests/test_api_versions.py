"""studio_api.routers.versions over FakeServices: every route answers and its events reach /events."""
import io
import time
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from studio_versions import NoCheckpoint
from tests.api_tap import EventTap


class ExportingServices(FakeServices):
    """FakeServices that writes what export_version promises, so the download has something to send."""

    def export_version(self, version, kind, path):
        super().export_version(version, kind, path)
        path = Path(path)
        if kind == "hf":
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_bytes(b"weights")
        else:
            path.write_bytes(b"GGUF")
        return {"path": str(path), "files": [], "note": ""}


def raising(error):
    def fail(*args, **kwargs):
        raise error
    return fail


class VersionsApiTest(unittest.TestCase):
    def setUp(self):
        self.services = ExportingServices()
        self.app = create_app(self.services, open=True, hosts=ANY)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def save(self, change="more heads"):
        return self.client.post("/versions", json={"spec": {"dims": {"n_ctx": 8}}, "change": change, "prediction": {"error": "lower"}})

    def job(self, response):
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        for _ in range(100):
            job = self.client.get(f"/jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                return job
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_nothing_saved_yet(self):
        self.assertEqual(self.client.get("/versions").json(), [])
        self.assertIsNone(self.client.get("/versions/current").json())

    def test_save_announces_the_version_on_events(self):
        made = self.save()
        self.assertEqual(made.status_code, 201)
        self.assertEqual(made.json()["number"], 1)
        self.assertEqual(made.json()["prediction"], {"error": "lower"})
        self.assertEqual([v["change"] for v in self.client.get("/versions").json()], ["more heads"])
        self.assertEqual(self.client.get("/versions/current").json()["number"], 1)
        self.assertEqual(self.tap.wait("versions")[0]["number"], 1)

    def test_a_design_the_services_refuse_is_a_400(self):
        self.services.save_version = raising(ValueError("That design has no layers."))
        refused = self.save()
        self.assertEqual((refused.status_code, refused.json()["detail"]), (400, "That design has no layers."))

    def test_slow_reads_are_jobs_whose_progress_reaches_events(self):
        self.save()
        for route, kind, result in (("/versions/best", "versions.best", None), ("/versions/1/checkpoint-step", "versions.checkpoint_step", None),
                                    ("/versions/1/run-facts", "versions.run_facts", {}), ("/versions/1/run-speed", "versions.run_speed", None)):
            with self.subTest(route):
                job = self.job(self.client.post(route))
                self.assertEqual((job["kind"], job["status"], job["result"]), (kind, "done", result))
        self.tap.wait_for(lambda: len([p for p in self.tap.of("job") if p["status"] == "done"]) == 4, "four finished jobs")

    def test_the_best_version_is_the_job_result(self):
        self.save()
        self.services._versions[0].best_checkpoint = "best.npz"
        job = self.job(self.client.post("/versions/best"))
        self.assertEqual(job["result"]["number"], 1)

    def test_a_failing_read_is_a_failed_job(self):
        self.save()
        self.services.run_speed = raising(ZeroDivisionError("nope"))
        job = self.job(self.client.post("/versions/1/run-speed"))
        self.assertEqual(job["status"], "failed")
        self.assertIn("ZeroDivisionError", job["error"])

    def test_per_version_routes_404_for_an_unknown_version(self):
        for route in ("/versions/9/checkpoint-step", "/versions/9/run-facts", "/versions/9/run-speed"):
            self.assertEqual(self.client.post(route).status_code, 404, route)
        self.assertEqual(self.client.get("/versions/9/export", params={"kind": "hf"}).status_code, 404)

    def test_export_hf_downloads_a_zip_of_the_folder(self):
        self.save()
        response = self.client.get("/versions/1/export", params={"kind": "hf"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertIn(".zip", response.headers["content-disposition"])
        names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
        self.assertEqual(sorted(names), ["config.json", "model.safetensors"])
        self.assertEqual(self.services.exports[0][:2], (1, "hf"))

    def test_export_gguf_downloads_the_file(self):
        self.save()
        response = self.client.get("/versions/1/export", params={"kind": "gguf"})
        self.assertEqual((response.status_code, response.content), (200, b"GGUF"))
        self.assertIn(".gguf", response.headers["content-disposition"])

    def test_export_cleans_up_after_itself(self):
        self.save()
        self.client.get("/versions/1/export", params={"kind": "gguf"})
        self.assertFalse(Path(self.services.exports[0][2]).exists())

    def test_export_of_an_unknown_kind_is_refused(self):
        self.save()
        self.assertEqual(self.client.get("/versions/1/export", params={"kind": "onnx"}).status_code, 422)

    def test_export_that_wrote_nothing_is_a_404(self):
        self.save()
        self.services.export_version = lambda version, kind, path: {}
        self.assertEqual(self.client.get("/versions/1/export", params={"kind": "gguf"}).status_code, 404)

    def test_continuing_a_version_queues_a_job_that_names_what_it_continues(self):
        self.save()
        for body, label in (({"kind": "same", "steps": 50}, "Version 1 continued from step 100"),
                            ({"kind": "new_data", "steps": 50, "dataset_ids": ["stories"]}, "Version 1 continued from step 100 on new data"),
                            ({"kind": "grow", "steps": 50, "width": 64, "layers": 4}, "Version 2 grown from version 1 (step 100)")):
            with self.subTest(body["kind"]):
                job = self.job(self.client.post("/versions/1/continue", json=body))
                self.assertEqual((job["kind"], job["status"]), ("versions.continue", "done"))
                self.assertEqual((job["result"]["label"], job["result"]["from_step"], job["result"]["kind"]), (label, 100, body["kind"]))
        self.assertEqual([j.label for j in self.services.continued][:1], ["Version 1 continued from step 100"])

    def test_continuing_needs_what_it_continues_on(self):
        self.save()
        for body in ({"kind": "new_data", "steps": 5}, {"kind": "grow", "steps": 5}, {"kind": "grow", "steps": 5, "width": 64}):
            self.assertEqual(self.client.post("/versions/1/continue", json=body).status_code, 422, body)
        for body in ({"kind": "same", "steps": 0}, {"kind": "sideways", "steps": 5}, {"steps": 5}):
            self.assertEqual(self.client.post("/versions/1/continue", json=body).status_code, 422, body)
        self.assertEqual(self.client.post("/versions/9/continue", json={"kind": "same", "steps": 5}).status_code, 404)

    def test_a_run_that_cannot_be_continued_fails_its_job_with_the_sentence(self):
        self.save()
        self.services.continue_run = raising(ValueError("This run has no full checkpoint (last.npz) to continue from."))
        job = self.job(self.client.post("/versions/1/continue", json={"kind": "same", "steps": 5}))
        self.assertEqual(job["status"], "failed")
        self.assertIn("no full checkpoint", job["error"])

    def test_export_without_a_checkpoint_is_a_409(self):
        self.save()
        self.services.export_version = raising(NoCheckpoint(1))
        self.assertEqual(self.client.get("/versions/1/export", params={"kind": "hf"}).status_code, 409)


if __name__ == "__main__":
    unittest.main()

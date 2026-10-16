"""studio_api.routers.post_training over FakeServices: every route answers and the events its methods emit reach /events."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_import
from studio_api.app import create_app
from studio_ia.services import FakeServices
from tests.api_tap import EventTap
from tests.api_work import finished, small_spec, trained


class DataServices(FakeServices):
    """FakeServices with something stored for the rounds to learn from."""

    def post_training(self, version):
        return {"rounds": {1: {"status": "done", "note": "live"}, 2: {"status": "idle"}}, "counts": {"conversations": 3, "problems": 2,
                "corrections": 1, "pairs": 2}, "gates": {1: {"ok": True, "judge": (3, 4)}}, "overlap": {"text": {"checked": 4}}}

    def post_rows(self, kind):
        return [{"kind": kind, "bpc": float("nan")}]


class PostTrainingApiTest(unittest.TestCase):
    def setUp(self):
        self.services = DataServices()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.services.root = self.root
        self.app = create_app(self.services, open=True)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def queued(self):
        return self.services.queue().up_next

    def test_status_is_a_job_shaped_like_post_training(self):
        trained(self.services)
        job, steps = finished(self.tap, self.client.post("/post-training/status", json={}))
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"]["counts"], {"conversations": 3, "problems": 2, "corrections": 1, "pairs": 2})
        self.assertEqual(job["result"]["rounds"]["1"], {"status": "done", "note": "live"})
        self.assertEqual(job["result"]["gates"]["1"]["judge"], [3, 4])
        self.assertEqual([p["status"] for p in steps][0], "queued")  # every state change reached /events

    def test_status_of_a_version_that_does_not_exist(self):
        self.assertEqual(self.client.post("/post-training/status", json={"version": 9}).status_code, 404)

    def test_rows_are_a_job_and_nan_becomes_null(self):
        for kind in ("conversations", "problems", "pairs", "corrections", "heldout_docs", "heldout_problems"):
            job, _ = finished(self.tap, self.client.post("/post-training/rows", json={"kind": kind}))
            self.assertEqual(job["status"], "done", job)
            self.assertEqual(job["result"], [{"kind": kind, "bpc": None}])
        self.assertEqual(self.client.post("/post-training/rows", json={"kind": "everything"}).status_code, 422)

    def test_options_say_why_lora_is_off_and_who_can_teach(self):
        first = trained(self.services)
        options = self.client.get("/post-training/options").json()
        self.assertIsNone(options["lora_blocker"])
        self.assertEqual(options["lora_ranks"], [4, 8, 16, 32])
        self.assertEqual([x["name"] for x in options["leash"]], ["Close", "Medium", "Far"])
        self.assertEqual(options["teachers"], [])
        second = self.services.save_version(small_spec(), "untrained")
        self.assertEqual(self.client.get("/post-training/options", params={"version": second.number}).json()["lora_blocker"],
                         "LoRA needs a trained version to start from.")
        third = trained(self.services)
        options = self.client.get("/post-training/options", params={"version": third.number}).json()
        self.assertEqual([t["number"] for t in options["teachers"]], [first.number])
        self.assertEqual(options["default_teacher"], first.number)
        self.assertEqual(self.client.get("/post-training/options", params={"version": 99}).status_code, 404)

    def test_a_round_is_queued_with_the_pages_parameters(self):
        version = trained(self.services)
        queued = self.client.post("/post-training/rounds", json={"round": 1, "lora": True, "lora_rank": 16})
        self.assertEqual(queued.status_code, 200, queued.text)
        job = self.queued()[-1]
        self.assertEqual((job.kind, job.id), ("post_round", queued.json()["id"]))
        self.assertEqual(job.params, {"round": 1, "version": version.number, "checkpoint": "best.npz", "config": {"lora": {"rank": 16}}})
        self.assertIn("LoRA rank 16", job.label)
        self.tap.wait("queue")

        self.client.post("/post-training/rounds", json={"round": 1, "lora": False})
        self.assertEqual(self.queued()[-1].params["config"], {"lora": None})
        self.client.post("/post-training/rounds", json={"round": 2})
        self.assertNotIn("config", self.queued()[-1].params)
        self.client.post("/post-training/rounds", json={"round": 4, "leash": "Far"})
        self.assertEqual(self.queued()[-1].params["beta"], .02)

    def test_a_correction_round_goes_to_the_front(self):
        teacher = trained(self.services)
        trained(self.services)
        self.client.post("/post-training/rounds", json={"round": 2})
        round_three = self.client.post("/post-training/rounds", json={"round": 3}).json()
        self.assertEqual(self.queued()[0].id, round_three["id"])
        self.assertEqual(round_three["params"]["teacher"], "best.npz")
        self.assertEqual(round_three["params"]["teacher_choice"], "previous best")
        self.client.post("/post-training/rounds", json={"round": 3, "teacher": "correction"})
        self.assertEqual(self.queued()[0].params["teacher_choice"], "correction")
        self.assertIsNone(self.queued()[0].params["teacher"])
        self.assertEqual(self.client.post("/post-training/rounds", json={"round": 3, "teacher_version": teacher.number + 5}).status_code, 400)

    def test_a_round_that_cannot_run_says_why(self):
        self.assertEqual(self.client.post("/post-training/rounds", json={"round": 1}).json()["detail"],
                         "There is no version yet. Build one first.")
        self.services.save_version(small_spec(), "untrained")
        self.assertEqual(self.client.post("/post-training/rounds", json={"round": 1}).json()["detail"], "Train this version in Pretraining first.")
        trained(self.services)
        self.services.post_training = lambda version: {"counts": {}}
        for number, sentence in ((1, "Add conversations in Your data first."), (2, "Make problems a rule can check first."),
                                 (3, "There is no other trained version to learn from yet."), (4, "Rate at least 2 pairs in Chat first.")):
            self.assertEqual(self.client.post("/post-training/rounds", json={"round": number}).json()["detail"], sentence)
        self.assertEqual(self.client.post("/post-training/rounds", json={"round": 5}).status_code, 422)
        del self.services.post_training
        self.assertEqual(self.client.post("/post-training/rounds", json={"round": 1, "lora_rank": 7}).status_code, 400)
        self.assertEqual(self.queued(), [])

    def test_import_saves_a_version_and_says_so_on_events(self):
        response = self.client.post("/post-training/imports", data={"name": "gpt2"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["number"], 1)
        self.assertEqual(self.services.imports, ["gpt2"])
        self.assertEqual(self.tap.wait("versions")[0]["number"], 1)
        self.assertNotIn("lora", self.services.versions()[0].spec["train"].get("run", {}))

    def test_import_with_lora_options(self):
        response = self.client.post("/post-training/imports", data={"name": "gpt2", "lora": "true", "lora_rank": "16"})
        self.assertEqual(response.json()["spec"]["train"]["run"]["lora"]["rank"], 16)
        self.assertEqual(self.client.post("/post-training/imports", data={"lora": "true", "lora_rank": "5"}).status_code, 400)
        self.assertEqual(self.client.post("/post-training/imports", data={"name": "bert"}).status_code, 400)

    def test_import_keeps_an_uploaded_file_only_when_it_is_the_published_one(self):
        weights = b"not really safetensors"
        dest = self.root / "data" / "pretrained" / "gpt2" / "model.safetensors"
        with mock.patch.dict(studio_import.MODELS["gpt2"], size=len(weights)), mock.patch.object(studio_import, "GPT2_SHA256", hashlib.sha256(weights).hexdigest()):
            wrong = self.client.post("/post-training/imports", files={"weights": ("model.safetensors", b"x" * len(weights))})
            self.assertEqual(wrong.status_code, 400)
            self.assertIn("not the published GPT-2", wrong.json()["detail"])
            self.assertFalse(dest.exists())
            self.assertEqual(list(dest.parent.iterdir()), [])  # the partial copy is gone too
            self.assertEqual(self.services.versions(), [])

            right = self.client.post("/post-training/imports", files={"weights": ("model.safetensors", weights)})
            self.assertEqual(right.status_code, 200, right.text)
            self.assertEqual(dest.read_bytes(), weights)

    def test_an_upload_needs_a_workspace_to_keep_it_in(self):
        del self.services.root
        response = self.client.post("/post-training/imports", files={"weights": ("model.safetensors", b"x")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.services.versions(), [])


if __name__ == "__main__":
    unittest.main()

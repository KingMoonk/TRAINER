"""studio_api.routers.post_training over FakeServices: every route answers and the events its methods emit reach /events."""
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import studio_import
from studio_api.app import create_app
from studio_api.hosts import ANY
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
    def test_import_catalogue_comes_from_the_server_registry(self):
        models = {"tiny": {"title": "Tiny model", "size": 1234}, "other": {"title": "Another model", "size": 9876}}
        with patch("studio_import.MODELS", models):
            response = self.client.get("/post-training/import-models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{"id": "tiny", "name": "Tiny model", "size": 1234},
                                           {"id": "other", "name": "Another model", "size": 9876}])

    def test_status_job_exposes_its_result_schema(self):
        schema = self.client.get("/openapi.json").json()
        response = schema["paths"]["/post-training/status"]["post"]["responses"]["202"]["content"]["application/json"]["schema"]
        self.assertEqual(response["$ref"], "#/components/schemas/PostTrainingJobOut")
        result = schema["components"]["schemas"]["PostTrainingJobOut"]["properties"]["result"]
        self.assertIn({"$ref": "#/components/schemas/PostTrainingOut"}, result["anyOf"])

    def setUp(self):
        self.services = DataServices()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.services.root = self.root
        self.app = create_app(self.services, open=True, hosts=ANY)
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
        for kind in ("conversations", "problems", "pairs", "corrections", "heldout_docs", "heldout_conversations", "heldout_problems"):
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
        self.assertIsNone(options["default_teacher"])   # Round 1 never ran on it: never an unchecked default
        self.assertEqual(options["teachers"][0]["chat_passed"], None)
        self.assertEqual(self.client.get("/post-training/options", params={"version": 99}).status_code, 404)

    def test_the_default_teacher_is_the_best_earlier_version_that_passed_round_1(self):
        from studio_api.routers.post_training import default_teacher
        weak, strong, failed, newest, student = (trained(self.services) for _ in range(5))
        gate = lambda ok, loss: {1: {"ok": ok, "after": {"conv_loss": loss}}}
        self.services._round_gates = {weak.number: gate(True, 2.0), strong.number: gate(True, 1.2),
                                      failed.number: gate(False, 0.5), newest.number: {}}
        options = self.client.get("/post-training/options", params={"version": student.number}).json()
        self.assertEqual(options["default_teacher"], strong.number)
        by_number = {t["number"]: t for t in options["teachers"]}
        self.assertEqual((by_number[strong.number]["chat_passed"], by_number[strong.number]["chat_loss"]), (True, 1.2))
        self.assertEqual((by_number[failed.number]["chat_passed"], by_number[failed.number]["chat_loss"]), (False, 0.5))
        self.assertIsNone(by_number[newest.number]["chat_passed"])
        # only a failed or unmeasured Round 1 earlier on: no default, and a round queued without a pick says so
        self.services._round_gates = {failed.number: gate(False, 0.5), weak.number: {1: {"ok": True, "after": {}}}}
        self.assertIsNone(default_teacher([weak, failed, newest], student, self.services.round_gates()))
        refused = self.client.post("/post-training/rounds", json={"round": 3, "version": student.number})
        self.assertEqual(refused.status_code, 400, refused.text)
        self.assertIn("no teacher is preselected", refused.text)
        picked = self.client.post("/post-training/rounds", json={"round": 3, "version": student.number, "teacher_version": failed.number})
        self.assertEqual(picked.status_code, 200, picked.text)

    def test_teachers_are_filtered_by_the_serialized_tokenizer_the_job_will_compare(self):
        from studio_api.routers.post_training import teacher_options
        student, same, other, unreadable = (trained(self.services) for _ in range(4))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        def write(version, tokenizer):
            path = Path(tmp.name) / f"v{version.number}.npz"
            np.savez(path, metadata=np.array(json.dumps({"tokenizer": tokenizer})))
            version.best_checkpoint = str(path)
        write(student, {"id": "bytes", "merges": [[1, 2]]})
        write(same, {"id": "bytes", "merges": [[1, 2]]})
        write(other, {"id": "bytes", "merges": [[3, 4]]})   # the same spec entry, a different serialized tokenizer
        unreadable.best_checkpoint = str(Path(tmp.name) / "missing.npz")
        listed =[v.number for v in teacher_options(self.services.versions(), student)]
        self.assertEqual(listed, [same.number, unreadable.number])   # the unreadable one falls back to the spec entry, equal here

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

    def test_round_two_settings_are_queued_and_bounded(self):
        version = trained(self.services)
        queued = self.client.post("/post-training/rounds", json={"round": 2, "steps": 50, "group": 4, "prompts_per_step": 6, "temperature": .7, "max_tokens": 32})
        self.assertEqual(queued.status_code, 200, queued.text)
        self.assertEqual(self.queued()[-1].params, {"round": 2, "version": version.number, "checkpoint": "best.npz", "steps": 50, "group": 4,
                                                    "prompts_per_step": 6, "temperature": .7, "max_tokens": 32})
        self.client.post("/post-training/rounds", json={"round": 2})
        self.assertEqual(self.queued()[-1].params, {"round": 2, "version": version.number, "checkpoint": "best.npz"})   # the round's own defaults apply
        self.assertEqual(self.client.post("/post-training/rounds", json={"round": 2, "group": 1}).status_code, 422)

    def test_a_correction_round_goes_to_the_front(self):
        teacher = trained(self.services)
        trained(self.services)
        self.services._round_gates = {teacher.number: {1: {"ok": True, "after": {"conv_loss": 1.0}}}}
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

    def test_a_loop_is_queued_listed_stopped_and_carried_on(self):
        version = trained(self.services)
        queued = self.client.post("/post-training/loops", json={"judge_model": "qwen", "config": {"max_rounds": 5, "patience": 2}})
        self.assertEqual(queued.status_code, 200, queued.text)
        job = self.queued()[-1]
        self.assertEqual((job.kind, job.id, queued.json()["job"]), ("rlhf_loop", job.id, job.id))
        self.assertEqual(job.params["config"], {"max_rounds": 5, "patience": 2})
        self.assertEqual((job.params["version"], job.params["judge_model"], job.params["loop_id"]), (version.number, "qwen", queued.json()["loop_id"]))
        listed = self.client.get("/post-training/loops").json()
        self.assertEqual([(x["id"], x["active"]) for x in listed], [(queued.json()["loop_id"], True)])
        again = self.client.post("/post-training/loops", json={"resume": queued.json()["loop_id"]})
        self.assertEqual(again.status_code, 400)
        self.assertIn("already", again.json()["detail"])
        self.assertEqual(self.client.post(f"/post-training/loops/{queued.json()['loop_id']}/stop").json(), {"stopped": True})
        self.assertEqual(self.client.post("/post-training/loops/nope/stop").status_code, 400)
        self.assertEqual(self.client.post("/post-training/loops", json={"config": {"max_rounds": 0}}).status_code, 422)

    def test_a_loop_needs_something_to_learn_from(self):
        trained(self.services)
        self.services.post_training = lambda version: {"counts": {"problems": 0}}
        response = self.client.post("/post-training/loops", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("rule can check", response.json()["detail"])
        self.assertEqual(self.client.post("/post-training/loops", json={"judge_model": "qwen"}).status_code, 200)

    def test_judges_say_why_ollama_could_not_be_read(self):
        with patch("studio_rlhf.OllamaTeacher.models", side_effect=RuntimeError("Cannot reach Ollama at http://localhost:11434. Is it running?")):
            body = self.client.get("/post-training/judges").json()
        self.assertEqual(body["models"], [])
        self.assertIn("Cannot reach Ollama", body["error"])
        with patch("studio_rlhf.OllamaTeacher.models", return_value=["a", "b"]):
            self.assertEqual(self.client.get("/post-training/judges").json()["models"], ["a", "b"])

    def test_adopting_a_loop_is_a_job_with_the_gate_result(self):
        job, _ = finished(self.tap, self.client.post("/post-training/loops/abc/adopt"))
        self.assertEqual(job["status"], "done", job)
        self.assertFalse(job["result"]["ok"])

    def test_a_kept_round_can_be_tried_then_adopted(self):
        trained(self.services)
        self.services.kept_rounds = {(1, 1): "/x/kept.npz"}
        tried = self.client.post("/chat/reply", json={"messages": [{"role": "user", "content": "hi"}], "round": 1})
        self.assertIn("kept checkpoint", tried.text)
        adopted = self.client.post("/post-training/rounds/1/adopt", json={"version": 1})
        self.assertEqual(adopted.status_code, 200, adopted.text)
        self.assertEqual(adopted.json(), {"version": 1, "round": 1, "checkpoint": "/x/kept.npz"})
        self.assertEqual(self.services.versions()[0].best_checkpoint, "/x/kept.npz")

    def test_adopting_a_round_with_nothing_kept_says_why(self):
        trained(self.services)
        refused = self.client.post("/post-training/rounds/2/adopt", json={})
        self.assertEqual(refused.status_code, 400, refused.text)
        self.assertIn("Round 2 has no kept checkpoint", refused.text)
        self.assertEqual(self.client.post("/post-training/rounds/9/adopt", json={}).status_code, 404)

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

import threading
import unittest
from pathlib import Path

from tests.sse_harness import ApiCase


class PretrainingRouterTests(ApiCase):
    prefix = "/pretraining"

    def test_every_route_answers(self):
        cached = self.call("GET", "/pretraining/preview", params={"dataset_ids": ["a", "b"], "rows": 4})
        self.assertEqual((cached["pieces"], len(cached["first_rows"])), (2_000_000, 4))

        preview = self.job("POST", "/pretraining/preview", json={"dataset_ids": ["a"], "rows": 3})
        self.assertEqual(preview["status"], "done")
        self.assertEqual((preview["result"]["pieces"], preview["result"]["context"], len(preview["result"]["first_rows"])),
                         (2_000_000, 256, 3))

        lists = self.job("POST", "/pretraining/word_lists")
        self.assertEqual(lists["status"], "done", lists)
        result = lists["result"]
        self.assertTrue(result["sentence"])
        self.assertEqual(len(result["lists"]), 4)
        letters = result["lists"][0]
        self.assertEqual((letters["id"], letters["count"] == len(letters["chips"])), ("letters", True))
        self.assertIsNone(result["lists"][2])  # GPT-2's list is not fetched by the fake
        self.assertEqual(set(letters["chips"][0]), {"label", "known"})

        found = self.job("POST", "/pretraining/lr_range_test", json={"spec": {"dims": {"n_ctx": 64}}})
        self.assertEqual(found["status"], "done")
        self.assertEqual(len(found["result"]["rates"]), 100)
        self.assertEqual(found["result"]["steepest"], 70)
        self.assertFalse(found["result"]["diverged"])
        self.run_page_reads()
        self.assertAllRoutesHit()

    def test_an_unknown_preview_is_null_not_computed(self):
        self.services.cached_preview = lambda ids, rows=16: None
        self.assertIsNone(self.call("GET", "/pretraining/preview"))

    def test_the_preview_without_a_design_is_empty(self):
        self.services.pretrain_preview = lambda ids, rows=16: {}
        done = self.job("POST", "/pretraining/preview", json={})
        self.assertEqual(done["result"], {})

    def test_the_range_test_gets_the_spec_and_a_stop_that_the_job_sets(self):
        seen, running = {}, threading.Event()

        def slow(spec, stop=None):
            seen["spec"] = spec
            running.set()
            stop.wait(10)
            return {"rates": [1e-4], "losses": [3.0], "cancelled": stop.is_set()}
        self.services.lr_range_test = slow
        started = self.call("POST", "/pretraining/lr_range_test", 202, json={"spec": {"x": 1}})
        self.assertTrue(running.wait(10))
        self.call("DELETE", "/jobs/{job_id}", path={"job_id": started["id"]})
        self.events.wait(lambda ev: any(k == "job" and p["id"] == started["id"] and p["status"] == "cancelled" for k, p in ev), "the cancel")
        self.assertEqual(seen["spec"], {"x": 1})
        self.assertTrue(self.client.get(f"/jobs/{started['id']}").json()["result"]["cancelled"])

    def test_a_failing_read_fails_its_job_with_the_sentence(self):
        def refuse(spec, stop=None):
            raise ValueError("There is no plain text to try it on. Add some in Your data.")
        self.services.lr_range_test = refuse
        failed = self.job("POST", "/pretraining/lr_range_test", json={"spec": {}})
        self.assertEqual(failed["status"], "failed")
        self.assertIn("There is no plain text to try it on.", failed["error"])

    def test_bad_requests(self):
        self.call("POST", "/pretraining/lr_range_test", 422, json={})
        self.call("POST", "/pretraining/preview", 422, json={"rows": 0})
        self.call("GET", "/pretraining/preview", 422, params={"rows": 1000})

    def run_page_reads(self):
        """The training-run page's reads find their run from a version or a job, never from a path."""
        from studio_ia.services import Job, QueueState

        asked = []
        self.services.run_curve = lambda run_dir: asked.append(("curve", run_dir)) or {
            "steps": [(10, 5.5)], "smooth": [(10, 5.4)], "heldout": [(10, 5.6)], "lr": [(10, 3e-4)],
            "phases": [(0, 5, "warm-up"), (5, 100, "steady")], "total": 100,
            "samples": [{"step": 10, "text": "once"}], "sample_every": 50, "prompt": "Once"}
        self.services.run_phases = lambda run_dir: asked.append(("phases", run_dir)) or [(0, 5, "warm-up"), (5, 100, "steady")]
        self.services.save_version({"dims": {"n_ctx": 64}}, "first")
        self.services.versions()[0].best_checkpoint = "/work/runs/v1/best.npz"
        running = Job("j1", "pretrain", params={"output_dir": "/work/runs/live"}, status="running")
        self.services.set_queue(QueueState(job=running))

        curve = self.call("GET", "/pretraining/versions/{number}/curve", path={"number": 1})
        self.assertEqual((curve["steps"], curve["total"], curve["samples"], curve["sample_every"]), ([[10, 5.5]], 100, [{"step": 10, "text": "once"}], 50))
        self.assertEqual(curve["phases"][1], [5, 100, "steady"])
        phases = self.call("GET", "/pretraining/versions/{number}/phases", path={"number": 1})
        self.assertEqual(phases[0], {"from_step": 0, "to_step": 5, "name": "warm-up"})
        self.call("GET", "/pretraining/jobs/{job_id}/curve", path={"job_id": "j1"})
        self.call("GET", "/pretraining/jobs/{job_id}/phases", path={"job_id": "j1"})
        self.assertEqual(sorted(asked), sorted([("curve", str(Path("/work/runs/v1"))), ("phases", str(Path("/work/runs/v1"))),
                                                ("curve", "/work/runs/live"), ("phases", "/work/runs/live")]))

        refs = self.call("GET", "/pretraining/versions/{number}/references", path={"number": 1})
        self.assertEqual(set(refs), {"uniform_bpc", "unigram_bpc", "best_so_far_bpc"})
        self.assertIsNone(self.call("GET", "/pretraining/versions/{number}/gate_result", path={"number": 1}))
        self.services.gate_result = lambda version: {"better": True, "diff": -0.1, "upper": -0.02, "vs": 0}
        self.assertEqual(self.call("GET", "/pretraining/versions/{number}/gate_result", path={"number": 1}),
                         {"better": True, "diff": -0.1, "upper": -0.02, "vs": 0})

    def test_a_run_that_does_not_exist_is_a_404_and_no_path_is_taken(self):
        self.services.save_version({}, "no run yet")
        for path in ("/pretraining/versions/9/curve", "/pretraining/versions/9/phases", "/pretraining/versions/9/references",
                     "/pretraining/versions/9/gate_result", "/pretraining/versions/1/curve", "/pretraining/jobs/nope/curve",
                     "/pretraining/jobs/nope/phases"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        self.assertEqual(self.client.get("/pretraining/versions/1/curve", params={"run_dir": "/etc"}).status_code, 404)

    def test_every_event_reaches_events(self):
        self.job("POST", "/pretraining/preview", json={"dataset_ids": [], "rows": 2})
        self.job("POST", "/pretraining/word_lists")
        done = self.job("POST", "/pretraining/lr_range_test", json={"spec": {}})
        self.assertDelivered()
        finished = [p for p in self.events.of("job") if p["id"] == done["id"]]
        self.assertEqual([p["status"] for p in finished], ["queued", "running", "done"])
        self.assertEqual(finished[-1]["result"]["suggestion"], done["result"]["suggestion"])  # the result rides the last event
        self.assertEqual({p["kind"] for p in self.events.of("job")}, {"pretrain_preview", "word_lists", "lr_range_test"})


if __name__ == "__main__":
    unittest.main()

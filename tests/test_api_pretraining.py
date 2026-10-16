import threading
import unittest

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

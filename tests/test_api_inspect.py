"""studio_api.routers.inspect over FakeServices: every read is a job whose result arrives on /events, as plain JSON."""
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_ia.services import FakeServices
from tests.api_tap import EventTap
from tests.api_work import finished


class InspectApiTest(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.app = create_app(self.services, open=True)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def read(self, route, body):
        job, steps = finished(self.tap, self.client.post(f"/inspect/{route}", json=body))
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["kind"], "inspect." + route.replace("-", "_"))
        self.assertEqual(steps[0]["status"], "queued")  # every state change reached /events
        return job["result"]

    def test_forward_is_a_job_with_the_arrays_as_lists(self):
        result = self.read("forward", {"text": "abcabc"})
        self.assertEqual(sorted(result), ["hooks", "ids", "logits", "tokens"])
        self.assertIsInstance(result["logits"], list)
        self.assertIn("block0.attn.pattern", result["hooks"])
        self.assertEqual(len(result["hooks"]["embed"][0]), 9)

    def test_forward_keeps_only_the_hooks_asked_for(self):
        result = self.read("forward", {"text": "abc", "names": ["embed", "block0.attn.pattern", "nothing"]})
        self.assertEqual(sorted(result["hooks"]), ["block0.attn.pattern", "embed"])
        self.assertEqual(self.read("forward", {"text": "abc", "names": []})["hooks"], {})

    def test_trace_prepared_loss_and_value_map(self):
        trace = self.read("trace", {"text": "abc", "pos": 3})
        self.assertEqual([row["stage"] for row in trace["rows"]], ["embedding", "block 1 mixer", "block 1 channel", "output"])
        self.assertIsNone(trace["rows"][0]["cosine"])
        self.assertEqual(self.read("prepared", {})["ids"][:3], [1, 5, 6])
        loss = self.read("loss", {"text": "abc"})
        self.assertEqual(loss["target"], 3)
        self.assertAlmostEqual(loss["probabilities"][3], 1 - .02 * 11)
        fresh = self.read("loss", {"text": "abc", "trained": False})
        self.assertAlmostEqual(fresh["probabilities"][3], 1 / 12)
        value_map = self.read("value-map", {"heads": 4, "block": 1, "head": 2})
        self.assertEqual((value_map["block"], value_map["head"], value_map["singular"]), (1, 2, [3.0, 1.5, .7, .1]))
        self.assertEqual(value_map["blocks"], {"0": 4, "1": 4})  # int keys become strings in JSON
        self.assertIsNone(self.read("value-map", {"heads": 8})["singular"])

    def test_rank_heads_and_heads_off(self):
        rows = self.read("rank-heads", {"mode": "zero"})
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[5], {"block": 1, "head": 1, "rise": .25})
        self.assertAlmostEqual(self.read("heads-off", {"off": [[0, 1], [1, 2]]})["after"], 2.2)
        self.assertAlmostEqual(self.read("heads-off", {"off": [[0, 1]], "mode": "zero"})["after"], 2.2)

    def test_reply_with_heads_off(self):
        self.assertEqual(self.read("reply", {"prompt": "Hi", "off": []}), "[reply]")
        self.assertEqual(self.read("reply", {"prompt": "Hi", "off": [[0, 0], [1, 1]]}), "[reply with 2 heads off]")

    def test_the_version_and_arguments_reach_the_services(self):
        seen = []
        self.services.inspect_trace = lambda version, text, pos=-1: seen.append((version, text, pos)) or {"tokens": [], "pos": 0, "rows": []}
        self.read("trace", {"version": 3, "text": "abc"})
        self.read("trace", {"text": "abc", "pos": 2})
        self.assertEqual(seen, [(3, "abc", -1), (None, "abc", 2)])

    def test_a_read_that_fails_is_a_failed_job_with_a_sentence(self):
        def refuse(version, text, hooks=True):
            raise ValueError("Version 4 has not trained yet.")
        self.services.inspect_forward = refuse
        job, _ = finished(self.tap, self.client.post("/inspect/forward", json={"text": "abc"}))
        self.assertEqual((job["status"], job["error"]), ("failed", "ValueError: Version 4 has not trained yet."))
        self.assertEqual(self.client.get(f"/jobs/{job['id']}").json()["error"], "ValueError: Version 4 has not trained yet.")

    def test_a_bad_request_never_starts_a_job(self):
        for route, body in (("forward", {"text": ""}), ("forward", {}), ("trace", {"text": "x", "pos": "end"}), ("value-map", {"heads": 0}),
                            ("rank-heads", {"mode": "median"}), ("heads-off", {"off": [[0]]}), ("heads-off", {"off": [[0, 1]], "mode": "x"}),
                            ("reply", {"prompt": ""}), ("loss", {"text": "x", "trained": "maybe"})):
            self.assertEqual(self.client.post(f"/inspect/{route}", json=body).status_code, 422, (route, body))
        self.assertEqual(self.tap.of("job"), [])


if __name__ == "__main__":
    unittest.main()

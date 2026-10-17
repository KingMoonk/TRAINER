"""studio_api.routers.inspect over FakeServices: every read is a job whose result arrives on /events, as plain JSON."""
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from tests.api_tap import EventTap
from tests.api_work import finished


class InspectApiTest(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.app = create_app(self.services, open=True, hosts=ANY)
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

    def test_forward_reads_piece_ids_as_well_as_text(self):
        seen = []
        real = self.services.inspect_forward
        self.services.inspect_forward = lambda version, text, hooks=True: seen.append(text) or real(version, text, hooks)
        self.assertEqual(sorted(self.read("forward", {"ids": [1, 5, 6], "names": []})), ["hooks", "ids", "logits", "tokens"])
        self.read("forward", {"text": "abc", "names": []})
        self.assertEqual(seen, [[1, 5, 6], "abc"])

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

    def test_logit_lens_is_a_job_with_one_row_per_stage(self):
        rows = self.read("logit-lens", {"text": "abc", "top_k": 3})
        self.assertEqual([row["hook"] for row in rows], ["embed", "block0.resid.post", "block1.resid.post"])
        self.assertEqual(len(rows[0]["top_ids"][0][0]), 3)
        self.assertEqual(len(rows[0]["probabilities"][0][0]), 12)
        self.assertAlmostEqual(sum(rows[0]["probabilities"][0][0]), 1.0)
        top = rows[1]["top_probabilities"][0][0]
        self.assertEqual(top, sorted(top, reverse=True))
        self.assertEqual(len(self.read("logit-lens", {"text": "abc"})[0]["top_ids"][0][0]), 5)

    def test_ablate_is_a_job_for_every_head_or_the_ones_asked_for(self):
        every = self.read("ablate", {"text": "abc"})
        self.assertEqual((every["mode"], every["baseline"], len(every["heads"])), ("zero", 2.0, 8))
        self.assertEqual(sorted(every["heads"][0]), ["block", "delta", "entropy", "first_token_mass", "head", "loss", "mean_distance", "previous_token_mass"])
        some = self.read("ablate", {"text": "abc", "mode": "mean", "heads": [[1, 2], [0, 0]]})
        self.assertEqual(some["mode"], "mean")
        self.assertEqual([(row["block"], row["head"]) for row in some["heads"]], [(1, 2), (0, 0)])
        self.assertAlmostEqual(some["heads"][0]["delta"], some["heads"][0]["loss"] - some["baseline"])

    def test_a_head_that_does_not_exist_fails_the_ablate_job_with_a_sentence(self):
        job, _ = finished(self.tap, self.client.post("/inspect/ablate", json={"text": "abc", "heads": [[5, 0]]}))
        self.assertEqual((job["status"], job["error"]), ("failed", "ValueError: Selected block/head has no attention output hook."))

    def test_head_stats_answers_at_once_for_the_posted_pattern(self):
        pattern = self.read("forward", {"text": "abcabc", "names": ["block0.attn.pattern"]})["hooks"]["block0.attn.pattern"]
        response = self.client.post("/inspect/head-stats", json={"pattern": pattern})
        self.assertEqual(response.status_code, 200, response.text)
        stats = response.json()
        self.assertEqual(sorted(stats), ["entropy", "first_token_mass", "mean_distance", "previous_token_mass"])
        self.assertTrue(all(len(values) == 4 for values in stats.values()))
        one_hot = [[[[1.0 if k == 0 else 0.0 for k in range(3)] for _ in range(3)]]]  # one head that always looks at the first piece
        self.assertEqual(self.client.post("/inspect/head-stats", json={"pattern": one_hot}).json()["first_token_mass"], [1.0])
        self.assertEqual({p["kind"] for p in self.tap.of("job")}, {"inspect.forward"})  # head-stats started no job

    def test_head_stats_refuses_a_pattern_that_is_not_square_attention(self):
        for pattern in ([[1.0]], [[[[0.5, 0.5]]]], [[[1.0], [1.0, 2.0]]], [["a"]]):
            self.assertEqual(self.client.post("/inspect/head-stats", json={"pattern": pattern}).status_code, 422, pattern)
        self.assertEqual(self.client.post("/inspect/head-stats", json={"pattern": []}).status_code, 422)

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
        for route, body in (("forward", {"text": ""}), ("forward", {}), ("forward", {"text": "a", "ids": [1]}), ("forward", {"ids": []}), ("forward", {"ids": ["a"]}), ("trace", {"text": "x", "pos": "end"}), ("value-map", {"heads": 0}),
                            ("rank-heads", {"mode": "median"}), ("heads-off", {"off": [[0]]}), ("heads-off", {"off": [[0, 1]], "mode": "x"}),
                            ("reply", {"prompt": ""}), ("logit-lens", {"text": "x", "top_k": 0}), ("logit-lens", {"text": ""}),
                            ("ablate", {"text": "x", "mode": "median"}), ("ablate", {"text": "x", "heads": [[0]]}), ("loss", {"text": "x", "trained": "maybe"})):
            self.assertEqual(self.client.post(f"/inspect/{route}", json=body).status_code, 422, (route, body))
        self.assertEqual(self.tap.of("job"), [])


if __name__ == "__main__":
    unittest.main()

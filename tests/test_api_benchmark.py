"""studio_api.routers.benchmark over FakeServices: the reads that score or parse data are jobs; the verdict answers at once."""
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_ia.services import FakeServices
from tests.api_tap import EventTap
from tests.api_work import finished

RUN = {"base": {"dims": {"d_model": 16}}, "variants": [{"name": "wide"}], "mode": "fair", "results": {"base": {"bpc": 2.5}, "wide": {"bpc": 2.4}},
       "finished": 1700000000.0, "added_by_engine": 1}


class SavedRuns(FakeServices):
    """FakeServices with a finished Compare runs experiment (it has none), and a verdict that compares."""

    def compare_runs(self):
        return [RUN]

    def verdict(self, a, b):
        return {"percent": 37.5, "comparable": True, "reason": "", "extra": [a, b]}


class BenchmarkApiTest(unittest.TestCase):
    def setUp(self):
        self.services = SavedRuns()
        self.app = create_app(self.services, open=True)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def read(self, route, body=None):
        job, steps = finished(self.tap, self.client.post(f"/benchmark/{route}", json=body))
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["kind"], "benchmark." + route.replace("-", "_"))
        self.assertEqual(steps[0]["status"], "queued")  # every state change reached /events
        return job["result"]

    def test_read_is_the_stored_track_one_row_per_trained_version(self):
        result = self.read("read")
        self.assertEqual((result["suite"], result["sizes"]), ("fake", {"docs": 24, "choice": 40, "problems": 30}))
        self.assertEqual([(row["number"], row["result"] and row["result"]["bpc"]) for row in result["rows"]], [(1, 2.6), (2, 2.2), (3, None)])
        self.assertEqual(result["rows"][0]["result"]["problems"], {"passed": 3, "total": 30, "rate": .1})

    def test_run_scores_one_version(self):
        result = self.read("run", {"version": 2})
        self.assertEqual((result["version"], result["bpc"]), (2, 2.2))
        self.assertEqual(self.client.post("/benchmark/run", json={}).status_code, 422)  # a run needs the version to score

    def test_a_run_that_fails_is_a_failed_job(self):
        def refuse(number):
            raise ValueError(f"Version {number} has no checkpoint.")
        self.services.run_benchmark = refuse
        job, _ = finished(self.tap, self.client.post("/benchmark/run", json={"version": 7}))
        self.assertEqual((job["status"], job["error"]), ("failed", "ValueError: Version 7 has no checkpoint."))

    def test_few_shot_rows_by_number_of_examples(self):
        result = self.read("few-shot", {})
        self.assertEqual((result["items"], result["tasks"], result["held_back"], result["chance"]), (40, 2, True, .3))
        self.assertEqual([(row["shots"], row["correct"]) for row in result["rows"]], [(0, 11), (1, 13), (4, 14)])
        self.assertEqual(self.read("few-shot", {"version": 2})["items"], 40)

    def test_compare_runs_keep_what_the_engine_saved(self):
        (run,) = self.read("compare-runs")
        self.assertEqual((run["mode"], run["results"]["wide"], run["finished"]), ("fair", {"bpc": 2.4}, 1700000000.0))
        self.assertEqual(run["added_by_engine"], 1)  # a key the router does not know is passed through
        self.services.compare_runs = lambda: []
        self.assertEqual(self.read("compare-runs"), [])

    def test_verdict_answers_at_once(self):
        response = self.client.get("/benchmark/verdict", params={"a": 1, "b": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"percent": 37.5, "comparable": True, "reason": "", "extra": [1, 2]})

    def test_verdict_with_nothing_to_compare_says_why(self):
        self.services.verdict = lambda a, b: FakeServices.verdict(self.services, a, b)
        self.assertEqual(self.client.get("/benchmark/verdict", params={"a": 1, "b": 2}).json(),
                         {"percent": None, "comparable": False, "reason": "A run has no points yet."})
        self.assertEqual(self.client.get("/benchmark/verdict", params={"a": 1}).status_code, 422)


if __name__ == "__main__":
    unittest.main()

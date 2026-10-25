import unittest

from tests.sse_harness import ApiCase
from studio_ia.services import FakeServices, Job

STEPS = 4


class FakeJobTests(ApiCase):
    def test_uploaded_text_is_available_to_the_next_page(self):
        payload = [("files", ("phase2-notes.txt", b"A small boat crossed the quiet lake.", "text/plain"))]
        first = self.job("POST", "/data/text/import", files=payload)
        self.assertEqual(first["status"], "done", first)
        self.assertIn("1 added", first["result"][0])
        rows = self.services.data("text").rows
        self.assertTrue(any(row["dataset"] == "phase2-notes" and row["b"] == "A small boat crossed the quiet lake." for row in rows))
        second = self.job("POST", "/data/text/import", files=payload)
        self.assertIn("0 added, 1 duplicates", second["result"][0])

    def test_a_job_with_its_own_thread_does_not_wait_behind_a_slow_one(self):
        import threading
        from studio_api.jobs import Jobs
        release, started, finished = threading.Event(), threading.Event(), threading.Event()
        jobs = Jobs(FakeServices(job_steps=STEPS, step_seconds=0, sleep=lambda s: None))
        self.addCleanup(jobs.close)
        self.addCleanup(release.set)
        jobs.submit("slow", lambda stop: (started.set(), release.wait(10)))
        self.assertTrue(started.wait(5))
        waiting = jobs.submit("shared", lambda stop: "late")
        quick = jobs.submit("quick", lambda stop: finished.set() or "now", own_thread=True)
        self.assertTrue(finished.wait(5))
        for _ in range(100):
            if quick.status == "done":
                break
            threading.Event().wait(.05)
        self.assertEqual((quick.status, quick.result, waiting.status), ("done", "now", "queued"))

    def test_snapshot_events_are_opt_in_and_name_a_step(self):
        self.assertEqual(self.services.snapshot_every, 0)  # the default fake emits only stats, so the sequence above is unchanged
        seen = []
        fake = FakeServices(job_steps=STEPS, step_seconds=0, sleep=lambda s: None, snapshot_every=2)
        fake.emit = lambda kind, payload=None: seen.append((kind, payload))
        fake.enqueue(Job("j1", "pretrain", "Train"))
        fake.start_queue()
        self.assertTrue(fake.wait_idle())
        snapshots = [p for k, p in seen if k == "train" and p.get("type") == "viz_snapshot"]
        self.assertEqual([p["step"] for p in snapshots], [2, 4])

    def test_fake_step_duration_is_configurable_and_validated(self):
        from studio_api.__main__ import parse
        self.assertEqual(parse(["--fake", "--fake-step-seconds", ".25"]).fake_step_seconds, .25)
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse(["--fake", "--fake-step-seconds", value])

    def make_services(self):
        return FakeServices(job_steps=STEPS, step_seconds=0, sleep=lambda s: None)

    def add(self, **params):
        return self.call("POST", "/queue/jobs", 201, json={"kind": "pretrain", "label": "Train", "params": params})

    def run_queue(self):
        self.call("POST", "/queue/start")
        self.assertTrue(self.services.wait_idle())

    def test_a_queued_job_runs_to_a_new_version(self):
        job = self.add()
        self.assertEqual(self.call("GET", "/versions"), [])
        self.run_queue()
        events = self.events.wait(lambda e: any(k == "versions" for k, _ in e), "the version")
        steps = [p for k, p in events if k == "train"]
        self.assertEqual([p["step"] for p in steps], list(range(1, STEPS + 1)))
        self.assertTrue(all(p["job"] == job["id"] and p["total"] == STEPS for p in steps))
        self.assertEqual({p["phase"] for p in steps}, {"warm-up", "steady", "cool-down"})
        kinds = [k for k, _ in events]
        self.assertLess(kinds.index("job"), kinds.index("train"))
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "train"), kinds.index("versions"))
        self.assertEqual([p["status"] for p in self.events.of("job")], ["running", "done"])
        versions = self.call("GET", "/versions")
        self.assertEqual([(v["number"], v["status"]) for v in versions], [(1, "trained")])
        state = self.call("GET", "/queue")
        self.assertIsNone(state["job"])
        self.assertEqual(state["up_next"], [])

    def test_the_job_shows_progress_while_it_runs(self):
        seen = []

        def sleep(seconds):
            q = fake.queue()
            seen.append((q.job.id, q.step, q.total, q.job.status))

        fake = FakeServices(job_steps=STEPS, step_seconds=0, sleep=sleep)
        fake.enqueue(Job("a", "pretrain", "Train"))
        fake.start_queue()
        self.assertTrue(fake.wait_idle())
        self.assertEqual(seen, [("a", step, STEPS, "running") for step in range(STEPS)])  # read before each step lands

    def test_a_job_for_a_saved_version_trains_that_version(self):
        self.services.save_version({}, "first")
        self.add(version=1)
        self.run_queue()
        self.assertEqual([(v["number"], v["status"]) for v in self.call("GET", "/versions")], [(1, "trained")])

    def test_jobs_run_in_order_and_a_paused_queue_does_not_start(self):
        self.call("POST", "/queue/pause")
        first, second = self.add(), self.add()
        self.run_queue()
        self.assertEqual(len(self.call("GET", "/queue")["up_next"]), 2)
        self.call("POST", "/queue/resume")
        self.assertTrue(self.services.wait_idle())
        self.assertEqual(len(self.call("GET", "/versions")), 2)
        self.events.wait(lambda events: sum(kind == "job" for kind, _ in events) == 4)
        self.assertEqual([p["job"] for p in self.events.of("job")], [first["id"], first["id"], second["id"], second["id"]])

    def test_a_held_job_is_skipped(self):
        job = self.add()
        self.call("POST", "/queue/jobs/{job_id}/hold", path={"job_id": job["id"]})
        self.run_queue()
        self.assertEqual(self.call("GET", "/versions"), [])


if __name__ == "__main__":
    unittest.main()

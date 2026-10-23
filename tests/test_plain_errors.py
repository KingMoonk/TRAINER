"""Errors in plain words, Chat only offering trained versions, and the disk check before a job (no display)."""
import io
import tempfile
import time
import unittest
import urllib.error
from collections import namedtuple
from pathlib import Path
from unittest import mock

import studio_queue
from studio_ia.busy import Busy
from studio_ia.services import FakeServices, NoCheckpoint, Version, WorkspaceServices
from studio_queue import Job, Plan, Runner

Usage = namedtuple("Usage", "total used free")
ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)


class BackgroundTest(unittest.TestCase):
    def test_a_failed_download_is_stored_as_a_sentence_naming_the_source(self):
        import threading

        import studio_catalog
        entry = studio_catalog.CATALOG[0]
        services = object.__new__(WorkspaceServices)
        services._lock, services._busy, services._failed, events = threading.Lock(), {}, {}, []
        services.busy = Busy(lambda kind, payload=None: None)
        services.emit = lambda kind, payload=None: events.append((kind, payload))

        def work(cid, task):
            raise urllib.error.HTTPError("http://x", 503, "Unavailable", {}, io.BytesIO())
        self.assertTrue(services._background([entry["id"]], work, "download_done", "Downloading"))
        services._worker.join(5)
        message = f"The server returned 503 for {entry['name']}. The file may have moved; try again later."
        self.assertEqual(services._failed[entry["id"]], message)
        self.assertIn(("download_failed", {"id": entry["id"], "error": message}), events)


class NoCheckpointTest(unittest.TestCase):
    def test_message_names_the_version_and_the_fix(self):
        message = str(NoCheckpoint(3))
        self.assertIn("Version 3 has not trained yet", message)
        self.assertIn("Build your model", message)
        self.assertIsInstance(NoCheckpoint(3), ValueError)

    def test_services_raise_it_for_an_untrained_version(self):
        services = object.__new__(WorkspaceServices)
        services._as_version = lambda version: Version(3)
        with self.assertRaises(NoCheckpoint) as caught:
            services._checkpoint(3)
        self.assertIn("Version 3", str(caught.exception))
        with self.assertRaises(NoCheckpoint):
            services.chat(3, [], lambda text: None, None)

    def test_a_round_without_a_checkpoint_raises_it_too(self):
        import studio_jobs
        import studio_rl
        services = mock.Mock()
        services.current_version.return_value = Version(2)
        services.root, services.workspace.run_dir = "w", "w/runs"
        with self.assertRaises(NoCheckpoint) as caught:  # in the app, before a worker starts: the queued job fails with this sentence
            studio_jobs.prepare(Job("post_round", params={}), services)
        self.assertIn("Version 2 has not trained yet", str(caught.exception))
        with self.assertRaises(NoCheckpoint):  # and a round started without one says the same
            studio_rl.run_round(1, Job("post_round", params={"version": 2}), services, None, None, None)


def note_ran(job, context, emit, stop, pause):
    with open(Path(context.root) / "ran.txt", "a") as f:
        f.write(job.label + "\n")


class DiskCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def runner(self):
        r = Runner(self.root / "data" / "queue.json", {"quicktest": note_ran}, tick=.01)
        r.set_plan(ANY)
        self.addCleanup(r.shutdown)
        return r

    def finish(self, free):
        r = self.runner()
        job = r.enqueue(Job("quicktest", "a"))
        with mock.patch.object(studio_queue.shutil, "disk_usage", return_value=Usage(10 ** 12, 0, free)):
            r.start()
            end = time.time() + 20
            while job.status in ("waiting", "running") and time.time() < end:
                time.sleep(.01)
        return job

    def ran(self):
        path = self.root / "ran.txt"
        return path.read_text().splitlines() if path.exists() else []

    def test_a_full_disk_fails_the_job_with_a_sentence_and_never_starts_it(self):
        job = self.finish(free=200 * 2 ** 20)
        self.assertEqual(job.status, "failed")
        self.assertEqual(self.ran(), [])
        self.assertIn("Not enough disk space for this run: 200 MB free, about 2.0 GB needed.", job.error)
        self.assertIn("Resources › Storage", job.error)

    def test_enough_room_runs_the_job(self):
        job = self.finish(free=50 * 2 ** 30)
        self.assertEqual((job.status, self.ran()), ("done", ["a"]))

    def test_need_is_three_times_the_largest_checkpoint_but_at_least_the_floor(self):
        self.assertEqual(studio_queue.disk_need(self.root), studio_queue.MIN_FREE)
        (self.root / "runs" / "r1").mkdir(parents=True)
        (self.root / "runs" / "r1" / "best.npz").write_bytes(b"x" * 1000)
        (self.root / "runs" / "r1" / "last.npz").write_bytes(b"x" * 400)
        self.assertEqual(studio_queue.disk_need(self.root), studio_queue.MIN_FREE)
        with mock.patch.object(studio_queue, "MIN_FREE", 0):
            self.assertEqual(studio_queue.disk_need(self.root), 3000)

    def test_storage_card_has_free_space(self):
        self.assertGreater(FakeServices().free_space(), 0)


if __name__ == "__main__":
    unittest.main()

"""A failed queue job says why in plain words (#415), and Start reports what really happened."""
import errno
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

import studio_worker
from studio_api.app import create_app
from studio_api.routers import queue as queue_router
from studio_failures import CHECKPOINT, DATA, DISK, FOLDER, GPU, RAM, explain
from tests.sse_harness import ANY as HOSTS
from studio_ia.services import FakeServices
from studio_queue import Job, Plan, Runner

ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)


class OutOfMemoryError(RuntimeError):
    """torch.cuda.OutOfMemoryError by name, so the test needs no GPU."""


class Explain(unittest.TestCase):
    def test_each_common_failure_gets_its_sentence_and_keeps_the_raw_text(self):
        cases = [
            (MemoryError(), RAM, "MemoryError"),
            (MemoryError("Unable to allocate 4.00 GiB"), RAM, "MemoryError: Unable to allocate 4.00 GiB"),
            (OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), GPU, "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"),
            (RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)"), GPU, "RuntimeError: MPS backend out of memory (MPS allocated: 9 GB)"),
            (OSError(errno.ENOSPC, "No space left on device"), DISK, "OSError: [Errno 28] No space left on device"),
            (OSError("There is not enough space on the disk"), DISK, "OSError: There is not enough space on the disk"),
            (json.JSONDecodeError("Unterminated string starting at", "{", 26), DATA, "JSONDecodeError: Unterminated string starting at: line 1 column 27 (char 26)"),
            (ValueError("Choose a new output folder to preserve the existing run."), FOLDER, "ValueError: Choose a new output folder to preserve the existing run."),
            (zipfile.BadZipFile("File is not a zip file"), CHECKPOINT, "BadZipFile: File is not a zip file"),
            (ValueError("Choose a Groundwork NumPy checkpoint (.npz)."), CHECKPOINT, "ValueError: Choose a Groundwork NumPy checkpoint (.npz)."),
            (ValueError("Resume must preserve architecture, tokenizer, split, seed, objective, optimizer, schedule and batch settings."), CHECKPOINT,
             "ValueError: Resume must preserve architecture, tokenizer, split, seed, objective, optimizer, schedule and batch settings."),
        ]
        for error, sentence, details in cases:
            with self.subTest(error=repr(error)):
                self.assertEqual(explain(error), (sentence, details))

    def test_text_alone_is_mapped_too(self):
        self.assertEqual(explain("CUDA out of memory"), (GPU, "CUDA out of memory"))
        self.assertEqual(explain("Unterminated string starting at: line 1 column 26")[0], DATA)

    def test_an_unknown_error_keeps_its_own_text(self):
        self.assertEqual(explain(ValueError("Give one temperature, or one per reply.")), ("Give one temperature, or one per reply.", ""))
        self.assertEqual(explain(KeyError()), ("KeyError", ""))
        self.assertEqual(explain("something odd"), ("something odd", ""))

    def test_every_sentence_says_what_to_do_next(self):
        for sentence in (RAM, GPU, DISK, CHECKPOINT, FOLDER, DATA):
            self.assertRegex(sentence, r"(?i)(start it again|then try|add (this job|it) again|remove (this|the) job|import it again)")
            self.assertNotIn("Error", sentence)


def out_of_memory(job, context, emit, stop, pause):
    raise MemoryError()


def own_text(job, context, emit, stop, pause):
    raise ValueError("Give one temperature, or one per reply.")


def wait(condition, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(.02)
    return False


class InTheQueue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def runner(self, handlers, **kwargs):
        runner = Runner(self.root / "data" / "queue.json", handlers, tick=.02, **kwargs)
        runner.set_plan(ANY)
        self.addCleanup(runner.shutdown)
        return runner

    def test_a_worker_failure_is_a_sentence_with_the_raw_text_beside_it(self):
        runner = self.runner({"quicktest": out_of_memory})
        job = runner.enqueue(Job("quicktest", "t"))
        runner.start()
        self.assertTrue(wait(lambda: job.status == "failed"))
        self.assertEqual((job.error, job.details), (RAM, "MemoryError"))
        self.assertEqual(runner.state().failed, [job])
        with runner.lock:  # the status is set before the queue is saved, under the same lock
            pass
        again = Runner(self.root / "data" / "queue.json", {})  # kept across a restart
        self.assertEqual((again.jobs[0].error, again.jobs[0].details), (RAM, "MemoryError"))

    def test_a_failure_in_the_app_before_the_worker_starts_is_mapped_too(self):
        def prepare(job):
            raise OSError(errno.ENOSPC, "No space left on device")
        runner = self.runner({"quicktest": out_of_memory}, prepare=prepare)
        job = runner.enqueue(Job("quicktest", "t"))
        runner.start()
        self.assertTrue(wait(lambda: job.status == "failed"))
        self.assertEqual(job.error, DISK)
        self.assertIn("No space left", job.details)

    def test_a_job_of_the_own_text_has_no_details(self):
        runner = self.runner({"quicktest": own_text})
        job = runner.enqueue(Job("quicktest", "t"))
        runner.start()
        self.assertTrue(wait(lambda: job.status == "failed"))
        self.assertEqual((job.error, job.details), ("Give one temperature, or one per reply.", ""))

    def test_a_lost_worker_says_so_and_what_to_do(self):
        self.assertIn("Start it again", studio_worker.crash_message(3))


class Backed(FakeServices):
    """The fake services with the queue taken from a real Runner, as WorkspaceServices has it."""

    def __init__(self, runner, fail_at_once):
        super().__init__()
        self.runner = runner

    def queue(self):
        return self.runner.state()

    def start_queue(self):
        if self.runner.thread is None:
            self.runner.start()


class StartReportsTheOutcome(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def post_start(self, prepare):
        runner = Runner(self.root / "data" / "queue.json", {"quicktest": out_of_memory}, prepare=prepare, tick=.02)
        runner.set_plan(ANY)
        self.addCleanup(runner.shutdown)
        runner.jobs.append(Job("quicktest", "Version 1 pretraining", id="j1"))
        client = TestClient(create_app(Backed(runner, True), open=True, hosts=HOSTS))
        return client.post("/queue/start")

    def test_a_job_that_fails_at_once_is_in_the_answer_with_its_sentence(self):
        def prepare(job):
            raise OSError(errno.ENOSPC, "No space left on device")
        started = time.monotonic()
        answer = self.post_start(prepare)
        self.assertEqual(answer.status_code, 200)
        failed = answer.json()["failed"]
        self.assertEqual([(j["id"], j["status"], j["error"]) for j in failed], [("j1", "failed", DISK)])
        self.assertIn("No space left", failed[0]["details"])
        self.assertLess(time.monotonic() - started, queue_router.SETTLE_S)

    def test_a_job_that_runs_is_not_reported_failed(self):
        def prepare(job):
            return {"root": str(self.root), "run_dir": str(self.root / "runs")}
        runner = Runner(self.root / "data" / "queue.json", {"quicktest": hold_on}, prepare=prepare, tick=.02)
        runner.set_plan(ANY)
        self.addCleanup(runner.shutdown)
        runner.jobs.append(Job("quicktest", "t", id="j2"))
        answer = TestClient(create_app(Backed(runner, False), open=True, hosts=HOSTS)).post("/queue/start").json()
        self.assertEqual((answer["failed"], answer["job"]["id"]), ([], "j2"))


def hold_on(job, context, emit, stop, pause):
    while not stop.is_set():
        time.sleep(.02)


if __name__ == "__main__":
    unittest.main()

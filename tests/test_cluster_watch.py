"""The watchdog of a distributed run (#479): a run whose driver stopped calling its coordinator, or whose owner job ended, is ended with a
reason and its members released. Fake clock, fake peers, a fake runner; no network, no bucket, no window."""
import tempfile
import threading
import unittest
from pathlib import Path

import studio_cluster as sc
import studio_cluster_watch as watch
import studio_distributed as runs
from studio_queue import Job
from tests import diloco_support as support
from tests.fake_cluster import Clock, Net


class Runner:
    """What the watchdog uses of studio_queue.Runner: the lock, the jobs and `remove`."""

    def __init__(self, *jobs):
        self.lock, self.jobs, self.removed = threading.RLock(), list(jobs), []

    def remove(self, job_id):
        self.removed.append(job_id)


class Watchdog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.clock, self.net, self.deleted = Clock(), Net(), []
        self.net.handlers["b"] = self.handle
        self.addCleanup(runs.RUNS.clear)
        self.coordinator = support.build(Path(self.tmp.name), clock=self.clock, rounds=4)
        self.job = Job("pretrain_distributed", "Run", id="run1", status="running")
        self.runner = Runner(self.job)
        runs.RUNS["run1"] = self.run = runs.Run("run1", "Run", 1, 0., {}, "laptop", [
            runs.Machine("laptop", sc.LOCAL), runs.Machine("b", "b", peer_job="member-b", state="invited")])
        for node in ("laptop", "b"):
            self.coordinator.join(node)

    def handle(self, method, path, body):
        self.deleted.append((method, path))

    def check(self, **options):
        return watch.check(self.runner, {"run1": self.coordinator}, self.net.client_for, **options)

    def test_a_driver_that_keeps_calling_is_left_alone(self):
        for _ in range(5):
            self.clock.advance(watch.DRIVER_TIMEOUT * .9)
            self.coordinator.round_view(0, "laptop")  # a member waiting for its round
            self.assertEqual(self.check(), [])
        self.assertFalse(self.coordinator.finished)

    def test_a_progress_report_is_a_heartbeat_even_when_the_round_is_closing(self):
        self.clock.advance(watch.DRIVER_TIMEOUT - 1)
        self.coordinator.lock.acquire()  # a round closing: the report is refused
        try:
            reported = []
            thread = threading.Thread(target=lambda: reported.append(self.coordinator.progress(0, "laptop", 1, 0, "training")))
            thread.start()
            thread.join()
        finally:
            self.coordinator.lock.release()
        self.assertEqual(reported, [False])
        self.clock.advance(watch.DRIVER_TIMEOUT - 1)
        self.assertEqual(self.check(), [])

    def test_a_silent_driver_ends_the_run_with_the_round_and_releases_the_peers(self):
        self.coordinator.round_view(0, "b")  # the other machine is fine; the driver is not
        self.clock.advance(watch.DRIVER_TIMEOUT + 1)
        self.assertEqual(self.check(), ["run1"])
        reason = "Ended: the driver stopped responding at round 0, so the run was stopped."
        self.assertTrue(self.coordinator.finished)
        self.assertEqual(self.coordinator.events[-1]["text"], reason)
        self.assertEqual(self.runner.removed, ["run1"])
        self.assertEqual(self.job.params["ended_reason"], reason)  # the runner makes it the job's error once the worker has gone
        self.assertEqual(self.deleted, [("DELETE", "queue/jobs/member-b")])
        self.assertTrue(self.run.stopped)
        self.assertEqual(self.check(), [])  # once, not on every tick

    def test_the_peers_are_told_the_run_is_over_at_their_next_call(self):
        self.clock.advance(watch.DRIVER_TIMEOUT + 1)
        self.check()
        self.assertEqual(self.coordinator.round_view(0, "b")["state"], "finished")

    def test_a_driver_not_yet_heard_from_gets_the_timeout_from_the_start_of_the_run(self):
        coordinator = support.build(Path(self.tmp.name) / "other", clock=self.clock)
        self.clock.advance(watch.DRIVER_TIMEOUT - 1)
        self.assertEqual(watch.reason_for(coordinator, self.job, "laptop"), "")
        self.clock.advance(2)
        self.assertIn("responding at round 0", watch.reason_for(coordinator, self.job, "laptop"))

    def test_an_owner_job_that_failed_while_the_coordinator_goes_ends_the_run_with_its_error(self):
        self.job.status, self.job.error = "failed", "CUDA out of memory"
        self.assertEqual(self.check(), ["run1"])
        self.assertEqual(self.coordinator.events[-1]["text"], "Ended: the driver stopped at round 0 (CUDA out of memory), so the run was stopped.")
        self.assertEqual(self.runner.removed, [])  # its worker is gone already
        self.assertEqual(self.deleted, [("DELETE", "queue/jobs/member-b")])

    def test_a_peer_that_cannot_be_reached_does_not_stop_the_run_from_ending(self):
        self.net.down.add("b")
        self.clock.advance(watch.DRIVER_TIMEOUT + 1)
        self.assertEqual(self.check(), ["run1"])
        self.assertTrue(self.coordinator.finished)

    def test_a_finished_run_a_waiting_job_and_a_run_without_a_job_are_left_alone(self):
        self.clock.advance(watch.DRIVER_TIMEOUT * 3)
        self.job.status = "waiting"  # resuming: no worker yet
        self.assertEqual(self.check(), [])
        self.job.status = "running"
        self.runner.jobs = []
        self.assertEqual(self.check(), [])
        self.runner.jobs = [self.job]
        self.coordinator.stop()
        self.assertEqual(self.check(), [])

    def test_a_run_without_a_local_member_has_no_driver_to_watch(self):
        self.run.machines = [runs.Machine("b", "b")]
        self.clock.advance(watch.DRIVER_TIMEOUT * 3)
        self.assertIsNone(watch.driver_of(self.run))
        self.assertEqual(self.check(), [])


if __name__ == "__main__":
    unittest.main()

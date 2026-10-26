"""A restart of the app (a code update, a crash) does not end a queue run (#414): the job carries on from its checkpoint with no
Start pressed, a pause or stop the person made holds, a dead run no longer says "training", and the dev loop does not restart the
API under a running job."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import studio_jobs
import studio_model
import studio_worker
import tasks
from studio_ia.services import WorkspaceServices
from studio_queue import Job, Plan, Runner, reconcile_run, run_folders
from tests.test_restart_resume import attempt, make, services

ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)


def wait(condition, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(.02)
    return False


def noop(job, context, emit, stop, pause):
    pass


def crashed_queue(root, job, **extra):
    """queue.json as an app killed mid-job leaves it: the job still "running"."""
    path = Path(root) / "data" / "queue.json"
    first = Runner(path, {})
    first.set_plan(ANY)
    first.jobs = [job]
    job.status = "running"
    first._save()
    if extra:
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**data, **extra}), encoding="utf-8")
    return path


class AppKilledMidJob(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_the_job_continues_from_its_checkpoint_with_no_start_pressed(self):
        job = make("a1b2c3d4")
        attempt(self.root, job, stop_after=3)  # trained a few steps, saved last.npz
        out = Path(job.params["output_dir"])
        step = studio_model.load_checkpoint(out / "last.npz")["step"]
        record = json.loads((out / "run.json").read_text(encoding="utf-8"))
        (out / "run.json").write_text(json.dumps({**record, "status": "training"}), encoding="utf-8")  # the app died before it could say otherwise
        path = crashed_queue(self.root, Job(**{**job.__dict__, "params": dict(job.params)}))

        runner = Runner(path, studio_jobs.HANDLERS, prepare=lambda j: studio_jobs.prepare(j, services(self.root)), tick=.05)
        self.addCleanup(runner.shutdown)
        self.assertEqual(runner.jobs[0].status, "waiting")
        runner.start()  # what resume_queue_at_start does: nothing else is asked of the person
        self.assertTrue(wait(lambda: runner.jobs[0].status == "done"), runner.jobs[0].error)
        self.assertIn(f"continued from its own checkpoint at step {step}", runner.jobs[0].note)
        aside = json.loads((out.with_name(f"{out.name}-before-restart") / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((aside["status"], aside["interrupted"]), ("stopped", True))  # not "training"
        self.assertEqual(json.loads((out / "run.json").read_text(encoding="utf-8"))["status"], "completed")

    def test_a_queue_the_person_paused_stays_paused(self):
        path = crashed_queue(self.root, Job("quicktest", "t", id="p1"), user_paused=True)
        runner = Runner(path, {"quicktest": noop}, tick=.02)
        self.addCleanup(runner.shutdown)
        runner.start()
        time.sleep(.5)
        self.assertEqual((runner.jobs[0].status, runner.state().paused, len(runner.handles)), ("waiting", True, 0))
        runner.resume()
        self.assertTrue(wait(lambda: runner.jobs[0].status == "done"))

    def test_a_job_the_person_stopped_is_not_run_again(self):
        path = crashed_queue(self.root, Job("quicktest", "t", id="s1"), stopping="s1")
        runner = Runner(path, {"quicktest": noop}, tick=.02)
        self.addCleanup(runner.shutdown)
        self.assertEqual(runner.jobs[0].status, "stopped")
        runner.start()
        time.sleep(.3)
        self.assertEqual(runner.jobs[0].status, "stopped")

    def test_stopping_the_running_job_is_saved_before_it_ends(self):
        runner = Runner(self.root / "data" / "queue.json", {})
        job = runner.enqueue(Job("quicktest", "t", id="r1"))
        job.status = "running"
        runner.running = job
        runner.remove("r1")
        self.assertEqual(json.loads((self.root / "data" / "queue.json").read_text(encoding="utf-8"))["stopping"], "r1")

    def test_a_started_job_has_its_output_folder_saved(self):
        def prepare(job):
            job.params["output_dir"] = str(self.root / "runs" / "queue_x")
            return {"root": str(self.root), "run_dir": str(self.root / "runs")}

        runner = Runner(self.root / "data" / "queue.json", {"quicktest": noop}, prepare=prepare, tick=.02)
        runner.set_plan(ANY)
        self.addCleanup(runner.shutdown)
        saved = []
        original = runner._save
        runner._save = lambda: (saved.append(Path(runner.jobs[-1].params.get("output_dir", ""))), original())[1]
        job = runner.enqueue(Job("quicktest", "t", id="o1"))
        runner.start()
        self.assertTrue(wait(lambda: job.status == "done"))
        self.assertIn(self.root / "runs" / "queue_x", saved[:-1])  # saved after prepare, before the job ended
        self.assertEqual(run_folders(self.root, job)[0], self.root / "runs" / "queue_x")


class DeadRunRecord(unittest.TestCase):
    def test_only_a_record_saying_training_is_changed(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            folder = Path(directory)
            self.assertFalse(reconcile_run(folder))  # no run.json
            for status in ("completed", "failed", "stopped"):
                (folder / "run.json").write_text(json.dumps({"status": status}))
                self.assertFalse(reconcile_run(folder))
            (folder / "run.json").write_text(json.dumps({"status": "training", "step": 7}))
            self.assertTrue(reconcile_run(folder))
            self.assertEqual(json.loads((folder / "run.json").read_text()), {"status": "stopped", "step": 7, "interrupted": True})
            (folder / "run.json").write_text("{not json")
            self.assertFalse(reconcile_run(folder))


class ReapSweepsTemporaries(unittest.TestCase):
    def test_reap_deletes_spools_and_part_files_of_runs_that_are_not_running_and_leaves_a_running_job_alone(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            runner = Runner(root / "data" / "queue.json", {}, tick=.05)
            self.addCleanup(runner.shutdown)
            idle, busy, elsewhere = root / "runs" / "queue_a", root / "runs" / "queue_b", root / "out"
            for folder in (idle, busy, elsewhere):
                folder.mkdir(parents=True)
                (folder / "corpus-x.spool").write_bytes(b"rows")
                (folder / "tokens.part").write_bytes(b"half")
                (folder / "last.npz").write_bytes(b"keep")
            runner.jobs = [Job("quicktest", "idle", id="a", status="stopped", params={"output_dir": str(idle)}),
                           Job("quicktest", "waiting", id="c", params={"output_dir": str(elsewhere)})]
            runner.running = Job("quicktest", "busy", id="b", status="running", params={"output_dir": str(busy)})
            runner.jobs.append(runner.running)
            runner._reap()
            for folder in (idle, elsewhere):
                self.assertEqual(sorted(p.name for p in folder.iterdir()), ["last.npz"])
            self.assertEqual(sorted(p.name for p in busy.iterdir()), ["corpus-x.spool", "last.npz", "tokens.part"])


class ReapGrace(unittest.TestCase):
    def test_a_worker_that_leaves_by_itself_is_not_killed(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            spec = Path(directory) / "a.json"
            spec.write_text(json.dumps({"pid": 5}))
            polls, stopped, slept = iter([True, True, True, False, False]), [], []
            result = studio_worker.reap(directory, command_line=lambda pid: f"studio_worker run {spec}", alive=lambda pid: next(polls),
                                        terminate=stopped.append, grace=5, sleep=slept.append)
            self.assertEqual((result, stopped, len(slept)), ([], [], 2))
            self.assertFalse(spec.exists())

    def test_a_worker_still_there_after_the_grace_is_stopped(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            spec = Path(directory) / "a.json"
            spec.write_text(json.dumps({"pid": 5}))
            stopped = []
            result = studio_worker.reap(directory, command_line=lambda pid: f"studio_worker run {spec}", alive=lambda pid: True,
                                        terminate=stopped.append, grace=1, sleep=lambda s: None)
            self.assertEqual((result, stopped), ([5], [5]))


class ServicesStart(unittest.TestCase):
    def test_waiting_jobs_from_the_last_session_start_the_runner(self):
        from studio_core import Workspace
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            crashed_queue(directory, Job("quicktest", "t", id="w1"))
            svc = WorkspaceServices(Workspace(directory))
            self.addCleanup(svc.close)
            with mock.patch.object(svc.runner, "start") as start:
                svc.resume_queue_at_start()
            start.assert_called_once()

    def test_an_empty_queue_starts_nothing(self):
        from studio_core import Workspace
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            svc = WorkspaceServices(Workspace(directory))
            self.addCleanup(svc.close)
            svc.resume_queue_at_start()
            self.assertIsNone(svc.runner.thread)


class DevLoopDefersTheRestart(unittest.TestCase):
    def test_a_change_under_a_running_job_waits_for_it(self):
        step = tasks.dev_step
        self.assertEqual(step(True, True, True, True, True), "defer")
        self.assertEqual(step(True, True, True, True, False), "restart")  # the job ended
        self.assertEqual(step(False, True, True, True, True), "restart")  # the API is down already: nothing to protect
        self.assertEqual(step(True, True, False, True, True), "keep")
        self.assertEqual(step(True, False, True, True, True), "stop")

    def test_busy_means_a_live_worker_naming_its_own_spec(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            folder = Path(directory)
            self.assertFalse(tasks.training_busy(folder, alive=lambda p: True, command_line=lambda p: ""))
            spec = folder / "j.json"
            spec.write_text(json.dumps({"pid": 9}))

            def ours(pid):
                return f"python -m studio_worker run {spec}"
            self.assertTrue(tasks.training_busy(folder, alive=lambda p: True, command_line=ours))
            self.assertFalse(tasks.training_busy(folder, alive=lambda p: False, command_line=ours))  # the worker has ended
            self.assertFalse(tasks.training_busy(folder, alive=lambda p: True, command_line=lambda p: "some other program"))  # a reused pid


if __name__ == "__main__":
    unittest.main()

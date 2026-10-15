"""Queue jobs in a worker process (studio_worker): events pass through unchanged, params and errors come back, stop and a crash
end the job plainly, a worker whose app has gone stops by itself, and one left by a crashed app is stopped at the next launch."""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import studio_worker
from studio_queue import Job, Plan, Runner

ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)


def wait(condition, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(.01)
    return False


# --------------------------------------------------------------------------- handlers (they run in the worker process)

def chatty(job, context, emit, stop, pause):
    import numpy as np
    emit({"type": "metric", "step": 1, "total": 2, "lr": .5, "nested": {"a": [1, 2.5, None, "x"]}, "flag": True})
    emit({"type": "odd", "value": np.float32(1.5), "array": np.arange(3), "path": Path("/tmp/x"), "nan": float("nan")})
    job.params["result"] = {"pid": os.getpid(), "root": str(context.root), "run_dir": str(context.run_dir)}
    job.params["output_dir"] = "elsewhere"


def threads_seen(job, context, emit, stop, pause):
    import torch
    job.params["result"] = {"torch": torch.get_num_threads(), "env": os.environ.get("OMP_NUM_THREADS")}


def until_stopped(job, context, emit, stop, pause):
    """Steps until stopped, and writes 'stopped' once it notices, so a test can see a worker stop by itself."""
    step = 0
    while not stop.is_set():
        if not pause.is_set():
            step += 1
            emit({"type": "metric", "step": step, "total": 10 ** 6})
        time.sleep(.01)
    (Path(context.root) / "stopped").write_text("yes")


def dies(job, context, emit, stop, pause):
    emit({"type": "metric", "step": 1, "total": 2})
    os._exit(7)


def killed(job, context, emit, stop, pause):
    os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(10)


def prints(job, context, emit, stop, pause):
    print("stray output from a library")
    print("more", file=sys.stderr)
    emit({"type": "metric", "step": 1, "total": 1})


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_job(self, handler, kind="quicktest", runner_kw=None, **params):
        events = []
        runner = Runner(self.root / "data" / "queue.json", {kind: handler}, **{"tick": .05, **(runner_kw or {})})
        runner.set_plan(ANY)
        runner.subscribe(lambda k, payload: events.append((k, payload)))
        self.addCleanup(runner.shutdown)
        job = runner.enqueue(Job(kind, "t", params=params))
        runner.start()
        self.assertTrue(wait(lambda: job.status not in ("waiting", "running") and ("job", {"job": job.id, "status": job.status, "error": job.error}) in events))
        return job, events, runner

    def test_events_pass_through_unchanged_and_params_come_back(self):
        job, events, _ = self.run_job(chatty)
        self.assertEqual((job.status, job.error), ("done", ""))
        train = [p for k, p in events if k == "train"]
        self.assertEqual(train[0], {"job": job.id, "type": "metric", "step": 1, "total": 2, "lr": .5, "nested": {"a": [1, 2.5, None, "x"]}, "flag": True})
        odd = train[1]
        self.assertEqual((odd["value"], odd["array"], odd["path"]), (1.5, [0, 1, 2], str(Path("/tmp/x"))))  # numpy as numbers, the rest as text
        self.assertNotEqual(odd["nan"], odd["nan"])
        self.assertNotEqual(job.params["result"]["pid"], os.getpid())  # it ran in a process of its own
        self.assertEqual(job.params["result"]["root"], str(self.root))
        self.assertEqual(job.params["result"]["run_dir"], str(self.root / "runs"))
        self.assertEqual(job.params["output_dir"], "elsewhere")
        self.assertIn("max_batch_size", job.params)  # what the handler did not change is kept
        kinds = [p["status"] for k, p in events if k == "job"]
        self.assertEqual(kinds, ["running", "done"])
        self.assertEqual(list((self.root / "data" / "workers").glob("*.json")), [])  # the spec goes when the job ends

    def test_stray_prints_never_reach_the_events(self):
        job, events, _ = self.run_job(prints)
        self.assertEqual((job.status, [p for k, p in events if k == "train"]), ("done", [{"job": job.id, "type": "metric", "step": 1, "total": 1}]))
        self.assertIn("stray output from a library", (self.root / "logs" / "worker.log").read_text())

    def test_a_prepare_hook_gives_the_context_and_its_failure_fails_the_job(self):
        def prepare(job):
            if job.params.get("bad"):
                raise ValueError("Version 2 has not trained yet.")
            job.params["output_dir"] = "set by the app"
            return {"root": str(self.root / "elsewhere"), "run_dir": str(self.root / "r")}
        job, _, _ = self.run_job(chatty, runner_kw={"prepare": prepare})
        self.assertEqual((job.params["result"]["root"], job.params["result"]["run_dir"]), (str(self.root / "elsewhere"), str(self.root / "r")))
        job, events, _ = self.run_job(chatty, runner_kw={"prepare": prepare}, bad=True)
        self.assertEqual((job.status, job.error), ("failed", "Version 2 has not trained yet."))
        self.assertNotIn("result", job.params)

    def test_a_worker_that_dies_fails_the_job_with_a_sentence(self):
        job, events, _ = self.run_job(dies)
        self.assertEqual((job.status, job.error), ("failed", studio_worker.crash_message(7)))
        self.assertIn(("train", {"job": job.id, "type": "metric", "step": 1, "total": 2}), events)  # what it sent before is kept
        self.assertIn("ended with code 7", job.error)

    @unittest.skipIf(sys.platform == "win32", "SIGKILL is POSIX")
    def test_a_killed_worker_fails_the_job(self):
        job, _, _ = self.run_job(killed)
        self.assertEqual((job.status, job.error), ("failed", studio_worker.crash_message(-signal.SIGKILL)))

    def test_stop_pause_and_resume_reach_the_worker(self):
        events = []
        runner = Runner(self.root / "data" / "queue.json", {"quicktest": until_stopped}, tick=.05)
        runner.set_plan(ANY)
        runner.subscribe(lambda k, p: events.append(p.get("step")) if k == "train" else None)
        self.addCleanup(runner.shutdown)
        job = runner.enqueue(Job("quicktest", "t"))
        runner.start()
        self.assertTrue(wait(lambda: len(events) > 3))
        runner.pause()
        time.sleep(.3)
        frozen = len(events)
        time.sleep(.3)
        self.assertEqual(len(events), frozen)
        runner.resume()
        self.assertTrue(wait(lambda: len(events) > frozen + 3))
        runner.remove(job.id)
        self.assertTrue(wait(lambda: job.status == "stopped"))
        self.assertEqual((self.root / "stopped").read_text(), "yes")  # the handler saw the stop and returned: no kill
        self.assertEqual(job.error, "")

    def test_the_worker_has_one_core_fewer_than_the_computer(self):
        job, _, _ = self.run_job(threads_seen)
        seen = job.params["result"]
        self.assertLessEqual(seen["torch"], int(seen["env"]))  # torch keeps under the cap it starts with (on Windows it may take fewer: physical cores)
        self.assertLessEqual(int(seen["env"]), studio_worker.threads())  # studio_model's own lower default of 4 may stand

    def test_thread_env_caps_but_keeps_a_lower_setting(self):
        self.assertEqual(studio_worker.threads(1), 1)
        self.assertEqual(studio_worker.threads(8), 7)
        env = studio_worker.thread_env({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "64", "OPENBLAS_NUM_THREADS": "x", "PATH": "p"}, cores=8)
        self.assertEqual(env, {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "7", "OPENBLAS_NUM_THREADS": "7", "PATH": "p"})

    def test_a_worker_whose_app_has_gone_stops_its_job(self):
        events, ended = [], threading.Event()
        handle = studio_worker.Handle(Job("quicktest", "t"), until_stopped, {"root": str(self.root), "run_dir": str(self.root / "runs")},
                                      events.append, self.root / "data" / "workers", self.root / "logs" / "worker.log", ended.set)
        self.assertTrue(wait(lambda: events))
        handle._conn.shutdown(socket.SHUT_RDWR)  # as when the app's process ends: the worker sees its socket close
        self.assertTrue(ended.wait(20))
        self.assertEqual((self.root / "stopped").read_text(), "yes")
        self.assertIsNotNone(handle.proc.returncode)

    def test_quit_stops_the_running_job_and_it_runs_again_next_time(self):
        runner = Runner(self.root / "data" / "queue.json", {"quicktest": until_stopped}, tick=.05)
        runner.set_plan(ANY)
        job = runner.enqueue(Job("quicktest", "t"))
        runner.start()
        self.assertTrue(wait(lambda: runner.handles))
        handle = next(iter(runner.handles))
        runner.shutdown(timeout=20)
        self.assertTrue(handle.done.is_set())
        self.assertEqual(job.status, "waiting")
        self.assertEqual(Runner(self.root / "data" / "queue.json", {}).jobs[0].status, "waiting")


class ReapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "data" / "workers"
        self.folder.mkdir(parents=True)

    def spec(self, name, pid):
        path = self.folder / f"{name}.json"
        path.write_text(json.dumps({"pid": pid} if pid else {}))
        return path

    def test_only_a_process_naming_its_own_spec_is_stopped(self):
        ours, gone, unread = self.spec("a", 11), self.spec("c", 13), self.spec("d", None)
        self.spec("b", 12)  # its pid now belongs to another program
        lines = {11: f"python -m studio_worker run {ours}", 12: "/usr/bin/some-other-program", 13: f"python -m studio_worker run {gone}"}
        stopped = []
        result = studio_worker.reap(self.folder, command_line=lines.get, alive=lambda pid: pid != 13, terminate=stopped.append)
        self.assertEqual((result, stopped), ([11], [11]))
        self.assertEqual(list(self.folder.glob("*.json")), [])
        self.assertFalse(unread.exists())

    def test_a_crashed_apps_worker_is_stopped_when_the_queue_next_starts(self):
        spec = self.folder / "left.json"
        stray = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "studio_worker", "run", str(spec)])
        self.addCleanup(stray.kill)
        spec.write_text(json.dumps({"pid": stray.pid}))
        runner = Runner(self.folder.parent / "queue.json", {}, tick=.05)
        self.addCleanup(runner.shutdown)
        runner.start()
        self.assertTrue(wait(lambda: stray.poll() is not None))
        self.assertTrue(wait(lambda: not spec.exists()))


if __name__ == "__main__":
    unittest.main()

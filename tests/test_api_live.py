import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from studio_api import __main__ as entry
from studio_api import live

RUNNING = {"job": {"id": "j1", "label": "pretrain a", "status": "running"}}
IDLE = {"job": None, "up_next": [{"id": "j2", "status": "waiting"}]}


def run(state, **extra):
    return {"id": "r1", "label": "diloco", "state": state, "round": 4, **extra}


class LiveReasonTests(unittest.TestCase):
    def test_idle(self):
        self.assertIsNone(live.live_reason(IDLE, []))
        self.assertIsNone(live.live_reason(None, None))

    def test_a_running_queue_job_is_live(self):
        self.assertIn("pretrain a", live.live_reason(RUNNING, []))

    def test_a_waiting_job_is_not(self):
        self.assertIsNone(live.live_reason({"job": {"id": "j", "status": "waiting"}}, []))

    def test_active_distributed_run_is_live(self):
        for state in ("starting", "waiting", "running"):
            self.assertIn(state, live.live_reason(IDLE, [run(state)]))

    def test_ended_or_queued_runs_are_not(self):
        for state in ("queued", "finished", "stopped", "failed"):
            self.assertIsNone(live.live_reason(IDLE, [run(state)]))

    def test_another_nodes_run_is_not(self):
        self.assertIsNone(live.live_reason(IDLE, [run("running", age_s=12.5, origin="other")]))
        self.assertIsNone(live.live_reason(IDLE, [run("running", origin_down=True)]))
        self.assertIn("running", live.live_reason(IDLE, [run("running", age_s=0.)]))


class MarkerTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "marker"
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def test_write_read_and_remove(self):
        self.assertIsNone(live.marker_reason(self.path))
        live.write_marker(self.path, "run is live")
        self.assertEqual(live.marker_reason(self.path), "run is live")
        live.write_marker(self.path, None)
        self.assertIsNone(live.marker_reason(self.path))
        live.write_marker(self.path, None)  # removing twice is fine

    def test_a_stale_marker_is_a_crashed_process(self):
        live.write_marker(self.path, "run is live")
        mtime = self.path.stat().st_mtime
        self.assertIsNone(live.marker_reason(self.path, now=mtime + live.STALE_S + 1))
        self.assertEqual(live.marker_reason(self.path, now=mtime + live.STALE_S - 1), "run is live")


class ProbeTests(unittest.TestCase):
    def test_probe_reads_both_routes(self):
        answers = {"http://x/queue": RUNNING, "http://x/distributed/runs": []}
        self.assertIn("pretrain a", live.probe("http://x", answers.__getitem__))
        answers["http://x/queue"] = IDLE
        answers["http://x/distributed/runs"] = [run("running")]
        self.assertIn("diloco", live.probe("http://x", answers.__getitem__))

    def test_exit_codes(self):
        def code(**kw):
            with mock.patch.object(live, "probe", **kw), mock.patch("builtins.print"):
                return live.main(["http://x/"])

        self.assertEqual(code(return_value="busy"), 0)
        self.assertEqual(code(return_value=None), 1)
        self.assertEqual(code(side_effect=ConnectionRefusedError()), 2)  # API down: the hook pulls as before
        self.assertEqual(code(side_effect=ValueError("not json")), 2)


class DeferredReloadTests(unittest.TestCase):
    def test_holds_while_live_and_releases_once_idle(self):
        reloader, state = entry.DeferredReload(), {"why": "busy"}
        file = Path("a.py")
        with self.assertLogs("uvicorn.error", "WARNING") as logged:
            self.assertIsNone(reloader.hold([file], lambda: state["why"]))
            self.assertIsNone(reloader.hold([file], lambda: state["why"]))  # StatReload reports the file again: one line only
            self.assertIsNone(reloader.hold(None, lambda: state["why"]))  # watchfiles reports nothing more: still held
        self.assertEqual(len(logged.records), 1)
        self.assertIn("Reload waiting: busy", logged.output[0])
        state["why"] = None
        self.assertEqual(reloader.hold(None, lambda: state["why"]), [file])
        self.assertIsNone(reloader.hold(None, lambda: state["why"]))  # applied once

    def test_idle_changes_restart_at_once(self):
        self.assertEqual(entry.DeferredReload().hold([Path("a.py")], lambda: None), [Path("a.py")])
        self.assertIsNone(entry.DeferredReload().hold(None, lambda: None))

    def test_the_reloader_reads_the_marker(self):
        from uvicorn.supervisors import ChangeReload

        marker = Path(tempfile.mkdtemp()) / "marker"
        self.addCleanup(lambda: marker.unlink(missing_ok=True))
        reloader = object.__new__(entry.deferred_reloader())
        with mock.patch.dict(os.environ, {live.MARKER_ENV: str(marker)}), \
                mock.patch.object(ChangeReload, "should_restart", return_value=[Path("a.py")]):
            live.write_marker(marker, "busy")
            with self.assertLogs("uvicorn.error", "WARNING"):
                self.assertIsNone(reloader.should_restart())
            live.write_marker(marker, None)
            self.assertEqual(reloader.should_restart(), [Path("a.py")])


class KeepMarkerTests(unittest.TestCase):
    def test_marker_follows_the_decision_and_survives_a_failed_read(self):
        marker = Path(tempfile.mkdtemp()) / "marker"
        self.addCleanup(lambda: marker.unlink(missing_ok=True))
        answers, stop, seen = iter(["busy", RuntimeError("boom"), None]), threading.Event(), []

        def reason(_services):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        def after_tick(_):
            seen.append(marker.exists())
            if len(seen) == 3:
                stop.set()

        with mock.patch("builtins.print"), mock.patch.object(stop, "wait", side_effect=after_tick):
            entry.keep_marker(None, marker, every=0, reason=reason, stop=stop)
        self.assertEqual(seen, [True, True, False])  # written, kept through the failed read, removed when idle


class LocalReasonTests(unittest.TestCase):
    def test_over_the_fake_services(self):
        from studio_ia.services import FakeServices

        services = FakeServices(sleep=lambda seconds: None)
        self.assertIsNone(entry.local_reason(services))
        services.demo_run()
        self.assertIn("running", entry.local_reason(services))

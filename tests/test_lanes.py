"""The shell's two worker lanes: a slow job never delays a quick one, each lane keeps its own order, and every busy lane holds the collector."""
import gc
import tempfile
import threading
import time
import unittest
from unittest import mock

from studio_core import Workspace
from studio_ia import threads
from studio_ia.services import WorkspaceServices
from studio_ia.shell import Studio

WAIT = 10  # seconds a test gives a lane before failing instead of hanging


def until(done):
    end = time.monotonic() + WAIT
    while not done() and time.monotonic() < end:
        time.sleep(.005)


def collect(lane, count):
    """Tk-thread side: poll the lane the way the shell does until `count` results are in."""
    got = []

    def poll():
        got.extend(lane.finished())
        return len(got) >= count
    until(poll)
    return got


class LaneTest(unittest.TestCase):
    def setUp(self):
        threads.attach()
        self.addCleanup(threads.detach)
        self.fast, self.slow = threads.Lane(), threads.Lane()
        self.addCleanup(self.fast.stop)
        self.addCleanup(self.slow.stop)

    def test_a_slow_job_does_not_delay_a_quick_one(self):
        started, go = threading.Event(), threading.Event()
        self.addCleanup(go.set)
        self.slow.submit(lambda: started.set() or go.wait(WAIT), "scan")
        self.assertTrue(started.wait(WAIT))
        self.fast.submit(lambda: "quick", "read")
        self.assertEqual(collect(self.fast, 1), [("read", "quick", None)])  # answered while the scan still runs
        self.assertEqual(self.slow.finished(), [])
        go.set()
        self.assertEqual(collect(self.slow, 1), [("scan", True, None)])

    def test_each_lane_runs_its_jobs_in_the_order_they_were_submitted(self):
        ran = {"fast": [], "slow": []}
        for i in range(20):
            self.fast.submit(lambda i=i: ran["fast"].append(i) or i, i)
            self.slow.submit(lambda i=i: ran["slow"].append(i) or i, i)
        self.assertEqual([tag for tag, _, _ in collect(self.fast, 20)], list(range(20)))
        self.assertEqual([tag for tag, _, _ in collect(self.slow, 20)], list(range(20)))
        self.assertEqual(ran, {"fast": list(range(20)), "slow": list(range(20))})

    def test_a_failed_job_comes_back_as_an_error_and_the_lane_goes_on(self):
        def fail():
            raise ValueError("no")
        self.fast.submit(fail, "bad")
        self.fast.submit(lambda: 1, "good")
        (bad, _, error), good = collect(self.fast, 2)
        self.assertEqual((bad, type(error), good), ("bad", ValueError, ("good", 1, None)))

    def test_each_busy_lane_is_one_hold_on_the_collector(self):
        go = threading.Event()
        self.addCleanup(go.set)
        self.fast.submit(lambda: go.wait(WAIT))
        self.assertFalse(gc.isenabled())
        self.slow.submit(lambda: go.wait(WAIT))
        self.slow.submit(lambda: 2)  # more jobs on a busy lane are not more holds
        self.assertEqual(threads._holds, 2)
        go.set()
        collect(self.fast, 1)
        self.assertEqual(threads._holds, 1)
        self.assertFalse(gc.isenabled())  # the slow lane still holds it
        collect(self.slow, 2)
        self.assertEqual(threads._holds, 0)
        self.assertTrue(gc.isenabled())

    def test_a_lane_holds_until_the_tk_thread_has_taken_its_last_result(self):
        self.fast.submit(lambda: 1)
        until(lambda: not self.fast._done.empty())
        self.assertFalse(gc.isenabled())  # the job is done, but only the Tk thread gives the collector back
        collect(self.fast, 1)
        self.assertTrue(gc.isenabled())


class FakeShell:
    """Studio's job methods on a stand-in, so the wiring runs without a window."""
    _submit, _poll, quietly, background, slow = Studio._submit, Studio._poll, Studio.quietly, Studio.background, Studio.slow

    def __init__(self):
        self.lanes = {"fast": threads.Lane(), "slow": threads.Lane()}
        self._waiting = self._quiet = 0
        self.services = mock.Mock(loading=("", None))
        self.loader, self.load_title, self.load_text, self.load_bar = (mock.Mock() for _ in range(4))
        self.errors = []

    def after(self, _ms, _callback):
        """The real one polls again in 60 ms; the tests poll themselves."""

    def report_callback_exception(self, kind, error, trace):
        self.errors.append(error)

    def drain(self):
        until(lambda: self._poll() or not self._waiting + self._quiet)


class ShellLanesTest(unittest.TestCase):
    def setUp(self):
        threads.attach()
        self.addCleanup(threads.detach)
        self.shell = FakeShell()
        for lane in self.shell.lanes.values():
            self.addCleanup(lane.stop)

    def test_quick_reads_go_fast_and_the_panel_waits_for_the_slow_lane(self):
        go, order = threading.Event(), []
        self.addCleanup(go.set)
        self.shell.background(lambda: go.wait(WAIT), lambda _: order.append("import"), "Importing")
        self.shell.quietly(lambda: "step", order.append)
        self.assertEqual((self.shell._waiting, self.shell._quiet), (1, 1))
        until(lambda: self.shell._poll() or "step" in order)
        self.assertEqual(order, ["step"])
        self.shell.loader.place_forget.assert_not_called()  # the panel stays up until the slow job answers
        self.assertFalse(gc.isenabled())
        go.set()
        self.shell.drain()
        self.assertEqual(order, ["step", "import"])
        self.shell.loader.place_forget.assert_called_once()
        self.assertEqual((self.shell._waiting, self.shell._quiet, threads._holds), (0, 0, 0))
        self.assertTrue(gc.isenabled())

    def test_quietly_can_be_sent_to_the_slow_lane_where_it_queues_behind_a_scan(self):
        go, order = threading.Event(), []
        self.addCleanup(go.set)
        self.shell.quietly(lambda: go.wait(WAIT), lambda _: order.append("scan"), lane="slow")
        self.shell.quietly(lambda: 1, lambda _: order.append("recount"), lane="slow")
        self.shell.quietly(lambda: 1, lambda _: order.append("read"))
        until(lambda: self.shell._poll() or "read" in order)
        self.assertEqual(order, ["read"])
        go.set()
        self.shell.drain()
        self.assertEqual(order, ["read", "scan", "recount"])

    def test_an_error_is_reported_on_the_tk_thread(self):
        def fail():
            raise ValueError("no")
        self.shell.quietly(fail, lambda _: None)
        self.shell.drain()
        self.assertEqual(([str(e) for e in self.shell.errors], self.shell._quiet, threads._holds), (["no"], 0, 0))


class SharedCacheTest(unittest.TestCase):
    """Two lanes can ask for the same cached read, or for different ones, at once."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = WorkspaceServices(Workspace(self.tmp.name))
        self.addCleanup(self.svc.close)

    def test_an_entry_both_lanes_ask_for_is_made_once(self):
        made, results, go = [], [], threading.Event()
        self.addCleanup(go.set)

        def make():
            made.append(1)
            go.wait(WAIT)
            return "summary"
        pair = [threading.Thread(target=lambda: results.append(self.svc._fresh("k", make))) for _ in range(2)]
        for t in pair:
            t.start()
        until(lambda: made)
        time.sleep(.1)  # the second is in by now: it waits on the first instead of making its own
        go.set()
        for t in pair:
            t.join(WAIT)
        self.assertEqual((made, results), ([1], ["summary", "summary"]))

    def test_a_long_make_does_not_hold_up_another_entry(self):
        inside, go = threading.Event(), threading.Event()

        def long():
            inside.set()
            go.wait(WAIT)
            return "late"
        slow = threading.Thread(target=lambda: self.svc._cached(("band", "a"), long))
        slow.start()
        self.addCleanup(slow.join, WAIT)
        self.addCleanup(go.set)  # cleanups run last-in first: release it, then wait for it
        self.assertTrue(inside.wait(WAIT))
        got = []
        quick = threading.Thread(target=lambda: got.append(self.svc._cached(("band", "b"), lambda: "now")))
        quick.start()
        quick.join(WAIT)
        self.assertEqual(got, ["now"])  # while "a" is still being made


if __name__ == "__main__":
    unittest.main()

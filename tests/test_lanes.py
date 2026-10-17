"""The shell's two worker lanes: a slow job never delays a quick one, each lane keeps its own order, and every busy lane holds the collector."""
import tempfile
import threading
import time
import unittest

from studio_core import Workspace
from studio_ia.services import WorkspaceServices

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

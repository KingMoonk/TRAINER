"""Background threads keep the collector off until the Tk thread settles them; no thread is started any other way."""
import gc
import re
import threading
import unittest
from pathlib import Path

from studio_ia import threads

ROOT = Path(__file__).resolve().parent.parent


class ThreadsTest(unittest.TestCase):
    def setUp(self):
        threads.attach()
        self.addCleanup(threads.detach)

    def test_collector_stays_off_until_settled(self):
        go = threading.Event()
        t = threads.start(go.wait)
        self.assertFalse(gc.isenabled())
        go.set()
        t.join()
        self.assertFalse(gc.isenabled())  # finished, but only the Tk thread gives the collector back
        threads.settle()
        self.assertTrue(gc.isenabled())

    def test_holds_are_counted(self):
        threads.hold()
        go = threading.Event()
        t = threads.start(go.wait)
        go.set()
        t.join()
        threads.settle()
        self.assertFalse(gc.isenabled())  # the shell's own hold is still there
        threads.release()
        self.assertTrue(gc.isenabled())

    def test_unattached_never_disables(self):
        threads.detach()
        threads.start(lambda: None).join()
        self.assertTrue(gc.isenabled())

    def test_detach_keeps_running_threads_held(self):
        """A destroyed app's threads still run: turning the collector back on then would free its Tk objects off the Tk thread."""
        go = threading.Event()
        t = threads.start(go.wait)
        threads.detach()
        self.assertFalse(gc.isenabled())
        go.set()
        t.join()
        threads.attach()  # the next app gives it back
        threads.settle()
        self.assertTrue(gc.isenabled())

    def test_unattached_thread_never_releases_a_later_hold(self):
        threads.detach()
        threads.start(lambda: None).join()
        threads.attach()
        threads.hold()
        threads.settle()
        self.assertFalse(gc.isenabled())
        threads.release()
        self.assertTrue(gc.isenabled())

    def test_no_bare_thread_starts(self):
        """A thread started outside threads.start() can free Tk objects off the Tk thread: crashes CI at random."""
        allowed = {"studio_ia/threads.py", "studio_ia/shell.py"}  # shell: its one long-lived worker holds per job
        bare = [p.relative_to(ROOT).as_posix() for p in (ROOT / "studio_ia").rglob("*.py")
                if p.relative_to(ROOT).as_posix() not in allowed and re.search(r"threading\.Thread\(", p.read_text(encoding="utf-8"))]
        self.assertEqual(bare, [], "start these with studio_ia.threads.start")


if __name__ == "__main__":
    unittest.main()

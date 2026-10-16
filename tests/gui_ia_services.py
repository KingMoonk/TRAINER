"""The shell on FixtureServices visits all 19 routes without error (skips without a display)."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import tempfile
import time
import tkinter as tk
import unittest

from studio_ia.services import FixtureServices
from studio_ia.shell import ROUTES, Studio


class ShellOnFixtureTest(unittest.TestCase):
    def test_all_routes(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())  # removed last, after the app and services close
        svc = FixtureServices(tmp)
        self.addCleanup(svc.close)
        try:
            app = Studio(svc)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(app.destroy)
        self.assertEqual(len(ROUTES), 19)
        for route in ROUTES:
            app.show(route)
            app.update()
        for _ in range(3000):  # the "versions" snapshot is weighed on a worker, which scores the held-back text first
            if "Chat uses version" in app.strip_text:
                break
            app.update()
            time.sleep(.02)
        self.assertIn("Chat uses version", app.strip_text)


if __name__ == "__main__":
    unittest.main()

"""The shell on FixtureServices visits all 16 routes without error (skips without a display)."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import tempfile
import tkinter as tk
import unittest

from studio_ia.services import FixtureServices
from studio_ia.shell import ROUTES, Studio


class ShellOnFixtureTest(unittest.TestCase):
    def test_all_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = FixtureServices(tmp)
            self.addCleanup(svc.close)
            try:
                app = Studio(svc)
            except tk.TclError as error:
                self.skipTest(f"no display: {error}")
            self.addCleanup(app.destroy)
            self.assertEqual(len(ROUTES), 16)
            for route in ROUTES:
                app.show(route)
                app.update()
            self.assertIn("Chat uses version", app.strip_text)


if __name__ == "__main__":
    unittest.main()

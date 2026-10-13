"""The 3D page renders on FakeServices, switches mode and zoom, and starts/stops the (fake) viewer."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import tkinter as tk
import unittest

from studio_ia.pages import inside3d
from studio_ia.services import FakeServices
from studio_ia.shell import Studio
from tests.gui_ia_shell import texts


class Inside3DPage(unittest.TestCase):
    def setUp(self):
        try:
            self.svc = FakeServices()
            self.app = Studio(self.svc)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)

    def test_empty_state_and_controls(self):
        page = self.app.show("inside3d")
        self.app.update()
        self.assertIsInstance(page, inside3d.Inside3DPage)
        self.assertTrue(any("demo model" in t for t in texts(page.body)))
        page.pick_mode("Train")
        page.pick_level("3 · Weights")
        self.assertEqual(self.svc.viewer_log, [])  # viewer not open: nothing sent
        page.launch.invoke()
        self.assertEqual(self.svc.viewer_log[0], ("start", {"level": "Weights", "mode": "Train"}))
        page.pick_mode("Ride a position")
        self.assertEqual(self.svc.viewer_log[-1][1]["mode"], "Ride a token")
        page.launch.invoke()
        self.assertEqual(self.svc.viewer_log[-1][0], "stop")


if __name__ == "__main__":
    unittest.main()

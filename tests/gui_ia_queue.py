"""Queue page Storage card: the confirm dialog is patched; CI-only."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import tkinter as tk
import unittest
from unittest import mock

from studio_ia.services import FakeServices
from studio_ia.shell import Studio


class Fake(FakeServices):
    def storage(self, days=None, data=False):
        return (2, 3_000_000)


class StorageCardTest(unittest.TestCase):
    def setUp(self):
        self.fake = Fake()
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)
        self.page = self.app.show("queue")
        self.app.update()

    def test_delete_asks_then_cleans_default_30_days(self):
        with mock.patch("tkinter.messagebox.askyesno", return_value=False) as ask:
            self.page._delete()
        self.assertIn("cannot be undone", ask.call_args.args[1])
        self.assertFalse(hasattr(self.fake, "cleaned"))
        with mock.patch("tkinter.messagebox.askyesno", return_value=True):
            self.page._delete()
        self.assertEqual(self.fake.cleaned, (30, False))


if __name__ == "__main__":
    unittest.main()

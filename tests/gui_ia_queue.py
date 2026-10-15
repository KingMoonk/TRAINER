"""Storage page (Resources) and the Queue page's Up next list: the confirm dialog is patched; CI-only."""
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
        self.page = self.app.show("storage")
        self.app.update()

    def test_delete_asks_then_cleans_default_30_days(self):
        with mock.patch("tkinter.messagebox.askyesno", return_value=False) as ask:
            self.page._delete()
        self.assertIn("cannot be undone", ask.call_args.args[1])
        self.assertFalse(hasattr(self.fake, "cleaned"))
        with mock.patch("tkinter.messagebox.askyesno", return_value=True):
            self.page._delete()
        self.assertEqual(self.fake.cleaned, (30, False))

    def test_queue_page_has_no_storage_card(self):
        page = self.app.show("queue")
        self.app.update()
        self.assertFalse(hasattr(page, "delete_btn"))

    def test_your_data_links_to_storage(self):
        self.app.show("your_data")
        self.app.update()
        self.app.pages["your_data"].navigate("storage")
        self.app.update()
        self.assertEqual(self.app.route, "storage")



class PassivePageTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServices()
        self.fake.passive_sessions = [{"id": "p1", "status": "running", "created": "2026-10-15T08:30:00+00:00", "stop": "Stops after 8 hours.",
                                       "trainer": "Running", "watchdog": "Running", "restarts": 0, "heard": 30.0, "error": ""}]
        self.fake.passive_signin = True
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)
        self.page = self.app.show("passive")
        self.app.update()

    def test_stop_asks_first_and_sign_in_can_be_removed(self):
        with mock.patch("tkinter.messagebox.askyesno", return_value=False):
            self.page._stop("p1")
        self.assertEqual(self.fake.passive()["sessions"][0]["status"], "running")
        with mock.patch("tkinter.messagebox.askyesno", return_value=True):
            self.page._stop("p1")
        self.assertEqual(self.fake.passive()["sessions"][0]["status"], "stopped")
        self.page._remove_signin()
        self.assertFalse(self.fake.passive()["signin"])

class UpNextTest(unittest.TestCase):
    def setUp(self):
        from studio_ia.services import Job
        self.fake = FakeServices()
        for n in "abc":
            self.fake.enqueue(Job(n, "quicktest", f"job {n}"))
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)
        self.page = self.app.show("queue")
        self.app.update()

    def order(self):
        return [(j.label, j.status) for j in self.fake.queue().up_next]

    def press(self, row, label):
        from tkinter import ttk
        button = next(w for w in self.page._rows[row].winfo_children() if isinstance(w, ttk.Button) and w.cget("text") == label)
        button.invoke()
        self.app.update()

    def test_each_item_can_be_reordered_held_and_removed(self):
        self.press(2, "Run next")
        self.assertEqual(self.order(), [("job c", "waiting"), ("job a", "waiting"), ("job b", "waiting")])
        self.press(0, "▼")
        self.assertEqual([x for x, _ in self.order()], ["job a", "job c", "job b"])
        self.press(1, "Hold")
        self.assertEqual(self.order()[1], ("job c", "held"))
        self.press(1, "Release")
        self.assertEqual(self.order()[1], ("job c", "waiting"))
        self.press(2, "Remove")
        self.assertEqual([x for x, _ in self.order()], ["job a", "job c"])

    def test_drag_moves_a_row(self):
        rows = self.page._rows
        self.page._drag_start(0)
        last = rows[-1]
        drop = type("E", (), {"y_root": last.winfo_rooty() + last.winfo_height()})()
        self.page._drag_drop(self.fake.queue().up_next[0].id, drop)
        self.app.update()
        self.assertEqual([j.label for j in self.fake.queue().up_next], ["job b", "job c", "job a"])


if __name__ == "__main__":
    unittest.main()

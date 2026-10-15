"""Shell check: all 16 routes render their h1, and a queue event updates the machine strip."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import time
import unittest
import tkinter as tk
from tkinter import ttk

from studio_ia.services import FakeServices, Job, QueueState
from studio_ia.shell import ROUTES, Studio


def texts(widget):
    own = [str(widget.cget("text"))] if isinstance(widget, (tk.Label, ttk.Label)) else []
    return own + [t for child in widget.winfo_children() for t in texts(child)]


class ShellTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServices()
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)

    def test_all_routes_show_their_h1(self):
        self.assertEqual(len(ROUTES), 16)
        for route, spec in ROUTES.items():
            page = self.app.show(route)
            self.app.update()
            self.assertIn(spec.title, texts(page), route)
            self.assertIn(spec.lead, texts(page), route)

    def test_sidebar_rows_for_queue_and_data_follow_the_page(self):
        from studio_ia.theme import C
        for route, area in (("queue", "machine"), ("your_data", "data"), ("make_data", "data"), ("builder", "designer")):
            self.app.show(route)
            self.app.update()
            on = [key for key, (row, *_rest) in self.app._area_rows.items() if row.cget("bg") == C["violet_bg"]]
            self.assertEqual(on, [area], route)

    def test_queue_event_updates_machine_strip(self):
        self.assertIn("Nothing is training", self.app.strip_text)
        self.fake.set_queue(QueueState(job=Job("1", "pretrain", "Version 5 catch-up"), step=50, total=200,
                                       phase="steady", eta_s=600, up_next=[Job("2", "quicktest")]))
        self.app.update()
        self.assertIn("Version 5 catch-up", self.app.strip_text)
        self.assertIn("1 queued", self.app.strip_text)
        self.assertAlmostEqual(self.app.strip_bar.fraction, 0.25)

    def wait_for(self, what):
        """Pump Tk until `what()` holds: the Next row is filled in by the worker thread."""
        for _ in range(100):
            self.app.update()
            if what():
                return
            time.sleep(.02)
        self.fail("timed out")

    def test_next_row_follows_the_workspace_and_opens_its_page(self):
        self.wait_for(lambda: "Design your first model" in self.app.next_text)
        self.fake.save_version({}, "")
        self.wait_for(lambda: "Queue the training" in self.app.next_text)
        self.app._next_click()
        self.assertEqual(self.app.route, "builder")
        self.fake._rows()["text"].clear()
        self.fake.emit("data")
        self.wait_for(lambda: "Add some text" in self.app.next_text)
        self.app._next_click()
        self.assertEqual(self.app.route, "your_data")

    def test_widgets_render(self):
        from studio_ia import widgets as w
        f = ttk.Frame(self.app.main)
        f.pack()
        plot = w.BpcPlot(f, 640, 300)
        plot.pack()
        for d in ({}, dict(steps=[(i, 5 - i / 50) for i in range(100)], smooth=[(i, 5 - i / 50) for i in range(100)],
                           heldout=[(50, 3.9)], refs=[("uniform", 6.6), ("counting words", 4.9), ("previous best", 3.5)],
                           total=200, progress=100, lr=[(i, 1) for i in range(0, 200, 10)],
                           phases=[(0, 20, "warm-up"), (20, 160, "steady"), (160, 200, "cool-down")], band=(3.4, 3.6))):
            plot.set(**d)
            self.app.update()
        grid = w.HeatGrid(f)
        grid.set([[0, .5, None], [1, .2, .3]], ["a", "b"], ["x", "y", "z"])
        guess = w.GuessChips(f, ["lower", "same", "higher"])
        self.assertIsNone(guess.value)
        guess.pick("same")
        toggle = w.Toggle(f, "x")
        toggle.flip()
        kofn = w.KofN(f, None, None, "passed")
        self.assertEqual(kofn.cget("text"), "not counted yet · passed")
        w.HowItWorks(f).toggle()
        w.StateTray(f, "live", "live", "text")
        w.Chip(f, "x", "training")
        w.Segmented(f, ["a", "b"], "a")
        w.Card(f)
        w.Progress(f).set(0.5)
        btn = w.button(f, "Go", lambda: None, primary=True)
        btn.pack()
        self.app.update()
        self.assertGreaterEqual(btn.winfo_reqheight(), 44)


if __name__ == "__main__":
    unittest.main()

"""The five look-inside pages render on FakeServices; step buttons change the grid; head lab experiments run."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import time
import tkinter as tk
import unittest

from studio_ia.services import FakeServices
from studio_ia.shell import Studio
from tests.gui_ia_shell import texts


class InsidePages(unittest.TestCase):
    def setUp(self):
        try:
            self.app = Studio(FakeServices())
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)

    def wait(self, condition, seconds=3):
        end = time.time() + seconds
        while not condition() and time.time() < end:
            self.app.update()
            time.sleep(.02)
        self.assertTrue(condition())

    def test_pages_render_with_numbers(self):
        for route in ("trace", "head_zoom", "value_map", "loss", "head_lab"):
            page = self.app.show(route)
            self.app.update()
            self.assertEqual(type(page).__name__ != "Page", True, route)
            self.assertNotIn("Numbers appear when the engine has run your prompt.", [t for t in texts(page.body)], route)
        self.assertEqual(len(self.app.show("trace").cells), 4)

    def test_step_buttons_change_the_grid(self):
        page = self.app.show("head_zoom")
        self.app.update()
        grids = []
        for button in page.step_buttons:
            button.invoke()
            self.app.update()
            grids.append([row[:] for row in page.grid_view.texts])
        self.assertEqual(len({str(g) for g in grids[:4]}), 4)  # step 5 reuses the step-4 weights and highlights a column
        self.assertEqual(grids[2][-1][0], "−∞")                       # step 3 masks the lower-left
        for j in range(len(grids[3])):                                  # step 4: each column is a distribution
            self.assertAlmostEqual(sum(float(grids[3][i][j]) for i in range(len(grids[3]))), 1, delta=.05)
        self.assertEqual(page.banner.winfo_ismapped(), False)

    def test_value_map_resplits_heads(self):
        page = self.app.show("value_map")
        self.app.update()
        self.assertIn("head_dim = 256 ÷ 4 = 64", [t for t in texts(page.body)] + ["head_dim = 256 ÷ 4 = 64"] if False else texts(page.body))
        page.set_heads(8)
        self.app.update()
        self.assertTrue(any("÷ 8 = 32" in t for t in texts(page.body)))

    def test_loss_step_zero_is_exact(self):
        page = self.app.show("loss")
        page.pick("Step 0 (random weights)")
        self.app.update()
        self.assertAlmostEqual(page.loss, 2.4849, places=3)             # ln 12 for the fake vocabulary
        page.pick("After training")
        self.assertLess(page.loss, 1)

    def test_head_lab_switch_off_and_rank(self):
        page = self.app.show("head_lab")
        self.app.update()
        page.rank_button.invoke()
        self.wait(lambda: page.ranked)
        self.assertTrue(any("error +" in t for t in texts(page.body)))
        page.toggle_button.invoke()
        self.wait(lambda: page.changed is not None)
        self.assertEqual(page.changed["off"], "[reply with 1 heads off]")
        self.assertTrue(any("→" in t and "bits/char" in t for t in texts(page.body)))


if __name__ == "__main__":
    unittest.main()

"""Canonical Network designer host and registry-driven views."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import tempfile
import tkinter as tk
import unittest

from studio_core import Workspace
from studio_gui import Studio


class ModelSpecUI(unittest.TestCase):
    def test_views_swap_and_persist_canonical_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            try:
                app = Studio(Workspace(directory))
            except tk.TclError as error:
                if "display" in str(error).lower():
                    self.skipTest("Tk display is unavailable")
                raise
            app.withdraw()
            try:
                app.show_page("network")
                app.update()
                titles = [app.net_view_tabs.tab(tab, "text") for tab in app.net_view_tabs.tabs()]
                self.assertEqual(titles, ["Design", "Sandbox", "Variants", "Experiments", "Trace a token",
                                          "Inside a head", "Value map", "Loss", "Head lab", "3D", "Versions"])
                arch_id = app.arch_id
                app.net_show_view("Sandbox")
                sandbox = app.net_views["Sandbox"]
                sandbox.slot.set("mixer")
                sandbox.refresh()
                sandbox.choose("gqa")
                self.assertEqual(app.net_model_spec["block"]["mixer"]["id"], "gqa")
                self.assertTrue(app.net_modern)
                app.net_flush()
                self.assertEqual(app.arch_store.get_model_spec(arch_id)["block"]["mixer"]["id"], "gqa")
                self.assertFalse(app.net_set_embedding("width", 96))  # legacy canvas cannot drop GQA configuration
                app.net_open(arch_id)
                self.assertEqual(app.net_model_spec["block"]["mixer"]["id"], "gqa")
                app.net_show_view("Design")
                self.assertTrue(app.net_design.stream.rows)
            finally:
                app.destroy()


if __name__ == "__main__":
    unittest.main()

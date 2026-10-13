"""Hot reload (dev.py): the window is rebuilt in place, without stacking handlers or timers, and bad code is rolled back."""
import shutil
import sys
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import dev
from studio_core import Workspace


class HotReload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dev.install()
        from studio_gui import Studio
        cls.directory = tempfile.mkdtemp()
        try:
            cls.app = Studio(Workspace(cls.directory))
        except tk.TclError as exc:
            shutil.rmtree(cls.directory, ignore_errors=True)
            if "display" in str(exc).lower():
                raise unittest.SkipTest("Tk display is unavailable")
            raise
        cls.app.withdraw()
        cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def shape(self):
        app = self.app
        time.sleep(.35)  # let one-shot startup timers (index_local_downloads at 300 ms) fire, so only recurring ones are counted
        app.update()
        return ({seq: len(app.bind_class("all", seq)) for seq in app.bind_class("all")}, len(app.winfo_children()),
                len(app.tk.splitlist(app.tk.call("after", "info"))))

    def test_rebuild_keeps_the_window_and_stacks_nothing(self):
        app = self.app
        window, built = app.winfo_id(), self.shape()
        shapes = []
        for _ in range(3):
            dev.rebuild(app, type(app), Workspace(self.directory))
            app.update()
            shapes.append(self.shape())
        self.assertEqual(app.winfo_id(), window)
        self.assertEqual(shapes[0][:2], built[:2])  # same global handlers and widgets as a first launch
        self.assertEqual(shapes[2], shapes[0])  # and repeating it adds no handlers, widgets or timers
        commands = len(app._tclCommands)
        dev.rebuild(app, type(app), Workspace(self.directory))
        self.shape()
        self.assertEqual(len(app._tclCommands), commands)  # nor Tcl callbacks
        app.show_page("training")

    def test_reload_swaps_in_fresh_code(self):
        app, old, modules = self.app, type(self.app), dev.project_modules()
        try:
            ok = dev.reload(app)
            self.assertTrue(ok)
            self.assertIsNot(type(app), old)
            self.assertIs(type(app), sys.modules["studio_gui"].Studio)
            self.assertEqual(str(app.workspace.root), self.directory)
            sys.modules["studio_registry"].get("mixer", "attention")  # studio_modules re-registered into the fresh registry
        finally:  # later test modules hold the original module objects, so put them back
            sys.modules.update(modules)
            dev.rebuild(app, old, Workspace(self.directory))

    def test_broken_code_keeps_the_old_code_running(self):
        app, old, module = self.app, type(self.app), sys.modules["studio_gui"]
        with patch("dev.importlib.import_module", side_effect=SyntaxError("bad edit")):
            ok = dev.reload(app)
        self.assertFalse(ok)
        self.assertIs(type(app), old)
        self.assertIs(sys.modules["studio_gui"], module)
        self.assertIn("bad edit", app.status_var.get())


if __name__ == "__main__":
    unittest.main()

"""Window tests are CI-only. `from tests import windows` as the first import of a gui_*.py file skips the whole module
unless CI is set (GitHub Actions does) or GROUNDWORK_GUI_TESTS=1. It runs before any Tk or Panda3D call."""
import os
import sys
import unittest

if not (os.environ.get("CI") or os.environ.get("GROUNDWORK_GUI_TESTS") == "1"):
    raise unittest.SkipTest("opens real windows: CI-only (set GROUNDWORK_GUI_TESTS=1 to opt in)")

if sys.platform == "darwin":
    # Tk on macOS keeps the first interpreter it ever made as the app's event interpreter (TKApplication's _eventInterp)
    # and asks it about the app menu whenever AppKit validates that menu, which macOS 26 does on its own from the run loop.
    # Each test builds and drops its own Tk root; once the first one is garbage-collected that pointer dangles and the next
    # validation segfaults inside update() (Tcl_FindCommand from -[TKApplication(TKMenus) validateUserInterfaceItem:]).
    # One hidden root held for the whole run keeps that first interpreter alive.
    import tkinter

    _FIRST_ROOT = tkinter.Tk()
    _FIRST_ROOT.withdraw()
    tkinter._default_root = None  # the tests' own roots stay the default one

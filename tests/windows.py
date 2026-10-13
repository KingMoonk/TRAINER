"""Window tests are CI-only. `from tests import windows` as the first import of a gui_*.py file skips the whole module
unless CI is set (GitHub Actions does) or GROUNDWORK_GUI_TESTS=1. It runs before any Tk or Panda3D call."""
import os
import unittest

if not (os.environ.get("CI") or os.environ.get("GROUNDWORK_GUI_TESTS") == "1"):
    raise unittest.SkipTest("opens real windows: CI-only (set GROUNDWORK_GUI_TESTS=1 to opt in)")

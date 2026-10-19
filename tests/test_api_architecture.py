"""The web server runs without tkinter."""
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ApiBoundaryTests(unittest.TestCase):
    def test_quicktest_and_training_run_work_without_tk(self):
        # A fresh interpreter prevents another test's cached desktop imports hiding a dependency.
        code = """
import importlib.abc
import sys
class NoDesktop(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tkinter' or fullname.startswith('tkinter.'):
            raise ImportError('Desktop dependency: ' + fullname)
sys.meta_path.insert(0, NoDesktop())
from studio_api.routers.quicktest import _describe, _cost
from studio_spec import default_spec, validate_spec
spec = validate_spec(default_spec())
assert 'blocks' in _describe(spec)
assert _cost(spec, spec) == 'same size · same work per step'
from studio_api.routers.training_run import training_run
from studio_ia.services import FakeServices
training_run(FakeServices())
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

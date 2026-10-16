"""The web server must remain usable after the desktop pages are removed."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ApiBoundaryTests(unittest.TestCase):
    def test_api_does_not_import_desktop_pages(self):
        violations = []
        for path in (ROOT / "studio_api").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                    modules += [f"{node.module}.{alias.name}" for alias in node.names]
                if any(name == "studio_ia.pages" or name.startswith("studio_ia.pages.") for name in modules):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(violations, [], "API imports desktop pages: " + ", ".join(violations))

    def test_quicktest_descriptions_work_without_tk(self):
        # A fresh interpreter prevents another test's cached desktop imports hiding a dependency.
        code = """
import importlib.abc
import sys
class NoDesktop(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tkinter' or fullname.startswith(('tkinter.', 'studio_ia.pages')):
            raise ImportError('Desktop dependency: ' + fullname)
sys.meta_path.insert(0, NoDesktop())
from studio_api.routers.quicktest import _describe, _cost
from studio_spec import default_spec, validate_spec
spec = validate_spec(default_spec())
assert 'blocks' in _describe(spec)
assert _cost(spec, spec) == 'same size · same work per step'
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

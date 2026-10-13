"""Check that the PyInstaller hook collects the registry's full module package."""
import pkgutil
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ModulePackagingTests(unittest.TestCase):
    def test_hook_collects_all_discoverable_studio_modules_without_pyinstaller(self):
        import studio_modules

        expected = sorted(info.name for info in pkgutil.walk_packages(
            studio_modules.__path__, "studio_modules."))
        calls = []

        def collect_submodules(package):
            calls.append(package)
            return expected

        pyinstaller = types.ModuleType("PyInstaller")
        utils = types.ModuleType("PyInstaller.utils")
        hooks = types.ModuleType("PyInstaller.utils.hooks")
        hooks.collect_submodules = collect_submodules
        module_map = {
            "PyInstaller": pyinstaller,
            "PyInstaller.utils": utils,
            "PyInstaller.utils.hooks": hooks,
        }
        hook_path = ROOT / "packaging" / "hook-studio_modules.py"
        with patch.dict(sys.modules, module_map):
            result = runpy.run_path(str(hook_path))

        self.assertEqual(calls, ["studio_modules"])
        self.assertEqual(result["hiddenimports"], expected)
        self.assertTrue(expected)
        self.assertIn("studio_modules.optimizer.sgd", expected)
        self.assertIn("studio_modules.objective.ce_z", expected)


if __name__ == "__main__":
    unittest.main()

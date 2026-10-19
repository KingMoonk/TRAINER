"""release.yml and the frozen app agree: the hidden imports app.py checks are the ones PyInstaller is given, and web/out is bundled where the API looks for it."""
import re
import unittest
from pathlib import Path

import app
from studio_api import app as api_app

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def pyinstaller_command():
    return next(line for line in WORKFLOW.splitlines() if "-m PyInstaller" in line)


class ReleaseWorkflow(unittest.TestCase):
    def test_hidden_imports_match_what_the_smoke_test_checks(self):
        self.assertEqual(set(re.findall(r"--hidden-import (\S+)", pyinstaller_command())), set(app.HIDDEN))

    def test_the_routers_studio_api_imports_by_name_are_collected(self):
        self.assertIn("--collect-submodules studio_api.routers", pyinstaller_command())

    def test_web_out_is_bundled_where_the_api_looks_for_it(self):
        """Frozen, studio_api/app.py lives under the bundle root (sys._MEIPASS), so WEB_OUT is <root>/web/out whatever the checkout."""
        wanted = api_app.WEB_OUT.relative_to(ROOT).as_posix()
        self.assertEqual(wanted, "web/out")
        found = re.findall(r'--add-data "([^"$]*)\$\{\{ matrix\.sep \}\}([^"]*)"', pyinstaller_command())
        self.assertEqual(found, [(wanted, wanted)])

    def test_the_add_data_separator_is_the_platform_path_separator(self):
        self.assertIn('{os: macos-latest, name: macos, sep: ":"}', WORKFLOW)
        self.assertIn('{os: windows-latest, name: windows, sep: ";"}', WORKFLOW)

    def test_the_web_bundle_is_built_before_pyinstaller(self):
        self.assertLess(WORKFLOW.index("pnpm build"), WORKFLOW.index("-m PyInstaller"))
        self.assertIn("pnpm install --frozen-lockfile", WORKFLOW)


if __name__ == "__main__":
    unittest.main()

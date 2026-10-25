"""release.yml packages the source and built web/out into Groundwork.zip, smoke tests the unpacked archive, and attaches it on release."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
AUTO_RELEASE_WORKFLOW = (ROOT / ".github" / "workflows" / "auto-release.yml").read_text(encoding="utf-8")


class ReleaseWorkflow(unittest.TestCase):
    def test_the_web_bundle_is_built_before_packaging(self):
        self.assertIn("pnpm install --frozen-lockfile", RELEASE_WORKFLOW)
        self.assertIn("pnpm build", RELEASE_WORKFLOW)
        self.assertLess(RELEASE_WORKFLOW.index("pnpm build"), RELEASE_WORKFLOW.index("Groundwork.zip"))

    def test_web_out_and_app_are_packaged_in_groundwork_zip(self):
        self.assertIn('"web" / "out"', RELEASE_WORKFLOW)
        self.assertIn("requirements*.txt", RELEASE_WORKFLOW)
        self.assertIn("Groundwork.zip", RELEASE_WORKFLOW)

    def test_smoke_test_runs_on_unpacked_release(self):
        self.assertIn('app.py", "--smoke"', RELEASE_WORKFLOW)

    def test_artifact_and_release_attachment(self):
        self.assertIn("name: Groundwork", RELEASE_WORKFLOW)
        self.assertIn("path: Groundwork.zip", RELEASE_WORKFLOW)
        self.assertIn("dist/Groundwork.zip", RELEASE_WORKFLOW)

    def test_auto_release_dispatches_release_workflow(self):
        self.assertIn("gh workflow run release.yml", AUTO_RELEASE_WORKFLOW)

    def test_no_legacy_packaging_or_frozen_references_in_workflow(self):
        for term in ("py" + "installer", "sys." + "frozen", "pack" + "aging/"):
            self.assertNotIn(term, RELEASE_WORKFLOW.lower())


if __name__ == "__main__":
    unittest.main()

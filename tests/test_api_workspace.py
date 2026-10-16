"""A worktree can serve the user's existing records instead of an empty checkout."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from studio_api.__main__ import make, parse
from studio_ia.services import FakeServices


class WorkspaceArgumentTest(unittest.TestCase):
    def test_explicit_workspace_reaches_services(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with mock.patch("studio_core.Workspace") as workspace, mock.patch(
                "studio_ia.services.WorkspaceServices", return_value=FakeServices()
            ) as services:
                app, host, _ = make(["--workspace", str(root)])
            self.addCleanup(app.state.jobs.close)
            self.addCleanup(app.state.broadcaster.close)
            workspace.assert_called_once_with(root)
            services.assert_called_once_with(workspace.return_value)
            self.assertEqual(host, "127.0.0.1")

    def test_fake_cannot_accidentally_use_real_workspace(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            parse(["--fake", "--workspace", "."])

    def test_misspelled_workspace_does_not_create_empty_data(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                parse(["--workspace", str(missing)])
            self.assertFalse(missing.exists())

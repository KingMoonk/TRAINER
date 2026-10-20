import subprocess
import unittest
from unittest import mock

from studio_api import version


class AppVersionTests(unittest.TestCase):
    def setUp(self):
        version.app_version.cache_clear()
        self.addCleanup(version.app_version.cache_clear)

    def test_the_hash_length_is_pinned_so_two_clones_of_one_commit_report_the_same_version(self):
        done = subprocess.CompletedProcess([], 0, stdout="v0.16.2-222-g515a0378e5b6\n")
        with mock.patch("studio_api.version.subprocess.run", return_value=done) as run:
            self.assertEqual(version.app_version(), "v0.16.2-222-g515a0378e5b6")
        self.assertIn("--abbrev=12", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()

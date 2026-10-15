import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from studio_timing import timed


class TimedTests(unittest.TestCase):
    def run_block(self, env):
        out = io.StringIO()
        with patch.dict(os.environ, env, clear=True), redirect_stderr(out):
            with timed("block"):
                pass

            @timed("call")
            def add(a, b):
                return a + b
            self.assertEqual(add(2, 3), 5)  # the decorator passes arguments and the result through
        return out.getvalue()

    def test_silent_when_unset(self):
        self.assertEqual(self.run_block({}), "")

    def test_silent_when_not_one(self):
        self.assertEqual(self.run_block({"GROUNDWORK_TIMINGS": "0"}), "")

    def test_logs_name_time_and_thread_when_set(self):
        lines = self.run_block({"GROUNDWORK_TIMINGS": "1"}).splitlines()
        self.assertEqual(len(lines), 2)
        for line, name in zip(lines, ("block", "call")):
            self.assertRegex(line, rf"^{name}: \d+\.\d ms \[.+\]$")

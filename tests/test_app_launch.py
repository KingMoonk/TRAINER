"""app.py starts the API and opens the browser at it. Window-free: webbrowser.open is a fake and the services are FakeServices."""
import io
import json
import socket
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import app
from studio_api import app as api_app


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Launch(unittest.TestCase):
    def test_starts_the_api_and_opens_the_browser_at_it(self):
        port, seen = free_port(), {}

        def after_open(server):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/node", timeout=10) as reply:
                seen["node"] = json.load(reply)
            server.should_exit = True

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "index.html").write_text("<h1>Groundwork</h1>", encoding="utf-8")
            with mock.patch.object(api_app, "WEB_OUT", Path(tmp)), mock.patch("webbrowser.open") as opened:
                code = app.main(["--fake", "--port", str(port)], after_open=after_open)
        self.assertEqual(code, 0)
        opened.assert_called_once_with(f"http://127.0.0.1:{port}/")
        self.assertIsInstance(seen["node"], dict)

    def test_without_the_web_build_it_says_so_and_opens_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api_app, "WEB_OUT", Path(tmp) / "out"), \
                mock.patch("webbrowser.open") as opened, redirect_stdout(io.StringIO()) as out:
            code = app.main(["--fake", "--port", str(free_port())])
        self.assertEqual(code, 1)
        self.assertIn("tasks.py web:build", out.getvalue())
        opened.assert_not_called()


class Smoke(unittest.TestCase):
    def test_smoke_flag_is_taken_out_of_the_arguments(self):
        self.assertEqual(app.split_smoke(["--smoke", "r.txt", "--allow-cpu"]), ("r.txt", ["--allow-cpu"]))
        self.assertEqual(app.split_smoke(["--allow-cpu"]), (None, ["--allow-cpu"]))
        with self.assertRaises(SystemExit):
            app.split_smoke(["--smoke"])

    def test_a_module_that_fails_to_import_is_named(self):
        real = app.importlib.import_module

        def fake(name):
            if name == "studio_worker":
                raise ModuleNotFoundError("No module named 'x'")
            return real(name)

        with mock.patch.object(app.importlib, "import_module", fake), mock.patch.object(app, "HIDDEN", ("studio_worker",)):
            lines = app.missing_modules()
        self.assertEqual(len(lines), 1)
        self.assertIn("studio_worker: ModuleNotFoundError", lines[0])

    def test_smoke_starts_the_api_without_a_browser_and_reports_ok(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("webbrowser.open") as opened, \
                mock.patch.object(app, "missing_modules", return_value=[]), redirect_stdout(io.StringIO()):
            report = Path(tmp) / "report.txt"
            code = app.main(["--smoke", str(report)])
            self.assertEqual(report.read_text(encoding="utf-8"), "OK\n")
        self.assertEqual(code, 0)
        opened.assert_not_called()

    def test_smoke_fails_loudly_on_a_missing_module(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(app, "missing_modules", return_value=["import studio_worker: boom"]), \
                redirect_stdout(io.StringIO()):
            report = Path(tmp) / "report.txt"
            code = app.main(["--smoke", str(report)])
            self.assertEqual(report.read_text(encoding="utf-8"), "import studio_worker: boom\nFAIL\n")
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()

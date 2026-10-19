"""app.py starts the API and opens the browser at it. Window-free: webbrowser.open is a fake and the services are FakeServices."""
import io
import json
import socket
import tempfile
import unittest
import urllib.request
from contextlib import contextmanager, redirect_stdout
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


class SplitSmoke(unittest.TestCase):
    def test_takes_the_flag_and_its_value_out(self):
        self.assertEqual(app.split_smoke(["--smoke", "r.txt", "--allow-cpu"]), ("r.txt", ["--allow-cpu"]))
        self.assertEqual(app.split_smoke(["--allow-cpu", "--smoke=r.txt"]), ("r.txt", ["--allow-cpu"]))
        self.assertEqual(app.split_smoke(["--allow-cpu"]), (None, ["--allow-cpu"]))

    def test_refuses_a_missing_empty_or_flag_like_value(self):
        for argv in (["--smoke"], ["--smoke", ""], ["--smoke="], ["--smoke", "--fake"], ["--smoke=--fake"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                app.split_smoke(argv)

    def test_an_empty_value_never_starts_the_normal_app(self):
        with mock.patch("webbrowser.open") as opened, self.assertRaises(SystemExit):
            app.main(["--smoke", ""])
        opened.assert_not_called()


@contextmanager
def smoke_run(web=True, modules=()):
    """app.main(["--smoke", report]) with the modules check and the web build faked; yields (exit code, report text, browser fake)."""
    with tempfile.TemporaryDirectory() as tmp, mock.patch("webbrowser.open") as opened, redirect_stdout(io.StringIO()), \
            mock.patch.object(app, "missing_modules", return_value=list(modules)):
        out = Path(tmp) / "out"
        if web:
            out.mkdir()
            (out / "index.html").write_text("<h1>Groundwork</h1>", encoding="utf-8")
        report = Path(tmp) / "report.txt"
        with mock.patch.object(api_app, "WEB_OUT", out):
            code = app.main(["--smoke", str(report)])
        yield code, report.read_text(encoding="utf-8"), opened


class Smoke(unittest.TestCase):
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

    def test_every_hidden_import_and_model_module_imports_from_source(self):
        self.assertEqual(app.missing_modules(), [])

    def test_starts_the_api_without_a_browser_and_reports_ok(self):
        with smoke_run() as (code, text, opened):
            pass
        self.assertEqual((code, text), (0, "OK\n"))
        opened.assert_not_called()

    def test_fails_loudly_on_a_missing_module(self):
        with smoke_run(modules=["import studio_worker: boom"]) as (code, text, _):
            pass
        self.assertEqual((code, text), (1, "import studio_worker: boom\nFAIL\n"))

    def test_fails_loudly_without_the_web_build(self):
        with smoke_run(web=False) as (code, text, _):
            pass
        self.assertEqual(code, 1)
        self.assertIn("web build missing", text)
        self.assertTrue(text.endswith("FAIL\n"))

    def test_fails_when_node_never_answers(self):
        with mock.patch.object(app, "OPEN_WITHIN", .2), mock.patch.object(app, "open_when_up"), smoke_run() as (code, text, _):
            pass
        self.assertEqual(code, 1)
        self.assertIn("/node did not answer", text)
        self.assertIn("within 0s", text)
        self.assertTrue(text.endswith("FAIL\n"))

    def test_any_exception_reaches_the_report(self):
        with mock.patch.object(app, "free_port", side_effect=SystemExit("cannot bind")), smoke_run() as (code, text, _):
            pass
        self.assertEqual(code, 1)
        self.assertIn("SystemExit: cannot bind", text)
        self.assertTrue(text.endswith("FAIL\n"))


if __name__ == "__main__":
    unittest.main()

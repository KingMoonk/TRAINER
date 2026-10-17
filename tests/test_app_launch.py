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


if __name__ == "__main__":
    unittest.main()

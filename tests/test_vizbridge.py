"""Checks for the GUI-to-viewer process bridge (studio_vizbridge)."""
import io
import shutil
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio_vizbridge import VizBridge, ViewerLink, parse_viewer_args  # noqa: E402

ECHO_VIEWER = '''import sys, time
sys.path.insert(0, {root!r})
from studio_vizbridge import ViewerLink, parse_viewer_args
port, key = parse_viewer_args()
link = ViewerLink(port, key)
while link.connected:
    for m in link.poll():
        if m.get("type") == "quit":
            link.send({{"type": "closed"}})
            link.close()
            sys.exit(0)
        link.send({{"type": "echo", "msg": m}})
    time.sleep(0.005)
'''

HANG_VIEWER = "import time\ntime.sleep(60)\n"


def _wait_for(bridge, predicate, timeout=10.0):
    """Poll the bridge until predicate(received) holds; returns everything received."""
    received = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        received.extend(bridge.poll())
        if predicate(received):
            return received
        time.sleep(0.02)
    return received


class VizBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="vizbridge-"))
        cls.echo = cls.dir / "echo_viewer.py"
        cls.echo.write_text(ECHO_VIEWER.format(root=str(ROOT)), encoding="utf-8")
        cls.hang = cls.dir / "hang_viewer.py"
        cls.hang.write_text(HANG_VIEWER, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _bridge(self, script=None):
        bridge = VizBridge(script or self.echo)
        self.addCleanup(bridge.stop)
        return bridge

    def test_roundtrip_spec_and_select(self):
        bridge = self._bridge()
        bridge.start()
        spec = {"format": "groundwork-arch/1", "blocks": [{"id": "blk_ab12cd"}]}
        bridge.send({"type": "spec", "spec": spec, "selected": None})
        bridge.send({"type": "select", "block_id": "blk_ab12cd"})
        got = _wait_for(bridge, lambda r: len([m for m in r if m.get("type") == "echo"]) >= 2)
        echoes = [m["msg"] for m in got if m.get("type") == "echo"]
        self.assertEqual([m["type"] for m in echoes], ["spec", "select"])
        self.assertEqual(echoes[0]["spec"], spec)
        self.assertEqual(echoes[1]["block_id"], "blk_ab12cd")
        self.assertTrue(bridge.running)

    def test_command_replaces_the_python_and_script_pair(self):
        bridge = VizBridge("ignored.py", command=[sys.executable, str(self.echo)])
        self.addCleanup(bridge.stop)
        bridge.start()
        bridge.send({"type": "select", "block_id": "blk_ab12cd"})
        got = _wait_for(bridge, lambda r: any(m.get("type") == "echo" for m in r))
        self.assertEqual([m["msg"]["block_id"] for m in got if m.get("type") == "echo"], ["blk_ab12cd"])

    def test_telemetry_coalesces_and_keeps_last(self):
        bridge = self._bridge()
        bridge.start()
        bridge.send({"type": "spec", "spec": {}, "selected": None})
        _wait_for(bridge, lambda r: any(m.get("type") == "echo" for m in r))
        total = 500
        for step in range(total):
            bridge.send({"type": "telemetry", "step": step, "total": total, "loss": 1.0 / (step + 1)})
        got = _wait_for(bridge, lambda r: any(m.get("type") == "echo" and m["msg"].get("step") == total - 1 for m in r))
        telemetry = [m["msg"] for m in got if m.get("type") == "echo" and m["msg"].get("type") == "telemetry"]
        self.assertTrue(telemetry, "no telemetry reached the viewer")
        self.assertLess(len(telemetry), total // 2, f"telemetry not coalesced: {len(telemetry)} received")
        self.assertEqual(telemetry[-1]["step"], total - 1)

    def test_poll_never_blocks(self):
        bridge = self._bridge()
        start = time.monotonic()
        self.assertEqual(bridge.poll(), [])
        bridge.start()
        self.assertEqual(bridge.poll(), [])
        self.assertLess(time.monotonic() - start, 0.5)

    def test_stop_terminates_within_three_seconds(self):
        bridge = self._bridge()
        bridge.start()
        bridge.send({"type": "spec", "spec": {}, "selected": None})
        _wait_for(bridge, lambda r: any(m.get("type") == "echo" for m in r))
        proc = bridge._proc
        start = time.monotonic()
        bridge.stop()
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertIsNotNone(proc.poll())
        self.assertFalse(bridge.running)
        bridge.stop()  # idempotent

    def test_stop_before_viewer_connects(self):
        bridge = self._bridge(self.hang)
        bridge.start()
        self.assertTrue(bridge.running)
        proc = bridge._proc
        start = time.monotonic()
        bridge.stop()
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertIsNotNone(proc.poll())

    def test_send_before_start_is_noop(self):
        bridge = self._bridge()
        bridge.send({"type": "spec", "spec": {}, "selected": None})
        bridge.send({"type": "telemetry", "step": 1})
        self.assertIsNone(bridge._proc)
        self.assertFalse(bridge.running)
        self.assertEqual(bridge.poll(), [])
        bridge.stop()

    def test_viewer_link_fails_cleanly_on_closed_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        key = "00" * 16
        start = time.monotonic()
        with self.assertRaises(ConnectionError):
            ViewerLink(port, key)
        self.assertLess(time.monotonic() - start, 7.0)

    def test_parse_viewer_args(self):
        self.assertEqual(parse_viewer_args(["--port", "5555", "--authkey", "abcd", "--extra", "x"]), (5555, "abcd"))
        self.assertEqual(parse_viewer_args([]), (None, None))
        self.assertEqual(parse_viewer_args(["--port", "5555"], stdin=io.StringIO("beef\n")), (5555, "beef"))

    def test_authkey_stays_off_the_command_line(self):
        bridge = self._bridge()
        bridge.start()
        self.assertNotIn(bridge._authkey.hex(), " ".join(bridge._proc.args))
        bridge.send({"type": "select", "block_id": "x"})
        self.assertTrue(any(m.get("type") == "echo" for m in _wait_for(bridge, lambda r: r)))

    def test_stop_without_waiting_returns_at_once(self):
        bridge = self._bridge(self.hang)
        bridge.start()
        proc, start = bridge._proc, time.monotonic()
        bridge.stop(wait=False)
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertFalse(bridge.running)
        proc.wait(timeout=5)  # terminated by the background thread

    def test_failed_launch_closes_the_listener(self):
        bridge = VizBridge(self.echo, command=[str(self.dir / "missing-executable")])
        with patch("studio_vizbridge.Listener") as listener:
            listener.return_value.address = ("127.0.0.1", 1)
            with self.assertRaises(OSError):
                bridge.start()
        listener.return_value.close.assert_called_once()

    def test_unpicklable_message_is_reported(self):
        bridge = self._bridge()
        bridge.start()
        bridge.send({"type": "select", "block_id": lambda: None})
        got = _wait_for(bridge, lambda r: any(m.get("type") == "send_error" for m in r))
        self.assertTrue(any(m.get("type") == "send_error" for m in got))


if __name__ == "__main__":
    unittest.main()

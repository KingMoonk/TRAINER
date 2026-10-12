"""Offscreen checks for the Panda3D network-brain viewer (demo selftest + link protocol)."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "studio_brain3d.py"

FAKE_LINK_SCRIPT = r"""
import sys
sys.path.insert(0, %r)
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen\nframebuffer-hardware 0\naudio-library-name null\nsync-video 0")
import studio_brain3d as sb

class Link:
    connected = True
    def __init__(self): self.inbox, self.sent = [], []
    def send(self, m): self.sent.append(m)
    def poll(self):
        out, self.inbox = self.inbox, []
        return out

link = Link()
app = sb.BrainApp(link=link, demo=False)
app.auto_orbit = False
spec = sb.sample_spec(4)
link.inbox += [{"type": "spec", "spec": spec, "selected": spec["blocks"][1]["id"]},
               {"type": "telemetry", **sb.synth_telemetry(spec, 5, 100, 1.0)}]
for _ in range(5):
    app.taskMgr.step()
assert app.sel_id == spec["blocks"][1]["id"], app.sel_id
assert len(app.ids) == 4
# GUI mode: edits are only sent, never applied locally
app.cmd_step("heads", 1)
app.cmd_toggle("mlp")
app.cmd_add()
app.cmd_remove()
kinds = [m["type"] for m in link.sent]
assert kinds == ["edit", "edit", "add_block", "remove_block"], kinds
assert link.sent[0]["path"] == "attention.heads" and link.sent[0]["value"] == 16, link.sent[0]  # 8 -> 16 (width 64)
assert len(app.ids) == 4
# invalid head values are skipped: width 64 has no divisor between 8 and 16 except 16
assert sb.valid_heads(64, 8, 1) == 16 and sb.valid_heads(48, 4, 1) == 6 and sb.valid_heads(64, 16, 1) is None
# block count change re-frames the column but keeps the orbit angle and pitch
cam = (app.cam_h, app.cam_p)
spec2 = sb.sample_spec(5)
link.inbox.append({"type": "spec", "spec": spec2, "selected": None})
app.taskMgr.step()
assert len(app.ids) == 5 and (app.cam_h, app.cam_p) == cam
# same block count (e.g. head edit echoed back): the user's zoom and pan are kept
app.zoom(0.5)
cam = (app.cam_h, app.cam_p, app.cam_d, app.cam_t.getZ())
link.inbox.append({"type": "spec", "spec": sb.sample_spec(5), "selected": None})
app.taskMgr.step()
assert (app.cam_h, app.cam_p, app.cam_d, app.cam_t.getZ()) == cam
link.inbox.append({"type": "quit"})
try:
    for _ in range(5):
        app.taskMgr.step()
except SystemExit:
    pass
assert link.sent[-1] == {"type": "closed"}, link.sent[-1]
print("FAKELINK-OK")
"""


class Brain3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Machines without a GPU (hosted CI runners) cannot open even an offscreen window: skip rather than fail there.
        with tempfile.TemporaryDirectory() as directory:
            probe = subprocess.run([sys.executable, str(SCRIPT), "--selftest", "1", "--screenshot", os.path.join(directory, "probe.png")],
                                   capture_output=True, text=True, timeout=180, cwd=str(ROOT))
        output = probe.stdout + probe.stderr
        if "Could not open window" in output or "No graphics pipe is available" in output:
            raise unittest.SkipTest("no usable OpenGL on this machine")

    def test_selftest_renders_screenshot(self):
        with tempfile.TemporaryDirectory() as directory:
            shot = os.path.join(directory, "brain.png")
            proc = subprocess.run([sys.executable, str(SCRIPT), "--selftest", "90", "--screenshot", shot],
                                  capture_output=True, text=True, timeout=180, cwd=str(ROOT))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue(os.path.exists(shot))
            self.assertGreater(os.path.getsize(shot), 10 * 1024)
            self.assertIn("fps(24 blocks)", proc.stdout)

    def test_link_protocol_with_fake_link(self):
        proc = subprocess.run([sys.executable, "-c", FAKE_LINK_SCRIPT % str(ROOT)], capture_output=True, text=True,
                              timeout=180, cwd=str(ROOT))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("FAKELINK-OK", proc.stdout)

    def test_real_bridge_quit_sends_closed(self):
        try:
            from studio_vizbridge import VizBridge
        except ImportError:
            self.skipTest("studio_vizbridge not available")
        bridge = VizBridge(SCRIPT, extra_args=["--offscreen"])
        bridge.start()
        try:
            spec = __import__("studio_brain3d").sample_spec(3)
            deadline = time.time() + 30
            while time.time() < deadline and not bridge.running:
                time.sleep(0.1)
            bridge.send({"type": "spec", "spec": spec, "selected": None})
            time.sleep(2.0)
            self.assertTrue(bridge.running)
            bridge.send({"type": "quit"})
            seen, deadline = [], time.time() + 15
            while time.time() < deadline and "closed" not in seen:
                seen += [m.get("type") for m in bridge.poll()]
                time.sleep(0.1)
            self.assertIn("closed", seen)
        finally:
            bridge.stop()
        self.assertFalse(bridge.running)


if __name__ == "__main__":
    unittest.main()

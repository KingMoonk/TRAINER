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
assert kinds == ["module_swap", "module_swap", "add_block", "remove_block"], kinds
assert link.sent[0]["slot"] == "mixer" and link.sent[0]["entry"]["config"]["heads"] == 16, link.sent[0]  # 8 -> 16 (width 64)
assert link.sent[1]["slot"] == "channel" and link.sent[1]["entry"]["id"] == "none", link.sent[1]
assert "block" not in link.sent[0], f"module_swap message should not contain 'block' key: {link.sent[0]}"
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
# a bad message is reported in the HUD instead of crashing the viewer
link.inbox.append({"type": "selection", "selection": {"bogus": 1}})
app.taskMgr.step()
assert "Ignored a selection message" in app.hud["text"], app.hud["text"]
# weight edits need a cell from a loaded snapshot, and are refused rather than sent blind
app.edit_weight(.01)
assert link.sent[-1]["type"] != "edit_weight" and not app.edit_pending
# the studio going away is noticed
link.connected = False
app.taskMgr.step()
assert app.hud["text"] == "studio disconnected", app.hud["text"]
link.connected = True
link.inbox.append({"type": "quit"})
try:
    for _ in range(5):
        app.taskMgr.step()
except SystemExit:
    pass
assert link.sent[-1] == {"type": "closed"}, link.sent[-1]
print("FAKELINK-OK")
"""


# Weight edits target an explicit highlighted cell, one at a time; a new snapshot keeps the chosen matrix.
EDIT_SCRIPT = r"""
import sys, tempfile
sys.path.insert(0, %r)
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen\nframebuffer-hardware 0\naudio-library-name null\nsync-video 0")
import studio_brain3d as sb
from studio_compile import CompiledModel
from studio_snapshot import one_step, save
from studio_spec import default_spec, validate_spec
spec = default_spec(); spec["dims"].update(d_model=16, n_layer=2, n_ctx=32, dropout=0.); spec = validate_spec(spec)
snap = one_step(CompiledModel(spec, 9, device="cpu"), [[1, 2, 3]], [[2, 3, 4]])
d = tempfile.mkdtemp(); p1 = str(save(d + "/a", snap)); p2 = str(save(d + "/b", snap))
class Link:
    connected = True
    def __init__(self): self.inbox, self.sent = [], []
    def send(self, m): self.sent.append(m)
    def poll(self):
        out, self.inbox = self.inbox, []; return out
link = Link(); app = sb.BrainApp(link=link, demo=False)
link.inbox.append({"type": "snapshot", "path": p1}); app.taskMgr.step()
names = app.scene.frame('Weights')['available']; pick = names[-1]
app.select_matrix(pick); app.set_level('Weights')
app.move_cell(1, 0); app.move_cell(0, 2)
app.edit_weight(.01); app.edit_weight(.01)
edits = [m for m in link.sent if m["type"] == "edit_weight"]
assert len(edits) == 1 and edits[0]["weight"] == pick and (edits[0]["row"], edits[0]["column"]) == (1, 2), edits
link.inbox.append({"type": "snapshot", "path": p2}); link.inbox.append({"type": "weight_edit_done", "ok": True, "message": "saved"}); app.taskMgr.step()
assert app.scene.selection.weight == pick, app.scene.selection.weight
assert not app.edit_pending
app.set_level('Head'); app.set_head_stage('scale'); app.set_head_stage('mask')
print("EDIT-OK")
app.destroy()
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

    def test_weight_edits_use_the_selected_cell(self):
        proc = subprocess.run([sys.executable, "-c", EDIT_SCRIPT % str(ROOT)], capture_output=True, text=True,
                              timeout=180, cwd=str(ROOT))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("EDIT-OK", proc.stdout)

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

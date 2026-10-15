"""Offscreen checks for the Panda3D network-brain viewer (demo selftest + link protocol)."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
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
import torch  # before panda3d: Windows DLL load order (see studio_brain3d)
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
# linked: a mixer swap is only sent (the studio saves it as a new version and sends the design back)
app.swap_mixer("linear")
assert link.sent[-1] == {"type": "module_swap", "block_id": spec["blocks"][1]["id"], "slot": "mixer", "entry": {"id": "linear"}}, link.sent[-1]
assert app.scene.gates[1]["entry"]["mixer"]["id"] != "linear"
sent = len(link.sent)
app.swap_mixer(app.scene.gates[1]["entry"]["mixer"]["id"])  # the block's own mixer: nothing to do
assert len(link.sent) == sent
# the snapshot buttons ask the studio; Run one real step is disabled until its answer arrives
app.load_snapshots()
assert link.sent[-1] == {"type": "load_replay"}
app.probe_step(); app.probe_step()
assert [m["type"] for m in link.sent].count("probe_step") == 1 and app.probing
assert app.probe_button["state"] == "disabled" and app.probe_button["text"] == "Running one step..."
app.handle_message({"type": "error", "message": "Train this version first."})
assert not app.probing and app.probe_button["state"] == "normal" and "Train this version first." in app.hud["text"]
app.handle_message({"type": "replay", "paths": []})
assert not app.snapshot_paths and app.play_button.isHidden() and app.position["text"] == "No snapshots loaded."
# the page's zoom and mode show in the menus without being sent back; a choice made here is sent to the page
sent = len(link.sent)
app.handle_message({"type": "scene", "level": "Window", "mode": "Ride a token"})
assert (app.scene.level, app.scene.mode) == ("Window", "Ride a token") and len(link.sent) == sent
assert app.level_menu.get() == "Window" and app.mode_menu.get() == "Ride a token"
assert app.module_menu.isHidden()  # the mixer menu is a Build-mode control
app.set_mode("Build")
assert link.sent[-1] == {"type": "scene", "level": "Window", "mode": "Build"} and not app.module_menu.isHidden()
assert app.module_menu.get() == app.scene.gates[1]["entry"]["mixer"]["id"]
# Tab selects the next block and tells the page
app.next_block()
assert link.sent[-1] == {"type": "select", "block_id": spec["blocks"][2]["id"], "index": 2}, link.sent[-1]
# the wheel zooms the camera and never changes the level
level, distance = app.scene.level, app.cam_d
app.zoom(.8)
assert app.scene.level == level and app.cam_d < distance
# page -> viewer: step the head's forward pass, jump the scrubber, show gradients, feed training stats
app.set_level("Head")
assert app.level_menu.get() == "Head" and link.sent[-1]["level"] == "Head" and not app.step_menu.isHidden()
app.handle_message({"type": "step", "delta": 1})
assert app.head_stage == "values" and app.step_menu.get() == "values", app.head_stage
app.handle_message({"type": "step", "stage": "scores"})
assert app.head_stage == "scores"
app.handle_message({"type": "show", "what": "gradient"})
assert app.weight_kind == "gradient" and app.tile_menu.get() == "gradient"
app.handle_message({"type": "stats", "step": 7, "total": 10, "loss": 1.5, "lr": 1e-3})
assert "step 7/10" in app.hud["text"] and "loss 1.5000" in app.hud["text"], app.hud["text"]
app.snapshot_paths = ["a", "b", "c"]
reads = []
app.read_snapshot = reads.append
app.handle_message({"type": "scrub", "index": 9})
assert reads == ["c"] and app.snapshot_index == 2 and link.sent[-1] == {"type": "replay_state", "index": 2, "count": 3}, link.sent[-1]
app.sync_controls()
assert app.position["text"] == "Snapshot 3 of 3" and app.next_button["state"] == "disabled" and app.previous_button["state"] == "normal"
app.handle_message({"type": "notice", "message": "Version 3 is ready."})
assert "Version 3 is ready." in app.hud["text"]
app.set_level("Model")
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
# block labels are flat screen chips; none overlaps another
chips = [c for c in app.gate_labels.values() if not c.isHidden()]
assert chips
boxes = [(c.getX() - c.getWidth() * c.getScale()[0] / 2, c.getX() + c.getWidth() * c.getScale()[0] / 2, c.getZ(), c.getZ() + c.getHeight() * c.getScale()[2]) for c in chips]
for i, a in enumerate(boxes):
    for b in boxes[i + 1:]:
        assert a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2], (a, b)
assert not hasattr(app, "edit_weight")
# the studio going away is noticed
link.connected = False
app.taskMgr.step()
assert app.hud["text"].endswith("studio disconnected"), app.hud["text"]
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


# The matrix menu offers the selected block's matrices and follows the block; a new snapshot keeps the chosen matrix.
EDIT_SCRIPT = r"""
import sys, tempfile
sys.path.insert(0, %r)
import torch  # before panda3d: Windows DLL load order (see studio_brain3d)
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
assert app.snapshot_paths == [p1] and app.position["text"] == "Snapshot 1 of 1"
app.set_level('Weights')
assert not app.tile_menu.isHidden() and not app.rows['matrix'][1].isHidden() and app.step_menu.isHidden()
offered = list(app.rows['matrix'][1]['items'])
assert offered and all(not n.startswith('block') or n.startswith('block0.') for n in offered), offered
assert app.scene.selection.weight in offered and app.scene.selection.weight.startswith('block0.'), app.scene.selection.weight
pick = [n for n in offered if n.startswith('block0.')][-1]
app.select_matrix(pick)
assert app.rows['matrix'][1].get() == pick
app.move_cell(1, 0); app.move_cell(0, 2)
assert app.cell == [1, 2]
link.inbox.append({"type": "snapshot", "path": p2}); app.taskMgr.step()
assert app.scene.selection.weight == pick, app.scene.selection.weight
app.next_block()  # the menu now offers block 1's matrices, and shows one of them
offered = list(app.rows['matrix'][1]['items'])
assert all(not n.startswith('block') or n.startswith('block1.') for n in offered) and app.scene.selection.weight.startswith('block1.'), offered
assert not hasattr(app, 'edit_weight')
app.set_level('Head'); app.set_head_stage('scale'); app.set_head_stage('mask')
assert app.step_menu.get() == 'mask' and app.tile_menu.isHidden()
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

    def test_matrix_menu_follows_the_block(self):
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

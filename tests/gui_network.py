"""Network page: circuit editing, revisions, training with telemetry, weight registry, backups and viewer messages."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tkinter as tk

from studio_arch import validate_spec
from studio_core import Workspace
from studio_gui import Studio
from studio_model import load_checkpoint

TINY = {"embedding": {"tokenizer": "character", "width": 16, "context": 32, "dropout": 0.0},
        "training": {"steps": 20, "batch_size": 2, "accumulation": 1, "learning_rate": 0.004, "warmup": 0, "weight_decay": 0.01,
                     "eval_every": 10, "validation": 0.0, "seed": 7, "objective": "all", "device": "cpu"}}


class FakeViz:
    running = connected = True
    exit_code = log_path = None

    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def poll(self):
        return []

    def stop(self, wait=True):
        self.running = False


class NetworkPage(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        workspace = Workspace(self.directory.name)
        self.dataset = workspace.create_dataset("Network test")
        workspace.add(self.dataset, [{"user": f"question {i}", "assistant": f"answer number {i}"} for i in range(8)])
        try:
            self.app = Studio(workspace)
        except tk.TclError as exc:
            self.directory.cleanup()
            if "display" in str(exc).lower():
                self.skipTest("Tk display is unavailable")
            raise
        self.app.withdraw()
        self.app.refresh_datasets(self.dataset)
        patcher = patch("studio_network.messagebox.showerror", side_effect=lambda *args, **kwargs: self.fail(str(args)))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("studio_gui.messagebox.showerror", side_effect=lambda *args, **kwargs: self.fail(str(args)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        app = self.app
        app.stop.set()
        app.pause.clear()
        if app.worker:
            app.worker.join(timeout=30)
        app.destroy()
        self.directory.cleanup()

    def settle(self, timeout=60):
        deadline = time.monotonic() + timeout
        while self.app.job_kind and time.monotonic() < deadline:
            self.app.update()
            time.sleep(.01)
        self.assertIsNone(self.app.job_kind, "Worker did not complete")

    def pump(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.app.update()
            time.sleep(.01)

    def open_page(self):
        self.app.show_page("network")
        self.app.update()
        return self.app, self.app.arch_store, self.app.arch_id

    def make_tiny(self, blocks=1):
        app = self.app
        def mutate(spec):
            spec["embedding"].update(TINY["embedding"])
            spec["training"].update(TINY["training"])
            spec["blocks"] = spec["blocks"][:blocks]
            spec["blocks"][0]["attention"]["heads"] = 2
        self.assertTrue(app.net_edit(mutate, "tiny test network"))
        app.net_flush()

    def test_page_order_and_default_architecture(self):
        pages = [key for key, *_ in self.app.PAGES]
        self.assertEqual(pages[pages.index("blueprint") + 1:pages.index("blueprint") + 3], ["network", "training"])
        app, store, arch = self.open_page()
        self.assertEqual(len(store.list()), 1)
        self.assertEqual(validate_spec(app.net_spec)["blocks"][0]["attention"]["enabled"], True)
        self.assertEqual(len(store.revisions(arch)), 1)
        app.next_step()
        self.assertEqual(app.active_page, "training")

    def test_edit_blocks_commit_undo_redo_and_invalid_states(self):
        app, store, arch = self.open_page()
        start = len(app.net_spec["blocks"])
        first = app.net_spec["blocks"][0]["id"]
        added = app.net_insert_block("mlp", 1)
        self.assertEqual(len(app.net_spec["blocks"]), start + 1)
        self.assertEqual(app.net_spec["blocks"][1]["id"], added)
        self.assertFalse(app.net_spec["blocks"][1]["attention"]["enabled"])
        self.assertEqual(app.net_spec["blocks"][1]["label"], "Block 2")
        self.assertEqual(app.net_selected, f"b:{added}")
        self.assertEqual(app.net_error, "")
        app.net_flush()
        self.assertEqual(len(store.revisions(arch)), 2)
        # reorder: the same id moves, labels follow the position
        self.assertTrue(app.net_move_block(first, 2))
        self.assertEqual([b["id"] for b in app.net_spec["blocks"]][2], first)
        self.assertEqual([b["label"] for b in app.net_spec["blocks"]], ["Block 1", "Block 2", "Block 3"])
        app.net_flush()
        self.assertEqual(len(store.revisions(arch)), 3)
        self.assertIn("move block", store.revisions(arch)[0]["note"])
        clone = app.net_duplicate_block(first)
        self.assertEqual(len(app.net_spec["blocks"]), start + 2)
        self.assertNotEqual(clone, first)
        self.assertTrue(app.net_remove_block(clone))
        self.assertEqual(len(app.net_spec["blocks"]), start + 1)
        # field edits go through the same path; heads are only offered when they divide the width
        self.assertTrue(app.net_set_block(first, "attention.heads", 8))
        from studio_network import valid_heads
        self.assertEqual(valid_heads(64), [1, 2, 4, 8, 16])
        app.net_flush()
        committed = len(store.revisions(arch))
        # undo / redo walk the snapshots and commit the restored spec
        self.assertTrue(app.net_undo())
        self.assertEqual(next(b for b in app.net_spec["blocks"] if b["id"] == first)["attention"]["heads"], 4)
        app.net_flush()
        self.assertEqual(len(store.revisions(arch)), committed + 1)
        self.assertTrue(app.net_redo())
        self.assertEqual(next(b for b in app.net_spec["blocks"] if b["id"] == first)["attention"]["heads"], 8)
        app.net_flush()
        self.assertEqual(len(store.revisions(arch)), committed + 2)
        committed += 2
        # an invalid draft is shown in red, never committed, and cannot be trained
        before = len(store.revisions(arch))
        self.assertTrue(app.net_set_training("steps", 0))
        self.assertIn("Steps", app.net_error)
        self.assertEqual(app.net_canvas.err_keys, {"train"})
        self.assertEqual(str(app.net_train_button.cget("state")), "disabled")
        app.net_flush()
        self.assertEqual(len(store.revisions(arch)), before)
        self.assertTrue(app.net_undo())
        self.assertEqual(app.net_error, "")
        # a width change keeps heads valid
        self.assertTrue(app.net_set_embedding("width", 48))
        self.assertTrue(all(48 % b["attention"]["heads"] == 0 for b in app.net_spec["blocks"] if b["attention"]["enabled"]))
        self.assertEqual(app.net_error, "")
        # deleting the last blocks is refused
        for block in list(app.net_spec["blocks"])[1:]:
            app.net_remove_block(block["id"])
        self.assertFalse(app.net_remove_block(app.net_spec["blocks"][0]["id"]))

    def test_debounced_autocommit_and_restore_revision(self):
        app, store, arch = self.open_page()
        before = len(store.revisions(arch))
        app.net_insert_block("transformer")
        self.assertEqual(len(store.revisions(arch)), before)
        self.pump(1.0)
        self.assertEqual(len(store.revisions(arch)), before + 1)
        self.assertEqual(store.revisions(arch)[0]["note"], "insert transformer block")
        app.net_remove_block()
        app.net_flush()
        head = app.net_restore_revision(before + 1)
        self.assertEqual(len(app.net_spec["blocks"]), len(store.revision(arch, before + 1)["blocks"]))
        self.assertGreater(head, before + 1)

    def test_inspector_controls_use_the_same_edit_path(self):
        app, store, arch = self.open_page()
        block = app.net_spec["blocks"][0]["id"]
        app.net_select(f"b:{block}:mlp")
        heads = next(e for e in app.net_ins if e["kind"] == "choice" and e["get"]() == app.net_spec["blocks"][0]["attention"]["heads"])
        self.assertEqual(list(heads["widget"].cget("values")), ["1", "2", "4", "8", "16"])
        heads["var"].set("2")
        heads["widget"].event_generate("<<ComboboxSelected>>")
        self.assertEqual(app.net_spec["blocks"][0]["attention"]["heads"], 2)
        app.net_select("train")
        steps = next(e for e in app.net_ins if e["kind"] == "spin")
        steps["widget"].event_generate("<<Increment>>")
        self.assertEqual(app.net_spec["training"]["steps"], 350)
        # disabling attention is blocked when the MLP is off
        app.net_select(f"b:{block}")
        app.net_set_block(block, "mlp.enabled", False)
        toggle = next(e for e in app.net_ins if e["kind"] == "bool" and e["get"]() is True)
        self.assertEqual(str(toggle["widget"].cget("state")), "disabled")

    def test_canvas_interaction_hit_testing_and_drag(self):
        app, store, arch = self.open_page()
        app.geometry("1300x860+40+40")
        app.deiconify()
        self.pump(.6)
        canvas = app.net_canvas
        canvas.fit()
        self.pump(.3)
        blocks = canvas.geo["blocks"]
        self.assertGreaterEqual(len(blocks), 2)
        def centre(i):
            d = blocks[i]
            return canvas._p((d["x0"] + d["x1"]) / 2, d["board"][1] + 10)
        x, y = centre(0)
        self.assertEqual(canvas.key_at(x, y), "b:" + blocks[0]["id"])
        # click selects, empty space deselects
        canvas._press(SimpleNamespace(x=x, y=y))
        canvas._release(SimpleNamespace(x=x, y=y))
        self.assertEqual(app.net_selected, "b:" + blocks[0]["id"])
        # dragging the first block past the second reorders the circuit
        ids = [b["id"] for b in app.net_spec["blocks"]]
        x2, y2 = centre(1)
        canvas._press(SimpleNamespace(x=x, y=y))
        canvas._motion(SimpleNamespace(x=x + 40, y=y + 20))
        canvas._motion(SimpleNamespace(x=x2 + 100, y=y2))
        canvas._release(SimpleNamespace(x=x2 + 100, y=y2))
        self.assertEqual([b["id"] for b in app.net_spec["blocks"]][:2], [ids[1], ids[0]])
        # palette drop at slot 0 inserts a new block at the front
        slot = canvas.slot_at(canvas._p(-40, 0)[0])
        self.assertEqual(slot, 0)
        before = len(app.net_spec["blocks"])
        app._pal_press(SimpleNamespace(x_root=0, y_root=0), "attention", "Attention-only block")
        app._pal = dict(app._pal, moved=True, slot=0)
        app._pal_release(SimpleNamespace())
        self.assertEqual(len(app.net_spec["blocks"]), before + 1)
        self.assertFalse(app.net_spec["blocks"][0]["mlp"]["enabled"])
        # zoom keeps the pointer's world point steady
        z = canvas.z
        canvas.zoom(1.25, 300, 200)
        self.assertAlmostEqual(canvas.z, z * 1.25, places=6)
        app.net_delete_selected()
        app.withdraw()

    def test_train_from_page_telemetry_weights_and_initial_weights(self):
        app, store, arch = self.open_page()
        self.make_tiny()
        self.assertTrue(app.net_train())
        self.settle()
        self.assertGreater(app.net_tele_count, 0)
        self.assertIsNotNone(app.net_canvas.tele)
        self.assertIn(app.net_tele["blocks"][0]["id"], [b["id"] for b in app.net_spec["blocks"]])
        weights = store.weights(arch)
        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0]["step"], 20)
        self.assertEqual(weights[0]["source"], "train")
        saved = load_checkpoint(store.weight_path(arch, weights[0]["version"]))
        self.assertEqual(len(saved["config"]["architecture"]["blocks"]), 1)
        # the blueprint's own config is untouched by a network run
        self.assertNotIn("architecture", app.workspace.state.get("studioConfig", {}) or {})
        app.read_config()
        # start another run from those weights
        self.assertEqual(app.net_use_weights(weights[0]["version"]), weights[0]["version"])
        self.assertIn("v1", app.net_init_var.get())
        self.assertTrue(app.net_train())
        self.settle()
        weights = store.weights(arch)
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0]["parent"], 1)
        # register an external checkpoint
        path = store.weight_path(arch, 1)
        self.assertEqual(app.net_register_checkpoint(path), 3)

    def test_apply_live_edit_continues_from_saved_weights(self):
        app, store, arch = self.open_page()
        self.make_tiny()
        app.net_set_training("steps", 4000)
        self.assertTrue(app.net_train())
        deadline = time.monotonic() + 30
        while app.net_tele_count < 2 and time.monotonic() < deadline:
            app.update()
            time.sleep(.01)
        self.assertGreaterEqual(app.net_tele_count, 2)
        self.assertTrue(app.net_is_training())
        app.net_insert_block("mlp")
        app.net_set_training("steps", 25)
        self.assertTrue(app.net_apply_live())
        self.settle(90)
        weights = store.weights(arch)
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0]["parent"], weights[1]["version"])
        newest = load_checkpoint(store.weight_path(arch, weights[0]["version"]))
        self.assertEqual(len(newest["config"]["architecture"]["blocks"]), 2)
        self.assertEqual(newest["step"], 25)

    def test_backup_and_restore(self):
        app, store, arch = self.open_page()
        self.make_tiny()
        app.net_train()
        self.settle()
        blocks = len(app.net_spec["blocks"])
        path = app.net_backup()
        self.assertTrue(Path(path).exists())
        self.assertEqual(len(store.backups()), 1)
        app.net_insert_block("transformer")
        app.net_flush()
        with patch("studio_network.messagebox.askyesno", return_value=False):
            self.assertIsNone(app.net_restore_backup(path))
        self.assertEqual(len(app.net_spec["blocks"]), blocks + 1)
        with patch("studio_network.messagebox.askyesno", return_value=True) as ask:
            report = app.net_restore_backup(path)
        self.assertTrue(ask.called)
        self.assertEqual(report["architectures"], 1)
        self.assertEqual(len(app.net_spec["blocks"]), blocks)
        self.assertEqual(len(store.backups()), 2)
        self.assertEqual(len(store.weights(arch)), 1)

    def test_viewer_messages_use_the_edit_path(self):
        app, store, arch = self.open_page()
        viz = app.viz = FakeViz()
        first = app.net_spec["blocks"][0]["id"]
        revisions = len(store.revisions(arch))
        self.assertTrue(app.net_handle_viewer({"type": "select", "block_id": first}))
        self.assertEqual(app.net_selected, f"b:{first}")
        self.assertEqual(viz.sent, [])
        self.assertTrue(app.net_handle_viewer({"type": "edit", "block_id": first, "path": "attention.heads", "value": 2}))
        self.assertEqual(app.net_spec["blocks"][0]["attention"]["heads"], 2)
        self.assertEqual(len(store.revisions(arch)), revisions + 1)
        self.assertEqual(viz.sent[-1]["type"], "spec")
        self.assertEqual(viz.sent[-1]["spec"]["blocks"][0]["attention"]["heads"], 2)
        # invalid edits are rejected and the viewer gets the unchanged spec back
        sent = len(viz.sent)
        self.assertFalse(app.net_handle_viewer({"type": "edit", "block_id": first, "path": "attention.heads", "value": 5}))
        self.assertEqual(app.net_spec["blocks"][0]["attention"]["heads"], 2)
        self.assertEqual(viz.sent[sent]["type"], "spec")
        self.assertFalse(app.net_handle_viewer({"type": "edit", "block_id": first, "path": "bogus", "value": 1}))
        self.assertTrue(app.net_handle_viewer({"type": "edit", "block_id": first, "path": "mlp.expansion", "value": 2}))
        count = len(app.net_spec["blocks"])
        self.assertTrue(app.net_handle_viewer({"type": "add_block", "after": first}))
        self.assertEqual(len(app.net_spec["blocks"]), count + 1)
        added = app.net_spec["blocks"][1]["id"]
        self.assertNotEqual(added, first)
        self.assertTrue(app.net_handle_viewer({"type": "add_block", "after": None}))
        self.assertEqual(len(app.net_spec["blocks"]), count + 2)
        self.assertTrue(app.net_handle_viewer({"type": "remove_block", "block_id": added}))
        self.assertNotIn(added, [b["id"] for b in app.net_spec["blocks"]])
        self.assertEqual(len(store.revisions(arch)), store.revisions(arch)[0]["number"])
        self.assertEqual(validate_spec(store.get(arch)["spec"])["blocks"], validate_spec(app.net_spec)["blocks"])
        # telemetry is forwarded to the viewer as it arrives
        app.net_telemetry({"type": "telemetry", "step": 1, "total": 2, "loss": 1.0, "lr": 0.1,
                           "embedding": {}, "blocks": [], "output": {}})
        self.assertEqual(viz.sent[-1]["type"], "telemetry")

    def test_session_state_jobs_and_viewer_startup(self):
        app, store, arch = self.open_page()
        # a failed run never restarts itself later
        app._net_run, app._net_live_pending = {"arch": arch}, True
        with patch("studio_gui.messagebox.showerror"):
            app.job_error(("boom", "trace"))
        self.assertEqual((app._net_run, app._net_live_pending), (None, False))
        # notes that could not be committed do not follow the user to another architecture
        app.net_error, app.net_notes = "broken", ["stale note"]
        app.net_open(arch)
        self.assertEqual(app.net_notes, [])
        # Versions › Backups restores its own selection, not the inspector's
        path = app.net_backup()
        versions = app.net_views["Versions"]
        versions.refresh()
        versions.trees["Backups"].selection_set(path)
        app.backup_tree.selection_remove(*app.backup_tree.selection())
        with patch("studio_network.messagebox.askyesno", return_value=True), \
                patch.object(store, "restore", return_value={"architectures": 1}) as restore:
            versions.restore_backup()
        restore.assert_called_once_with(path)
        # Experiments is a host job: refused while another one runs
        app.job_kind = "training"
        try:
            app.net_views["Experiments"].start()
        finally:
            app.job_kind = None
        self.assertIn("Another task", app.net_views["Experiments"].message.get())
        # only the newest inspection request is delivered
        done, view = Mock(), app.net_views["Trace a token"]
        view.work(lambda: time.sleep(.2) or "old", done)
        view.work(lambda: "new", done)
        self.pump(.5)
        done.assert_called_once_with(True, "new")
        # weight edits from the viewer are saved one at a time
        viz = app.viz = FakeViz()
        app._net_weight_edit_busy = True
        app.net_edit_viewer_weight({"weight": "x", "row": 0, "column": 0, "delta": .1})
        self.assertIn("still being saved", viz.sent[-1]["message"])
        app._net_weight_edit_busy = False
        # a viewer that dies while starting shows the end of its log
        log = Path(self.directory.name) / "viewer.log"
        log.write_text("Traceback\nImportError: no panda3d\n", encoding="utf-8")
        viz.connected, viz.exit_code, viz.log_path = False, 1, log
        app._net_viewer_seen = False
        with patch("studio_network.messagebox.showerror") as error:
            app.net_poll_viewer()
        self.assertIn("no panda3d", error.call_args[0][1])
        self.assertIsNone(app._net_viewer_job)  # and polling stops with the process

    def test_architecture_management(self):
        app, store, arch = self.open_page()
        second = app.net_new("Second network")
        self.assertEqual(len(store.list()), 2)
        self.assertEqual(app.arch_id, second)
        app.net_rename("Renamed")
        self.assertEqual(store.get(second)["name"], "Renamed")
        export = Path(self.directory.name) / "export.json"
        app.net_export(str(export))
        app.net_import(str(export))
        self.assertEqual(len(store.list()), 3)
        app.net_open(arch)
        self.assertEqual(app.arch_id, arch)
        with patch("studio_network.messagebox.askyesno", return_value=True):
            self.assertTrue(app.net_delete())
        self.assertEqual(len(store.list()), 2)


if __name__ == "__main__":
    unittest.main()

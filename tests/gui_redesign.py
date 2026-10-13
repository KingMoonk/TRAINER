"""Redesigned pages: drill-downs, Start here checklist, sidebar groups, dataset modes, Beginner/Advanced visibility.

One Studio window is shared by the whole class (building it is the slow part); each test restores what it changes.
The class skips when Tk has no display.
"""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import json
import os
import shutil
import tempfile
import time
import unittest
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch

from studio_align import Collapsible, TAB_AUTO, TAB_CHECK, TAB_COMPARE, TAB_IMPROVE, TAB_RATE, TAB_SAVED
from studio_core import Workspace
from studio_gui import Studio
from studio_widgets import ACCENT


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def drawers(page):
    return {w.title: w for w in walk(page) if isinstance(w, Collapsible)}


def buttons(page, text):
    return [w for w in walk(page) if isinstance(w, ttk.Button) and str(w.cget("text")) == text]


class RedesignedUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        cls.workspace = Workspace(cls.directory)
        try:
            cls.app = Studio(cls.workspace)
        except tk.TclError as exc:
            shutil.rmtree(cls.directory, ignore_errors=True)
            if "display" in str(exc).lower():
                raise unittest.SkipTest("Tk display is unavailable")
            raise
        cls.app.withdraw()
        cls.app.update()
        # A modal dialog would hang the suite: record instead of showing.
        cls.dialogs = []
        cls.patches = [patch(f"{module}.messagebox.{name}", side_effect=lambda *a, _n=name, **k: cls.dialogs.append((_n, a)))
                       for module in ("studio_gui", "studio_align", "studio_flow", "studio_curriculum_page") for name in ("showerror", "showinfo")]
        for item in cls.patches:
            item.start()

    @classmethod
    def tearDownClass(cls):
        for item in cls.patches:
            item.stop()
        app = cls.app
        app.stop.set()
        if app.worker:
            app.worker.join(timeout=20)
        app.destroy()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def setUp(self):
        self.app.set_ui_mode("beginner")
        self.app.update()

    def page(self, name):
        self.app.show_page(name)
        self.app.update()
        return getattr(self.app.pages[name], "body", self.app.pages[name])

    def settle(self):
        deadline = time.monotonic() + 20
        while self.app.job_kind and time.monotonic() < deadline:
            self.app.update()
            time.sleep(.01)
        self.app.update()
        self.assertIsNone(self.app.job_kind)

    def add_dataset(self, name, count):
        dataset = self.workspace.create_dataset(name)
        if count:
            self.workspace.add(dataset, [{"user": f"{name} {i}", "assistant": "ok"} for i in range(count)])
        self.app.refresh_datasets(dataset)
        return dataset

    def make_run(self, name, **fields):
        run = self.workspace.run_dir / name
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"name": name, "status": "completed", "createdAt": "2030", **fields}), encoding="utf-8")
        self.addCleanup(shutil.rmtree, run, True)
        return run

    # ------------------------------------------------------------ drill-down widgets
    def test_collapsible_toggles(self):
        box = Collapsible(self.app, "Details")
        try:
            self.assertFalse(box.opened)
            self.assertEqual(box.body.winfo_manager(), "")
            self.assertTrue(str(box.toggle_button.cget("text")).startswith("▸"))
            box.toggle()
            self.assertTrue(box.opened)
            self.assertEqual(box.body.winfo_manager(), "pack")
            self.assertTrue(str(box.toggle_button.cget("text")).startswith("▾"))
            box.toggle(True)                      # explicit open is idempotent
            self.assertEqual(box.body.winfo_manager(), "pack")
            box.toggle(False)
            self.assertFalse(box.opened)
            self.assertEqual(box.body.winfo_manager(), "")
            box.toggle_button.invoke()            # the button itself drives it
            self.assertTrue(box.opened)
        finally:
            box.destroy()

    def test_drilldown_helper_shows_and_hides_panel(self):
        row, panel = ttk.Frame(self.app), ttk.Frame(self.app)
        row.pack()
        try:
            button = self.app.drilldown(row, "More", panel)
            self.assertEqual(str(button.cget("text")), "▸ More")
            self.assertEqual(panel.winfo_manager(), "")
            button.invoke()
            self.assertEqual(panel.winfo_manager(), "pack")
            self.assertEqual(str(button.cget("text")), "▾ More")
            button.base = "Less"
            button.sync()
            self.assertEqual(str(button.cget("text")), "▾ Less")
            button.invoke()
            self.assertEqual(panel.winfo_manager(), "")
            self.assertEqual(str(button.cget("text")), "▸ Less")
        finally:
            row.destroy()
            panel.destroy()

    # ------------------------------------------------------------ sidebar
    def test_sidebar_groups_precede_their_first_page(self):
        order = self.app.sidebar.pack_slaves()

        def text(widget):
            return str(widget.cget("text")) if isinstance(widget, (ttk.Label, ttk.Button)) else None
        for group, page in (("DATA", "catalog"), ("MODEL", "blueprint"), ("IMPROVE", "align")):
            self.assertEqual(text(order[order.index(self.app.nav[page]) - 1]), group)
        self.assertEqual([text(w) for w in order if text(w) in {"DATA", "MODEL", "IMPROVE"}], ["DATA", "MODEL", "IMPROVE"])

    def test_network_designer_hidden_in_beginner_and_restored_in_place(self):
        app = self.app
        self.assertEqual(app.nav["network"].winfo_manager(), "")
        self.assertEqual(app.hidden_pages(), {"network"})
        app.set_ui_mode("advanced")
        order = app.sidebar.pack_slaves()
        self.assertEqual(app.nav["network"].winfo_manager(), "pack")
        self.assertEqual(order.index(app.nav["network"]), order.index(app.nav["blueprint"]) + 1)
        self.assertLess(order.index(app.nav["training"]), order.index(app.nav["align"]))
        self.assertEqual(app.hidden_pages(), set())
        # visible pages are numbered without gaps in both modes
        def numbers():
            return [int(str(app.nav[k].cget("text"))[:2]) for k, *_ in app.PAGES if k not in app.hidden_pages() and str(app.nav[k].cget("text"))[:2].isdigit()]
        self.assertEqual(numbers(), list(range(1, len(numbers()) + 1)))
        app.set_ui_mode("beginner")
        self.assertEqual(numbers(), list(range(1, len(numbers()) + 1)))

    def test_leaving_advanced_from_network_page_moves_to_training_setup(self):
        self.app.set_ui_mode("advanced")
        self.page("network")
        self.app.set_ui_mode("beginner")
        self.assertEqual(self.app.active_page, "blueprint")
        with self.assertRaises(ValueError):
            self.app.set_ui_mode("expert")
        self.assertEqual(self.app.ui_mode, "beginner")

    def test_mode_is_remembered_in_the_workspace(self):
        self.app.set_ui_mode("advanced")
        self.assertEqual(json.loads(self.workspace.path.read_text(encoding="utf-8"))["uiMode"], "advanced")

    # ------------------------------------------------------------ Start here
    def test_start_here_rows_show_status_and_only_next_has_a_button(self):
        app = self.app
        keys = list(app.home_rows)
        self.assertTrue(all({"marker", "title", "button"} <= set(row) for row in app.home_rows.values()))

        def read():
            app.refresh_home()
            progress = {item["key"]: item for item in app.home_progress()}
            words = {k: str(app.home_rows[k]["marker"].cget("text")) for k in keys}
            shown = [k for k in keys if app.home_rows[k]["button"].winfo_manager()]
            return progress, words, shown

        def check(progress, words, shown):
            upcoming = next((k for k in keys if not progress[k]["done"] and not progress[k]["optional"]), None)
            for k in keys:
                item = progress[k]
                expected = "Done" if item["done"] else "Next" if k == upcoming else "Optional" if item["optional"] else "To do"
                self.assertEqual(words[k], expected, k)
            self.assertEqual(shown, [upcoming] if upcoming else [])
            return upcoming
        check(*read())
        self.add_dataset("Home data", 2)
        progress, words, shown = read()
        self.assertEqual(words["data"], "Done")
        check(progress, words, shown)
        self.make_run("run_home")
        self.app.chat_model = object()
        try:
            progress, words, shown = read()
            self.assertTrue(progress["train"]["done"] and progress["chat"]["done"])
            check(progress, words, shown)
        finally:
            self.app.chat_model = None

    def test_dpo_run_counts_as_feedback_not_as_a_trained_model(self):
        before = {item["key"]: item["done"] for item in self.app.home_progress()}
        self.make_run("run_dpo", stage="dpo")
        after = {item["key"]: item["done"] for item in self.app.home_progress()}
        self.assertEqual(after["train"], before["train"])
        self.assertTrue(after["improve"])

    # ------------------------------------------------------------ Training setup: dataset mode
    def test_dataset_mode_headline_and_rotation_text(self):
        app = self.app
        first, second = self.add_dataset("Mode A", 3), self.add_dataset("Mode B", 2)
        self.add_dataset("Mode empty", 0)
        known = [d["id"] for d in self.workspace.state["datasets"] if app.dataset_stats(d["id"])["records"] > 0]
        a, b = known.index(first), known.index(second)
        total = sum(app.dataset_stats(k)["records"] for k in known)
        original = app.dataset_mode.get()
        try:
            app.dataset_mode.set("everything")
            app.dataset_mode_changed()
            self.assertEqual(app.data_summary.get(), f"Train on everything  ·  {total:,} examples")
            self.assertIn(f"all {len(known)} dataset(s) stage by stage", app.rotation_text.get())
            self.assertEqual(app.manual_frame.winfo_manager(), "")
            self.assertEqual(app.run_sources(), known)

            app.dataset_mode.set("rotate")
            app.workspace.state["rotationIndex"] = a
            app.dataset_mode_changed()
            self.assertEqual(app.data_summary.get(), "One dataset per run  ·  3 examples")
            self.assertTrue(app.rotation_text.get().startswith(f"Next run: Mode A  ({a + 1} of {len(known)})"))
            self.assertEqual(app.manual_frame.winfo_manager(), "")
            app.workspace.state["rotationIndex"] = b
            app.refresh_rotation_text()
            self.assertTrue(app.rotation_text.get().startswith("Next run: Mode B"))
            self.assertEqual(app.run_sources(), [second])

            app.dataset_mode.set("manual")
            app.dataset_mode_changed()
            self.assertEqual(app.manual_frame.winfo_manager(), "pack")
            self.assertTrue(app.rotation_text.get().startswith("Choosing by hand"))
            app.refresh_mixture()
            app.mixture_tree.selection_set([second])
            self.assertEqual(app.run_sources(), [second])
            app.update_estimate()
            self.assertEqual(app.data_summary.get(), "Chosen by hand  ·  2 examples")
            self.assertEqual(Workspace(self.directory).state["datasetMode"], "manual")   # survives a restart
        finally:
            app.dataset_mode.set(original)
            app.dataset_mode_changed()

    def test_start_training_dispatches_to_stage_plan_only_in_everything_mode(self):
        app = self.app
        calls = []
        app.start_curriculum = lambda: calls.append(1)     # the plan itself is covered by test_curriculum and test_gui
        original = app.dataset_mode.get()
        try:
            app.dataset_mode.set("everything")
            self.assertFalse(app.start_training())          # stub started no job
            self.assertEqual(calls, [1])
            self.assertEqual(app.active_page, "training")
            app.busy = lambda: True                         # a busy app never starts anything
            self.assertFalse(app.start_training())
            self.assertEqual(calls, [1])
        finally:
            del app.start_curriculum
            if "busy" in app.__dict__:
                del app.busy
            app.dataset_mode.set(original)

    # ------------------------------------------------------------ Training setup drawers
    def test_training_setup_drawers_closed_and_hold_working_fields(self):
        page = self.page("blueprint")
        boxes = drawers(page)
        self.assertTrue({"Name and training steps", "Model shape", "Learning settings", "Device"} <= set(boxes))
        self.assertFalse(any(b.opened for b in boxes.values()))
        name = boxes["Name and training steps"]
        name.toggle(True)
        used = {str(e.cget("textvariable")) for e in walk(name.body) if isinstance(e, ttk.Entry)}
        self.assertTrue({str(self.app.config_vars["name"]), str(self.app.config_vars["steps"])} <= used)
        device = boxes["Device"]
        device.toggle(True)
        combo = next(w for w in walk(device.body) if isinstance(w, ttk.Combobox))
        self.assertEqual(str(combo.cget("textvariable")), str(self.app.config_vars["device"]))
        self.assertEqual(list(combo.cget("values")), ["auto", "cpu", "cuda"])
        for box in boxes.values():
            box.toggle(False)

    def test_beginner_hides_expert_blueprint_controls_and_advanced_restores_them(self):
        page = self.page("blueprint")
        boxes = drawers(page)
        for title in ("Model shape", "Learning settings"):
            self.assertEqual(boxes[title].winfo_manager(), "", title)
        for title in ("Name and training steps", "Device"):
            self.assertEqual(boxes[title].winfo_manager(), "pack", title)
        self.assertEqual([b.winfo_manager() for b in buttons(page, "Save blueprint")], [""])
        self.assertEqual([b.winfo_manager() for b in buttons(page, "Design the layers yourself")], [""])
        hints = [w for w in walk(page) if isinstance(w, ttk.Label) and "unlock the Network designer" in str(w.cget("text"))]
        self.assertEqual([h.winfo_manager() for h in hints], ["pack"])
        self.assertEqual(self.app.next_action.cget("text"), "Go to Training →")
        self.app.set_ui_mode("advanced")
        self.app.update()
        for title in ("Model shape", "Learning settings"):
            self.assertEqual(boxes[title].winfo_manager(), "pack", title)
        self.assertEqual([b.winfo_manager() for b in buttons(page, "Save blueprint")], ["pack"])
        self.assertEqual([h.winfo_manager() for h in hints], [""])
        self.assertEqual(self.app.next_action.cget("text"), "Design the network →")

    def test_advanced_drawers_return_to_their_original_order(self):
        page = self.page("blueprint")
        self.app.set_ui_mode("advanced")
        self.app.update()
        order = [w.title for w in page.pack_slaves() if isinstance(w, Collapsible)]
        self.assertEqual(order[:4], ["Name and training steps", "Model shape", "Learning settings", "Device"])

    # ------------------------------------------------------------ Training page
    def test_stage_card_sits_above_drawers_and_is_not_advanced_only(self):
        app = self.app
        page = self.page("training")
        slaves = page.pack_slaves()
        self.assertEqual(app.curriculum_box.winfo_manager(), "pack")      # visible in Beginner mode
        for title, box in drawers(page).items():
            self.assertLess(slaves.index(app.curriculum_box), slaves.index(box), title)
        self.assertFalse(app.curriculum_open)
        self.assertEqual(app.curriculum_body.winfo_manager(), "")
        app.toggle_curriculum()
        self.assertEqual(app.curriculum_body.winfo_manager(), "pack")
        self.assertTrue(str(app.curriculum_toggle.cget("text")).startswith("▾"))
        app.toggle_curriculum()
        self.assertEqual(app.curriculum_body.winfo_manager(), "")
        app.toggle_curriculum(True)
        app.toggle_curriculum(True)
        self.assertTrue(app.curriculum_open)
        app.toggle_curriculum(False)
        self.assertFalse(app.curriculum_open)
        app.set_ui_mode("advanced")
        self.assertEqual(app.curriculum_box.winfo_manager(), "pack")

    def test_training_drawers_hold_their_controls_and_follow_the_mode(self):
        app = self.app
        page = self.page("training")
        boxes = drawers(page)
        self.assertFalse(any(b.opened for b in boxes.values()))
        saved = boxes["Saved runs"]
        saved.toggle(True)
        self.assertIn(app.runs_tree, list(walk(saved.body)))
        self.assertTrue({"Refresh", "Set up a new run", "Chat with this model"} <= {str(b.cget("text")) for b in walk(saved.body) if isinstance(b, ttk.Button)})
        self.assertEqual([b.winfo_manager() for b in buttons(saved.body, "Resume checkpoint")], [""])
        self.assertEqual([b.winfo_manager() for b in buttons(saved.body, "Open folder")], [""])
        boxes["Speed and run details"].toggle(True)
        self.assertEqual(app.run_log.winfo_manager(), "")                  # log is Advanced-only
        app.set_ui_mode("advanced")
        self.assertEqual(app.run_log.winfo_manager(), "pack")
        self.assertEqual([b.winfo_manager() for b in buttons(saved.body, "Resume checkpoint")], ["pack"])
        for box in boxes.values():
            box.toggle(False)

    def test_chat_button_appears_on_done_and_hides_on_new_metrics(self):
        app = self.app
        self.assertFalse(app.run_map)
        self.assertEqual(app.chat_button.winfo_manager(), "")
        app.training_event({"type": "done", "run": {"status": "completed", "steps": 10}, "directory": "x"})
        self.assertEqual(app.chat_button.winfo_manager(), "pack")
        self.assertEqual(str(app.chat_button.cget("text")), "Chat with this model")
        app.training_event({"type": "metric", "step": 1, "total": 10, "loss": 2.5, "lr": .001})
        self.assertEqual(app.chat_button.winfo_manager(), "")
        self.assertEqual(app.metric_vars["step"].get(), "1")
        self.assertEqual(app.metric_vars["loss"].get(), "2.5000")
        self.assertEqual(app.metric_vars["val_loss"].get(), "—")
        app.training_event({"type": "done", "run": {"status": "stopped", "steps": 1}, "directory": "x"})
        self.assertEqual(app.chat_button.winfo_manager(), "pack")
        app.history = []

    # ------------------------------------------------------------ Improve with feedback
    def test_align_steps_hidden_tab_strip_and_highlight(self):
        app = self.app
        self.page("align")
        self.assertEqual([str(app.step_widgets[i][1].cget("text")) for i in range(4)], ["Check the model", "Vote", "Train", "Compare"])
        self.assertEqual(app.align_tabs.cget("style"), "Hidden.TNotebook")
        self.assertEqual([part for part, _ in ttk.Style(app).layout("Hidden.TNotebook.Tab")], ["null"])   # the tab strip is drawn empty
        expected = {TAB_CHECK: 0, TAB_RATE: 1, TAB_AUTO: 1, TAB_SAVED: 1, TAB_IMPROVE: 2, TAB_COMPARE: 3}
        for tab, active in expected.items():
            app.align_tabs.select(tab)
            app.update()
            lit = [i for i, (_, name) in enumerate(app.step_widgets) if str(name.cget("foreground")) == str(ACCENT)]
            self.assertEqual(lit, [active], tab)
        app.align_tabs.select(TAB_CHECK)

    def test_align_clicking_a_step_selects_its_tab(self):
        app = self.app
        self.page("align")
        for index, tab in enumerate((TAB_CHECK, TAB_RATE, TAB_IMPROVE, TAB_COMPARE)):
            app.step_widgets[index][1].event_generate("<Button-1>")
            app.update()
            self.assertEqual(app.align_tabs.index(app.align_tabs.select()), tab)
        app.align_tabs.select(TAB_CHECK)

    def test_align_navigation_buttons(self):
        app = self.app
        self.page("align")
        frames = [app.align_tabs.nametowidget(name) for name in app.align_tabs.tabs()]

        def current():
            return app.align_tabs.index(app.align_tabs.select())
        for tab in (TAB_AUTO, TAB_SAVED):
            app.align_tabs.select(tab)
            found = buttons(frames[tab], "← Back to voting")
            self.assertEqual(len(found), 1)
            found[0].invoke()
            self.assertEqual(current(), TAB_RATE)
        for text, tab in (("Train on these →", TAB_IMPROVE), ("Let a teacher model vote for me", TAB_AUTO), ("Saved votes", TAB_SAVED)):
            app.align_tabs.select(TAB_RATE)
            buttons(frames[TAB_RATE], text)[0].invoke()
            self.assertEqual(current(), tab)
        app.align_tabs.select(TAB_CHECK)
        buttons(frames[TAB_CHECK], "Next: vote →")[0].invoke()
        self.assertEqual(current(), TAB_RATE)
        app.align_tabs.select(TAB_IMPROVE)
        buttons(frames[TAB_IMPROVE], "Compare with the original →")[0].invoke()
        self.assertEqual(current(), TAB_COMPARE)
        app.align_tabs.select(TAB_CHECK)

    def test_align_model_name_follows_checkpoint(self):
        app = self.app
        before = app.align_checkpoint.get()
        try:
            for blank in ("", "   "):
                app.align_checkpoint.set(blank)
                self.assertEqual(app.align_model_name.get(), "Your votes polish (no model chosen yet)")
            app.align_checkpoint.set(os.path.join(self.directory, "runs", "base", "last.npz"))
            self.assertEqual(app.align_model_name.get(), "Your votes polish base · last.npz")
            app.align_checkpoint.set(os.path.join(self.directory, "runs", "other", "best.npz"))
            self.assertEqual(app.align_model_name.get(), "Your votes polish other · best.npz")
        finally:
            app.align_checkpoint.set(before)

    def test_align_use_latest_checkpoint_prefers_best_then_last(self):
        app = self.app
        before = app.align_checkpoint.get()
        run = self.make_run("run_ckpt")
        try:
            (run / "last.npz").write_bytes(b"")
            app.use_latest_checkpoint()
            self.assertEqual(app.align_checkpoint.get(), str(run / "last.npz"))
            (run / "best.npz").write_bytes(b"")
            app.use_latest_checkpoint()
            self.assertEqual(app.align_checkpoint.get(), str(run / "best.npz"))
            self.assertEqual(app.align_model_name.get(), "Your votes polish run_ckpt · best.npz")
            (run / "best.npz").unlink()
            (run / "last.npz").unlink()
            app.align_checkpoint.set("")
            app.use_latest_checkpoint()
            self.assertEqual(app.align_checkpoint.get(), "")
            self.assertIn("No checkpoints yet", app.status_var.get())
        finally:
            app.align_checkpoint.set(before)

    def test_align_change_model_drawer_closed_and_holds_working_controls(self):
        app = self.app
        boxes = drawers(self.page("align"))
        change = boxes["Change model"]
        self.assertFalse(change.opened)
        change.toggle(True)
        entry = next(w for w in walk(change.body) if isinstance(w, ttk.Entry))
        self.assertEqual(str(entry.cget("textvariable")), str(app.align_checkpoint))
        combo = next(w for w in walk(change.body) if isinstance(w, ttk.Combobox))
        self.assertEqual(str(combo.cget("textvariable")), str(app.inference_device))
        change.toggle(False)

    # ------------------------------------------------------------ Chat page drawers
    def test_chat_drawers_closed_and_hold_working_controls(self):
        app = self.app
        boxes = drawers(self.page("chat"))
        self.assertEqual(set(boxes), {"Change model", "Reply settings, system prompt, device"})
        self.assertFalse(any(b.opened for b in boxes.values()))
        change = boxes["Change model"]
        change.toggle(True)
        entry = next(w for w in walk(change.body) if isinstance(w, ttk.Entry))
        self.assertEqual(str(entry.cget("textvariable")), str(app.chat_checkpoint))
        self.assertEqual({str(b.cget("text")) for b in walk(change.body) if isinstance(b, ttk.Button)},
                         {"Use latest run", "Choose file…", "Load model"})
        more = boxes["Reply settings, system prompt, device"]
        more.toggle(True)
        used = {str(e.cget("textvariable")) for e in walk(more.body) if type(e) is ttk.Entry}
        self.assertEqual(used, {str(v) for v in (app.temperature, app.top_k, app.max_tokens, app.system_prompt)})
        combo = next(w for w in walk(more.body) if isinstance(w, ttk.Combobox))
        self.assertEqual(str(combo.cget("textvariable")), str(app.inference_device))
        app.chat_messages = [{"role": "user", "content": "hi"}]
        buttons(more.body, "Clear conversation")[0].invoke()
        self.assertEqual(app.chat_messages, [])
        for box in boxes.values():
            box.toggle(False)

    def test_chat_use_latest_run_fills_checkpoint(self):
        app = self.app
        run = self.make_run("run_chat")
        (run / "last.npz").write_bytes(b"")
        self.page("chat")
        buttons(drawers(app.pages["chat"])["Change model"].body, "Use latest run")[0].invoke()
        self.settle()              # it also tries to load the (empty, invalid) file in the background
        self.assertEqual(app.chat_checkpoint.get(), str(run / "last.npz"))
        app.chat_checkpoint.set("")

    def test_load_model_with_missing_file_reports_and_keeps_no_model(self):
        app = self.app
        self.page("chat")
        app.chat_checkpoint.set(os.path.join(self.directory, "nope.npz"))
        self.dialogs.clear()
        app.load_chat_model()
        self.settle()
        self.assertIsNone(app.chat_model)
        self.assertTrue(self.dialogs)
        app.chat_checkpoint.set("")


if __name__ == "__main__":
    unittest.main()

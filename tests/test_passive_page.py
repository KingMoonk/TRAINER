"""Passive tab on the Training page: pure helpers (no display) and the three views (skipped without a display)."""
import shutil
import tempfile
import unittest
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from unittest.mock import Mock, patch

from studio_core import Workspace
from studio_gui import Studio
from studio_passive import DEFAULT_PLAN, PassiveStore, merge_plan
from studio_passive_page import (chart_points, checkpoint_story, choose_view, duration_words, form_defaults, interruptions, journey, journey_layout,
                                 plan_from_form, stop_words, summary_sentence)

AT = "2026-01-01T22:10:00+00:00"


def story(base="/nonexistent"):
    """The engine's story: errors 5, 4, 2, 3, 3.5, 1.5 where 4 and 5 are discarded off 3 and 6 continues from 3."""
    errors, parents = [5, 4, 2, 3, 3.5, 1.5], [None, 1, 2, 3, 4, 3]
    return [{"n": i, "round": i, "path": str(Path(base) / f"{i}.npz"), "error": e, "elapsed": i * 600.0, "at": AT, "parent": p, "kept": True,
             "discarded": i in (4, 5), "pivot": "halve_lr" if i == 6 else None, "teacher": {"polished": i == 2, "win_rate": None, "lessons": 2 if i == 1 else 0}}
            for i, (e, p) in enumerate(zip(errors, parents), 1)]


class Helpers(unittest.TestCase):
    def test_choose_view(self):
        done = {"id": "a", "status": "finished"}
        self.assertEqual(choose_view({"id": "b", "status": "running"}, {"id": "b", "status": "running"}, None), "running")
        self.assertEqual(choose_view({"id": "b"}, done, "a"), "running")
        for status in ("finished", "stopped", "failed"):
            self.assertEqual(choose_view(None, {"id": "a", "status": status}, None), "catchup")
        self.assertEqual(choose_view(None, done, "a"), "start")
        self.assertEqual(choose_view(None, done, "other"), "catchup")
        self.assertEqual(choose_view(None, None, None), "start")

    def test_plan_from_form_round_trips(self):
        form = form_defaults()
        self.assertEqual(merge_plan(plan_from_form(form)), DEFAULT_PLAN)
        form.update(duration="30", unit="minutes", win="60", target="", improve="2.5", memory="A quarter of this computer's memory", awake=False, pivots=["halve_lr"])
        plan = merge_plan(plan_from_form(form))
        self.assertEqual((plan["stop"]["duration"], plan["stop"]["unit"], plan["stop"]["win_rate"], plan["stop"]["target_error"]), (30, "minutes", 0.6, None))
        self.assertEqual((plan["plateau"]["min_improvement"], plan["limits"]["memory_fraction"], plan["limits"]["keep_awake"], plan["worse"]["pivots"]), (0.025, 0.25, False, ["halve_lr"]))
        form.update(target="1.5")
        self.assertEqual(merge_plan(plan_from_form(form))["stop"]["target_error"], 1.5)

    def test_bad_numbers_surface_the_plans_message(self):
        for key, bad, word in (("duration", "lots", "time limit"), ("duration", "0", "time limit"), ("patience", "2.5", "Rounds"), ("win", "150", "head-to-head"), ("steps", "", "step limit")):
            with self.assertRaises(ValueError, msg=key) as caught:
                merge_plan(plan_from_form({**form_defaults(), key: bad}))
            self.assertIn(word, str(caught.exception))

    def test_chart_points_stay_inside(self):
        cps = story()
        for width, height in ((600, 260), (120, 100), (30, 30)):
            pts = chart_points(cps, [[3000.0, 1.2], [99999.0, 1.1]], 3600, width, height)
            self.assertEqual(len(pts["rings"]), 6)
            for _, x, y, _ in pts["rings"]:
                self.assertTrue(0 <= x <= max(width, 80) and 0 <= y <= max(height, 80))
            for x, y in pts["live"]:
                self.assertTrue(0 <= x <= max(width, 80) and 0 <= y <= max(height, 80))
        self.assertEqual(pts["live"][1][0], pts["box"][2])  # past the limit is clamped to the right edge

    def test_chart_points_edge_cases(self):
        self.assertEqual(chart_points([], [], 3600, 400, 200)["rings"], [])
        one = chart_points(story()[:1], [], 3600, 400, 200)
        self.assertEqual(len(one["rings"]), 1)
        same = chart_points([{"n": i, "error": 2.0, "elapsed": i * 10.0} for i in (1, 2, 3)], [[5.0, 2.0]], 3600, 400, 200)
        self.assertEqual(len({y for _, _, y, _ in same["rings"]}), 1)
        chart_points(story()[:2], [], 0, 400, 200)  # a zero limit must not divide by zero
        layout = journey_layout(story(), 500, 240)
        self.assertEqual(set(layout), {1, 2, 3, 4, 5, 6})
        self.assertEqual(journey_layout(story()[:1], 500, 240).keys(), {1})
        self.assertTrue(all(0 <= x <= 500 and 0 <= y <= 240 for x, y in layout.values()))

    def test_journey(self):
        tale = journey(story())
        self.assertEqual([c["n"] for c in tale["kept"]], [1, 2, 3, 6])
        self.assertEqual(len(tale["branches"]), 1)
        branch = tale["branches"][0]
        self.assertEqual((branch["parent"], [c["n"] for c in branch["items"]], branch["pivot"]), (3, [4, 5], "halve_lr"))
        self.assertEqual(journey(story()[:3]), {"kept": story()[:3], "branches": []})
        self.assertEqual(journey([]), {"kept": [], "branches": []})

    def session(self, **state):
        base = {"active_seconds": 8 * 3600, "best": 6, "checkpoints": story(), "stop_reason": "duration"}
        return {"status": "finished", "error": "", "state": {**base, **state}}

    def test_summary_sentence(self):
        text = summary_sentence(self.session())
        for part in ("8 hours", "6 checkpoints", "checkpoint 6", "1.500", "1 step back", "time limit was reached"):
            self.assertIn(part, text)
        for reason, words in (("patience", "stopped getting better"), ("target", "target test error"), ("user", "you stopped it"), ("pivots", "ran out of things to try"), ("win_rate", "beat the starting model")):
            self.assertIn(words, summary_sentence(self.session(stop_reason=reason)))
        failed = summary_sentence({**self.session(stop_reason=None), "status": "failed", "error": "It crashed."})
        self.assertIn("stopped early: It crashed.", failed)
        self.assertIn("gave up after repeated crashes", summary_sentence({**self.session(), "status": "failed"}))
        self.assertIn("no checkpoints", summary_sentence(self.session(checkpoints=[], best=None)))
        self.assertEqual(stop_words("nonsense"), "it came to an end")

    def test_duration_words(self):
        self.assertEqual([duration_words(s) for s in (1, 45, 120, 3600, 3900, 8 * 3600, 90000)], ["1 second", "45 seconds", "2 minutes", "1 hour", "1 hour 5 minutes", "8 hours", "1 day 1 hour"])

    def test_duration_words_survives_huge_durations(self):
        self.assertIn("longer", duration_words(float("inf")))
        self.assertIn("longer", duration_words(1e308 * 3600))

    def test_interruptions(self):
        self.assertEqual(interruptions([]), [])
        self.assertEqual(interruptions([{"type": "round"}, {"type": "check"}, {"type": "checkpoint"}]), [])
        lines = interruptions([{"type": "pause", "at": AT}, {"type": "resume", "at": AT}, {"type": "restart", "at": AT}, {"type": "memory", "at": AT, "round": 2},
                               {"type": "error", "at": AT, "message": "boom"}, {"type": "teacher", "at": AT, "skipped": "offline"}])
        self.assertEqual(len(lines), 6)
        for part, line in zip(("Paused", "Resumed", "restarted", "memory", "boom", "teacher"), lines):
            self.assertIn(part, line)

    def test_each_restart_shows_once(self):
        at = lambda minute: f"2026-01-01T22:{minute:02d}:00+00:00"  # noqa: E731
        watchdog = {"type": "restart", "reason": "stopped", "pid": 9, "at": at(10)}
        engine = {"type": "restart", "round": 3, "best": 2, "at": at(10)}
        for order in ((watchdog, engine), (engine, watchdog)):  # either process may journal first
            lines = interruptions(list(order))
            self.assertEqual(len(lines), 1)
            self.assertIn("its process had stopped", lines[0])
        self.assertEqual(len(interruptions([watchdog])), 1)  # the trainer died before it could say anything
        self.assertEqual(len(interruptions([engine])), 1)
        self.assertEqual(len(interruptions([watchdog, {**engine, "at": at(40)}])), 2)  # a later restart is a different one
        self.assertIn("stopped reporting", interruptions([{**watchdog, "reason": "silent"}])[0])

    def test_battery_pauses_say_why_without_doubling(self):
        power = lambda reason, at: {"type": "power", "reason": reason, "at": at}  # noqa: E731
        lines = interruptions([{"type": "power", "reason": "battery_paused", "at": "2026-01-01T22:10:01+00:00"}, {"type": "pause", "at": "2026-01-01T22:10:02+00:00"},
                               {"type": "resume", "at": "2026-01-01T23:10:00+00:00"}, power("battery_resumed", "2026-01-01T23:10:01+00:00"),
                               {"type": "pause", "at": "2026-01-02T01:00:00+00:00"}])
        self.assertEqual(len(lines), 3)
        self.assertIn("because the computer went on battery", lines[0])
        self.assertIn("because the computer was plugged in again", lines[1])
        self.assertTrue(lines[2].startswith("Paused at"))
        self.assertNotIn("because", lines[2])  # a pause the user made keeps the plain line

    def test_repeated_teacher_skips_are_one_line(self):
        skip = {"type": "teacher", "at": AT, "skipped": "offline"}
        self.assertEqual(len(interruptions([skip] * 40)), 1)
        self.assertIn("in 40 rounds", interruptions([skip] * 40)[0])

    def test_failed_summary_has_one_full_stop(self):
        text = summary_sentence({**self.session(stop_reason=None), "status": "failed", "error": "It gave up. See trainer.log."})
        self.assertTrue(text.endswith("See trainer.log."))
        self.assertNotIn("..", text)

    def test_checkpoint_story(self):
        cps = story()
        self.assertIn("Discarded", checkpoint_story(cps[3], 6))
        self.assertIn("Polished by the teacher", checkpoint_story(cps[1], 6))
        self.assertIn("2 new lessons", checkpoint_story(cps[0], 6))
        self.assertTrue(checkpoint_story(cps[5], 6).endswith("best"))


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


class Views(unittest.TestCase):
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
        cls.store = PassiveStore(cls.directory)
        cls.dialogs = []
        cls.patches = [patch(f"studio_passive_page.messagebox.{name}", side_effect=lambda *a, _n=name, **k: cls.dialogs.append(_n) or True) for name in ("showerror", "showinfo", "askyesno")]
        for item in cls.patches:
            item.start()
        app = cls.app
        app._passive_launch = Mock(return_value=1)  # never spawn a real process
        app._passive_health = Mock(return_value={"trainer": "Running", "heard": 3.0, "watchdog": "Running", "restarts": 0})
        app._passive_open_chat = Mock()

    @classmethod
    def tearDownClass(cls):
        for item in cls.patches:
            item.stop()
        cls.app.destroy()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def setUp(self):
        app = self.app
        if app._passive_job:
            app.after_cancel(app._passive_job)
            app._passive_job = None
        with self.store.connect() as db:
            db.execute("DELETE FROM sessions")
        app.workspace.state["datasets"], app.workspace.state["examples"] = [], []
        app.workspace.state.pop("passiveDismissed", None)
        app._stats_cache.clear()
        app.passive_base = None
        app._passive_launch.reset_mock()
        app._passive_open_chat.reset_mock()
        self.dialogs.clear()
        app.passive_show_tab("now")
        app.show_page("training")
        app._passive_key = None
        app.passive_show_tab("passive")
        app.update()

    def buttons(self, text, root=None):
        return [w for w in walk(root or self.app.passive_frame) if isinstance(w, ttk.Button) and str(w.cget("text")) == text]

    def seed(self, status="finished", **state):
        sid = self.store.create({"data": {"sources": ["x"]}})
        base = {"round": 6, "phase": "idle", "best": 6, "active_seconds": 8 * 3600, "stop_reason": "duration", "checkpoints": story(self.directory), "live": [],
                "teacher": {"model": "tutor", "reachable": True, "done": 0, "total": 0}, "memory": {"used": 2 ** 30, "limit": 2 ** 32}}
        self.store.update(sid, status=status, state={**base, **state})
        return sid

    def test_training_page_has_both_tabs_and_keeps_train_now(self):
        app = self.app
        self.assertEqual({str(b.cget("text")) for b in app._passive_buttons.values()}, {"Train now", "Passive"})
        self.assertIs(app.pages["training"].body, app.train_now_frame)
        for widget in (app.runs_tree, app.loss_plot, app.progress, app.curriculum_box, app.reading_drawer):
            self.assertTrue(str(widget).startswith(str(app.train_now_frame)), widget)
        app.passive_show_tab("now")
        app.update()
        self.assertEqual(app.train_now_frame.winfo_manager(), "pack")
        self.assertEqual(app.passive_frame.winfo_manager(), "")
        app.passive_show_tab("passive")
        self.assertEqual(app.train_now_frame.winfo_manager(), "")

    def test_empty_record_shows_start(self):
        self.assertEqual(self.app._passive_key, ("start", None))
        self.assertEqual(len(self.buttons("Start passive training")), 1)
        self.assertIsNone(self.app._passive_job)

    def test_start_without_data_is_refused(self):
        self.app.passive_start()
        self.assertEqual(self.dialogs, ["showerror"])
        self.assertEqual(self.store.list(), [])
        self.app._passive_launch.assert_not_called()

    def test_bad_number_is_refused_in_plain_words(self):
        dataset = self.workspace.create_dataset("Tiny")
        self.workspace.add(dataset, [{"user": "a", "assistant": "b"}])
        self.app._stats_cache.clear()
        self.app._pf["duration"].set("lots")
        with patch("studio_passive_page.messagebox.showerror") as shown:
            self.app.passive_start()
        self.assertIn("time limit must be a number", shown.call_args[0][1])
        self.assertEqual(self.store.list(), [])

    def test_valid_start_creates_one_queued_session_and_launches_once(self):
        app = self.app
        dataset = self.workspace.create_dataset("Tiny")
        self.workspace.add(dataset, [{"user": "a", "assistant": "b"}, {"user": "c", "assistant": "d"}])
        app._stats_cache.clear()
        app._pf["duration"].set("30")
        app._pf["unit"].set("minutes")
        app.passive_start()
        self.assertEqual(self.dialogs, [])
        sessions = self.store.list()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["status"], "queued")
        self.assertEqual((sessions[0]["plan"]["stop"]["duration"], sessions[0]["plan"]["stop"]["unit"]), (30, "minutes"))
        self.assertEqual(sessions[0]["plan"]["data"]["sources"], [dataset])
        self.assertEqual(sessions[0]["plan"]["config"]["name"], app.config_vars["name"].get())
        app._passive_launch.assert_called_once_with(sessions[0]["id"])
        self.assertEqual(app._passive_key, ("running", sessions[0]["id"]))
        app.passive_start()  # a second start while one is active is refused
        self.assertEqual(len(self.store.list()), 1)

    def test_failed_launch_marks_the_session_failed(self):
        dataset = self.workspace.create_dataset("Tiny")
        self.workspace.add(dataset, [{"user": "a", "assistant": "b"}])
        self.app._stats_cache.clear()
        self.app._passive_launch.side_effect = ImportError("no watchdog")
        self.app.status_var.set("")
        try:
            self.app.passive_start()
        finally:
            self.app._passive_launch.side_effect = None
        self.assertEqual(self.dialogs, ["showerror"])
        self.assertEqual(self.store.list()[0]["status"], "failed")
        self.assertIsNone(self.store.active())
        self.assertNotIn("started", self.app.status_var.get())

    def test_failed_build_is_retried_and_tick_survives_errors(self):
        app = self.app
        self.seed("failed", stop_reason=None)
        with patch.object(app, "_passive_build_catchup", side_effect=ValueError("bad build")):
            with self.assertRaises(ValueError):
                app.refresh_passive()
        self.assertIsNone(app._passive_key)
        app.refresh_passive()
        self.assertEqual(app._passive_key[0], "catchup")
        app.active_page = "training"
        with patch.object(app, "refresh_passive", side_effect=ValueError("tick boom")):
            app._passive_tick()
        self.assertIsNotNone(app._passive_job)
        self.assertIn("tick boom", app.status_var.get())

    def test_done_keeps_pending_lessons_reviewable(self):
        app = self.app
        sid = self.seed()
        self.store.set_lessons(sid, [{"id": "l1", "round": 1, "checkpoint": 1, "prompt": "Q", "reply": "r", "answer": "A", "messages": [{"role": "user", "content": "Q"}], "status": "pending", "at": AT}])
        app.refresh_passive()
        self.buttons("Done")[0].invoke()
        self.assertEqual(app._passive_key, ("catchup", sid))
        self.assertIn("showinfo", self.dialogs)
        self.assertEqual(self.store.lessons(sid)[0]["status"], "pending")

    def test_running_view_pause_resume_stop(self):
        app = self.app
        sid = self.seed("running", phase="teach", live=[[300.0, 2.0]], checkpoints=story(self.directory)[:2],
                        teacher={"model": "tutor", "reachable": True, "done": 3, "total": 8})
        self.store.append(sid, {"type": "start"})
        app.refresh_passive()
        self.assertEqual(app._passive_key, ("running", sid))
        self.assertIsNotNone(app._passive_job)
        self.assertIn("voted on 3 of 8", app._pv["round"].get())
        self.assertEqual(app._pv["strip"].active, 2)
        self.assertEqual(len(app._pv["tree"].get_children()), 2)
        self.assertEqual(app._pv["keeping"].get(), "All well · no restarts this session")
        self.buttons("Pause")[0].invoke()
        self.assertEqual(self.store.get(sid)["command"], "pause")
        self.store.update(sid, status="paused", command="")
        app.refresh_passive()
        self.assertEqual(len(self.buttons("Resume")), 1)
        self.buttons("Resume")[0].invoke()
        self.assertEqual(self.store.get(sid)["command"], "resume")
        self.buttons("Stop and keep the best")[0].invoke()
        self.assertEqual(self.store.get(sid)["command"], "stop")
        self.assertIn("askyesno", self.dialogs)

    def test_polling_stops_when_hidden_and_on_destroy(self):
        app = self.app
        self.seed("running")
        app.refresh_passive()
        job = app._passive_job
        self.assertIsNotNone(job)
        app.passive_show_tab("now")
        app._passive_tick()
        self.assertIsNone(app._passive_job)  # the Train now tab is showing: no more polling
        app.passive_show_tab("passive")
        self.assertIsNotNone(app._passive_job)

    def test_catchup_view_lessons_keep_and_drop(self):
        app = self.app
        sid = self.seed()
        lessons = [{"id": f"l{i}", "round": 1, "checkpoint": 1, "prompt": f"Q{i}", "reply": "bad", "answer": f"A{i}", "messages": [{"role": "user", "content": f"Q{i}"}], "status": "pending", "at": AT}
                   for i in (1, 2, 3)]
        self.store.set_lessons(sid, lessons)
        for event in ({"type": "pause"}, {"type": "resume"}, {"type": "restart", "round": 2}):
            self.store.append(sid, event)
        app.refresh_passive()
        self.assertEqual(app._passive_key, ("catchup", sid))
        labels = [str(w.cget("text")) for w in walk(app.passive_frame) if isinstance(w, ttk.Label)]
        self.assertIn("Welcome back.", labels)
        self.assertTrue(any("checkpoint 6" in text for text in labels))
        self.assertTrue(any("restarted" in text for text in labels))
        self.assertEqual(len(app._pv["tree"].get_children()), 6)
        self.assertIn("3 new lessons", app._pv["lessons_summary"].get())
        before = self.workspace.state.get("selectedDatasetId")
        app._pv["lessons"].selection_set("l1")
        self.buttons("Keep")[0].invoke()
        dataset = next(d["id"] for d in self.workspace.state["datasets"] if d["name"] == "Passive lessons")
        self.assertEqual(self.workspace.count(dataset), 1)
        self.assertEqual(self.workspace.state.get("selectedDatasetId"), before)
        self.assertEqual({r["id"]: r["status"] for r in self.store.lessons(sid)}, {"l1": "kept", "l2": "pending", "l3": "pending"})
        app._pv["lessons"].selection_set("l2")
        self.buttons("Drop")[0].invoke()
        self.assertEqual(self.workspace.count(dataset), 1)
        self.assertEqual({r["id"]: r["status"] for r in self.store.lessons(sid)}["l2"], "dropped")
        self.buttons("Keep all 1")[0].invoke()
        self.assertEqual(self.workspace.count(dataset), 2)
        self.assertEqual({r["status"] for r in self.store.lessons(sid)}, {"kept", "dropped"})
        self.assertEqual(sum(d["name"] == "Passive lessons" for d in self.workspace.state["datasets"]), 1)

    def test_collapsed_summaries_are_filled_in(self):
        app = self.app
        shown = lambda: [str(w.cget("textvariable")) and w.getvar(str(w.cget("textvariable"))) for w in walk(app.passive_frame) if isinstance(w, ttk.Label) and str(w.cget("textvariable"))]  # noqa: E731
        self.assertTrue(all(shown()), shown())  # the start view, including "If something crashes"
        self.assertIn("It restarts itself, even with Groundwork closed.", shown())
        sid = self.seed()
        self.store.append(sid, {"type": "pause", "at": AT})
        app.refresh_passive()
        self.assertTrue(all(shown()), shown())  # "Checkpoint by checkpoint" and "Interruptions" are not blank
        self.assertIn("1 during the session", shown())

    def test_keep_and_drop_are_not_offered_while_a_session_is_active(self):
        sid = self.seed("running")
        self.store.set_lessons(sid, [{"id": "l1", "round": 1, "checkpoint": 1, "prompt": "Q", "reply": "r", "answer": "A", "messages": [], "status": "pending", "at": AT}])
        self.app.refresh_passive()
        self.assertEqual(self.app._passive_key, ("running", sid))
        self.assertEqual((self.buttons("Keep"), self.buttons("Drop")), ([], []))  # the trainer may be appending to lessons.jsonl
        self.assertNotIn("lessons", self.app._pv)

    def test_a_picked_checkpoint_survives_the_refresh(self):
        app = self.app
        self.seed("running", checkpoints=story(self.directory)[:3])
        app.refresh_passive()
        app._pv["tree"].selection_set("2")
        app.refresh_passive()
        self.assertEqual(app._pv["tree"].selection(), ("2",))

    def test_continue_from_the_best_presets_the_base_checkpoint(self):
        app = self.app
        best = Path(self.directory) / "6.npz"
        best.write_bytes(b"x")
        sid = self.seed()
        app.refresh_passive()
        self.buttons("Chat with the best checkpoint")[0].invoke()
        app._passive_open_chat.assert_called_once_with(str(best))
        self.buttons("Continue from the best")[0].invoke()
        self.assertEqual(app._passive_key, ("start", None))
        self.assertEqual(app.passive_base, str(best))
        self.assertEqual(self.workspace.state["passiveDismissed"], sid)
        self.assertTrue(any("Starting from the best checkpoint" in str(w.cget("text")) for w in walk(app.passive_frame) if isinstance(w, ttk.Label)))
        best.unlink()

    def test_done_returns_to_start_and_stays_dismissed(self):
        app = self.app
        sid = self.seed("failed", stop_reason=None)
        self.store.update(sid, error="The trainer crashed 5 times.")
        app.refresh_passive()
        self.assertEqual(app._passive_key, ("catchup", sid))
        self.assertTrue(any("The trainer crashed 5 times." in str(w.cget("text")) for w in walk(app.passive_frame) if isinstance(w, ttk.Label)))
        self.buttons("Done")[0].invoke()
        self.assertEqual(app._passive_key, ("start", None))
        app.refresh_passive()
        app.show_page("training")
        self.assertEqual(app._passive_key, ("start", None))
        self.assertEqual(Workspace(self.directory).state.get("passiveDismissed"), sid)  # saved to disk


if __name__ == "__main__":
    unittest.main()

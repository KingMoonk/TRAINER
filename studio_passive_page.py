"""Training-page tab "Passive": start a background session, watch it run, catch up when it ends. The record is studio_passive.PassiveStore."""
from __future__ import annotations

import math
import traceback
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from studio_align import Collapsible, setting_rows
from studio_passive import DEFAULT_PLAN, PIVOTS, UNITS, PassiveStore, duration_seconds, merge_plan, plan_summary
from studio_widgets import ACCENT, BG, EDGE, FONT, MUTED, PANEL, TEXT, button, label, tree

DIM, LIVE, WARN = "#6b7f9e", "#17342f", "#ff9a8a"
MEMORY = {"A quarter of this computer's memory": 0.25, "Half of this computer's memory": 0.5, "Three quarters of it": 0.75}
PIVOT_WORDS = {"halve_lr": "halved the learning rate", "next_dataset": "switched data", "drop_lessons": "left the teacher's lessons out"}
PIVOT_ROWS = {"halve_lr": "Halve the learning rate", "next_dataset": "Switch to the next dataset", "drop_lessons": "Leave out the lessons added since the best checkpoint"}
STOP_WORDS = {"duration": "the time limit was reached", "patience": "it stopped getting better", "target": "it reached your target test error",
              "win_rate": "it beat the starting model often enough", "pivots": "it ran out of things to try", "user": "you stopped it",
              "failed": "it gave up after repeated crashes"}
STEPS = (("Train", "until the error settles"), ("Save", "a checkpoint"), ("Learn", "from the teacher"), ("Go again", "or step back if worse"))
PHASES = {"train": 0, "save": 1, "teach": 2, "idle": 3}
CRASHES = ("If the trainer stops (no heartbeat for 60 seconds, or its process is gone), the watchdog starts it again from the last saved step. "
           "After 5 restarts in 10 minutes it gives up and tells you why when you are back. If you close Groundwork nothing happens; reopening it "
           "picks the session up where it is. If the computer restarts, the watchdog comes back when you sign in and resumes an unfinished session.")


# --------------------------------------------------------------------------- pure helpers (no Tk)

def choose_view(active, latest, dismissed):
    if active:
        return "running"
    if latest and latest["status"] not in ("queued", "running", "paused") and latest["id"] != dismissed:
        return "catchup"
    return "start"


def form_defaults():
    d = DEFAULT_PLAN
    return {"duration": f"{d['stop']['duration']:g}", "unit": d["stop"]["unit"], "patience": str(d["stop"]["patience_rounds"]), "target": "", "win": "",
            "memory": next(k for k, v in MEMORY.items() if v == d["limits"]["memory_fraction"]), "keep": str(d["limits"]["keep_checkpoints"]),
            "awake": d["limits"]["keep_awake"], "battery": d["limits"]["pause_on_battery"], "improve": f"{d['plateau']['min_improvement'] * 100:g}",
            "checks": str(d["plateau"]["checks"]), "steps": str(d["plateau"]["max_steps"]), "polish": d["teacher"]["polish"], "lessons": d["teacher"]["lessons"],
            "worse_limit": str(d["worse"]["limit"]), "pivots": list(d["worse"]["pivots"])}


def _number(text, scale=1):
    """Blank -> None. A number -> number. Anything else is passed through so merge_plan rejects it in plain words."""
    text = str(text).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        return text
    value = round(value * scale, 9)
    return int(value) if value == int(value) and scale == 1 else value


def plan_from_form(v):
    """Form values -> overrides for merge_plan. Percentages become fractions; blank optional numbers become None."""
    return {"stop": {"duration": _number(v["duration"]), "unit": v["unit"], "patience_rounds": _number(v["patience"]), "target_error": _number(v["target"]), "win_rate": _number(v["win"], .01)},
            "limits": {"memory_fraction": MEMORY.get(v["memory"], .5), "keep_checkpoints": _number(v["keep"]), "keep_awake": bool(v["awake"]), "pause_on_battery": bool(v["battery"])},
            "plateau": {"min_improvement": _number(v["improve"], .01), "checks": _number(v["checks"]), "max_steps": _number(v["steps"])},
            "teacher": {"polish": bool(v["polish"]), "lessons": bool(v["lessons"])},
            "worse": {"limit": _number(v["worse_limit"]), "pivots": [p for p in PIVOTS if p in v["pivots"]]}}


def duration_words(seconds):
    if not math.isfinite(seconds or 0):
        return "longer than anyone will wait"
    seconds = int(seconds or 0)
    unit = lambda n, word: f"{n} {word}" + ("" if n == 1 else "s")  # noqa: E731
    if seconds < 60:
        return unit(seconds, "second")
    if seconds < 3600:
        return unit(seconds // 60, "minute")
    if seconds < 86400:
        hours, minutes = seconds // 3600, seconds % 3600 // 60
        return unit(hours, "hour") + (" " + unit(minutes, "minute") if minutes else "")
    days, hours = seconds // 86400, seconds % 86400 // 3600
    return unit(days, "day") + (" " + unit(hours, "hour") if hours else "")


def stop_words(reason):
    return STOP_WORDS.get(reason, "it came to an end")


def _ys(values, top, bottom):
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    return lambda e: (top + bottom) / 2 if high == low else bottom - (e - low) / (high - low) * (bottom - top)


def chart_points(checkpoints, live, limit_seconds, width, height):
    """Canvas coordinates for the running chart: one ring per checkpoint (by elapsed time), the live round's points, and the plot box."""
    left, top = 56, 40
    right, bottom = max(width - 30, left + 20), max(height - 40, top + 20)
    y = _ys([c["error"] for c in checkpoints] + [e for _, e in live], top, bottom)
    x = lambda t: left + (min(max(t / limit_seconds, 0.0), 1.0) if limit_seconds > 0 else 0.0) * (right - left)  # noqa: E731
    return {"rings": [(c["n"], x(c.get("elapsed", 0)), y(c["error"]), bool(c.get("discarded"))) for c in checkpoints],
            "live": [(x(t), y(e)) for t, e in live], "box": (left, top, right, bottom)}


def journey_layout(checkpoints, width, height):
    """{checkpoint number: (x, y)} for the catch-up chart: evenly spaced in order, height by test error."""
    left, top, right, bottom = 48, 44, max(width - 40, 100), max(height - 36, 80)
    y = _ys([c["error"] for c in checkpoints], top, bottom)
    step = (right - left) / max(len(checkpoints) - 1, 1)
    return {c["n"]: (left + i * step if len(checkpoints) > 1 else (left + right) / 2, y(c["error"])) for i, c in enumerate(checkpoints)}


def journey(checkpoints):
    """The kept line, and each rolled-back stretch as a branch off its parent with the change tried afterwards."""
    kept = [c for c in checkpoints if not c.get("discarded")]
    branches = []
    for c in checkpoints:
        if not c.get("discarded"):
            continue
        if branches and branches[-1]["items"][-1]["n"] == c.get("parent"):
            branches[-1]["items"].append(c)
        else:
            branches.append({"parent": c.get("parent"), "items": [c], "pivot": None})
    for b in branches:
        child = next((c for c in kept if c.get("parent") == b["parent"] and c["n"] > b["items"][-1]["n"]), None)
        b["pivot"] = child.get("pivot") if child else None
    return {"kept": kept, "branches": branches}


def summary_sentence(session):
    state = session["state"]
    cps, count = state.get("checkpoints", []), len(state.get("checkpoints", []))
    text = f"While you were away it trained for {duration_words(state.get('active_seconds'))} and saved " + (f"{count} checkpoint{'' if count == 1 else 's'}." if count else "no checkpoints.")
    best = next((c for c in cps if c["n"] == state.get("best")), None)
    if best:
        text += f" The best is checkpoint {best['n']}, with a test error of {best['error']:.3f}."
    back = len(journey(cps)["branches"])
    if back:
        text += f" It took {back} step{'' if back == 1 else 's'} back along the way."
    if session["status"] == "failed":
        return text + f" It stopped early: {(session.get('error') or stop_words('failed')).rstrip('.')}."
    return text + f" It stopped because {stop_words(state.get('stop_reason'))}."


def clock(at):
    try:
        return datetime.fromisoformat(at).astimezone().strftime("%H:%M")
    except (TypeError, ValueError):
        return "?"


def _seconds(at):
    try:
        return datetime.fromisoformat(at).timestamp()
    except (TypeError, ValueError):
        return None


def _near(events, when, within, **match):
    """True when one of `events` (all fields in `match` equal) happened within `within` seconds of `when`."""
    return when is not None and any(all(e.get(k) == v for k, v in match.items()) and (t := _seconds(e.get("at"))) is not None and abs(t - when) <= within for e in events)


def interruptions(journal):
    """One line per interruption. The watchdog and the trainer both note a restart, and the watchdog says why a pause happened; each shows once."""
    out, skipped = [], []
    watchdog = [e for e in journal if e.get("type") == "restart" and "pid" in e]
    for e in journal:
        when, kind, t = clock(e.get("at")), e.get("type"), _seconds(e.get("at"))
        if kind in ("pause", "resume"):
            if not _near(journal, t, 60, type="power", reason=f"battery_{kind}d"):  # else the power line below says it, with the reason
                out.append(f"Paused at {when}." if kind == "pause" else f"Resumed at {when}.")
        elif kind == "power":
            out.append(f"Paused at {when} because the computer went on battery." if e.get("reason") == "battery_paused" else f"Resumed at {when} because the computer was plugged in again.")
        elif kind == "restart":
            if "pid" not in e and _near(watchdog, t, 120):
                continue  # the trainer's own note of a restart the watchdog already recorded
            why = {"silent": " because it stopped reporting", "stopped": " because its process had stopped"}.get(e.get("reason"), "")
            out.append(f"The trainer was restarted at {when}{why} and carried on from the last saved step.")
        elif kind == "memory":
            out.append(f"It ran low on memory at {when} (round {e.get('round')}) and carried on with smaller batches.")
        elif kind == "error":
            out.append(f"A problem at {when}: {e.get('message', 'unknown')}")
        elif kind == "teacher" and e.get("skipped"):
            skipped.append(when)
    if skipped:
        out.append(f"The teacher could not be used at {skipped[0]}, so that round's lesson was skipped." if len(skipped) == 1
                   else f"The teacher could not be used in {len(skipped)} rounds (first at {skipped[0]}), so those rounds' lessons were skipped.")
    return out


def checkpoint_story(cp, best):
    teacher = cp.get("teacher") or {}
    what = "Discarded after a step back" if cp.get("discarded") else "Polished by the teacher" if teacher.get("polished") else "Kept"
    if teacher.get("lessons"):
        what += f" · {teacher['lessons']} new lesson{'' if teacher['lessons'] == 1 else 's'}"
    return what + (" · best" if cp["n"] == best else "")


def size_words(n):
    return f"{n / 1024 ** 3:.1f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} MB"


# --------------------------------------------------------------------------- canvas drawing

def draw_strip(c):
    c.delete("all")
    w = max(c.winfo_width(), 480)
    tile = (w - 42) / 4
    for i, (head, sub) in enumerate(STEPS):
        x, on = i * (tile + 14), i == c.active
        col = ACCENT if on else MUTED
        c.create_rectangle(x + 1, 2, x + tile, 66, fill=LIVE if on else PANEL, outline=ACCENT if on else EDGE, width=2 if on else 1)
        cx, cy = x + 28, 34
        if i == 0:
            c.create_line(cx - 12, cy - 10, cx - 4, cy - 2, cx + 4, cy + 4, cx + 12, cy + 8, fill=col, width=2)
        elif i == 1:
            c.create_rectangle(cx - 10, cy - 10, cx + 10, cy + 10, outline=col, width=2)
            c.create_rectangle(cx - 5, cy - 10, cx + 5, cy - 3, outline=col, width=2)
        elif i == 2:
            c.create_polygon(cx - 12, cy - 10, cx + 12, cy - 10, cx + 12, cy + 5, cx - 2, cy + 5, cx - 8, cy + 12, cx - 7, cy + 5, cx - 12, cy + 5, outline=col, fill="", width=2)
        else:
            c.create_arc(cx - 10, cy - 10, cx + 10, cy + 10, start=40, extent=280, style="arc", outline=col, width=2)
            c.create_polygon(cx + 6, cy - 12, cx + 15, cy - 9, cx + 9, cy - 3, fill=col, outline=col)
        c.create_text(x + 54, 26, anchor="w", text=head, fill=TEXT, font=(FONT, 11, "bold"))
        c.create_text(x + 54, 46, anchor="w", text=sub, fill=MUTED, font=(FONT, 9))


def strip(parent, active=None):
    c = tk.Canvas(parent, bg=BG, height=70, highlightthickness=0)
    c.active = active
    c.bind("<Configure>", lambda _e: draw_strip(c))
    return c


def draw_cup(c):
    c.create_polygon(12, 22, 40, 22, 37, 44, 15, 44, outline=ACCENT, fill="", width=2)
    c.create_arc(34, 26, 50, 40, start=-90, extent=180, style="arc", outline=ACCENT, width=2)
    c.create_line(9, 48, 44, 48, fill=ACCENT, width=2)
    c.create_line(21, 6, 24, 11, 21, 16, smooth=True, fill=ACCENT, width=2)
    c.create_line(30, 6, 33, 11, 30, 16, smooth=True, fill=ACCENT, width=2)


def draw_running(c, session):
    c.delete("all")
    w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 200)
    state, plan = session["state"], session["plan"]
    limit = duration_seconds(plan)
    cps, live = state.get("checkpoints", []), state.get("live", [])
    pts = chart_points(cps, live, limit, w, h)
    left, top, right, bottom = pts["box"]
    c.create_text(20, 18, anchor="w", text="TEST ERROR (LOWER IS BETTER)", fill=MUTED, font=(FONT, 9, "bold"))
    c.create_line(left, bottom + 16, right, bottom + 16, fill=EDGE)
    c.create_text(left, h - 10, anchor="w", text="start", fill=MUTED, font=(FONT, 9))
    c.create_text(right, h - 10, anchor="e", text=f"{duration_words(limit)} at most", fill=MUTED, font=(FONT, 9))
    rings = pts["rings"]
    if not rings and not live:
        c.create_text(w / 2, h / 2, text="The first checkpoint appears here when the test error settles.", fill=MUTED, font=(FONT, 12))
        return
    kept = [(x, y) for _, x, y, d in rings if not d]
    if len(kept) > 1:
        c.create_line(*[v for p in kept for v in p], fill=ACCENT, width=2)
    tail = ([kept[-1]] if kept else []) + pts["live"]
    if len(tail) > 1:
        c.create_line(*[v for p in tail for v in p], fill=ACCENT, width=2, dash=(4, 3))
    for n, x, y, gone in rings:
        c.create_oval(x - 10, y - 10, x + 10, y + 10, outline=DIM if gone else ACCENT, fill=BG, width=2, dash=(2, 2) if gone else ())
        c.create_text(x, y, text=str(n), fill=MUTED if gone else TEXT, font=(FONT, 9, "bold"))
    if tail:
        x, y = tail[-1]
        c.create_oval(x - 5, y - 5, x + 5, y + 5, fill=ACCENT, outline="")
        c.create_text(x, y - 16, text="now", fill=ACCENT, font=(FONT, 9, "bold"))


def draw_journey(c, session):
    c.delete("all")
    w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 200)
    state = session["state"]
    cps = state.get("checkpoints", [])
    c.create_text(20, 18, anchor="w", text="TEST ERROR (LOWER IS BETTER)", fill=MUTED, font=(FONT, 9, "bold"))
    if not cps:
        c.create_text(w / 2, h / 2, text="No checkpoints were saved.", fill=MUTED, font=(FONT, 12))
        return
    c.create_text(w - 20, 18, anchor="e", text="- - discarded", fill=DIM, font=(FONT, 9))
    pos, tale = journey_layout(cps, w, h), journey(cps)
    if len(tale["kept"]) > 1:
        c.create_line(*[v for k in tale["kept"] for v in pos[k["n"]]], fill=ACCENT, width=2)
    for b in tale["branches"]:
        path = ([pos[b["parent"]]] if b["parent"] in pos else []) + [pos[i["n"]] for i in b["items"]]
        if len(path) > 1:
            c.create_line(*[v for p in path for v in p], fill=DIM, width=2, dash=(5, 4))
        x, y = pos[b["items"][-1]["n"]]
        c.create_text(x, y + 22, text="stepped back: " + PIVOT_WORDS.get(b["pivot"], "tried again"), fill=MUTED, font=(FONT, 9))
    for cp in cps:
        x, y = pos[cp["n"]]
        gone, best = cp.get("discarded"), cp["n"] == state.get("best")
        c.create_oval(x - 11, y - 11, x + 11, y + 11, outline=DIM if gone else ACCENT, fill=ACCENT if best else BG, width=2, dash=(2, 2) if gone else ())
        c.create_text(x, y, text=str(cp["n"]), fill="#081c19" if best else MUTED if gone else TEXT, font=(FONT, 9, "bold"))
        if best:
            c.create_text(x, y - 22, text="best", fill=ACCENT, font=(FONT, 9, "bold"))


def drill(parent, title, var):
    box = Collapsible(parent, title)
    box.var = var  # Tk drops a variable when its Python object is collected, which would blank an inline StringVar's label
    box.pack(fill="x", pady=3)
    label(box, "", "CardMuted.TLabel", textvariable=var, wraplength=900, justify="left").pack(anchor="w", padx=24)
    return box


def rows(box, spec):
    grid = ttk.Frame(box.body, style="Card.TFrame")
    grid.pack(fill="x")
    setting_rows(grid, spec)


def list_checkpoints(widget, session):
    picked = widget.selection()
    widget.delete(*widget.get_children())
    state = session["state"]
    for c in reversed(state.get("checkpoints", [])):
        widget.insert("", "end", iid=str(c["n"]), values=(c["n"], clock(c.get("at")), f"{c['error']:.3f}", checkpoint_story(c, state.get("best"))))
    keep = [i for i in picked if widget.exists(i)]  # the running view redraws every 2 seconds; a pick must survive that
    if keep:
        widget.selection_set(keep)


CP_COLUMNS = [("n", "#", 50), ("at", "TIME", 80), ("error", "TEST ERROR", 100), ("what", "WHAT HAPPENED", 420)]


# --------------------------------------------------------------------------- the mixin

class PassiveMixin:
    def _build_training(self):
        page = self.pages["training"]
        bar = ttk.Frame(page.body)
        bar.pack(fill="x", pady=(0, 10))
        self._passive_buttons = {}
        for key, text in (("now", "Train now"), ("passive", "Passive")):
            self._passive_buttons[key] = ttk.Button(bar, text=text, command=lambda k=key: self.passive_show_tab(k))
            self._passive_buttons[key].pack(side="left", padx=(0, 6))
        self.train_now_frame, self.passive_frame = ttk.Frame(page.body), ttk.Frame(page.body)
        self.train_now_frame.pack(fill="both", expand=True)
        # Everything the Training page builds (and looks up later via pages["training"].body) lives in the Train now frame, so `.body` stays pointing there.
        page.body = self.train_now_frame
        super()._build_training()
        self._passive_key, self._passive_job, self._pv, self._passive_tab = None, None, {}, "now"
        self.passive_base = None
        self.passive_show_tab("now")

    def show_page(self, name):
        super().show_page(name)
        if name == "training" and getattr(self, "_passive_tab", None) == "passive":
            self.refresh_passive()

    def destroy(self):
        job, self._passive_job = getattr(self, "_passive_job", None), None
        if job:
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass
        super().destroy()

    def passive_show_tab(self, key):
        self._passive_tab = key
        for k, b in self._passive_buttons.items():
            b.configure(style="ActiveNav.TButton" if k == key else "Nav.TButton")
        if key == "now":
            self.passive_frame.pack_forget()
            self.train_now_frame.pack(fill="both", expand=True)
        else:
            self.train_now_frame.pack_forget()
            self.passive_frame.pack(fill="both", expand=True)
            self.refresh_passive()

    # ---- seams the tests replace
    def _passive_launch(self, session_id):
        import studio_passive_watch
        return studio_passive_watch.start(self.workspace.root, session_id)

    def _passive_health(self, session):
        import studio_passive_watch
        return studio_passive_watch.status(PassiveStore(self.workspace.root), session)

    def _passive_open_chat(self, path):
        self.chat_checkpoint.set(str(path))
        self.show_page("chat")
        self.load_chat_model()

    # ---- which view
    def _passive_store(self):
        return PassiveStore(self.workspace.root)

    def _passive_teacher(self):
        from studio_rlhf import load_settings
        return (load_settings(self.workspace.root).get("teacher_model") or "").strip()

    def refresh_passive(self):
        store = self._passive_store()
        active, latest = store.active(), store.latest()
        view = choose_view(active, latest, self.workspace.state.get("passiveDismissed"))
        session = active or (latest if view == "catchup" else None)
        key = (view, session["id"] if session else None)
        if key != self._passive_key:
            for child in self.passive_frame.winfo_children():
                child.destroy()
            self._pv, self._passive_key = {"id": key[1]}, None
            {"start": self._passive_build_start, "running": self._passive_build_running, "catchup": self._passive_build_catchup}[view](self.passive_frame, session)
            self._passive_key = key
        if view == "running":
            self._passive_update_running(session)
        if self._passive_job is None and view == "running" and self._passive_tab == "passive" and getattr(self, "active_page", None) == "training":
            self._passive_job = self.after(2000, self._passive_tick)

    def _passive_tick(self):
        self._passive_job = None
        try:
            self.refresh_passive()
        except tk.TclError:
            pass
        except Exception as exc:
            self.log(traceback.format_exc())
            self.status_var.set(f"Passive page could not refresh: {exc}")
            if self._passive_tab == "passive" and getattr(self, "active_page", None) == "training":
                self._passive_job = self.after(2000, self._passive_tick)

    def _passive_dismiss(self, session_id=None):
        self.workspace.state["passiveDismissed"] = session_id or self._pv["id"]
        self.workspace.save()
        self.refresh_passive()

    # ---- start view
    def _passive_values(self):
        out = {k: v.get() for k, v in self._pf.items()}
        out["pivots"] = [p for p in PIVOTS if out.pop("pivot_" + p)]
        if not self._pv["teacher"]:
            out["polish"] = out["lessons"] = False
        return out

    def _passive_data(self):
        ids = [k for k in self.run_sources() if k]
        if not ids:
            raise ValueError("There is no data to train on yet. Download or import a dataset first, then come back.")
        try:
            limit = int(self.train_limit.get())
        except ValueError:
            raise ValueError("The record limit on the Training setup page must be a whole number.") from None
        data = {"sources": ids, "max_records": limit, "use_all": self.use_all_records.get(), "balance": self.balance_var.get(),
                "reviewed_only": self.reviewed_only.get(), "weights": {k: v for k, v in self.mixture_weights.items() if k in ids}}
        return {"data": data, "config": self.read_config(), "base_checkpoint": self.passive_base}

    def _passive_summaries(self, *_):
        try:
            text = plan_summary(merge_plan(plan_from_form(self._passive_values())))
        except (ValueError, KeyError, TypeError, OverflowError):
            return
        teacher = self._pv["teacher"]
        out = dict(text, teacher=f"{teacher} · {text['teacher']}" if teacher else "No teacher set up — the Learn step is skipped.")
        for key, var in self._pv["summary"].items():
            if key in out:
                var.set(out[key])

    def _passive_build_start(self, f, _session):
        pv = self._pv
        pv["teacher"] = teacher = self._passive_teacher()
        d = form_defaults()
        pf = self._pf = {k: tk.BooleanVar(value=v) if isinstance(v, bool) else tk.StringVar(value=v) for k, v in d.items() if k != "pivots"}
        for p in PIVOTS:
            pf["pivot_" + p] = tk.BooleanVar(value=p in d["pivots"])
        pv["summary"] = {k: tk.StringVar() for k in ("stop", "limits", "plateau", "teacher", "worse")}
        name = getattr(self, "config_vars", {}).get("name")
        label(f, "Train while you're away", font=(FONT, 18, "bold")).pack(anchor="w")
        label(f, f"{name.get() if name else 'The model'} repeats this loop on its own until one of your stop conditions is met.", "Muted.TLabel").pack(anchor="w", pady=(4, 10))
        strip(f).pack(fill="x")
        if self.passive_base:
            label(f, f"Starting from the best checkpoint of the last session: {Path(self.passive_base).parent.name}/{Path(self.passive_base).name}", foreground=ACCENT).pack(anchor="w", pady=(8, 0))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=14)
        label(row, "Stop after").pack(side="left")
        ttk.Entry(row, textvariable=pf["duration"], width=6).pack(side="left", padx=8)
        ttk.Combobox(row, textvariable=pf["unit"], values=list(UNITS), state="readonly", width=9).pack(side="left")
        label(row, "or when it stops getting better").pack(side="left", padx=8)
        pv["start_button"] = ttk.Button(row, text="Start passive training", style="Accent.TButton", command=self.passive_start)
        pv["start_button"].pack(side="right")
        s = pv["summary"]
        box = drill(f, "When to stop", s["stop"])
        rows(box, [("Stop after", pf["duration"], "How long the session may run."), ("Unit", pf["unit"], "", list(UNITS)),
                                ("No better checkpoint in (rounds)", pf["patience"], "0 turns this off."), ("Test error reaches", pf["target"], "Leave blank to turn this off."),
                                ("Beats the model I started with in (%)", pf["win"], "Share of head-to-head replies; needs a teacher. Leave blank to turn this off.")])
        label(box.body, "It stops at whichever comes first, and always keeps the best checkpoint.", "CardMuted.TLabel").pack(anchor="w", pady=(6, 0))
        box = drill(f, "Limits", s["limits"])
        rows(box, [("Memory it may use", pf["memory"], "The trainer slows down rather than go past this.", list(MEMORY)), ("Checkpoints to keep", pf["keep"], "Older ones are cleared; the best is always kept."),
                                ("Keep the computer awake", pf["awake"], ""), ("Pause on battery power", pf["battery"], "")])
        box = drill(f, "When to save a checkpoint", s["plateau"])
        rows(box, [("Save when the test error improves by less than (%)", pf["improve"], "Across the checks below."), ("across (checks in a row)", pf["checks"], ""),
                                ("or after (steps)", pf["steps"], "Whichever comes first. Smaller numbers mean fewer, more settled checkpoints.")])
        box = drill(f, "Teacher", s["teacher"])
        label(box.body, f"Teacher: {teacher}." if teacher else "No teacher is set up (Improve with feedback), so this step is skipped and training simply continues.", "Card.TLabel").pack(anchor="w")
        for key, text, note in (("polish", "Polish the checkpoint", "The teacher picks the better of the model's own replies and the model is nudged toward them. Kept only if it then beats the checkpoint head to head."),
                                ("lessons", "Carry the lessons forward", 'Better answers the teacher wrote join the next round. They stay in "Passive lessons" until you approve them.')):
            ttk.Checkbutton(box.body, text=text, variable=pf[key], state="normal" if teacher else "disabled").pack(anchor="w", pady=(6, 0))
            label(box.body, note, "CardMuted.TLabel", wraplength=800, justify="left").pack(anchor="w", padx=24)
        box = drill(f, "If it gets worse", s["worse"])
        rows(box, [("After (checkpoints in a row worse than the best)", pf["worse_limit"], "Go back to the best checkpoint and try the next change below.")]
                     + [(f"{i}. {PIVOT_ROWS[p]}", pf["pivot_" + p], "") for i, p in enumerate(PIVOTS, 1)])  # ponytail: fixed order; reordering and adding changes need a list editor
        label(box.body, "When the list runs out it stops and keeps the best checkpoint. Worse is judged on test examples set aside before the session starts.", "CardMuted.TLabel", wraplength=800).pack(anchor="w", pady=(6, 0))
        box = drill(f, "If something crashes", tk.StringVar(value="It restarts itself, even with Groundwork closed."))
        label(box.body, CRASHES, "CardMuted.TLabel", wraplength=800, justify="left").pack(anchor="w")
        for var in pf.values():
            var.trace_add("write", self._passive_summaries)
        self._passive_summaries()

    def passive_start(self):
        try:
            plan = merge_plan({**plan_from_form(self._passive_values()), **self._passive_data()})
            session_id = self._passive_store().create(plan)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Cannot start passive training", str(exc), parent=self)
            return
        try:
            self._passive_launch(session_id)
        except Exception as exc:
            self._passive_store().update(session_id, status="failed", error=f"The background trainer could not be started: {exc}")
            messagebox.showerror("Cannot start passive training", f"Groundwork could not start the background trainer.\n\n{exc}", parent=self)
            self.refresh_passive()
            return
        self.passive_base = None
        self.status_var.set("Passive training started")
        self.refresh_passive()

    # ---- running view
    def _passive_build_running(self, f, session):
        pv = self._pv
        pv.update({k: tk.StringVar() for k in ("lead", "status", "problem", "round", "cps", "keeping", "trainer", "trainer_note", "watchdog", "watchdog_note", "memory", "memory_note", "teacher", "teacher_note")})
        label(f, "Training in the background", font=(FONT, 18, "bold")).pack(anchor="w")
        label(f, "", "Muted.TLabel", textvariable=pv["lead"], wraplength=900, justify="left").pack(anchor="w", pady=(4, 10))
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        pv["pause"] = button(bar, "Pause", self.passive_pause)
        button(bar, "Stop and keep the best", self.passive_stop)
        label(bar, "", "Muted.TLabel", textvariable=pv["status"]).pack(side="left", padx=12)
        label(f, "", foreground=WARN, textvariable=pv["problem"], wraplength=900, justify="left").pack(anchor="w")
        pv["chart"] = c = tk.Canvas(f, bg=PANEL, height=260, highlightthickness=0)
        c.pack(fill="x", pady=10)
        c.bind("<Configure>", lambda _e: self._pv.get("id") and draw_running(c, self._passive_session()))
        pv["strip"] = strip(f)
        pv["strip"].pack(fill="x", pady=(0, 10))
        box = drill(f, "This round", pv["round"])
        pv["round_text"] = label(box.body, "", "CardMuted.TLabel", wraplength=800, justify="left")
        pv["round_text"].pack(anchor="w")
        box = drill(f, "Checkpoints", pv["cps"])
        pv["tree"] = tree(box.body, CP_COLUMNS, 5)
        row = ttk.Frame(box.body, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        button(row, "Chat with it", lambda: self.passive_chat(self._passive_picked(pv["tree"])))
        box = drill(f, "Keeping it going", pv["keeping"])
        grid = ttk.Frame(box.body, style="Card.TFrame")
        grid.pack(fill="x")
        for i, (title, key) in enumerate((("Trainer", "trainer"), ("Watchdog", "watchdog"), ("Memory", "memory"), ("Teacher", "teacher"))):
            label(grid, title, "Card.TLabel", font=(FONT, 10, "bold")).grid(row=i, column=0, sticky="w", padx=(0, 16), pady=3)
            label(grid, "", "Card.TLabel", textvariable=pv[key]).grid(row=i, column=1, sticky="w", padx=(0, 16))
            label(grid, "", "CardMuted.TLabel", textvariable=pv[key + "_note"]).grid(row=i, column=2, sticky="w")

    def _passive_session(self):
        return self._passive_store().get(self._pv["id"])

    def _passive_update_running(self, session):
        pv, state, plan = self._pv, session["state"], session["plan"]
        cps, teacher = state.get("checkpoints", []), state.get("teacher", {})
        pv["lead"].set(f"Stops after {duration_words(duration_seconds(plan))}, or sooner if it stops getting better. You can close Groundwork; it carries on.")
        pv["pause"].configure(text="Resume" if session["status"] == "paused" else "Pause")
        pv["status"].set({"queued": "Starting…", "paused": "Paused"}.get(session["status"], ""))
        pv["problem"].set(f"Last problem: {session['error']}" if session.get("error") else "")
        pv["strip"].active = PHASES.get(state.get("phase"))
        draw_strip(pv["strip"])
        draw_running(pv["chart"], session)
        phase = state.get("phase")
        vote = f" · the teacher has voted on {teacher.get('done', 0)} of {teacher.get('total', 0)} prompts" if phase == "teach" and teacher.get("total") else ""
        pv["round"].set(f"Round {state.get('round', 0)}{vote}" if state.get("round") else "Getting ready")
        name = teacher.get("model") or "the teacher"
        pv["round_text"].configure(text=f"The model trains until its test error settles, a checkpoint is saved, then {name} votes on the model's answers and writes better ones. "
                                        "The model is nudged toward the winners and kept only if it then beats the checkpoint head to head. Better answers join the next round's lessons.")
        back = len(journey(cps)["branches"])
        best = state.get("best")
        pv["cps"].set(f"{len(cps)} saved" + (f" · best is checkpoint {best}" if best else "") + (f" · {back} step{'' if back == 1 else 's'} back" if back else " · no steps back"))
        list_checkpoints(pv["tree"], session)
        try:
            health = self._passive_health(session) or {}
        except Exception:
            health = {}
        restarts = health.get("restarts", len(session.get("restarts", [])))
        trainer, heard = health.get("trainer", "Unknown"), health.get("heard")
        pv["trainer"].set(trainer)
        pv["trainer_note"].set("No heartbeat yet" if heard is None else f"Last heard from {heard:.0f} seconds ago")
        pv["watchdog"].set(health.get("watchdog", "Unknown"))
        pv["watchdog_note"].set("Restarts the trainer if it stops")
        memory = state.get("memory") or {}
        pv["memory"].set(f"{size_words(memory['used'])} of {size_words(memory['limit'])}" if memory.get("limit") else "Not measured yet")
        pv["memory_note"].set("Over the limit, the round restarts with smaller batches")
        reach = teacher.get("reachable")
        pv["teacher"].set("Not set up" if not teacher.get("model") else "Reachable" if reach else "Not reachable" if reach is False else "Not checked yet")
        pv["teacher_note"].set("If it can't be reached, that round skips the lesson")
        pv["keeping"].set("All well · no restarts this session" if trainer == "Running" and not restarts else f"Trainer {trainer.lower()} · {restarts} restart{'' if restarts == 1 else 's'} this session")

    def passive_pause(self):
        session = self._passive_store().active()
        if session:
            self._passive_store().command(session["id"], "resume" if session["status"] == "paused" else "pause")
        self.refresh_passive()

    def passive_stop(self):
        session = self._passive_store().active()
        if session and messagebox.askyesno("Stop and keep the best", "Stop training now? The best checkpoint so far is kept.", parent=self):
            self._passive_store().command(session["id"], "stop")
        self.refresh_passive()

    def _passive_picked(self, widget):
        picked = widget.selection()
        return int(picked[0]) if picked else None

    def passive_chat(self, number=None):
        state = (self._passive_session() or {}).get("state", {})
        number = number or state.get("best")
        cp = next((c for c in state.get("checkpoints", []) if c["n"] == number), None)
        if cp is None:
            messagebox.showinfo("Chat with it", "There is no checkpoint to chat with yet.", parent=self)
        elif not Path(cp["path"]).exists():
            messagebox.showinfo("Chat with it", "That checkpoint's file was cleared to save space. Pick another one.", parent=self)
        else:
            self._passive_open_chat(cp["path"])

    # ---- catch-up view
    def _passive_build_catchup(self, f, session):
        pv, store = self._pv, self._passive_store()
        state = session["state"]
        head = ttk.Frame(f)
        head.pack(fill="x")
        cup = tk.Canvas(head, width=56, height=56, bg=BG, highlightthickness=0)
        cup.pack(side="left", padx=(0, 14))
        draw_cup(cup)
        words = ttk.Frame(head)
        words.pack(side="left", fill="x")
        label(words, "Welcome back.", font=(FONT, 18, "bold")).pack(anchor="w")
        label(words, summary_sentence(session), "Muted.TLabel", wraplength=900, justify="left").pack(anchor="w", pady=(4, 0))
        label(f, "What happened", font=(FONT, 12, "bold")).pack(anchor="w", pady=(14, 4))
        c = tk.Canvas(f, bg=PANEL, height=240, highlightthickness=0)
        c.pack(fill="x", pady=(0, 10))
        c.bind("<Configure>", lambda _e: draw_journey(c, session))
        cps, best = state.get("checkpoints", []), state.get("best")
        back = len(journey(cps)["branches"])
        box = drill(f, "Checkpoint by checkpoint", tk.StringVar(value=f"{len(cps)} saved" + (f" · {back} step{'' if back == 1 else 's'} back" if back else "") + (f" · best is checkpoint {best}" if best else "")))
        pv["tree"] = tree(box.body, CP_COLUMNS, 6)
        list_checkpoints(pv["tree"], session)
        row = ttk.Frame(box.body, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        button(row, "Chat with it", lambda: self.passive_chat(self._passive_picked(pv["tree"])))
        pv["lessons_summary"] = tk.StringVar()
        box = drill(f, "What the teacher taught it", pv["lessons_summary"])
        pv["lessons"] = tree(box.body, [("prompt", "PROMPT", 200), ("reply", "THE MODEL'S REPLY", 260), ("answer", "THE TEACHER'S BETTER ANSWER", 260)], 5)
        row = ttk.Frame(box.body, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        button(row, "Keep", lambda: self.passive_lessons(pv["lessons"].selection(), "kept"))
        button(row, "Drop", lambda: self.passive_lessons(pv["lessons"].selection(), "dropped"))
        pv["keep_all"] = button(row, "Keep all", lambda: self.passive_lessons(pv["lessons"].get_children(), "kept"))
        self._passive_fill_lessons()
        events = interruptions(store.journal(session["id"]))
        box = drill(f, "Interruptions", tk.StringVar(value=f"{len(events)} during the session" if events else "None"))
        for line in events:
            label(box.body, line, "CardMuted.TLabel", wraplength=800, justify="left").pack(anchor="w", pady=2)
        label(f, "What next", font=(FONT, 12, "bold")).pack(anchor="w", pady=(14, 4))
        row = ttk.Frame(f)
        row.pack(fill="x")
        button(row, "Chat with the best checkpoint", lambda: self.passive_chat(), True)
        button(row, "Continue from the best", self.passive_continue)
        button(row, "Done", self.passive_done)

    def passive_done(self):
        waiting = sum(r.get("status") == "pending" for r in self._passive_store().lessons(self._pv["id"]))
        if waiting:
            messagebox.showinfo("Lessons are waiting", f"{waiting} lesson{'' if waiting == 1 else 's'} from the teacher {'is' if waiting == 1 else 'are'} waiting for your review. Keep or drop them first, then press Done.", parent=self)
            return
        self._passive_dismiss()

    @staticmethod
    def _one_line(text, size=70):
        text = " ".join(str(text).split())
        return text if len(text) <= size else text[:size - 1] + "…"

    def _passive_fill_lessons(self):
        pv = self._pv
        rows = [r for r in self._passive_store().lessons(pv["id"]) if r.get("status") == "pending"]
        pv["lessons"].delete(*pv["lessons"].get_children())
        for r in rows:
            pv["lessons"].insert("", "end", iid=r["id"], values=tuple(self._one_line(r.get(k, "")) for k in ("prompt", "reply", "answer")))
        pv["lessons_summary"].set(f"{len(rows)} new lesson{'' if len(rows) == 1 else 's'} waiting for your review" if rows else "Nothing waiting for review")
        pv["keep_all"].configure(text=f"Keep all {len(rows)}" if rows else "Keep all")

    def passive_lessons(self, ids, status):
        store, session_id, ids = self._passive_store(), self._pv["id"], set(ids)
        rows = store.lessons(session_id)
        chosen = [r for r in rows if r["id"] in ids and r.get("status") == "pending"]
        if not chosen:
            return
        if status == "kept":
            workspace = self.workspace
            dataset = next((d["id"] for d in workspace.state["datasets"] if d["name"] == "Passive lessons"), None)
            if dataset is None:
                previous = workspace.state.get("selectedDatasetId")
                dataset = workspace.create_dataset("Passive lessons", "Better answers the teacher wrote during passive training.")
                workspace.state["selectedDatasetId"] = previous  # creating a dataset selects it; keep the user's choice
            workspace.add(dataset, [{"messages": r["messages"] + [{"role": "assistant", "content": r["answer"]}]} for r in chosen], source="passive")
            self.refresh_datasets()
        for r in chosen:
            r["status"] = status
        store.set_lessons(session_id, rows)
        self._passive_fill_lessons()

    def passive_continue(self):
        session = self._passive_session()
        best = next((c for c in session["state"].get("checkpoints", []) if c["n"] == session["state"].get("best")), None)
        if best is None or not Path(best["path"]).exists():
            messagebox.showinfo("Continue from the best", "There is no saved best checkpoint to continue from.", parent=self)
            return
        self.passive_base = best["path"]
        self._passive_dismiss(session["id"])

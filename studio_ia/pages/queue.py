"""This computer: Now / Up next / When it may train, rendered from services.queue()."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from studio_ia.copy import QUEUE as T
from studio_ia.pages.base import Page
from studio_ia.theme import C, GUTTER, tk_label
from studio_ia.widgets import Toggle

HOURS = (("any", "Any time"), ("nights", "Nights only"), ("set", "Set hours"))
MEMORY = (("half", "Half"), ("most", "Most"), ("all", "All"))
TOGGLES = (("idle_only", "Only when I’m not using the computer", "Pauses within a few seconds of keyboard or mouse use."),
           ("pause_on_battery", "Pause on battery", "Resumes when plugged in."),
           ("keep_awake", "Keep the computer awake while training", "Otherwise it pauses when the computer sleeps."),
           ("chat_follows_best", "Chat follows the best checkpoint",
            "Chat uses the checkpoint with the lowest error on held-back text, not simply the newest."))


def minutes(seconds):
    return "" if seconds is None else f"about {max(1, round(seconds / 60))} min"


CHOICES = {"Older than 1 day": (1, False), "Older than 7 days": (7, False), "Older than 30 days": (30, False), "Older than 90 days": (90, False), "All history": (None, False)}
ALL = "All history"


def size_text(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def storage_state(services, choice, data=False):
    """Window-free state of the Storage card: (days, data, checkbox enabled, line text, button enabled).
    `data` only counts with "All history"."""
    days = CHOICES[choice][0]
    data = bool(data) and choice == ALL
    items, size = services.storage(days, data)
    line = T["storage_frees"].format(size=size_text(size), n=items) if items else T["storage_nothing"]
    return days, data, choice == ALL, line, items > 0


def total_hours(state):
    """Hours until everything now running or waiting is done; None unless every job has an estimate."""
    jobs = ([state.job] if state.job else []) + list(state.up_next)
    etas = [state.eta_s if j is state.job else j.eta_s for j in jobs]
    return None if not jobs or any(e is None for e in etas) else sum(etas) / 3600


class QueuePage(Page):
    title = "This computer"
    lead = "What it’s training now, what’s waiting, and when it’s allowed to work. Every “Train” or “Run” button in the app adds to this queue."

    hour_boxes: dict = {}
    choice, data = "Older than 30 days", False

    def refresh(self):
        self.hour_boxes, self._live, self._ticking = {}, None, False
        for child in self.body.winfo_children():
            child.destroy()
        s = self.services.queue()
        self._now(s)
        self._up_next(s)
        self._plan(s.plan)
        self._storage()

    def on_event(self, kind, payload):
        """Training steps arrive several times a second: they update the NOW card in place, at most twice a second.
        Rebuilding the whole page for each one made it flash; only queue, job and plan changes rebuild it."""
        if kind != "train":
            return self.refresh()
        if not getattr(self, "_ticking", False):
            self._ticking = True
            self.after(500, self._tick)

    def _tick(self):
        self._ticking = False
        s, live = self.services.queue(), getattr(self, "_live", None)
        if live is None or not s.job or live["job"] != s.job.id:
            return self.refresh()
        live["state"].configure(text=self._state_line(s))
        live["progress"].configure(text=self._progress_line(s))
        live["bar"].configure(maximum=max(1, s.total), value=s.step)

    def _state_line(self, s):
        state = "paused" if s.paused else f"waiting: {s.blocked}" if s.blocked else "running"
        return f"{s.job.label or s.job.kind} · {state}"

    def _progress_line(self, s):
        lr = "auto" if s.lr is None else f"auto ({s.lr:.2g})"
        phase = s.phase or self._phase(s)
        parts = [f"step {s.step:,} of {s.total:,}" if s.total else "starting", f"schedule: {phase}" if phase else "",
                 f"learning rate: {lr}", f"{minutes(s.eta_s)} left" if s.eta_s is not None else ""]
        return " · ".join(p for p in parts if p)

    def _card(self, kicker):
        card = ttk.Frame(self.body, padding=GUTTER)
        card.pack(fill="x", pady=(0, GUTTER))
        tk_label(card, kicker, 12, "muted", bg="bg").pack(anchor="w")
        return card

    def _button(self, parent, text, command, **kw):
        return ttk.Button(parent, text=text, command=command, **kw)

    def _now(self, s):
        card = self._card("NOW")
        if not s.job:
            tk_label(card, "Nothing is training.", 15, "text2").pack(anchor="w")
            return
        state = tk_label(card, self._state_line(s), 15)
        state.pack(anchor="w")
        progress = tk_label(card, self._progress_line(s), 13, "muted")
        progress.pack(anchor="w", pady=4)
        bar = ttk.Progressbar(card, maximum=max(1, s.total), value=s.step)
        bar.pack(fill="x", pady=4)
        self._live = {"job": s.job.id, "state": state, "progress": progress, "bar": bar}
        row = ttk.Frame(card)
        row.pack(anchor="w", pady=4)
        self._button(row, "Resume" if s.paused else "Pause", self._call("resume" if s.paused else "pause"), style="Primary.TButton").pack(side="left")
        self._button(row, "Make a checkpoint now", self._call("checkpoint_now")).pack(side="left", padx=GUTTER)
        self._button(row, "Stop", self._call("remove", s.job.id)).pack(side="left")
        tk_label(card, "“Make a checkpoint now” branches off a short cool-down and keeps the main run going. "
                 "Pausing keeps everything; the run resumes exactly where it stopped.", 13, "dim", wraplength=720).pack(anchor="w")

    def _phase(self, s):
        """Schedule phase from the run's own schedule (the engine's progress events carry none)."""
        from studio_ia.pages.training_run import phase_at
        run = s.job.params.get("output_dir")
        try:
            if run and getattr(self, "_phases", (None,))[0] != (run, s.total):  # fixed for a run of a given length: read once
                self._phases = ((run, s.total), self.services.run_curve(run)["phases"])
            return phase_at(self._phases[1], s.step) if run else ""
        except (OSError, ValueError, KeyError, NotImplementedError):
            return ""

    def _up_next(self, s):
        card = self._card("UP NEXT · IN ORDER")
        hours = total_hours(s)
        if hours is not None:
            tk_label(card, f"all done in about {hours:.1f} h at the current plan", 13, "muted").pack(anchor="w")
        elif s.up_next:
            tk_label(card, "the finish time shows once each waiting job has a time estimate", 13, "muted").pack(anchor="w")
        if not s.up_next:
            tk_label(card, "Nothing waiting. When the current job ends, the computer keeps improving the current version in its steady phase, "
                     "within the plan below.", 14, "text2", wraplength=720).pack(anchor="w")
            return
        tk_label(card, "Drag ⠿ to reorder. A held job keeps its place and is skipped until you release it.", 12, "muted").pack(anchor="w", pady=(0, 4))
        self._rows = []
        last = len(s.up_next) - 1
        for n, job in enumerate(s.up_next):
            held = getattr(job, "status", "waiting") == "held"
            row = tk.Frame(card, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=8, pady=6)
            row.pack(fill="x", pady=3)
            grip = tk_label(row, "⠿", 18, "muted", "panel", cursor="fleur")
            grip.pack(side="left", padx=(0, 8))
            tk_label(row, str(n + 1), 15, "muted" if held else "violet", "panel").pack(side="left", padx=(0, GUTTER))
            text = tk.Frame(row, bg=C["panel"])
            text.pack(side="left", fill="x", expand=True)
            tk_label(text, (job.label or job.kind) + (" · on hold" if held else ""), 15, "muted" if held else "text", "panel").pack(anchor="w")
            if job.note:
                tk_label(text, job.note, 13, "muted", "panel", wraplength=480).pack(anchor="w")
            if job.eta_s is not None:
                tk_label(row, minutes(job.eta_s), 13, "muted", "panel").pack(side="left", padx=GUTTER)
            buttons = (("▲", "move", (job.id, n - 1), n > 0), ("▼", "move", (job.id, n + 1), n < last),
                       ("Run next", "move", (job.id, 0), n > 0), ("Release" if held else "Hold", "hold", (job.id, not held), True),
                       ("Remove", "remove", (job.id,), True))
            for label, method, args, enabled in buttons:
                b = self._button(row, label, self._call(method, *args), width=len(label) + 2 if len(label) > 1 else 3)
                b.pack(side="left", padx=2)
                if not enabled:
                    b.state(["disabled"])
            for w in (grip,):
                w.bind("<ButtonPress-1>", lambda e, i=n: self._drag_start(i))
                w.bind("<B1-Motion>", self._drag_motion)
                w.bind("<ButtonRelease-1>", lambda e, j=job.id: self._drag_drop(j, e))
            self._rows.append(row)

    # drag to reorder ----------------------------------------------------------------
    def _drag_start(self, index):
        self._dragging = index
        self._rows[index].configure(highlightbackground=C["violet"], highlightthickness=2)

    def _drop_index(self, y_root):
        """Row the pointer is over (by vertical position), clamped to the list."""
        for i, row in enumerate(self._rows):
            if y_root < row.winfo_rooty() + row.winfo_height() / 2:
                return i if i <= self._dragging else i - 1
        return len(self._rows) - 1

    def _drag_motion(self, event):
        target = self._drop_index(event.y_root)
        for i, row in enumerate(self._rows):
            on = i == target and i != self._dragging
            row.configure(highlightbackground=C["violet"] if on or i == self._dragging else C["border"],
                          highlightthickness=2 if on or i == self._dragging else 1)

    def _drag_drop(self, job_id, event):
        target, start = self._drop_index(event.y_root), self._dragging
        self._dragging = None
        if target != start:
            self.services.move(job_id, target)  # the queue event repaints the list
        else:
            self.refresh()

    def _plan(self, plan):
        card = self._card("WHEN IT MAY TRAIN")

        def segmented(title, key, options):
            row = ttk.Frame(card)
            row.pack(anchor="w", pady=4)
            tk_label(row, title, 14, "text2").pack(side="left", padx=(0, GUTTER))
            for value, label in options:
                on = plan.get(key) == value
                self._button(row, label, lambda v=value: self._set(plan, key, v), style="Primary.TButton" if on else "TButton").pack(side="left", padx=2)
        segmented("Hours", "hours", HOURS)
        if plan.get("hours") == "set":
            row = ttk.Frame(card)
            row.pack(anchor="w", pady=4)
            for key, label in (("start_hour", "From hour"), ("end_hour", "to hour")):
                tk_label(row, label, 13, "muted").pack(side="left", padx=(0, 6))
                box = ttk.Spinbox(row, from_=0, to=23, width=4, command=lambda k=key: self._hour(plan, k))
                box.set(plan.get(key, 0))
                box.pack(side="left", padx=(0, GUTTER))
                self.hour_boxes[key] = box
        segmented("Memory it may use", "memory", MEMORY)
        for key, name, note in TOGGLES:
            row = ttk.Frame(card)
            row.pack(fill="x", pady=4)
            text = ttk.Frame(row)
            text.pack(side="left")
            Toggle(text, name, plan.get(key), lambda v, k=key: self._set(plan, k, v), bg="bg").pack(anchor="w")
            tk_label(text, note, 12, "muted").pack(anchor="w", padx=(46, 0))
        tk_label(card, "Answering in Chat never stops training. It shares the computer, so training slows while it answers.", 13, "dim").pack(anchor="w", pady=(8, 0))

    def _storage(self):
        card = self._card(T["storage"].upper())
        tk_label(card, T["storage_desc"], 13, "muted", wraplength=720).pack(anchor="w")
        choice = tk.StringVar(value=self.choice)
        data = tk.BooleanVar(value=self.data)
        self.line = tk_label(card, "", 14, "text2")
        self.delete_btn = self._button(card, T["storage_delete"], self._delete)
        self.data_box = ttk.Checkbutton(card, text=T["storage_data"], variable=data, command=lambda: self._pick(choice, data))
        for label in CHOICES:
            ttk.Radiobutton(card, text=T["storage_all"] if label == ALL else label,
                            value=label, variable=choice, command=lambda: self._pick(choice, data)).pack(anchor="w")
        self.data_box.pack(anchor="w", pady=(6, 0))
        self.line.pack(anchor="w", pady=6)
        self.delete_btn.pack(anchor="w")
        self._vars = (choice, data)
        self._paint_storage()

    def _pick(self, choice, data):
        self.choice, self.data = choice.get(), data.get()
        self._paint_storage()

    def _paint_storage(self, text=None):
        days, self.data, box, line, ok = storage_state(self.services, self.choice, self.data)
        if not box:
            self._vars[1].set(False)
        self.data_box.state(["!disabled" if box else "disabled"])
        self.delete_btn.state(["!disabled" if ok else "disabled"])
        self.line.configure(text=text or line)

    def _delete(self):
        from tkinter import messagebox
        days, data, _box, _line, _ok = storage_state(self.services, self.choice, self.data)
        items, size = self.services.storage(days, data)
        if not messagebox.askyesno(T["storage"], T["storage_confirm"].format(n=items, size=size_text(size))):
            return
        try:
            _n, freed = self.services.clean_storage(days, data)
        except RuntimeError as error:
            return self._paint_storage(str(error))
        self.refresh()
        self._paint_storage(T["storage_freed"].format(size=size_text(freed)))

    def _hour(self, plan, key):
        try:
            self._set(plan, key, int(self.hour_boxes[key].get()) % 24)
        except ValueError:
            pass

    def _set(self, plan, key, value):
        self.services.set_plan({**plan, key: value})
        self.refresh()

    def _call(self, name, *args):
        def run():
            getattr(self.services, name)(*args)
            self.refresh()
        return run

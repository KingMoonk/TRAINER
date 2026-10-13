"""This computer: Now / Up next / When it may train, rendered from services.queue()."""
from __future__ import annotations

from tkinter import ttk

from studio_ia.pages.base import Page
from studio_ia.theme import GUTTER, tk_label

HOURS = (("any", "Any time"), ("nights", "Nights only"), ("set", "Set hours"))
MEMORY = (("half", "Half"), ("most", "Most"), ("all", "All"))
TOGGLES = (("idle_only", "Only when I’m not using the computer", "Pauses within a few seconds of keyboard or mouse use."),
           ("pause_on_battery", "Pause on battery", "Resumes when plugged in."),
           ("keep_awake", "Keep the computer awake while training", "Otherwise it pauses when the computer sleeps."),
           ("chat_follows_best", "Chat follows the best checkpoint",
            "Chat uses the checkpoint with the lowest error on held-back text, not simply the newest."))


def minutes(seconds):
    return "about [N] min" if seconds is None else f"about {max(1, round(seconds / 60))} min"


class QueuePage(Page):
    title = "This computer"
    lead = "What it’s training now, what’s waiting, and when it’s allowed to work. Every “Train” or “Run” button in the app adds to this queue."

    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        s = self.services.queue()
        self._now(s)
        self._up_next(s)
        self._plan(s.plan)

    def on_event(self, kind, payload):
        self.refresh()

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
        state = "paused" if s.paused else "running"
        tk_label(card, f"{s.job.label or s.job.kind} · {state}", 15).pack(anchor="w")
        lr = "auto" if s.lr is None else f"auto ({s.lr:.2g})"
        tk_label(card, f"step {s.step if s.total else '[N]'} of {s.total or '[N]'} · schedule: {s.phase or 'steady phase'} · learning rate: {lr} · "
                 f"{minutes(s.eta_s)} left", 13, "muted").pack(anchor="w", pady=4)
        bar = ttk.Progressbar(card, maximum=max(1, s.total), value=s.step)
        bar.pack(fill="x", pady=4)
        row = ttk.Frame(card)
        row.pack(anchor="w", pady=4)
        self._button(row, "Resume" if s.paused else "Pause", self._call("resume" if s.paused else "pause"), style="Primary.TButton").pack(side="left")
        self._button(row, "Make a checkpoint now", self._call("checkpoint_now")).pack(side="left", padx=GUTTER)
        tk_label(card, "“Make a checkpoint now” branches off a short cool-down and keeps the main run going. "
                 "Pausing keeps everything; the run resumes exactly where it stopped.", 13, "dim", wraplength=720).pack(anchor="w")

    def _up_next(self, s):
        card = self._card("UP NEXT · IN ORDER")
        if not s.up_next:
            tk_label(card, "Nothing waiting. When the current job ends, the computer keeps improving the current version in its steady phase, "
                     "within the plan below.", 14, "text2", wraplength=720).pack(anchor="w")
            return
        for n, job in enumerate(s.up_next, 1):
            row = ttk.Frame(card)
            row.pack(fill="x", pady=4)
            tk_label(row, str(n), 15, "violet").pack(side="left", padx=(0, GUTTER))
            text = ttk.Frame(row)
            text.pack(side="left", fill="x", expand=True)
            tk_label(text, job.label or job.kind, 15).pack(anchor="w")
            if job.note:
                tk_label(text, job.note, 13, "muted", wraplength=520).pack(anchor="w")
            tk_label(row, minutes(job.eta_s), 13, "muted").pack(side="left", padx=GUTTER)
            if n > 1:
                self._button(row, "Move up", self._call("move_up", job.id)).pack(side="left")
            self._button(row, "Remove", self._call("remove", job.id)).pack(side="left", padx=(8, 0))

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
        segmented("Memory it may use", "memory", MEMORY)
        for key, name, note in TOGGLES:
            row = ttk.Frame(card)
            row.pack(fill="x", pady=4)
            self._button(row, "On" if plan.get(key) else "Off", lambda k=key: self._set(plan, k, not plan.get(k)),
                         style="Primary.TButton" if plan.get(key) else "TButton").pack(side="left", padx=(0, GUTTER))
            text = ttk.Frame(row)
            text.pack(side="left")
            tk_label(text, name, 14).pack(anchor="w")
            tk_label(text, note, 12, "muted").pack(anchor="w")
        tk_label(card, "Answering in Chat never stops training. It shares the computer, so training slows while it answers.", 13, "dim").pack(anchor="w", pady=(8, 0))

    def _set(self, plan, key, value):
        self.services.set_plan({**plan, key: value})
        self.refresh()

    def _call(self, name, *args):
        def run():
            getattr(self.services, name)(*args)
            self.refresh()
        return run

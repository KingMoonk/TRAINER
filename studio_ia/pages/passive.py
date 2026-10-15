"""Passive training (in Resources): the passive sessions with their status and Stop, and whether the watchdog starts at sign-in.
Rendered from services.passive(); the trainer itself lives in studio_passive and its watchdog in studio_passive_watch."""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import ttk

from studio_ia.copy import PASSIVE as T
from studio_ia.pages.base import Page
from studio_ia.theme import C, GUTTER, tk_label
from studio_ia.widgets import info_icon
from studio_ux import format_duration

ACTIVE = ("queued", "running", "paused")  # studio_passive.ACTIVE, without importing the trainer into the page


def when(stamp):
    """A stored UTC timestamp as local day and time, e.g. "15 Oct 09:30"; the stamp itself if it does not parse."""
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%d %b %H:%M")
    except (TypeError, ValueError):
        return stamp or ""


def session_view(row):
    """(headline, facts line, Stop enabled) for one passive session row of services.passive()."""
    status = T.get(f"status_{row['status']}", row["status"])
    facts = T["facts"].format(trainer=row["trainer"], watchdog=row["watchdog"], restarts=row["restarts"])
    if row["heard"] is not None and row["status"] in ACTIVE:
        facts += T["heard"].format(ago=format_duration(row["heard"]).removeprefix("about "))
    return T["started"].format(status=status, when=when(row["created"])), facts, row["status"] in ACTIVE


def competes(view, queue):
    """The warning when a passive session and a queue job both want the computer, else ""."""
    running = any(r["status"] == "running" for r in view["sessions"])
    return T["competes"] if running and queue.job is not None and not queue.paused else ""


class PassivePage(Page):
    title = T["h1"]
    lead = T["lead"]

    view = None
    note = ""

    def preload(self):
        self.view = self.services.passive()

    def refresh(self):
        self.view = self.view or self.services.passive()
        view, self.view = self.view, None  # the next refresh reads afresh
        for child in self.body.winfo_children():
            child.destroy()
        warning = competes(view, self.services.queue())
        if warning:
            tk_label(self.body, warning, 14, "amber", wraplength=760).pack(anchor="w", pady=(0, GUTTER))
        card = self._card(T["kicker"], T["desc"])
        if not view["sessions"]:
            tk_label(card, T["none"], 14, "text2").pack(anchor="w", pady=(6, 0))
        for row in view["sessions"]:
            headline, facts, can_stop = session_view(row)
            line = tk.Frame(card, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=8, pady=6)
            line.pack(fill="x", pady=3)
            text = tk.Frame(line, bg=C["panel"])
            text.pack(side="left", fill="x", expand=True)
            tk_label(text, headline, 15, "text", "panel").pack(anchor="w")
            for part in (row["stop"], facts, row["error"]):
                if part:
                    tk_label(text, part, 13, "muted", "panel", wraplength=600).pack(anchor="w")
            if can_stop:
                ttk.Button(line, text=T["stop"], command=lambda sid=row["id"]: self._stop(sid)).pack(side="right")
        card = self._card(T["signin_kicker"])
        tk_label(card, T["signin_on"] if view["signin"] else T["signin_off"], 14, "text2").pack(anchor="w", pady=(6, 0))
        if view["signin"]:
            ttk.Button(card, text=T["signin_remove"], command=self._remove_signin).pack(anchor="w", pady=(6, 0))
        if self.note:
            tk_label(self.body, self.note, 14, "text2").pack(anchor="w")
            self.note = ""

    def on_event(self, kind, payload):
        if kind == "queue":
            self.refresh()

    def _card(self, kicker, hint=None):
        card = ttk.Frame(self.body, padding=GUTTER)
        card.pack(fill="x", pady=(0, GUTTER))
        head = tk.Frame(card, bg=C["bg"])
        head.pack(anchor="w")
        tk_label(head, kicker, 12, "muted", bg="bg").pack(side="left")
        if hint:
            info_icon(head, hint, "bg").pack(side="left", padx=(4, 0))
        return card

    def _stop(self, session_id):
        from tkinter import messagebox
        if not messagebox.askyesno(T["h1"], T["stop_confirm"]):
            return
        status = self.services.stop_passive(session_id)
        self.note = "" if status == "stopped" else T["stopping"]
        self.refresh()

    def _remove_signin(self):
        self.services.remove_passive_signin()
        self.note = T["signin_removed"]
        self.refresh()

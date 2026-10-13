"""Quick test: a throwaway micro run of a design idea against the current version."""
from __future__ import annotations

import tkinter as tk
import uuid

from studio_ia.copy import QUICKTEST as T
from studio_ia.pages.base import Page
from studio_ia.services import Job
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, GuessChips, HowItWorks, Kicker, Progress, Segmented, button
from studio_spec import default_spec, validate_spec
from studio_versions import change_text

DATASETS = (("stories", "short_stories", "short_stories_note"), ("slice", "slice_your_data", "slice_note"),
            ("problems", "practice_problems", "problems_note"))
MINUTES = (("1_min", 1), ("3_min", 3), ("10_min", 10))
GUESSES = ("Lower", "Same", "Higher")
CHECKS = (("starts_near_uniform", T["starts_where_should"], "First error near the uniform-guess level."),
          ("finite", T["numbers_finite"], T["no_overflow"]),
          ("memorises_one_batch", T["memorize_batch_check"], T["drives_error_zero"]),
          ("error_falls", T["error_falls"], T["smoothed_drops"]))
GUARD = "A design that never gets below counting word frequencies hasn't learned to use context."


class QuickTestPage(Page):
    title = T["h1"]
    lead = T["lead"]
    crumb = ("Network designer",)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.job_id, self.result, self.running, self.version_name = None, None, False, "Current version"
        b = self.body
        banner = Card(b)
        banner.pack(fill="x", pady=(0, GUTTER))
        Kicker(banner, T["throwaway"], "amber", "panel").pack(anchor="w")
        tk_label(banner, T["throwaway_desc"], 14, "text2", "panel", wraplength=720).pack(anchor="w", pady=(4, 0))
        what = Card(b)
        what.pack(fill="x", pady=(0, GUTTER))
        Kicker(what, "What you're testing", bg="panel").pack(anchor="w")
        row = tk.Frame(what, bg=C["panel"])
        row.pack(fill="x", pady=(6, 0))
        self.current_box, self.draft_box = (self._design(row, T["current_design"], "violet"), self._design(row, T["your_idea"], "amber"))
        button(what, T["change_designer"], lambda: self.navigate("builder")).pack(anchor="w", pady=(GUTTER, 0))
        choices = Card(b)
        choices.pack(fill="x", pady=(0, GUTTER))
        self.data = self._choice(choices, "Tiny dataset", Segmented(choices, [T[d[1]] for d in DATASETS], T["short_stories"], self.note_data))
        self.data_note = self._note(choices)
        self.time = self._choice(choices, T["time_each"], Segmented(choices, [T[m[0]] for m in MINUTES], T["3_min"]))
        self._note(choices).configure(text=T["both_get_time"])
        self.size = self._choice(choices, T["size_label"], Segmented(choices, [T["full"], T["shrunk_quarter_label"]], T["full"], self.note_size))
        self.size_note = self._note(choices)
        self.note_data(T["short_stories"])
        self.note_size(T["full"])
        guess = tk.Frame(b, bg=C["bg"])
        guess.pack(fill="x", pady=(0, GUTTER))
        tk_label(guess, T["your_guess"], 15).pack(anchor="w")
        tk_label(guess, T["result_shows"], 13, "muted").pack(anchor="w")
        self.guess = GuessChips(guess, GUESSES, bg="bg")
        self.guess.pack(anchor="w", pady=6)
        self.run_btn = button(guess, "Run quick test", self.start, primary=True)
        self.run_btn.pack(anchor="w", pady=(6, 0))
        self.status = tk_label(guess, "", 13, "pink")
        self.status.pack(anchor="w", pady=(6, 0))
        self.result_card = Card(b)
        self.how = HowItWorks(b, T["how_quick_tests"], bg="bg")
        self.how.pack(anchor="w", pady=GUTTER)
        for head, text in ((T["fair_by_time"], T["fair_note"]), (T["shrunk_quarter"], T["shrunk_note"]),
                           (T["memorize_batch"], T["memorize_note"]), (T["starts_where"], T["starts_note"])):
            tk_label(self.how.body, head, 14, "text", "bg", font=font(14, bold=True)).pack(anchor="w", pady=(6, 0))
            tk_label(self.how.body, text, 13, "muted", "bg", wraplength=720).pack(anchor="w")

    # helpers ----------------------------------------------------------------
    def _design(self, parent, kicker, color):
        box = tk.Frame(parent, bg=C["panel2"], padx=GUTTER, pady=GUTTER)
        box.pack(side="left", fill="x", expand=True, padx=(0, GUTTER))
        Kicker(box, kicker, color, "panel2").pack(anchor="w")
        title = tk_label(box, "", 17, "text", "panel2", font=font(17, bold=True))
        title.pack(anchor="w")
        desc = tk_label(box, "", 13, "muted", "panel2", wraplength=300)
        desc.pack(anchor="w")
        return title, desc

    def _choice(self, card, name, seg):
        tk_label(card, name, 14, "text2", "panel").pack(anchor="w", pady=(6, 2))
        seg.pack(anchor="w")
        return seg

    def _note(self, card):
        note = tk_label(card, "", 13, "muted", "panel", wraplength=720)
        note.pack(anchor="w", pady=(2, 6))
        return note

    def note_data(self, label):
        self.data_note.configure(text=T[next(d[2] for d in DATASETS if T[d[1]] == label)])

    def note_size(self, label):
        self.size_note.configure(text=T["faster_hint"] if label == T["shrunk_quarter_label"] else T["exact_designs"])

    def specs(self):
        v = self.services.current_version()
        current = validate_spec(v.spec) if v and v.spec else validate_spec(default_spec())
        draft = getattr(self.services, "draft_spec", None)
        draft = draft() if callable(draft) else None
        return current, (validate_spec(draft) if draft else None), (f"Version {v.number}" if v else "Current version")

    def refresh(self):
        current, draft, name = self.specs()
        self.version_name = name
        self.current_box[0].configure(text=name)
        self.current_box[1].configure(text=f"{current['dims']['n_layer']} blocks · d_model {current['dims']['d_model']}")
        self.draft_box[0].configure(text=f"{name} with your change")
        self.draft_box[1].configure(text=change_text(current, draft) if draft else "No change staged in the designer yet.")

    # run --------------------------------------------------------------------
    def start(self):
        current, draft, _name = self.specs()
        if draft is None:
            self.status.configure(text="Stage a change in the designer first.")
        elif not self.guess.value:
            self.status.configure(text="Pick your guess first.")
        elif not self.running:
            data = next(d[0] for d in DATASETS if T[d[1]] == self.data.value)
            minutes = next(m[1] for m in MINUTES if T[m[0]] == self.time.value)
            self.job_id, self.running, self.result = uuid.uuid4().hex[:8], True, None
            self.services.enqueue(Job(id=self.job_id, kind="quicktest", label=f"Quick test · {minutes} min each",
                                      params={"current_spec": current, "draft_spec": draft, "dataset": data, "minutes": minutes,
                                              "shrink": self.size.value == T["shrunk_quarter_label"], "guess": self.guess.value}))
            self.run_btn.configure(text=T["running"] + "…")
            self.status.configure(text="Queued: it runs when the computer is free.")
            self.show_result()

    def on_event(self, kind, payload):
        if kind != "train" or not isinstance(payload, dict) or payload.get("job") != self.job_id:
            return
        if payload.get("type") == "quicktest_done":
            self.result, self.running = payload["results"], False
            self.run_btn.configure(text="Run again")
            self.status.configure(text="")
        self.show_result(payload)

    def show_result(self, event=None):
        card = self.result_card
        for child in card.winfo_children():
            child.destroy()
        card.pack(fill="x", pady=(0, GUTTER), before=self.how)
        r = self.result
        Kicker(card, "RESULT · SAME DATA, ORDER AND TIME" if r else "RUNNING · ONE AFTER THE OTHER, SAME DATA ORDER", bg="panel").pack(anchor="w")
        for label, color in (("current", "violet"), ("draft", "amber")):
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", pady=4)
            tk_label(row, getattr(self, "version_name", "Version") if label == "current" else "Your idea", 14, "text", "panel", width=16).pack(side="left")
            bar = Progress(row, color=color)
            bar.pack(side="left", fill="x", expand=True, padx=GUTTER)
            if r:
                bar.set(1)
                text = f"done · {r[label]['steps']} steps"
            elif event and event.get("which") == label and event.get("total"):
                bar.set(event["step"] / event["total"])
                text = f"step {event['step']}"
            else:
                text = "waiting"
            tk_label(row, text, 13, "muted", "panel").pack(side="left")
        if not r:
            return
        plot = BpcPlot(card, 640, 200, 14)
        plot.pack(anchor="w", pady=8)
        plot.set(steps=[tuple(p) for p in r["current"]["curve"]], smooth=[tuple(p) for p in r["draft"]["curve"]],
                 refs=[("uniform guess", r["references"]["uniform"]), ("counting word frequencies", r["references"]["unigram"])])
        tk_label(card, f"grey: {self.version_name} · violet: your idea", 12, "muted", "panel").pack(anchor="w")
        guess = (self.guess.value or "nothing").lower()
        Kicker(card, T["early_signal"], bg="panel").pack(anchor="w", pady=(GUTTER, 0))
        tk_label(card, f"After the same {r['minutes']:g} minutes, your idea's error is {r['verdict']} than {self.version_name}'s. "
                 f"You guessed {guess}.", 17, "text", "panel", wraplength=720).pack(anchor="w", pady=4)
        tk_label(card, T["one_short_run"], 13, "muted", "panel", wraplength=720).pack(anchor="w")
        if not r["learned_context"]:
            tk_label(card, GUARD, 13, "amber", "panel", wraplength=720).pack(anchor="w", pady=(4, 0))
        Kicker(card, T["does_work"], bg="panel").pack(anchor="w", pady=(GUTTER, 0))
        by_id = {c["id"]: c for c in r["checks"]}
        for key, name, what in CHECKS:
            ok = by_id[key]["ok"]
            line = tk.Frame(card, bg=C["panel"])
            line.pack(fill="x", pady=2)
            tk_label(line, "✓" if ok else "…" if ok is None else "✗", 15, "violet" if ok else "muted" if ok is None else "coral", "panel", width=3).pack(side="left")
            tk_label(line, f"{name} — {what} ({by_id[key]['detail']})", 13, "text2", "panel", wraplength=680).pack(side="left")
        Kicker(card, T["what_each_writes"], bg="panel").pack(anchor="w", pady=(GUTTER, 0))
        for name, key in ((self.version_name, "current"), ("Your idea", "draft")):
            tk_label(card, name, 13, "muted", "panel").pack(anchor="w")
            tk_label(card, r[key]["sample"], 13, "text", "panel", wraplength=700, font=font(13, mono=True)).pack(anchor="w", pady=(0, 6))
        actions = tk.Frame(card, bg=C["panel"])
        actions.pack(anchor="w", pady=(GUTTER, 0))
        button(actions, T["save_as_version"].replace("version 6", "the next version"), self.save, primary=True).pack(side="left")
        button(actions, T["run_real_comparison"], lambda: self.navigate("compare")).pack(side="left", padx=GUTTER)
        button(actions, T["discard_idea"], self.discard).pack(side="left", padx=GUTTER)

    def save(self):
        current, draft, _name = self.specs()
        if draft:
            self.services.save_version(draft, change_text(current, draft))
            self.navigate("builder")

    def discard(self):
        self.result, self.running, self.job_id = None, False, None
        self.result_card.pack_forget()
        self.run_btn.configure(text="Run quick test")

"""Quick test: a throwaway micro run of a design idea against the current version."""
from __future__ import annotations

import time
import tkinter as tk
import uuid

import studio_quicktest
from studio_ia.copy import QUICKTEST as T
from studio_ia.pages.builder import C_, WIDTHS, P, cfg, read_spec, say, text_sets
from studio_ia.pages.base import Page
from studio_ia.services import Job
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, Chip, GuessChips, HowItWorks, Kicker, Progress, Segmented, button
from studio_spec import default_spec, validate_spec
from studio_versions import spec_diff

DATASETS = (("stories", "short_stories", "short_stories_note"), ("slice", "slice_your_data", "slice_note"),
            ("problems", "practice_problems", "problems_note"))
MINUTES = (("1_min", 1), ("3_min", 3), ("10_min", 10))
GUESSES = ("Lower", "Same", "Higher")
CHECKS = (("starts_near_uniform", T["starts_where_should"], "First error near the uniform-guess level."),
          ("finite", T["numbers_finite"], T["no_overflow"]),
          ("memorises_one_batch", T["memorize_batch_check"], T["drives_error_zero"]),
          ("error_falls", T["error_falls"], T["smoothed_drops"]))
LOOK_AT = {  # a failed check names the part to look at
    "starts_near_uniform": "Look at the starting setup: initial weight scale and the size of the word list.",
    "finite": "Look at the learning rate and the normalisation: something overflows.",
    "memorises_one_batch": "Look at the parts you changed: the design cannot even fit one small batch, so something in it or its maths is broken.",
    "error_falls": "Look at the learning rate and schedule: the error did not fall by more than its wobble."}
GUARD = "A design that never gets below counting word frequencies hasn't learned to use context."
NO_TEXT = "Your data has no plain text yet. Add some in Your data, or pick another tiny dataset."


def describe(spec):
    """'4 blocks · 4 look-back heads · normal thinking room' for a ModelSpec."""
    s = read_spec(spec)
    return f"{s['L']} blocks · {s['H']} look-back heads · {WIDTHS[s['W']].lower()} thinking room"


def cost_sentence(current, draft):
    """'same size · same work per step', or how much bigger/slower, from the exact cost formulas."""
    a, b = cfg(read_spec(current)), cfg(read_spec(draft))
    size, work = say(P(a), P(b), "bigger", "smaller"), say(C_(a), C_(b), "slower", "faster")
    same = "about the same"
    return f"{'same size' if size == same else size} · {'same work per step' if work == same else work}"


def seconds_left(started, step, total, now):
    """Time left from the pace so far, or None before the first step."""
    return None if not step or not total else max(0., (now - started) / step * (total - step))


def check_mark(ok):
    return "✓" if ok else "…" if ok is None else "✗"


def signal_text(result, name, guess):
    """The early-signal sentence, repeating the guess; a failed check replaces it with the part to look at."""
    if result.get("broken"):
        check = next(n for k, n, _ in CHECKS if k == result["broken"])
        return f"A check failed: {check}. {LOOK_AT[result['broken']]}"
    rel = "about the same as" if result["verdict"] == "about the same" else f"{result['verdict']} than"
    return (f"After the same {result['minutes']:g} minutes, your idea's error is {rel} {name}'s. "
            f"You guessed {(guess or 'nothing').lower().replace('same', 'about the same')}.")


class QuickTestPage(Page):
    title = T["h1"]
    lead = T["lead"]
    crumb = ("Network designer",)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.job_id, self.result, self.running, self.version_name = None, None, False, "Current version"
        self.started, self.paces, self.progress = 0., {}, {}
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
        self.chip_row = tk.Frame(self.draft_box[1].master, bg=C["panel2"])
        self.chip_row.pack(anchor="w", before=self.draft_box[1])
        button(what, T["change_designer"], lambda: self.navigate("builder")).pack(anchor="w", pady=(GUTTER, 0))
        choices = Card(b)
        choices.pack(fill="x", pady=(0, GUTTER))
        self.data = self._choice(choices, T["tiny_dataset"], Segmented(choices, [T[d[1]] for d in DATASETS], T["short_stories"], self.note_data))
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
        draft = self.services.draft_spec()
        return current, (validate_spec(draft) if draft else None), (f"Version {v.number}" if v else "Current version")

    def next_number(self):
        v = self.services.current_version()
        return (v.number if v else 0) + 1

    def refresh(self):
        current, draft, name = self.specs()
        self.version_name = name
        self.current_box[0].configure(text=name)
        self.current_box[1].configure(text=describe(current))
        self.draft_box[0].configure(text=f"{name} with your change")
        for child in self.chip_row.winfo_children():
            child.destroy()
        diff = spec_diff(current, draft) if draft else []
        for text in diff:
            Chip(self.chip_row, text, "unsaved").pack(side="left", padx=(0, 4))
        self.draft_box[1].configure(text=cost_sentence(current, draft) if draft else "No change staged in the designer yet.")
        if self.job_id and self.result is None:  # the run finished while another page was open
            self.result = studio_quicktest.load_last(self.services, self.started)
            if self.result:
                self.running = False
                self.run_btn.configure(text="Run again")
        if self.result or self.running:
            self.show_result()

    # run --------------------------------------------------------------------
    def start(self):
        own = self.data.value == T["slice_your_data"]
        self.slow(lambda: not own or bool(text_sets(self.services.data("text"))), self._start)

    def _start(self, has_text):
        current, draft, _name = self.specs()
        if draft is None:
            self.status.configure(text="Stage a change in the designer first.")
        elif not self.guess.value:
            self.status.configure(text="Pick your guess first.")
        elif not has_text:
            self.status.configure(text=NO_TEXT)
        elif not self.running:
            data = next(d[0] for d in DATASETS if T[d[1]] == self.data.value)
            minutes = next(m[1] for m in MINUTES if T[m[0]] == self.time.value)
            self.job_id, self.running, self.result, self.started = uuid.uuid4().hex[:8], True, None, time.time()
            self.paces, self.progress = {}, {}
            self.services.enqueue(Job(id=self.job_id, kind="quicktest", label=f"Quick test · {minutes} min each",
                                      params={"current_spec": current, "draft_spec": draft, "dataset": data, "minutes": minutes,
                                              "shrink": self.size.value == T["shrunk_quarter_label"], "guess": self.guess.value}))
            self.run_btn.configure(text=T["running"] + "…")
            self.run_btn.state(["disabled"])
            self.status.configure(text="Queued: it runs when the computer is free.")
            self.show_result()

    def on_event(self, kind, payload):
        if not isinstance(payload, dict) or payload.get("job") != self.job_id:
            return
        if kind == "job" and payload.get("status") in ("failed", "stopped"):
            self.running = False
            self.run_btn.state(["!disabled"])
            self.run_btn.configure(text="Run quick test")
            self.status.configure(text=f"The quick test {payload['status']}" + (f": {payload['error']}" if payload.get("error") else "."))
            return
        if kind != "train":
            return
        if payload.get("type") == "quicktest_done":
            self.result, self.running = payload["results"], False
            self.run_btn.state(["!disabled"])
            self.run_btn.configure(text="Run again")
            self.status.configure(text="")
        elif payload.get("type") == "quicktest" and payload.get("step"):
            which = payload["which"]
            self.paces.setdefault(which, (time.time(), payload["step"]))
            self.progress[which] = (payload["step"], payload.get("total"))
            self.status.configure(text="")
        self.show_result(payload)

    def show_result(self, event=None):
        card = self.result_card
        for child in card.winfo_children():
            child.destroy()
        card.pack(fill="x", pady=(0, GUTTER), before=self.how)
        r = self.result
        if r and r.get("stopped"):
            Kicker(card, "STOPPED", bg="panel").pack(anchor="w")
            tk_label(card, "The quick test was stopped before it finished; nothing was kept.", 14, "text2", "panel").pack(anchor="w")
            return
        Kicker(card, "RESULT · SAME DATA, ORDER AND TIME" if r else "RUNNING · ONE AFTER THE OTHER, SAME DATA ORDER", bg="panel").pack(anchor="w")
        for label, color in (("current", "violet"), ("draft", "amber")):
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", pady=4)
            tk_label(row, self.version_name if label == "current" else "Your idea", 14, color, "panel", width=16).pack(side="left")
            bar = Progress(row, color=color)
            bar.pack(side="left", fill="x", expand=True, padx=GUTTER)
            step, total = self.progress.get(label, (0, None))
            if r and r.get(label):
                bar.set(1)
                text = f"done · {r[label]['steps']} steps"
            elif r:
                bar.set(0)
                text = "not run"
            elif total and step:
                bar.set(step / total)
                t0, step0 = self.paces.get(label, (0, step))
                left = seconds_left(t0, step - step0, total - step0, time.time())
                text = f"step {step}" + (f" · {left:.0f} s left" if left is not None else "")
            else:
                text = "done" if label == "current" and self.progress.get("draft") else "waiting"
            tk_label(row, text, 13, "muted", "panel").pack(side="left")
        if not r:
            return
        if r.get("current"):
            plot = BpcPlot(card, 640, 200, 14)
            plot.pack(anchor="w", pady=8)
            plot.set(steps=[tuple(p) for p in r["current"]["curve"]], smooth=[tuple(p) for p in r["draft"]["curve"]],
                     refs=[("uniform guess", r["references"]["uniform"]), ("counting word frequencies", r["references"]["unigram"])])
            legend = tk.Frame(card, bg=C["panel"])
            legend.pack(anchor="w")
            tk_label(legend, f"faint grey: {self.version_name}", 12, "muted", "panel").pack(side="left", padx=(0, 16))
            tk_label(legend, "violet: your idea", 12, "violet", "panel").pack(side="left")
        Kicker(card, T["early_signal"], bg="panel").pack(anchor="w", pady=(GUTTER, 0))
        tk_label(card, signal_text(r, self.version_name, self.guess.value), 17, "text", "panel", wraplength=720).pack(anchor="w", pady=4)
        tk_label(card, T["one_short_run"].split(" A design")[0], 13, "muted", "panel", wraplength=720).pack(anchor="w")
        if r.get("references") and not r.get("learned_context"):
            tk_label(card, GUARD, 13, "amber", "panel", wraplength=720).pack(anchor="w", pady=(4, 0))
        Kicker(card, T["does_work"], bg="panel").pack(anchor="w", pady=(GUTTER, 0))
        by_id = {c["id"]: c for c in r["checks"]}
        for key, name, what in CHECKS:
            ok = by_id[key]["ok"]
            line = tk.Frame(card, bg=C["panel"])
            line.pack(fill="x", pady=2)
            tk_label(line, check_mark(ok), 15, "violet" if ok else "muted" if ok is None else "coral", "panel", width=3).pack(side="left")
            tk_label(line, f"{name} — {what} ({by_id[key]['detail']})", 13, "text2", "panel", wraplength=680).pack(side="left")
        if r.get("current"):
            Kicker(card, T["what_each_writes"], bg="panel").pack(anchor="w", pady=(GUTTER, 0))
            for name, key in ((self.version_name, "current"), ("Your idea", "draft")):
                tk_label(card, name, 13, "muted", "panel").pack(anchor="w")
                tk_label(card, r[key]["sample"], 13, "text", "panel", wraplength=700, font=font(13, mono=True)).pack(anchor="w", pady=(0, 6))
        actions = tk.Frame(card, bg=C["panel"])
        actions.pack(anchor="w", pady=(GUTTER, 0))
        button(actions, T["save_as_version"].replace("version 6", f"version {self.next_number()}"), self.save, primary=True).pack(side="left")
        button(actions, T["run_real_comparison"], lambda: self.navigate("compare")).pack(side="left", padx=GUTTER)
        button(actions, T["discard_idea"], self.discard).pack(side="left", padx=GUTTER)

    def save(self):
        """Save the idea as the next version; the designer then shows it as saved and needing training."""
        current, draft, _name = self.specs()
        if draft:
            self.services.save_version(draft, ", ".join(spec_diff(current, draft)) or "no change")
            self.services.stage_draft(None)
            self.navigate("builder")

    def discard(self):
        self.services.stage_draft(None)
        self.result, self.running, self.job_id = None, False, None
        self.result_card.pack_forget()
        self.run_btn.state(["!disabled"])
        self.run_btn.configure(text="Run quick test")
        self.navigate("builder")

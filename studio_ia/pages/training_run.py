"""Pretraining run: readable at a glance. Everything in bits per character."""
from __future__ import annotations

import math
import tkinter as tk

import studio_eval
from studio_ia.copy import TRAINING as T
from studio_ia.pages.base import Page
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, Chip, HowItWorks, Kicker, button, info_icon, tip

STATUS = {"learning": ("Learning", "training"), "flat": ("Flat", "unsaved"), "memorising": ("Memorising", "needs_training")}
CONTEXT = 256  # pieces per row, matches the Pretraining page


def phase_at(phases, step):
    """Schedule phase (warm-up | steady | cool-down) of `step` from run_curve's [(start, end, name)]; '' without a schedule."""
    return next((name for _s, end, name in phases if step <= end), phases[-1][2]) if phases else ""


def history_of(steps, heldout):
    """Engine-style history rows ({"loss", "val_loss"}) from the curve's per-step and held-back points."""
    held = dict(heldout)
    return [{"loss": y, "val_loss": held.get(x)} for x, y in steps]


def sample_rows(curve, step, pn):
    """(label, text) lines under 'what it writes now': step 0, the logged samples, the newest marked 'now'; before the first sample, a wait note."""
    rows = [(T["step_0"], T["identical_to"].format(parent=f"version {pn}") if pn else T["random_start"])]
    rows += [(f"step {x['step']}", x["text"]) for x in (curve.get("samples") or [])]
    if len(rows) > 1:
        rows[-1] = (T["now_step"].format(step=curve["samples"][-1]["step"]), rows[-1][1])
    else:
        rows.append((T["now_step"].format(step=f"{step:,}") if step is not None else "now", T["no_sample_yet"]))
    return rows


def switch_note(running, last_held, best, gate):
    """(copy key, {gap}) for the line about Chat switching. While the run goes it is only a first look at the held-back curve against
    the earlier best; once it ends and Chat has weighed it (services.gate_result), the paired test's verdict."""
    if not running and gate and gate.get("reason") == "few_docs":
        return "gate_few_docs", {"docs": gate["docs"], "need": studio_eval.MIN_DOCS}
    if not running and gate and math.isfinite(gate["diff"]):
        gap = f"{abs(gate['diff']):.3f}"
        return ("gate_better" if gate["better"] else "gate_not_sure" if gate["diff"] < 0 else "gate_not_better"), {"gap": gap}
    if last_held is None or best is None:
        return "no_band_yet", {}
    return ("held_back_below" if last_held < best else "held_back_inside"), {}


def num(x):
    return "–" if x is None else f"{x:.3f}"


class TrainingRunPage(Page):
    title = "Pretraining"
    lead = T["step"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pending = False
        b = self.body
        self.change = tk_label(b, "", 14, "text2", wraplength=760)
        self.change.pack(anchor="w")
        self.see = tk_label(b, T["see_change"], 13, "violet", cursor="hand2", takefocus=1)
        for seq in ("<Button-1>", "<Return>", "<space>"):
            self.see.bind(seq, lambda _e: self.navigate("builder"))
        row = tk.Frame(b, bg=C["bg"])
        row.pack(anchor="w", pady=GUTTER)
        self.pause_btn = button(row, T["pause"], self.toggle_pause, primary=True)
        self.pause_btn.pack(side="left")
        self.stop_btn = button(row, T["stop_cooldown"], lambda: self.services.stop_after_cooldown())
        self.stop_btn.pack(side="left", padx=GUTTER)
        self.progress = tk_label(b, "", 14, "text2", wraplength=760)
        self.progress.pack(anchor="w", pady=(0, GUTTER))
        card = Card(b)
        card.pack(fill="x", pady=(0, GUTTER))
        nums = tk.Frame(card, bg=C["panel"])
        nums.pack(fill="x")
        self.nums = []
        for label in (T["error_training"], T["error_held_back"], T["version_best"].format(parent="the earlier version")):
            col = tk.Frame(nums, bg=C["panel"])
            col.pack(side="left", expand=True, fill="x", anchor="w")
            value = tk_label(col, "–", 28, "text", "panel", font=font(28, mono=True))
            value.pack(anchor="w")
            name = tip(tk_label(col, label, 12, "muted", "panel"), T["bits_per_char_note"])
            name.pack(anchor="w")
            self.nums.append((value, name))
        self.chip = Chip(card, "Learning", "training")
        self.chip.pack(anchor="w", pady=(10, 2))
        self.plot = tip(BpcPlot(card, 700, 280), T["cooldown_drop"])
        self.plot.pack(fill="x", pady=(6, 0))
        legend = tk_label(card, f"{T['training_text']} · {T['every_step']} (faint) · {T['held_back']} (dots)", 12, "muted", "panel")
        legend.pack(anchor="w", pady=(4, 0))
        self.switch = tk.Frame(b, bg=C["blue_bg"], highlightthickness=1, highlightbackground=C["blue"], padx=GUTTER, pady=12)
        self.switch.pack(fill="x", pady=(0, GUTTER))
        self.switch_head = tk_label(self.switch, "", 15, "text", "blue_bg", font=font(15, bold=True))
        self.switch_head.pack(anchor="w")
        self.switch_text = tk_label(self.switch, "", 13, "text2", "blue_bg", wraplength=700)
        self.switch_text.pack(anchor="w")
        self.compare = tk_label(self.switch, T["compare_chat"], 13, "violet", "blue_bg", cursor="hand2", takefocus=1)
        for seq in ("<Button-1>", "<Return>", "<space>"):
            self.compare.bind(seq, lambda _e: self.navigate("chat"))
        self.compare.pack(anchor="w", pady=(4, 0))
        how = HowItWorks(b, T["how_status_work"], bg="bg")
        how.pack(fill="x", pady=(0, GUTTER))
        text = (f"{T['status_one_of']} {T['status_learning']} {T['learning_desc']} {T['status_flat']} {T['flat_desc']} "
                f"{T['status_memorising']} {T['memorising_desc']}\n\n{T['schedule_note']}\n\n{T['lr_auto_note']}\n\n{T['gate_how']}")
        tk_label(how.body, text, 13, "text2", "bg", wraplength=760).pack(anchor="w")
        self.samples = Card(b)
        self.samples.pack(fill="x")

    # actions -------------------------------------------------------------
    def toggle_pause(self):
        paused = self.services.queue().paused
        (self.services.resume if paused else self.services.pause)()
        self.refresh()

    def on_event(self, kind, payload):
        """Live events redraw at most every 1.5 s (the curve is re-read from the run's metrics file); showing the page again
        refreshes from disk, so nothing is lost while it is hidden."""
        if kind in ("queue", "train", "job") and not self._pending:
            self._pending = True
            self.after(1500, self._redraw)

    def _redraw(self):
        self._pending = False
        self.refresh()

    # state ---------------------------------------------------------------
    def refresh(self):
        s = self.services
        q = s.queue()
        try:
            v = s.current_version()
        except NotImplementedError:
            v = None
        parent = next((p for p in s.versions() if v and p.number == v.parent), None) if v else None
        n, pn = (v.number if v else None), (parent.number if parent else None)
        self.h1.configure(text=("Pretraining" if n is None else f"Pretraining · version {n}") + (" catching up" if q.job else ""))
        self.set_lead("Step k of n, the schedule phase and the error so far.")
        if v is None:
            change = "No version yet: design one in the Network designer, then train it in Pretraining."
        elif pn is None:
            change = f"Version {n} starts from random weights: there is no earlier version to continue from."
        else:
            change = (f"Version {n} = version {pn}" + (f" with {v.change}" if v.change else "") +
                      f". The new parts start at zero, so it began exactly at version {pn}’s error.")
        self.change.configure(text=change)
        if pn is None:  # no earlier version, so no change to see
            self.see.pack_forget()
        else:
            self.see.pack(anchor="w", pady=(2, 0), after=self.change)
        self.nums[2][1].configure(text=T["version_best"].format(parent=f"version {pn}" if pn else "the earlier version"))
        self.pause_btn.configure(text="Resume" if q.paused else T["pause"])
        self.pause_btn.state(["!disabled"] if q.job or q.paused else ["disabled"])  # pausing an empty queue only blocked the next job
        self.stop_btn.state(["!disabled"] if q.job else ["disabled"])
        params = q.job.params if q.job else {}
        run_dir = params.get("output_dir")
        curve = s.run_curve(run_dir) if run_dir else {}
        steps = curve.get("steps") or []
        smooth = curve.get("smooth") or []
        held = curve.get("heldout") or []
        try:
            refs = s.references(parent or v) if v else {}
        except NotImplementedError:
            refs = {}
        best = refs.get("best_so_far_bpc")
        try:
            gate = s.gate_result(v) if v and not q.job else None
        except NotImplementedError:
            gate = None
        rows = params.get("config", {}).get("batch_size", 16)
        seen = q.step * rows * params.get("config", {}).get("context", CONTEXT)
        per_pass = curve.get("pieces_per_pass") or params.get("pieces_per_pass")
        lr = "auto" if q.lr is None else f"auto ({q.lr:.2g})"
        phase = q.phase or phase_at(curve.get("phases") or [], q.step)
        parts = [f"{T['step']} {q.step:,} {T['of']} {q.total:,}" if q.total else "Starting"]
        parts.append((f"{T['pass']} {seen // per_pass + 1} {T['over_text']} " if per_pass else "") + f"{seen:,} {T['pieces_seen']}")
        if phase:
            parts.append(f"{T['schedule']} {phase}")
        parts.append(T["lr_auto"].replace("auto", lr))
        if q.eta_s is not None:
            parts.append(f"about {max(1, round(q.eta_s / 60))} min {T['left']}")
        self.progress.configure(text=" · ".join(parts) + (" · cooling down, then stopping" if q.stopping else "")
                                + (f" · waiting: {q.blocked}" if q.blocked else "") if q.job else "Nothing is training.")
        history = curve.get("history") or history_of(steps, held)
        if history:
            self.chip.set(*STATUS[studio_eval.run_status(history)])
            self.chip.pack(anchor="w", pady=(10, 2), before=self.plot)
        else:
            self.chip.pack_forget()  # no data, no status
        last_s = smooth[-1][1] if smooth else None
        last_h = held[-1][1] if held else None
        for (value, _), x in zip(self.nums, (last_s, last_h, best)):
            value.configure(text=num(x))
        reflines = [(name, val) for name, val in (("uniform guess", refs.get("uniform_bpc")),
                    ("counting word frequencies", refs.get("unigram_bpc")), (f"version {pn}’s best" if pn else "earlier best", best)) if val is not None]
        self.plot.set(steps=steps, smooth=smooth, heldout=held, refs=reflines, total=curve.get("total") or q.total or None,
                      progress=q.step, lr=curve.get("lr") or [], phases=curve.get("phases") or [])
        chat = s.best_version() if v else None
        cn = chat.number if chat else pn
        self.switch_head.configure(text=T["chat_still_uses"].format(chat=f"version {cn}" if cn else "no trained version yet"))
        words = {"Version": f"Version {n}" if n else "This version", "version": f"version {n}" if n else "this version",
                 "parent": f"version {gate['vs']}" if gate and not q.job else f"version {pn}" if pn else "the earlier best"}
        key, extra = switch_note(bool(q.job), last_h, best, gate)
        self.switch_text.configure(text=T[key].format(**words, **extra) + (" " + T["gate_seeds"] if key in ("gate_not_sure", "gate_not_better") else ""))
        self.samples_view(curve, q, n, pn)

    def samples_view(self, curve, q, n, pn):
        every = curve.get("sample_every")
        rows = sample_rows(curve, q.step if q.total else None, pn)
        # The last label ("now · step N") moves every step; it is updated in place, so only new samples rebuild the card,
        # which would otherwise flash on every 1.5 s refresh.
        shown = (every, curve.get("prompt"), tuple(rows[:-1]), rows[-1][1])
        if shown == getattr(self, "_samples_shown", None):
            self._now_label.configure(text=rows[-1][0])
            return
        self._samples_shown = shown
        for child in self.samples.winfo_children():
            child.destroy()
        head = tk.Frame(self.samples, bg=C["panel"])
        head.pack(anchor="w")
        Kicker(head, T["what_writes_now"].format(every=f"{every:,}") if every else T["what_writes_now_plain"], bg="panel").pack(side="left")
        info_icon(head, T["always_picks"]).pack(side="left", padx=(4, 0))
        tk_label(self.samples, curve.get("prompt") or T["example_prompt"], 14, "text2", "panel", font=font(14, serif=True)).pack(anchor="w", pady=6)
        for label, text in rows:
            line = tk.Frame(self.samples, bg=C["panel"])
            line.pack(fill="x", pady=2)
            self._now_label = tk_label(line, label, 12, "muted", "panel", width=14)
            self._now_label.pack(side="left", anchor="n")
            tk_label(line, text, 14, "text", "panel", font=font(14, serif=True), wraplength=620).pack(side="left")

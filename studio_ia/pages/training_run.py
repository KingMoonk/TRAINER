"""Pretraining run: readable at a glance. Everything in bits per character."""
from __future__ import annotations

import tkinter as tk

import studio_eval
from studio_ia.copy import TRAINING as T
from studio_ia.pages.base import Page
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, Chip, HowItWorks, Kicker, button

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
    rows = [(T["step_0"], T["identical_to"].replace("version 4", f"version {pn}"))]
    rows += [(f"step {x['step']}", x["text"]) for x in (curve.get("samples") or [])]
    if len(rows) > 1:
        rows[-1] = (T["now_step"].format(step=curve["samples"][-1]["step"]), rows[-1][1])
    else:
        rows.append((T["now_step"].format(step=step if step is not None else "[N]"), T["no_sample_yet"]))
    return rows


def num(x):
    return "[N]" if x is None else f"{x:.3f}"


class TrainingRunPage(Page):
    title = "Pretraining · version [N]"
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
        self.see.pack(anchor="w", pady=(2, GUTTER))
        row = tk.Frame(b, bg=C["bg"])
        row.pack(anchor="w", pady=(0, GUTTER))
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
        for label in (T["error_training"], T["error_held_back"], T["version_best"]):
            col = tk.Frame(nums, bg=C["panel"])
            col.pack(side="left", expand=True, fill="x", anchor="w")
            value = tk_label(col, "[N]", 28, "text", "panel", font=font(28, mono=True))
            value.pack(anchor="w")
            name = tk_label(col, label, 12, "muted", "panel")
            name.pack(anchor="w")
            self.nums.append((value, name))
        self.chip = Chip(card, "Learning", "training")
        self.chip.pack(anchor="w", pady=(10, 2))
        tk_label(card, T["bits_per_char_note"], 12, "dim", "panel").pack(anchor="w", pady=(0, 6))
        self.plot = BpcPlot(card, 700, 280)
        self.plot.pack(fill="x")
        tk_label(card, "Error usually drops most in the cool-down, where the learning rate falls to near zero.", 12, "dim", "panel").pack(anchor="w", pady=(4, 0))
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
                f"{T['status_memorising']} {T['memorising_desc']}\n\n{T['schedule_note']}\n\n{T['lr_auto_note']}")
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
        n, pn = (v.number if v else "[N]"), (parent.number if parent else "[N]")
        self.h1.configure(text=f"Pretraining · version {n}" + (" catching up" if q.job else ""))
        self.lead_label.configure(text="Step k of n, the schedule phase and the error so far.")
        change = v.change if v and v.change else "[N]"
        self.change.configure(text=f"Version {n} = version {pn} with {change}. The new parts start at zero, so it began exactly at version {pn}’s error.")
        self.pause_btn.configure(text="Resume" if q.paused else T["pause"])
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
        best, band = refs.get("best_so_far_bpc"), refs.get("noise_band")
        rows = params.get("config", {}).get("batch_size", 16)
        seen = q.step * rows * params.get("config", {}).get("context", CONTEXT)
        per_pass = curve.get("pieces_per_pass") or params.get("pieces_per_pass")
        pass_no = f"{seen // per_pass + 1}" if per_pass else "[N]"
        left = f"about {max(1, round(q.eta_s / 60))} min" if q.eta_s is not None else "about [N] min"
        lr = "auto" if q.lr is None else f"auto ({q.lr:.2g})"
        self.progress.configure(text=f"{T['step']} {q.step if q.total else '[N]'} {T['of']} {q.total or '[N]'} · {T['pass']} {pass_no} "
                                f"{T['over_text']} {seen:,} {T['pieces_seen']} · {T['schedule']} {q.phase or phase_at(curve.get('phases') or [], q.step) or '[N]'} · "
                                f"{T['lr_auto'].replace('auto', lr)} · {left} {T['left']}"
                                + (" · cooling down, then stopping" if q.stopping else "") + (f" · waiting: {q.blocked}" if q.blocked else "")
                                if q.job else "Nothing is training.")
        history = curve.get("history") or history_of(steps, held)
        status, state = STATUS[studio_eval.run_status(history)]
        self.chip.set(status, state)
        last_s = smooth[-1][1] if smooth else None
        last_h = held[-1][1] if held else None
        for (value, _), x in zip(self.nums, (last_s, last_h, best)):
            value.configure(text=num(x))
        reflines = [(name, val) for name, val in (("uniform guess", refs.get("uniform_bpc")),
                    ("counting word frequencies", refs.get("unigram_bpc")), (f"version {pn}’s best", best)) if val is not None]
        self.plot.set(steps=steps, smooth=smooth, heldout=held, refs=reflines, total=curve.get("total") or q.total or None,
                      progress=q.step, lr=curve.get("lr") or [], phases=curve.get("phases") or [], band=band)
        chat = s.best_version() if v else None
        cn = chat.number if chat else pn
        below = last_h is not None and band is not None and last_h < band[0]
        self.switch_head.configure(text=T["chat_still_uses"].replace("version 4", f"version {cn}"))
        self.switch_text.configure(text=(f"Version {n}’s held-back error is below version {pn}’s noise band. Chat switches at the next checkpoint."
                                         if below else T["held_back_inside"].replace("Version 5", f"Version {n}").replace("version 4", f"version {pn}")))
        self.samples_view(curve, q, n, pn)

    def samples_view(self, curve, q, n, pn):
        for child in self.samples.winfo_children():
            child.destroy()
        every = curve.get("sample_every")
        Kicker(self.samples, T["what_writes_now"].replace("[N]", str(every or "[N]")), bg="panel").pack(anchor="w")
        tk_label(self.samples, T["always_picks"], 12, "dim", "panel").pack(anchor="w")
        tk_label(self.samples, curve.get("prompt") or T["example_prompt"], 14, "text2", "panel", font=font(14, serif=True)).pack(anchor="w", pady=6)
        rows = sample_rows(curve, q.step if q.total else None, pn)
        for label, text in rows:
            line = tk.Frame(self.samples, bg=C["panel"])
            line.pack(fill="x", pady=2)
            tk_label(line, label, 12, "muted", "panel", width=14).pack(side="left", anchor="n")
            tk_label(line, text, 14, "text", "panel", font=font(14, serif=True), wraplength=620).pack(side="left")

"""Compare runs: change one thing, hold the rest equal, run several seeds, read the spread."""
from __future__ import annotations

import tkinter as tk
import uuid
from dataclasses import asdict

import studio_experiments as ex
from studio_ia.copy import COMPARE as T
from studio_ia.pages.base import Page
from studio_ia.pages.builder import text_sets
from studio_ia.services import Job
from studio_ia.theme import C, GUTTER, entry, font, tk_label
from studio_ia.widgets import BpcPlot, Card, GuessChips, Kicker, Segmented, Toggle, _activate, button, tip
from studio_spec import default_spec, validate_spec
from studio_versions import change_text

LOOKS = ("Pick your guess first.", "Add a variant first.")


def queue_block(plan, error, variants, guess):
    """Why "Queue the runs" cannot work yet, in words; "" when it can. Checked on every change, so the button is only live when a click will queue."""
    if plan is None:
        return error
    return LOOKS[1] if not variants else LOOKS[0] if not guess else ""


NO_TEXT = "Your data has no plain text yet. Add some in Your data, then queue the runs."

MODE_KEYS = (("params", "params_mode", "params_note", "x_axis_text"), ("compute", "compute_mode", "compute_note", "x_axis_compute"),
             ("tokens", "tokens_mode", "tokens_note", "x_axis_text"), ("wall-clock", "time_mode", "time_note", "x_axis_time"))
SEEDS = ("1", "3", "5")
COLORS = ("text", "violet", "amber", "blue", "pink", "sage")
STEPS, WALL_SECONDS = 300, 120.  # ponytail: fixed length per run; make it a choice when the board has one


AXES = (("steps", "axis_steps", "x_axis_steps", 0, 1), ("tokens", "axis_tokens", "x_axis_text", 2, 1), ("time", "axis_time", "x_axis_time", 3, 60.))


def axis_points(points, axis):
    """[(x, bpc)] on the chosen axis, or None when any point lacks it. Points are [step, bpc(, trained_tokens, elapsed_seconds)]."""
    _key, _label, _title, col, div = next(a for a in AXES if a[0] == axis)
    if any(len(p) <= col or p[col] is None for p in points):
        return None
    return [(p[col] / div, p[1]) for p in points]


def axis_for(curves, axis):
    """(axis actually used, missing): the asked-for axis if every arm's curve has it, else steps and the axis that was missing."""
    if axis != "steps" and any(axis_points(points, axis) is None for points in curves):
        return "steps", axis
    return axis, None


def arm_series(n, points, live):
    """Style of one arm's line: the arm's colour; a seed still running is dashed, the finished mean of seeds is solid."""
    return {"points": points, "color": COLORS[n % len(COLORS)], "width": 2 if live else 3, "dash": (6, 4) if live else None}


def million(n):
    return f"{n / 1e6:.2f}M"


def final_cell(s):
    """'mean · spread x' (or one seed) and, under it, speed and peak memory; a dash before results."""
    if not s:
        return "–"
    head = f"{s['mean']:.3f} · spread {s['spread']:.3f}" if len(s["bpc"]) > 1 else f"{s['mean']:.3f} · one seed"
    return f"{head}\n{s['pieces_per_s']:,.0f} pieces/s · {s['peak_mb']:.0f} MB peak"


def delta_cell(s):
    """Difference from the base; it counts only when larger than the spread across seeds."""
    if not s:
        return "–"
    if s["delta"] is None:
        return "—"
    note = {True: "inside the spread", False: "beyond the spread", None: "one seed: no spread"}[s["inside_spread"]]
    return f"{s['delta']:+.3f} · {note}"


def guess_line(guess, letter, row):
    """What happened to the guessed variant next to the guess; no verdict without a spread to compare against."""
    outcome = ex.variant_outcome(row)
    if outcome is None:
        return f"You guessed {guess.lower()}. No verdict on {letter}: " + ("it needs at least two seeds to have a spread." if row else "it has no result yet.")
    return f"You guessed {guess.lower()}; {letter} came out {outcome}" + (" (inside the spread)." if outcome == "about the same" else " (beyond the spread).")


class ComparePage(Page):
    title = T["h1"]
    lead = T["lead"]
    crumb = ("Network designer",)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.variants = []  # [{"name", "spec"}]
        self.mode, self.job_id, self.live, self.results, self.error = "params", None, {}, None, ""
        self.done_guess = ""
        b = self.body
        setup = Card(b)
        setup.pack(fill="x", pady=(0, GUTTER))
        Kicker(setup, T["question"], bg="panel").pack(anchor="w")
        self.question = entry(setup)
        self.question.pack(fill="x", pady=(4, GUTTER))
        Kicker(setup, T["hold_equal"], bg="panel").pack(anchor="w")
        self.modes = Segmented(setup, [T[m[1]] for m in MODE_KEYS], T["params_mode"], self.pick_mode)
        self.modes.pack(anchor="w", pady=(4, GUTTER))
        self.mode_tip = self.seeds_tip = ""
        tip(self.modes, lambda: self.mode_tip)
        row = tk.Frame(setup, bg=C["panel"])
        row.pack(anchor="w")
        tk_label(row, T["seeds"], 14, "text2", "panel").pack(side="left", padx=(0, GUTTER))
        self.seeds = Segmented(row, SEEDS, "3", lambda _v: self.update_plan())
        self.seeds.pack(side="left")
        self.retune = Toggle(setup, T["retune_lr"], True, lambda _v: self.update_plan())
        self.retune.pack(anchor="w", pady=(GUTTER, 4))
        for part in (row, self.retune):
            tip(part, lambda: self.seeds_tip)
        self.table_card = Card(b)
        self.table_card.pack(fill="x", pady=(0, GUTTER))
        self.table = tk.Frame(self.table_card, bg=C["panel"])
        self.table.pack(fill="x")
        add = tk.Frame(self.table_card, bg=C["panel"])
        add.pack(anchor="w", pady=(GUTTER, 0))
        button(add, T["add_variant"], self.add_from_designer).pack(side="left")
        self.problem = tk_label(self.table_card, "", 13, "amber", "panel", wraplength=720)
        self.problem.pack(anchor="w", pady=(6, 0))
        run = tk.Frame(b, bg=C["bg"])
        run.pack(fill="x", pady=(0, GUTTER))
        self.guess_label = tk_label(run, "", 14, "text2")
        self.guess_label.pack(anchor="w")
        self.guess = GuessChips(run, [T["better"], T["about_same"], T["worse"]], lambda _v: self.update_plan(), bg="bg")
        self.guess.pack(anchor="w", pady=(6, GUTTER))
        self.queue_btn = button(run, "", self.queue_it, primary=True)
        self.queue_btn.pack(anchor="w")
        self.status = tk_label(run, "", 13, "pink")
        self.status.pack(anchor="w", pady=(6, 0))
        self.curves = Card(b)
        self.curves.pack(fill="x")
        self.curves_title = Kicker(self.curves, "", bg="panel")
        self.curves_title.pack(anchor="w")
        tk_label(self.curves, T["curves_desc"], 13, "muted", "panel").pack(anchor="w")
        self.x_axis, self.axis_note = "steps", tk_label(self.curves, "", 13, "amber", "panel", wraplength=720)
        self.axes = Segmented(self.curves, [T[a[1]] for a in AXES], T["axis_steps"], self.pick_axis)
        self.axes.pack(anchor="w", pady=(6, 0))
        tip(self.axes, T["axis_fair"])
        self.axis_note.pack(anchor="w")
        self.legend = tk.Frame(self.curves, bg=C["panel"])
        self.legend.pack(anchor="w", pady=6)
        self.plots = tk.Frame(self.curves, bg=C["panel"])
        self.plots.pack(fill="x")
        self.hint = tk_label(self.curves, T["curves_stream"], 13, "dim", "panel", wraplength=720)
        self.hint.pack(anchor="w")
        self.update_plan()

    # model ------------------------------------------------------------------
    @property
    def base(self):
        v = self.services.current_version()
        try:
            return validate_spec(v.spec) if v and v.spec else validate_spec(default_spec())
        except ValueError:
            return validate_spec(default_spec())

    @property
    def base_number(self):
        v = self.services.current_version()
        return v.number if v else None

    def experiment(self):
        names = ["Base"] + [v["name"] for v in self.variants]
        return ex.ExperimentSpec(self.base, [v["spec"] for v in self.variants], seeds=int(self.seeds.value), mode=self.mode,
                                 steps=STEPS, wall_seconds=WALL_SECONDS, question=self.question.get().strip(),
                                 lr_retune=self.retune.value, guess=self.guess.value or "", names=names)

    def add_variant(self, name=None, spec=None):
        spec = validate_spec(spec)
        self.variants.append({"name": name or change_text(self.base, spec) or f"Variant {len(self.variants) + 1}", "spec": spec})
        self.update_plan()

    def remove_variant(self, index):
        del self.variants[index]
        self.update_plan()

    def add_from_designer(self):
        draft = self.services.draft_spec()
        if draft and change_text(self.base, draft) != "no change":
            self.add_variant(spec=draft)
        else:
            self.status.configure(text="Stage a change in the designer first, then add it here.")
            self.navigate("builder")

    def reopen(self, n):
        """Any row reopens in the designer: its spec is staged there as the draft."""
        self.services.stage_draft(self.variants[n - 1]["spec"] if n else None)
        self.navigate("builder")

    def pick_mode(self, label):
        self.mode = next(m[0] for m in MODE_KEYS if T[m[1]] == label)
        self.update_plan()

    def pick_axis(self, label):
        self.x_axis = next(a[0] for a in AXES if T[a[1]] == label)
        self.draw_curves()

    def label_for(self, row, plan, lr_on):
        if not row["variant"]:
            return "—"
        base, mode = plan["variants"][0], self.mode
        if mode == "params":
            text = (T["width_resolved"] if row["adjustments"] else T["none_needed"]).replace("[N]M", million(row["parameters"]))
            text += f" · within {row['relative_error']:.1%}"
        elif mode == "compute":
            more = row["steps"] / base["steps"] - 1
            text = T["stops_at_base"].replace("[N]%", f"{more:.0%}") if more > .005 else T["same_compute"]
        elif mode == "tokens":
            text = T["parameters"].replace("[N]M", million(row["parameters"]))
        else:
            text = T["min_budget"].replace("[N]", f"{WALL_SECONDS / 60:g}")
        return text + ("" if lr_on else " · learning rate not retuned")

    def refresh(self):
        if self.job_id and self.results is None:  # the runs finished while another page was open
            run_dir = getattr(self.services.workspace, "run_dir", None)
            self.results = ex.load_results(run_dir, self.job_id) if run_dir else None
            if self.results:
                self.status.configure(text=self.finished_text())
        self.update_plan()

    def finished_text(self):
        rows = {v["variant"]: v for v in (self.results or {}).get("variants", [])}
        return guess_line(self.guess.value or self.done_guess, "B", rows.get(1)) if (self.guess.value or self.done_guess) else ""

    # drawing ----------------------------------------------------------------
    def update_plan(self):
        _key, _mode, note, _axis = next(m for m in MODE_KEYS if m[0] == self.mode)
        lr_on = self.retune.value
        self.mode_tip = T[note]
        self.seeds_tip = T["seeds_note"].format(lr_note=T["lr_note_on" if lr_on else "lr_note_off"])
        self.guess_label.configure(text=T["before_runs"])
        try:
            self.plan = ex.plan(self.experiment())
            self.error = ""
        except ValueError as error:
            self.plan, self.error = None, str(error)
        block = queue_block(self.plan, self.error, self.variants, self.guess.value)
        self.problem.configure(text=self.error or next((f"Variant {r['variant']}: {r['reason']}" for r in self.plan["variants"] if not r["matched"]), block))
        self.queue_btn.state(["disabled"] if block else ["!disabled"])
        runs, sweeps = (self.plan["n_runs"], self.plan["n_sweeps"]) if self.plan else (0, 0)
        self.queue_btn.configure(text=T["queue_runs"].format(count=runs, sweeps=sweeps) if sweeps else T["queue_runs_no_sweep"].format(count=runs))
        self.draw_table(lr_on)
        self.draw_curves()

    def draw_table(self, lr_on):
        for child in self.table.winfo_children():
            child.destroy()
        for col, head in enumerate((T["run"], "", T["change_from_base"], T["matching"], T["final_error"], T["vs_base"])):
            tk_label(self.table, head, 12, "muted", "panel").grid(row=0, column=col, sticky="w", padx=(0, GUTTER), pady=(0, 6))
        rows = self.plan["variants"] if self.plan else []
        stats = {v["variant"]: v for v in (self.results or {}).get("variants", [])}
        number = self.base_number
        for i, row in enumerate(rows, 1):
            n, color = row["variant"], COLORS[row["variant"] % len(COLORS)]
            name = T["base_title"].replace("version 4", f"version {number}" if number else "this design") if not n else row["name"]
            diff = T["base_diff"] if not n else " · ".join(ex.spec_changes(self.base, self.variants[n - 1]["spec"])) or "no change"
            s = stats.get(n)
            vs = "—" if not n else delta_cell(s)
            cells = (chr(ord("A") + n), name, diff, self.label_for(row, self.plan, lr_on), final_cell(s), vs)
            for col, text in enumerate(cells):
                dim = col == 5 and s and s["inside_spread"]
                lbl = tk_label(self.table, text, 14 if col < 2 else 13, color if col == 0 else "dim" if dim else "text" if col < 2 else "text2", "panel",
                               wraplength=240 if col in (2, 3, 4) else 0, font=font(12, mono=True) if col == 2 else font(14 if col < 2 else 13))
                lbl.grid(row=i, column=col, sticky="w", padx=(0, GUTTER), pady=4)
                if col == 1:
                    lbl.configure(cursor="hand2", takefocus=1)
                    _activate(lbl, lambda k=n: self.reopen(k))
            if n:
                gone = tk_label(self.table, "remove", 12, "muted", "panel", cursor="hand2", takefocus=1)
                gone.grid(row=i, column=6, sticky="w")
                _activate(gone, lambda k=n - 1: self.remove_variant(k))

    def draw_curves(self):
        for child in self.legend.winfo_children() + self.plots.winfo_children():
            child.destroy()
        final = {v["variant"]: v for v in (self.results or {}).get("variants", [])}
        curves = {n: v["curve"] for n, v in final.items()} or {n: pts for n, (_seed, pts) in self.live.items()}
        self.hint.pack_forget() if curves else self.hint.pack(anchor="w")
        live = not final
        axis, missing = axis_for(curves.values(), self.x_axis)
        title = next(a[2] for a in AXES if a[0] == axis)
        self.curves_title.configure(text=T["curves_title"].format(xAxis=T[title]).upper())
        self.axis_note.configure(text=T["axis_fallback"].format(what=T[next(a[1] for a in AXES if a[0] == missing)].lower()) if missing
                                 else "")
        refs = [(k, v) for k, v in ((self.results or {}).get("references") or {}).items()]
        names = {"uniform": "uniform guess", "unigram": "counting word frequencies"}
        for n, points in sorted(curves.items()):
            name = self.plan["variants"][n]["name"] if self.plan and n < len(self.plan["variants"]) else f"Variant {n}"
            tk_label(self.legend, f"{chr(ord('A') + n)} · {name}", 13, COLORS[n % len(COLORS)], "panel").pack(side="left", padx=(0, GUTTER))
            plot = BpcPlot(self.plots, 640, 140, 14)
            plot.pack(anchor="w", pady=4)
            spread = final.get(n)
            plot.set(series=[arm_series(n, axis_points(points, axis), live)], refs=[(names[k], v) for k, v in refs],
                     band=(spread["min"], spread["max"]) if spread and len(spread["bpc"]) > 1 else None)

    # queue ------------------------------------------------------------------
    def queue_it(self):
        if not self.guess.value:
            self.status.configure(text=LOOKS[0])
            return
        if self.plan is None or len(self.variants) < 1:
            self.status.configure(text=LOOKS[1] if self.plan is not None else self.error)
            return
        self.slow(lambda: text_sets(self.services.data("text")), self._queue)

    def _queue(self, sets):
        if not sets:
            self.status.configure(text=NO_TEXT)
            return
        exp = self.experiment()
        label = f"Compare runs: {self.plan['n_runs']} runs" + (f" + {self.plan['n_sweeps']} short sweeps" if self.plan["n_sweeps"] else "")
        self.job_id = uuid.uuid4().hex[:8]
        self.live, self.results, self.done_guess = {}, None, self.guess.value
        job = Job(id=self.job_id, kind="compare", label=label, note=exp.question,
                  params={"experiment": asdict(exp), "dataset_kind": "text", "datasets": sets, "guess": self.guess.value})
        self.services.enqueue(job)
        self.status.configure(text=f"Queued: {label}. {self.services.queue_outlook(job)}.")
        self.draw_table(self.retune.value)
        self.draw_curves()

    def on_event(self, kind, payload):
        if not isinstance(payload, dict) or payload.get("job") != self.job_id:
            return
        if kind == "job" and payload.get("status") in ("failed", "stopped"):
            self.status.configure(text=f"The runs {payload['status']}" + (f": {payload['error']}" if payload.get("error") else "."))
            return
        if kind != "train":
            return
        letter = chr(ord("A") + payload.get("variant", 0))
        if payload.get("type") == "compare" and payload.get("phase") == "run":
            seed, points = self.live.get(payload["variant"], (payload["seed"], []))
            points = [] if seed != payload["seed"] else points
            self.live[payload["variant"]] = (payload["seed"], (points + [(payload["step"], payload["bpc"], payload.get("trained_tokens"), payload.get("elapsed_seconds"))])[-400:])
            self.status.configure(text=f"Running {letter}, seed {payload['seed']}: step {payload['step']} of {payload.get('total')}.")
            self.draw_curves()
        elif payload.get("type") == "compare" and payload.get("phase") == "sweep":
            self.status.configure(text=f"Short learning-rate sweep for {letter}: step {payload['step']} of {payload.get('total')}.")
        elif payload.get("type") == "compare_done":
            self.results = payload["results"]
            self.status.configure(text=self.finished_text())
            self.draw_table(self.retune.value)
            self.draw_curves()

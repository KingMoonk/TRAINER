"""Compare runs: change one thing, hold the rest equal, run several seeds, read the spread."""
from __future__ import annotations

import tkinter as tk
import uuid
from dataclasses import asdict

import studio_experiments as ex
from studio_ia.copy import COMPARE as T
from studio_ia.pages.base import Page
from studio_ia.services import Job
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, GuessChips, Kicker, Segmented, Toggle, button
from studio_spec import default_spec, validate_spec
from studio_versions import change_text

MODE_KEYS = (("params", "params_mode", "params_note", "x_axis_text"), ("compute", "compute_mode", "compute_note", "x_axis_compute"),
             ("tokens", "tokens_mode", "tokens_note", "x_axis_text"), ("wall-clock", "time_mode", "time_note", "x_axis_time"))
SEEDS = ("1", "3", "5")
COLORS = ("text", "violet", "amber", "blue", "pink", "sage")
STEPS, WALL_SECONDS = 300, 120.  # ponytail: fixed length per run; make it a choice when the board has one


def million(n):
    return f"{n / 1e6:.2f}M"


class ComparePage(Page):
    title = T["h1"]
    lead = T["lead"]
    crumb = ("Network designer",)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.variants = []  # [{"name", "spec"}]
        self.mode, self.job_id, self.live, self.results, self.error = "params", None, {}, None, ""
        b = self.body
        setup = Card(b)
        setup.pack(fill="x", pady=(0, GUTTER))
        Kicker(setup, T["question"], bg="panel").pack(anchor="w")
        self.question = tk.Entry(setup, bg=C["panel2"], fg=C["text"], insertbackground=C["text"], relief="flat", font=font(15))
        self.question.pack(fill="x", pady=(4, GUTTER), ipady=6)
        Kicker(setup, T["hold_equal"], bg="panel").pack(anchor="w")
        self.modes = Segmented(setup, [T[m[1]] for m in MODE_KEYS], T["params_mode"], self.pick_mode)
        self.modes.pack(anchor="w", pady=(4, 8))
        self.mode_note = tk_label(setup, "", 13, "muted", "panel", wraplength=720)
        self.mode_note.pack(anchor="w", pady=(0, GUTTER))
        row = tk.Frame(setup, bg=C["panel"])
        row.pack(anchor="w")
        tk_label(row, T["seeds"], 14, "text2", "panel").pack(side="left", padx=(0, GUTTER))
        self.seeds = Segmented(row, SEEDS, "3", lambda _v: self.update_plan())
        self.seeds.pack(side="left")
        self.retune = Toggle(setup, T["retune_lr"], True, lambda _v: self.update_plan())
        self.retune.pack(anchor="w", pady=(GUTTER, 4))
        self.seeds_note = tk_label(setup, "", 13, "muted", "panel", wraplength=720)
        self.seeds_note.pack(anchor="w")
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

    def experiment(self):
        names = ["Base"] + [v["name"] for v in self.variants]
        return ex.ExperimentSpec(self.base, [v["spec"] for v in self.variants], seeds=int(self.seeds.value), mode=self.mode,
                                 steps=STEPS, wall_seconds=WALL_SECONDS, question=self.question.get().strip(),
                                 lr_retune=self.retune.value, guess=self.guess.value or "", names=names)

    def add_variant(self, name=None, spec=None):
        spec = validate_spec(spec)
        self.variants.append({"name": name or change_text(self.base, spec) or f"Variant {len(self.variants) + 1}", "spec": spec})
        self.update_plan()

    def add_from_designer(self):
        draft = getattr(self.services, "draft_spec", None)
        draft = draft() if callable(draft) else None
        if draft and change_text(self.base, draft) != "no change":
            self.add_variant(spec=draft)
        else:
            self.navigate("builder")

    def pick_mode(self, label):
        self.mode = next(m[0] for m in MODE_KEYS if T[m[1]] == label)
        self.update_plan()

    def label_for(self, row, plan, lr_on):
        if not row["variant"]:
            return "—"
        base, mode = plan["variants"][0], self.mode
        if mode == "params":
            text = (T["width_resolved"] if row["adjustments"] else T["none_needed"]).replace("[N]M", million(row["parameters"]))
        elif mode == "compute":
            more = row["steps"] / base["steps"] - 1
            text = T["stops_at_base"].replace("[N]%", f"{more:.0%}") if more > .005 else T["same_compute"]
        elif mode == "tokens":
            text = T["parameters"].replace("[N]M", million(row["parameters"]))
        else:
            text = T["min_budget"].replace("[N]", f"{WALL_SECONDS / 60:g}")
        return text + ("" if lr_on else " · learning rate not retuned")

    # drawing ----------------------------------------------------------------
    def update_plan(self):
        _key, _mode, note, axis = next(m for m in MODE_KEYS if m[0] == self.mode)
        lr_on = self.retune.value
        self.mode_note.configure(text=T[note])
        self.seeds_note.configure(text=T["seeds_note"].format(lr_note=T["lr_note_on" if lr_on else "lr_note_off"]))
        self.curves_title.configure(text=T["curves_title"].format(xAxis=T[axis]).upper())
        letter = chr(ord("B") + len(self.variants) - 1) if self.variants else "B"
        self.guess_label.configure(text=T["before_runs"].replace("for B:", f"for {letter}:"))
        try:
            self.plan = ex.plan(self.experiment())
            self.error = ""
        except ValueError as error:
            self.plan, self.error = None, str(error)
        self.problem.configure(text=self.error or next((f"Variant {r['variant']}: {r['reason']}" for r in self.plan["variants"] if not r["matched"]), ""))
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
        for i, row in enumerate(rows, 1):
            n, color = row["variant"], COLORS[row["variant"] % len(COLORS)]
            name = T["base_title"].replace("version 4", "this version") if not n else row["name"]
            diff = T["base_diff"] if not n else change_text(self.base, row["spec"])
            s = stats.get(n)
            final = T["error_spread"] if not s else (f"{s['mean']:.3f} · spread {s['spread']:.3f}" if len(s["bpc"]) > 1 else f"{s['mean']:.3f} · one seed")
            vs = "—" if not n else "[N]" if not s or s["delta"] is None else f"{s['delta']:+.3f}" + (" · inside the spread" if s["inside_spread"] else "")
            cells = (chr(ord("A") + n), name, diff, self.label_for(row, self.plan, lr_on), final, vs)
            for col, text in enumerate(cells):
                dim = col == 5 and s and s["inside_spread"]
                lbl = tk_label(self.table, text, 14 if col < 2 else 13, color if col == 0 else "dim" if dim else "text" if col < 2 else "text2", "panel",
                               wraplength=240 if col in (2, 3) else 0)
                lbl.grid(row=i, column=col, sticky="w", padx=(0, GUTTER), pady=4)
                if col == 1:
                    lbl.configure(cursor="hand2")
                    lbl.bind("<Button-1>", lambda _e: self.navigate("builder"))

    def draw_curves(self):
        for child in self.legend.winfo_children() + self.plots.winfo_children():
            child.destroy()
        curves = {v["variant"]: v["curve"] for v in (self.results or {}).get("variants", [])} or {k: v for k, v in self.live.items()}
        self.hint.pack_forget() if curves else self.hint.pack(anchor="w")
        refs = [(k, v) for k, v in ((self.results or {}).get("references") or {}).items()]
        names = {"uniform": "uniform guess", "unigram": "counting word frequencies"}
        for n, points in sorted(curves.items()):
            name = self.plan["variants"][n]["name"] if self.plan and n < len(self.plan["variants"]) else f"Variant {n}"
            tk_label(self.legend, f"{chr(ord('A') + n)} · {name}", 13, COLORS[n % len(COLORS)], "panel").pack(side="left", padx=(0, GUTTER))
            plot = BpcPlot(self.plots, 640, 140, 14)
            plot.pack(anchor="w", pady=4)
            plot.set(smooth=[tuple(p) for p in points], refs=[(names[k], v) for k, v in refs])

    # queue ------------------------------------------------------------------
    def queue_it(self):
        if not self.guess.value:
            self.status.configure(text="Pick your guess first.")
            return
        if self.plan is None or len(self.variants) < 1:
            self.status.configure(text="Add a variant first." if self.plan is not None else self.error)
            return
        exp = self.experiment()
        label = f"Compare runs: {self.plan['n_runs']} runs" + (f" + {self.plan['n_sweeps']} short sweeps" if self.plan["n_sweeps"] else "")
        self.job_id = uuid.uuid4().hex[:8]
        self.live, self.results = {}, None
        self.services.enqueue(Job(id=self.job_id, kind="compare", label=label, note=exp.question,
                                  params={"experiment": asdict(exp), "dataset_kind": "text", "guess": self.guess.value}))
        self.status.configure(text=f"Queued: {label}.")
        self.draw_table(self.retune.value)

    def on_event(self, kind, payload):
        if kind != "train" or not isinstance(payload, dict) or payload.get("job") != self.job_id:
            return
        if payload.get("type") == "compare" and payload.get("phase") == "run":
            self.live.setdefault(payload["variant"], []).append((payload["step"], payload["bpc"]))
            self.live[payload["variant"]] = self.live[payload["variant"]][-400:]
            self.draw_curves()
        elif payload.get("type") == "compare_done":
            self.results = payload["results"]
            self.draw_table(self.retune.value)
            self.draw_curves()

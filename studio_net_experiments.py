"""Paired-seed network comparisons using the experiment service."""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from studio_arch import model_spec_presets
from studio_core import uid
from studio_experiments import ExperimentSpec, run, seed_stats
from studio_net_common import NetworkView
from studio_planning import build_training_set
from studio_widgets import label


class ExperimentsView(NetworkView):
    title = "Experiments"

    def __init__(self, parent, host):
        super().__init__(parent, host)
        self.stop = None
        line = ttk.Frame(self)
        line.pack(fill="x")
        self.variant = tk.StringVar()
        self.mode = tk.StringVar(value="tokens")
        self.seeds = tk.StringVar(value="7,42,123")
        self.budget = tk.StringVar(value="10000")
        self.steps = tk.StringVar(value="100")
        self.runs_by_key = {}
        for title, var, values, width in (("Variant", self.variant, (), 24), ("Hold equal", self.mode, ("tokens", "params", "compute", "wall-clock"), 12),
                                          ("Seeds", self.seeds, (), 12), ("Budget", self.budget, (), 10), ("Steps", self.steps, (), 8)):
            group = ttk.Frame(line)
            group.pack(side="left", padx=(0, 8))
            label(group, title).pack(anchor="w")
            (ttk.Combobox(group, textvariable=var, values=values, state="readonly", width=width) if values else
             ttk.Entry(group, textvariable=var, width=width)).pack()
        ttk.Button(line, text="Run comparison", command=self.start).pack(side="left", padx=8)
        ttk.Button(line, text="Stop", command=self.cancel).pack(side="left")
        self.chart = tk.Canvas(self, height=150, background="#0c1626", highlightthickness=0)
        self.chart.pack(fill="x", pady=(12, 0))
        self.chart.bind("<Configure>", lambda _event: self.draw_chart())
        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Load run design", command=self.load_selected_design).pack(side="left")
        ttk.Button(actions, text="Open checkpoint in Playground", command=self.open_selected_checkpoint).pack(side="left", padx=8)
        self.table = ttk.Treeview(self, columns=("variant", "seed", "step", "loss", "speed"), show="headings", height=8)
        for key, title in (("variant", "VARIANT"), ("seed", "SEED"), ("step", "STEP"), ("loss", "LOSS"), ("speed", "TOKENS/S")):
            self.table.heading(key, text=title)
            self.table.column(key, width=110)
        self.table.pack(fill="both", expand=True, pady=(12, 0))
        self.table.bind("<Double-1>", lambda _event: self.load_selected_design())

    def refresh(self):
        names = list(model_spec_presets())
        if self.variant.get() not in names:
            self.variant.set(names[0])
        self.message.set("Each seed trains the base and variant on the same data. Results report achieved budgets.")

    def cancel(self):
        if self.stop is not None:
            self.stop.set()

    def start(self):
        if self.stop is not None and not self.stop.is_set():
            self.message.set("A comparison is already running.")
            return
        try:
            seeds = [int(value.strip()) for value in self.seeds.get().split(",")]
            budget = float(self.budget.get())
            mode = self.mode.get()
            variant = copy.deepcopy(model_spec_presets()[self.variant.get()])
            experiment = ExperimentSpec(base=copy.deepcopy(self.spec), variants=[variant], seeds=seeds, mode=mode,
                                        token_budget=max(1, int(budget)), wall_seconds=budget, steps=int(self.steps.get())).normalized()
            ids = [row["id"] for row in self.host.workspace.state["datasets"]]
            if not ids:
                raise ValueError("Add a dataset before running a comparison.")
        except (ValueError, KeyError) as error:
            self.message.set(str(error))
            return
        self.table.delete(*self.table.get_children())
        self.runs_by_key.clear()
        self.draw_chart([])
        self.stop = threading.Event()
        output = self.host.workspace.run_dir / uid("experiment")
        self.message.set("Preparing shared training data…")
        def worker():
            try:
                examples = build_training_set(self.host.workspace, ids, max_records=2000, seed=seeds[0])["examples"]
                def emit(event):
                    self.host.net_view_events.put((self, self.event, (True, event)))
                results = run(experiment, examples, output, emit=emit, stop=self.stop)
                self.host.net_view_events.put((self, self.complete, (True, results)))
            except Exception as error:
                self.host.net_view_events.put((self, self.complete, (False, str(error))))
        threading.Thread(target=worker, daemon=True, name="network-experiment").start()

    def event(self, ok, event):
        if event.get("type") == "metric":
            key = f"{event['variant']}:{event['seed']}"
            values = (event["variant"], event["seed"], event["step"], f"{event['loss']:.4f}",
                      f"{event.get('tokens_per_second', 0):.0f}")
            if self.table.exists(key):
                self.table.item(key, values=values)
            else:
                self.table.insert("", "end", iid=key, values=values)
            self.table.see(key)
        elif event.get("type") == "experiment_run":
            self.message.set(f"Variant {event['variant']} · seed {event['seed']} · adjustment {event['adjustments'] or 'none'}")

    def complete(self, ok, result):
        self.stop = None
        if not ok:
            self.message.set(str(result))
            return
        groups = {}
        for row in result:
            self.runs_by_key[f"{row['variant']}:{row['seed']}"] = row
            loss = row["record"].get("val_loss")
            if loss is not None:
                groups.setdefault(row["variant"], []).append(loss)
        self.draw_chart(result)
        self.message.set(" · ".join(f"variant {key}: mean validation loss {seed_stats(values)['mean']:.4f}, "
                                    f"range {seed_stats(values)['min']:.4f}–{seed_stats(values)['max']:.4f}"
                                    for key, values in groups.items()) or "No validation losses were available.")

    def draw_chart(self, results=None):
        if results is not None:
            self.chart_results = results
        results = getattr(self, "chart_results", [])
        canvas = self.chart
        canvas.delete("all")
        width, height = max(240, canvas.winfo_width()), max(100, canvas.winfo_height())
        vals = [(row["variant"], row["record"].get("val_loss")) for row in results
                if row["record"].get("val_loss") is not None]
        if not vals:
            canvas.create_text(width / 2, height / 2, text="Validation loss by variant · min–max across seeds",
                               fill="#9aa9bd", font=("TkDefaultFont", 10))
            return
        groups = {}
        for variant, value in vals:
            groups.setdefault(variant, []).append(float(value))
        low, high = min(min(v) for v in groups.values()), max(max(v) for v in groups.values())
        span = max(high - low, 1e-9)
        left, right, top, bottom = 48, width - 24, 22, height - 34
        canvas.create_line(left, top, left, bottom, fill="#40516b")
        canvas.create_line(left, bottom, right, bottom, fill="#40516b")
        for i, (variant, values) in enumerate(sorted(groups.items())):
            x = left + (i + .5) * (right - left) / len(groups)
            y = lambda value: bottom - (value - low) / span * (bottom - top)
            mean = sum(values) / len(values)
            canvas.create_line(x, y(min(values)), x, y(max(values)), fill="#62d6b5", width=3)
            canvas.create_line(x - 7, y(min(values)), x + 7, y(min(values)), fill="#62d6b5", width=2)
            canvas.create_line(x - 7, y(max(values)), x + 7, y(max(values)), fill="#62d6b5", width=2)
            canvas.create_oval(x - 5, y(mean) - 5, x + 5, y(mean) + 5, fill="#f4c76b", outline="")
            canvas.create_text(x, bottom + 15, text=f"variant {variant} · n={len(values)}", fill="#c4cfdf")
        canvas.create_text(left, 10, anchor="w", text=f"Validation loss {low:.4g}–{high:.4g} · dot = mean, bar = seed range",
                           fill="#9aa9bd", font=("TkDefaultFont", 9))

    def _selected_run(self):
        selected = self.table.selection()
        if not selected or selected[0] not in self.runs_by_key:
            self.message.set("Select a completed experiment seed first.")
            return None
        return self.runs_by_key[selected[0]]

    def load_selected_design(self):
        row = self._selected_run()
        if not row:
            return
        try:
            spec = json.loads((Path(row["directory"]) / "spec.json").read_text(encoding="utf-8"))
            self.host.net_apply_model_edit(lambda draft: (draft.clear(), draft.update(copy.deepcopy(spec))),
                                           f"Load experiment variant {row['variant']} seed {row['seed']}")
            self.host.net_show_view("Design")
            self.message.set(f"Loaded experiment variant {row['variant']} seed {row['seed']} into Design.")
        except (OSError, ValueError, TypeError) as error:
            self.message.set(f"Could not load run design: {error}")

    def open_selected_checkpoint(self):
        row = self._selected_run()
        if not row:
            return
        directory = Path(row["directory"])
        checkpoint = next((directory / name for name in ("best.npz", "last.npz") if (directory / name).is_file()), None)
        if checkpoint is None:
            self.message.set("This run has no saved checkpoint to open.")
            return
        self.host.chat_checkpoint.set(str(checkpoint))
        self.host.show_page("chat")
        self.host.load_chat_model()

"""Paired-seed network comparisons using the experiment service."""
from __future__ import annotations

import copy
import threading
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
        for title, var, values, width in (("Variant", self.variant, (), 24), ("Hold equal", self.mode, ("tokens", "params", "compute", "wall-clock"), 12),
                                          ("Seeds", self.seeds, (), 12), ("Budget", self.budget, (), 10), ("Steps", self.steps, (), 8)):
            group = ttk.Frame(line)
            group.pack(side="left", padx=(0, 8))
            label(group, title).pack(anchor="w")
            (ttk.Combobox(group, textvariable=var, values=values, state="readonly", width=width) if values else
             ttk.Entry(group, textvariable=var, width=width)).pack()
        ttk.Button(line, text="Run comparison", command=self.start).pack(side="left", padx=8)
        ttk.Button(line, text="Stop", command=self.cancel).pack(side="left")
        self.table = ttk.Treeview(self, columns=("variant", "seed", "step", "loss", "speed"), show="headings", height=12)
        for key, title in (("variant", "VARIANT"), ("seed", "SEED"), ("step", "STEP"), ("loss", "LOSS"), ("speed", "TOKENS/S")):
            self.table.heading(key, text=title)
            self.table.column(key, width=110)
        self.table.pack(fill="both", expand=True, pady=(12, 0))

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
            self.table.insert("", "end", values=(event["variant"], event["seed"], event["step"],
                                                   f"{event['loss']:.4f}", f"{event.get('tokens_per_second', 0):.0f}"))
            children = self.table.get_children()
            self.table.see(children[-1])
        elif event.get("type") == "experiment_run":
            self.message.set(f"Variant {event['variant']} · seed {event['seed']} · adjustment {event['adjustments'] or 'none'}")

    def complete(self, ok, result):
        self.stop = None
        if not ok:
            self.message.set(str(result))
            return
        groups = {}
        for row in result:
            loss = row["record"].get("val_loss")
            if loss is not None:
                groups.setdefault(row["variant"], []).append(loss)
        self.message.set(" · ".join(f"variant {key}: mean validation loss {seed_stats(values)['mean']:.4f}, "
                                    f"range {seed_stats(values)['min']:.4f}–{seed_stats(values)['max']:.4f}"
                                    for key, values in groups.items()) or "No validation losses were available.")

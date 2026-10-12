"""Training-page card: "Train on everything" - sequential (curriculum) training across all datasets."""
from __future__ import annotations

import copy
import queue
import tkinter as tk
from tkinter import messagebox, ttk

from studio_curriculum import (DEFAULT_CONTEXT, DEFAULT_REPLAY, default_plan, latest_plan, resume_plan, run_plan, validate_plan)
from studio_widgets import BG, INPUT, TEXT, Tooltip, button, card, help_text, label, tree

CONTEXTS = ("128", "256", "512", "1024")
STATUS_TEXT = {"pending": "waiting", "running": "training…", "evaluating": "checking…", "completed": "done",
               "stopped": "stopped", "failed": "failed", "skipped": "skipped"}
INTRO = ("Sequential training teaches one model in steps, like school: plain language first, then short instructions, code, reasoning, "
         "conversations and finally long reading. Each step starts from what the last one learned, and after every step the model is re-tested "
         "on all earlier material so you can see what it forgot.")


def _num(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


class StageDialog(tk.Toplevel):
    """Edit one stage: its datasets, steps, record budget and learning rate."""

    def __init__(self, parent, stage, datasets):
        super().__init__(parent)
        self.title("Edit stage")
        self.configure(bg=BG)
        self.transient(parent)
        self.result = None
        self.datasets = datasets
        body = ttk.Frame(self, padding=18)
        body.pack(fill="both", expand=True)
        label(body, "Stage name").grid(row=0, column=0, sticky="w")
        self.name = tk.StringVar(value=stage["name"])
        ttk.Entry(body, textvariable=self.name, width=34).grid(row=0, column=1, sticky="we", pady=4)
        label(body, "Datasets").grid(row=1, column=0, sticky="nw", pady=4)
        self.listbox = tk.Listbox(body, selectmode="extended", exportselection=False, height=8, width=44, bg=INPUT, fg=TEXT,
                                  selectbackground="#355473", relief="flat", highlightthickness=0)
        self.listbox.grid(row=1, column=1, sticky="we", pady=4)
        for index, (key, name) in enumerate(datasets):
            self.listbox.insert("end", name)
            if key in stage["dataset_ids"]:
                self.listbox.selection_set(index)
        self.vars = {}
        for row, (key, title) in enumerate((("steps", "Steps"), ("max_records", "Records"), ("learning_rate", "Learning rate")), 2):
            label(body, title).grid(row=row, column=0, sticky="w")
            self.vars[key] = tk.StringVar(value=str(stage[key]))
            ttk.Entry(body, textvariable=self.vars[key], width=14).grid(row=row, column=1, sticky="w", pady=4)
        footer = ttk.Frame(body)
        footer.grid(row=5, column=0, columnspan=2, sticky="e", pady=(12, 0))
        button(footer, "Cancel", self.destroy)
        button(footer, "Save stage", self.save, True)
        body.columnconfigure(1, weight=1)
        self.bind("<Escape>", lambda _e: self.destroy())

    def save(self):
        try:
            ids = [self.datasets[i][0] for i in self.listbox.curselection()]
            if not ids:
                raise ValueError("Select at least one dataset.")
            steps, records, rate = int(self.vars["steps"].get()), int(self.vars["max_records"].get()), float(self.vars["learning_rate"].get())
            if steps < 1 or records < 1 or not 1e-7 <= rate <= .1:
                raise ValueError("Steps and records must be positive and the learning rate between 1e-7 and 0.1.")
        except ValueError as exc:
            messagebox.showerror("Check the stage", str(exc), parent=self)
            return
        self.result = {"name": self.name.get().strip() or "Stage", "dataset_ids": ids, "steps": steps, "max_records": records, "learning_rate": rate}
        self.destroy()


class CurriculumMixin:
    def _build_training(self):
        super()._build_training()
        self.curriculum = {"stages": [], "context": DEFAULT_CONTEXT}
        self.curriculum_base = None
        self.curriculum_register = None
        self._build_curriculum_card()
        self.refresh_curriculum()

    # ------------------------------------------------------------------ card

    def _build_curriculum_card(self):
        page = self.pages["training"].body
        box = card(page)
        box.pack(fill="x", pady=(0, 8), before=self.reading_drawer)
        self.curriculum_box = box
        self.curriculum_open = False
        self.curriculum_toggle = ttk.Button(box, text="▸  Train on everything: stage-by-stage training", command=self.toggle_curriculum)
        self.curriculum_toggle.pack(anchor="w")
        Tooltip(self.curriculum_toggle, help_text("curriculum"))
        body = self.curriculum_body = ttk.Frame(box, style="Card.TFrame")
        label(body, INTRO, "CardMuted.TLabel", wraplength=860, justify="left").pack(anchor="w", pady=(0, 8))
        options = ttk.Frame(body, style="Card.TFrame")
        options.pack(fill="x", pady=(0, 6))
        self.curriculum_use_net = tk.BooleanVar(value=False)
        self.curriculum_replay = tk.BooleanVar(value=True)
        self.curriculum_context = tk.StringVar(value=str(DEFAULT_CONTEXT))
        ttk.Checkbutton(options, text="Use Network architecture", variable=self.curriculum_use_net).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(options, text="Replay earlier data", variable=self.curriculum_replay).pack(side="left", padx=(0, 18))
        label(options, "Context", "CardMuted.TLabel").pack(side="left", padx=(0, 6))
        ttk.Combobox(options, textvariable=self.curriculum_context, values=CONTEXTS, state="readonly", width=6).pack(side="left")
        label(options, "(fixed for every stage)", "CardMuted.TLabel").pack(side="left", padx=(8, 0))
        columns = [("n", "#", 36), ("stage", "STAGE", 170), ("datasets", "DATASETS", 260), ("steps", "STEPS", 60),
                   ("records", "RECORDS", 80), ("status", "STATUS", 90), ("loss", "LOSS", 70), ("eval", "HELD-OUT", 80)]
        self.curriculum_tree = tree(body, columns, 4)
        self.curriculum_tree.master.pack_configure(fill="x", expand=False)
        controls = ttk.Frame(body, style="Card.TFrame")
        controls.pack(fill="x", pady=(8, 0))
        button(controls, "Build default plan", self.build_curriculum_plan)
        button(controls, "Edit stage", self.edit_curriculum_stage)
        button(controls, "Move up", lambda: self.move_curriculum_stage(-1))
        button(controls, "Move down", lambda: self.move_curriculum_stage(1))
        button(controls, "Remove", self.remove_curriculum_stage)
        button(controls, "Start", self.start_curriculum, True)
        button(controls, "Stop", self.stop_job)
        button(controls, "Resume last plan", self.resume_curriculum)
        self.curriculum_note = tk.StringVar(value="Press Build default plan to line up your datasets.")
        label(body, textvariable=self.curriculum_note, style="CardMuted.TLabel", wraplength=860, justify="left").pack(anchor="w", pady=(6, 0))
        self.forget_frame = ttk.Frame(body, style="Card.TFrame")
        label(self.forget_frame, "Did it forget? Held-out loss on each earlier stage, after learning it and now (lower is better).",
              "CardMuted.TLabel").pack(anchor="w", pady=(6, 4))
        self.forget_tree = tree(self.forget_frame, [("stage", "STAGE", 200), ("learned", "WHEN LEARNED", 110), ("latest", "NOW", 90),
                                                    ("change", "CHANGE", 90), ("verdict", "VERDICT", 200)], 3)
        self.forget_tree.master.pack_configure(fill="x", expand=False)

    def toggle_curriculum(self, open_=None):
        self.curriculum_open = (not self.curriculum_open) if open_ is None else open_
        self.curriculum_toggle.configure(text=("▾" if self.curriculum_open else "▸") + "  Train on everything: stage-by-stage training")
        if self.curriculum_open:
            self.curriculum_body.pack(fill="x", pady=(10, 0))
        else:
            self.curriculum_body.pack_forget()

    # ------------------------------------------------------------------ table

    def dataset_choices(self):
        return [(d["id"], d["name"]) for d in self.workspace.state["datasets"]]

    def refresh_curriculum(self):
        names = dict(self.dataset_choices())
        selected = self.curriculum_tree.selection()
        self.curriculum_tree.delete(*self.curriculum_tree.get_children())
        evals = {row["after"]: row["evals"] for row in self.curriculum.get("eval_table", [])}
        for index, stage in enumerate(self.curriculum["stages"]):
            own = (evals.get(index, {}).get(str(index)) or {}).get("loss")
            loss = stage.get("val_loss") if stage.get("val_loss") is not None else stage.get("loss")
            ds = ", ".join(names.get(key, key) for key in stage["dataset_ids"])
            self.curriculum_tree.insert("", "end", iid=str(index), values=(
                index + 1, stage["name"], ds, stage["steps"], f"{stage['max_records']:,}",
                STATUS_TEXT.get(stage.get("status", "pending"), stage.get("status")), _num(loss), _num(own)))
        keep = [s for s in selected if self.curriculum_tree.exists(s)]
        if keep:
            self.curriculum_tree.selection_set(keep)
        rows = self.curriculum.get("forgetting") or []
        self.forget_tree.delete(*self.forget_tree.get_children())
        if rows and any(r["after_stage"] > r["stage"] for r in rows):
            for r in rows:
                if r["after_stage"] == r["stage"]:
                    verdict = "just learned"
                else:
                    verdict = "kept it" if r["delta"] < .1 else "forgot a little" if r["delta"] < .5 else "forgot a lot"
                self.forget_tree.insert("", "end", values=(f"{r['stage'] + 1}. {r['name']}", _num(r["learned_loss"]), _num(r["latest_loss"]),
                                                           f"{r['delta']:+.3f}", verdict))
            self.forget_frame.pack(fill="x")
        else:
            self.forget_frame.pack_forget()

    def _selected_stage(self):
        selection = self.curriculum_tree.selection()
        return int(selection[0]) if selection else None

    def _editable(self):
        if self.job_kind:
            self.status_var.set("Stop the running job before editing the plan.")
            return False
        return True

    def build_curriculum_plan(self):
        if not self._editable():
            return
        try:
            base, context, register = self.curriculum_settings()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Cannot build the plan", str(exc), parent=self)
            return
        plan = default_plan(self.workspace, base, context=context, replay=DEFAULT_REPLAY if self.curriculum_replay.get() else 0.0)
        self.curriculum, self.curriculum_base, self.curriculum_register = plan, base, register
        self.refresh_curriculum()
        note = f"{len(plan['stages'])} stages · byte tokenizer · context {context} (same in every stage)."
        if plan["warnings"]:
            note += "  " + "  ".join(plan["warnings"])
        if not plan["stages"]:
            note = "No datasets with records yet. Download or import some on the Catalog and Datasets pages first."
        self.curriculum_note.set(note)

    def curriculum_settings(self):
        """(base config, context, architecture id) from the Blueprint form or, if asked, the Network page."""
        base = dict(self.read_config())
        base.update(tokenizer="byte", architecture=None)
        context, register = int(self.curriculum_context.get()), None
        if self.curriculum_use_net.get():
            self.net_ensure()
            self.net_flush()
            if self.net_error:
                raise ValueError("Fix the Network circuit first: " + self.net_error)
            from studio_arch import config_from_spec, validate_spec
            spec = validate_spec(self.net_spec)
            context = int(spec["embedding"]["context"])
            self.curriculum_context.set(str(context))
            base = dict(config_from_spec(spec), device=base["device"], tokenizer="byte")
            register = self.arch_id
        base["context"] = context
        return base, context, register

    def edit_curriculum_stage(self):
        index = self._selected_stage()
        if index is None or not self._editable():
            self.status_var.set("Select a stage to edit.")
            return
        dialog = StageDialog(self, self.curriculum["stages"][index], self.dataset_choices())
        self.wait_window(dialog)
        if dialog.result:
            self.curriculum["stages"][index].update(dialog.result)
            self.refresh_curriculum()

    def move_curriculum_stage(self, step):
        index = self._selected_stage()
        if index is None or not self._editable():
            return
        stages, target = self.curriculum["stages"], index + step
        if 0 <= target < len(stages):
            stages[index], stages[target] = stages[target], stages[index]
            self.refresh_curriculum()
            self.curriculum_tree.selection_set(str(target))

    def remove_curriculum_stage(self):
        index = self._selected_stage()
        if index is None or not self._editable():
            return
        del self.curriculum["stages"][index]
        self.refresh_curriculum()

    # ------------------------------------------------------------------ running

    def _curriculum_emit(self, event):
        self.events.put(("curriculum" if event.get("type") == "curriculum" else "training", event))

    def start_curriculum(self):
        if self.busy():
            return
        if not self.curriculum["stages"]:
            self.build_curriculum_plan()
        try:
            base, context, register = self.curriculum_settings() if self.curriculum_base is None else (self.curriculum_base, self.curriculum["context"], self.curriculum_register)
            plan = copy.deepcopy({k: v for k, v in self.curriculum.items() if k in {"name", "stages", "warnings"}})
            plan.update(context=context, replay=DEFAULT_REPLAY if self.curriculum_replay.get() else 0.0)
            for stage in plan["stages"]:
                stage["context"] = context
            validate_plan(self.workspace, plan, base)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Cannot start", str(exc), parent=self)
            return
        self.curriculum_base = base
        self.history = []
        self.draw_loss()
        self.toggle_curriculum(True)
        self.curriculum_note.set("Starting… preparing the first stage.")
        self.start_job("curriculum", lambda: run_plan(self.workspace, plan, base, self._curriculum_emit, self.stop, self.pause,
                                                     register_arch_id=register), self.curriculum_finished)

    def resume_curriculum(self):
        if self.busy():
            return
        saved = latest_plan(self.workspace)
        if saved is None:
            messagebox.showinfo("Nothing to resume", "There is no stopped or unfinished plan yet.", parent=self)
            return
        self.curriculum = {k: v for k, v in saved.items() if k != "base_config"}
        self.curriculum_base = saved.get("base_config")
        self.refresh_curriculum()
        self.curriculum_note.set(f"Resuming {saved['name']} from its first unfinished stage.")
        self.start_job("curriculum", lambda: resume_plan(self.workspace, saved["id"], self._curriculum_emit, self.stop, self.pause),
                       self.curriculum_finished)

    def curriculum_finished(self, state):
        if isinstance(state, dict):
            self.curriculum = {k: v for k, v in state.items() if k != "base_config"}
            self.refresh_curriculum()
            self.curriculum_note.set({"completed": "All stages finished. The table below shows what each stage remembers.",
                                      "stopped": "Stopped. Press Resume last plan to continue where it left off."}.get(state["status"], state["status"]))
        self.refresh_runs()

    def curriculum_event(self, event):
        state = event.get("state")
        if state:
            self.curriculum = {k: v for k, v in state.items() if k != "base_config"}
        stage = event.get("stage")
        if stage is not None and event.get("status") == "running":
            row = self.curriculum["stages"][stage]
            self.history = []
            self.draw_loss()
            self.progress.configure(value=0, maximum=row["steps"])
            self.curriculum_note.set(f"Stage {stage + 1} of {len(self.curriculum['stages'])}: {row['name']}")
        elif stage is not None and event.get("status") == "evaluating":
            self.curriculum_note.set(f"Stage {stage + 1} trained · re-testing on earlier material…")
        self.refresh_curriculum()

    # ------------------------------------------------------------------ plumbing

    def toggle_pause(self):
        if self.job_kind == "curriculum":
            self.job_kind = "training"
            try:
                super().toggle_pause()
            finally:
                self.job_kind = "curriculum"
        else:
            super().toggle_pause()

    def poll(self):
        deferred = []
        try:
            for _ in range(200):
                kind, value = self.events.get_nowait()
                if kind == "curriculum":
                    self.curriculum_event(value)
                else:
                    deferred.append((kind, value))
        except queue.Empty:
            pass
        for item in deferred:
            self.events.put(item)
        super().poll()

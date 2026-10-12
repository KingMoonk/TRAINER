"""Guided native workflow: acquire, curate, synthesize, blend, train and teach."""
from __future__ import annotations

import copy
import json
import math
import queue
import threading
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from studio_core import atomic_json, fingerprint, now, uid
from studio_model import DEFAULT_CONFIG, Tokenizer, parameter_estimate, train, validate_config as validate_model_config
import studio_catalog
from studio_catalog import CATALOG, download_dataset, list_downloads, register_download
from studio_synthesis import (DEFAULT_TEACHER_URL, RECIPE_PRESETS, TEACHER_MODES, apply_draft_edit, explain_reason,
                              generate_batch, summarize_batch, validate_config)
from studio_teacher_synth import list_teacher_models
from studio_planning import build_training_set
from studio_widgets import (BG, PANEL, INPUT, EDGE, TEXT, MUTED, ACCENT, BLUE, FONT,
                            label, button, text_box, content, set_text, card, field, tree, ExampleEditor,
                            Tooltip, help_icon, with_help, help_text)
from studio_align import Collapsible
from studio_ux import SIZE_PRESETS, device_kind, estimate_seconds, format_duration, step_seconds


QUICK_START_SECONDS = 120          # target length of the Quick start run
QUICK_START_RECORDS = 10000
DOWNLOAD_MBPS = 50                  # assumed connection speed for the catalog time estimate


def _first_number(text, unit=None):
    """First number in a catalog hint such as '64 MB download' or '143K instruction conversations'."""
    import re
    match = re.search(r"([\d][\d,]*\.?\d*)\s*(K|M)?\b" if unit is None else r"([\d][\d,]*\.?\d*)\s*(MB)", text or "")
    if not match:
        return 0.0
    value = float(match.group(1).replace(",", ""))
    if unit is None and match.group(2):
        value *= {"K": 1e3, "M": 1e6}[match.group(2)]
    return value


def catalog_summary(identifiers=None):
    """Totals for a set of catalog sources: count, download megabytes, examples and a rough time on a typical connection."""
    chooser = getattr(studio_catalog, "default_catalog_ids", None)
    wanted = set(identifiers if identifiers is not None else (chooser() if chooser else []))
    chosen = [source for source in CATALOG if source["id"] in wanted]
    megabytes = sum(_first_number(source.get("size_hint", ""), "MB") for source in chosen)
    records = sum(_first_number(source.get("records_hint", "")) for source in chosen)
    minutes = (megabytes * 8 / DOWNLOAD_MBPS + 25 * len(chosen)) / 60     # transfer time plus ~25 s per source to unpack and index
    return {"count": len(chosen), "megabytes": megabytes, "records": int(records), "minutes": minutes}


PANE_ORDER = ["templates", "practice", "teach", "grow", "remix", "model"]   # index 1 (practice) is the legacy "arithmetic" test hook
GOALS = [
    ("practice", "Practice problems", "Computed math, logic, code and date problems. Answers are calculated, so they are always correct."),
    ("teach", "Teach from a teacher model", "A local Ollama teacher writes new questions and answers, or short conversations, about topics you choose."),
    ("grow", "Grow my dataset", "Use the teacher on your working dataset: evolve harder prompts, reword prompts, or write fresh answers."),
    ("remix", "Remix downloaded data", "Reformat records you already have: final answers only, clean solutions, JSON answers, new phrasings."),
    ("model", "Use my own model", "Let a model trained in this studio draft answers to your prompts, then review them."),
    ("templates", "Templates", "Fill in a prompt and answer template from your own list of facts or values."),
]
TEACHER_GOALS = {"teach", "grow"}
LENGTH_LABELS = {"short": "Short · up to 220 characters", "medium": "Medium · up to 600 characters", "long": "Long · up to 1,500 characters"}
FILTER_LABELS = {"all": "All prompts", "review": "Only prompts not yet marked Good", "missing": "Only prompts without an answer"}
PRACTICE_KINDS = [("arithmetic", "Arithmetic"), ("word_problems", "Word problems"), ("logic", "Logic & comparisons"), ("list_ops", "List operations"),
                  ("string_ops", "Text transformations"), ("json_extract", "JSON extraction"), ("table_qa", "Table questions"),
                  ("code_trace", "Code tracing"), ("calendar", "Date arithmetic")]
REMIX_KINDS = [("final_answer", "Final answer only"), ("clean_solution", "Clean worked solution"), ("json_answer", "JSON-formatted answers"),
               ("multi_turn", "Multi-turn chains"), ("context_first", "Context-first reading")]


class ModeCard(tk.Frame):
    """Big radio-like button: title plus a one-sentence explanation."""
    def __init__(self, parent, key, title, blurb, command):
        super().__init__(parent, bg=PANEL, highlightthickness=2, highlightbackground=EDGE, highlightcolor=EDGE, cursor="hand2", padx=14, pady=12)
        self.key = key
        self.title_label = tk.Label(self, text=title, bg=PANEL, fg=TEXT, font=(FONT, 12, "bold"), anchor="w", cursor="hand2")
        self.blurb_label = tk.Label(self, text=blurb, bg=PANEL, fg=MUTED, font=(FONT, 10), anchor="w", justify="left", wraplength=280, cursor="hand2")
        self.title_label.pack(fill="x")
        self.blurb_label.pack(fill="x", pady=(6, 0))
        for widget in (self, self.title_label, self.blurb_label):
            widget.bind("<Button-1>", lambda _e, k=key: command(k))
        self.bind("<Configure>", lambda e: self.blurb_label.configure(wraplength=max(120, e.width - 40)))

    def set_selected(self, on):
        self.configure(highlightbackground=ACCENT if on else EDGE, highlightcolor=ACCENT if on else EDGE)
        self.title_label.configure(fg=ACCENT if on else TEXT)


class PaneSwitcher(ttk.Frame):
    """Notebook-compatible (select/index) container that shows one pane and sizes to it."""
    def __init__(self, parent, on_change=None):
        super().__init__(parent, style="Card.TFrame")
        self.panes, self.current, self.on_change = [], None, on_change

    def add(self, pane):
        self.panes.append(pane)

    def select(self, tab=None):
        if tab is None:
            return self.current
        index = tab if isinstance(tab, int) else self.panes.index(tab)
        for pane in self.panes:
            pane.pack_forget()
        self.panes[index].pack(fill="x")
        self.current = index
        if self.on_change:
            self.on_change(index)

    def index(self, tab):
        return tab if isinstance(tab, int) else self.panes.index(tab)


class WorkflowMixin:
    PAGES = [("home", "↗", "Start here", "Five steps, all on this computer. Follow the list; optional steps can be skipped."),
             ("catalog", "01", "Get data", "Download published training datasets. Their licenses and sources stay attached."),
             ("datasets", "02", "Your data", "Read your examples, fix mistakes, and mark the good ones."),
             ("synthesis", "03", "Make extra data", "Optional: generate more examples. Try 5 samples first, then add the good ones to your data."),
             ("blueprint", "04", "Training setup", "Choose your data and a model size, then start training."),
             ("network", "05", "Network designer", "Advanced: snap together the model's blocks, watch signals flow, and version every change."),
             ("training", "06", "Training", "Watch the model learn. Lower error is better."),
             ("align", "07", "Improve with feedback", "Optional: tell the model which replies are better (or let a teacher do it) and polish it."),
             ("chat", "08", "Chat & teach", "Talk to your model and save corrections that make it better next time.")]

    def __init__(self, workspace=None):
        self.record_page = 0
        self.record_page_size = 100
        self.draft_page = 0
        self.draft_page_size = 100
        self.search_after = None
        self._stats_cache = {}
        self.last_generation = None
        super().__init__(workspace)
        draft_file = self.workspace.root / "data" / "drafts.json"
        if draft_file.exists():
            try:
                saved = json.loads(draft_file.read_text(encoding="utf-8"))
                if isinstance(saved, list):
                    self.drafts = saved
            except (OSError, ValueError):
                self.status_var.set("Draft file could not be read; existing workspace drafts were kept.")
        self.refresh_drafts()
        self.show_page("home")
        self.after(300, self.index_local_downloads)

    def index_local_downloads(self):
        """Register completed downloads that are not yet in this workspace's library."""
        removed_file = self.workspace.root / "data" / "catalog_removed.json"
        try:
            removed = set(json.loads(removed_file.read_text(encoding="utf-8"))) if removed_file.exists() else set()
        except (OSError, ValueError):
            removed = set()
        known = {d["id"] for d in self.workspace.state["datasets"]}
        pending = [r for r in list_downloads(self.workspace.root)
                   if f"catalog_{r['catalog_id']}" not in known | removed]
        if not pending or self.busy():
            return
        def work():
            done = []
            for result in pending:
                if self.stop.is_set():
                    break
                done.append(register_download(self.workspace, result, stop=self.stop,
                                              emit=lambda event: self.events.put(("catalog_progress", event))))
                self.events.put(("catalog_registered", done[-1]))
            return done
        def finished(identifiers):
            self.refresh_datasets(identifiers[-1] if identifiers else None)
            self.catalog_status.set(f"{len(identifiers)} downloaded collection(s) indexed and ready.")
        self.start_job("library indexing", work, finished)

    def _layout(self):
        super()._layout()
        self.active_page = "home"
        # A persistent footer makes background tasks and the next action visible on every page.
        parent = self.page_host.master
        self.flow_bar = ttk.Frame(parent)
        self.flow_bar.pack(fill="x", before=self.page_host, pady=(0, 14))
        with_help(self.flow_bar, "WORKING DATASET", "working_dataset", "Muted.TLabel", font=(FONT, 9, "bold")).pack(side="left", padx=(0, 12))
        self.dataset_picker(self.flow_bar)
        self.next_action = ttk.Button(self.flow_bar, text="Discover data →", command=self.next_step, style="Accent.TButton")
        self.next_action.pack(side="right")
        self.task_progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.task_progress.pack(side="bottom", fill="x", before=self.page_host, pady=(6, 0))
        self._build_home()
        self._build_catalog()

    NEXT_LABELS = {"home": "", "catalog": "Review your data \u2192", "datasets": "Make extra data \u2192", "synthesis": "Set up training \u2192",
                   "blueprint": "Design the network \u2192", "network": "View training \u2192", "training": "Improve with feedback \u2192",
                   "align": "Chat with your model \u2192", "chat": "Add more data \u2192"}

    def next_page_of(self, name):
        mapping = {"home": "catalog", "catalog": "datasets", "datasets": "synthesis", "synthesis": "blueprint",
                   "blueprint": "network" if self.is_advanced() else "training", "network": "training", "training": "align", "align": "chat", "chat": "datasets"}
        return mapping[name]

    def refresh_next_label(self):
        name = getattr(self, "active_page", "home")
        if name == "home":
            self.next_action.pack_forget()      # the home page has its own single Start here path
            return
        if not self.next_action.winfo_manager():
            self.next_action.pack(side="right")
        text = self.NEXT_LABELS[name]
        if name == "blueprint" and not self.is_advanced():
            text = "Go to Training \u2192"
        self.next_action.configure(text=text)

    def show_page(self, name):
        super().show_page(name)
        self.active_page = name
        self.refresh_next_label()
        if name == "home":
            self.refresh_home()
        elif name == "catalog":
            self.refresh_catalog()

    def next_step(self):
        self.show_page(self.next_page_of(self.active_page))

    def _build_home(self):
        page = self.pages["home"].body
        start = ttk.Frame(page)
        start.pack(fill="x", pady=(0, 14))
        self.home_rows = {}
        for item in self.home_steps():
            row = ttk.Frame(start)
            row.pack(fill="x", pady=2, ipady=6)
            status = label(row, "", "Muted.TLabel", width=9)
            status.pack(side="left", padx=(0, 10))
            text = ttk.Frame(row)
            text.pack(side="left", fill="x", expand=True)
            title = label(text, item["title"], font=(FONT, 12, "bold"))
            title.pack(anchor="w")
            label(text, item["desc"], "Muted.TLabel").pack(anchor="w")
            go = ttk.Button(row, text=item["action"], command=lambda p=item["page"]: self.show_page(p), style="Accent.TButton")
            self.home_rows[item["key"]] = {"marker": status, "title": title, "button": go}
            for widget in (row, text, title):
                widget.bind("<Button-1>", lambda _e, p=item["page"]: self.show_page(p))
        links = ttk.Frame(page)
        links.pack(fill="x")
        self.quick_button = ttk.Button(links, text="Quick start: train a first model for me", command=self.quick_start)
        self.quick_button.pack(side="left", padx=(0, 8))
        Tooltip(self.quick_button, "Picks your biggest instruction dataset and a Small model, then trains for about two minutes. You confirm first.")
        self.home_metrics = {key: tk.StringVar(value="0") for key in ("datasets", "records", "drafts", "runs")}
        self.home_totals, self.home_runs = tk.StringVar(), tk.StringVar()
        ttk.Button(links, textvariable=self.home_totals, command=lambda: self.show_page("datasets")).pack(side="left", padx=(0, 8))
        ttk.Button(links, textvariable=self.home_runs, command=lambda: self.show_page("training")).pack(side="left")
        self.home_hint = tk.StringVar()
        label(page, textvariable=self.home_hint, style="Muted.TLabel", wraplength=950, justify="left").pack(anchor="w", pady=12)

    @staticmethod
    def home_steps():
        return [{"key": "data", "title": "Get data", "desc": "Download a starter collection or import your own examples.", "page": "catalog", "action": "Get data", "optional": False},
                {"key": "extra", "title": "Make extra data", "desc": "Generate more practice examples to fill gaps in your data.", "page": "synthesis", "action": "Make data", "optional": True},
                {"key": "train", "title": "Train", "desc": "Pick a model size and let it learn from your data.", "page": "blueprint", "action": "Set up training", "optional": False},
                {"key": "improve", "title": "Improve with feedback", "desc": "Say which replies are better to polish the model.", "page": "align", "action": "Give feedback", "optional": True},
                {"key": "chat", "title": "Chat", "desc": "Talk to your model and teach it corrections.", "page": "chat", "action": "Open chat", "optional": False}]

    def home_progress(self):
        """Which checklist steps are done, derived from what exists in the workspace. Returns [{key, done, ...}] in order."""
        datasets = self.workspace.state["datasets"]
        runs = self.workspace.runs()
        batches = self.workspace.root / "data" / "synthetic_batches"
        done = {"data": any(self.dataset_stats(d["id"])["records"] > 0 for d in datasets),
                "extra": bool(self.drafts) or (batches.exists() and any(batches.glob("*.json"))),
                "train": any(run.get("stage") != "dpo" for run in runs),
                "improve": len(self.preferences) > 0 or any(run.get("stage") == "dpo" for run in runs),
                "chat": self.chat_model is not None}
        steps = []
        for item in self.home_steps():
            item = dict(item, done=done[item["key"]])
            steps.append(item)
        return steps

    def refresh_home(self):
        if not hasattr(self, "home_metrics"):
            return
        datasets = self.workspace.state["datasets"]
        runs = self.workspace.runs()
        counts = {"datasets": len(datasets), "records": sum(self.dataset_stats(d["id"])["records"] for d in datasets),
                  "drafts": len(self.drafts), "runs": len([r for r in runs if r.get("stage") != "dpo"])}
        for key, var in self.home_metrics.items():
            var.set(f"{counts[key]:,}")
        self.home_totals.set(f"{counts['datasets']:,} datasets \u00b7 {counts['records']:,} examples")
        self.home_runs.set(f"{counts['runs']:,} trained models")
        progress = self.home_progress()
        upcoming = next((item["key"] for item in progress if not item["done"] and not item["optional"]), None)
        for item in progress:
            row = self.home_rows[item["key"]]
            is_next = item["key"] == upcoming
            row["marker"].configure(text="Done" if item["done"] else "Next" if is_next else "Optional" if item["optional"] else "To do",
                                    foreground=ACCENT if item["done"] or is_next else MUTED)
            row["title"].configure(foreground=ACCENT if is_next else TEXT)
            if is_next:
                row["button"].pack(side="right")
            else:
                row["button"].pack_forget()
        selected = self.selected_dataset()
        if not selected:
            text = "Next: download a starter collection or import your own examples."
        else:
            stats = self.dataset_stats(selected)
            text = f"Working dataset: {self.dataset_var.get()} \u00b7 {stats['records']:,} examples \u00b7 {stats['review']:,} not yet checked."
        self.home_hint.set(text)

    # ---- quick start
    def quick_start_plan(self):
        """Pick sensible defaults for a first run: biggest instruction dataset, Small model, about two minutes."""
        sized = [(d, self.dataset_stats(d["id"])["records"]) for d in self.workspace.state["datasets"]]
        sized = [(d, n) for d, n in sized if n > 0]
        if not sized:
            raise ValueError("Quick start needs some data first. Open Get data to download a starter collection, or import your own examples.")
        stage_of = {source["id"]: source.get("stage") for source in CATALOG}

        def stage(dataset):
            key = dataset.get("catalog_id", dataset.get("catalogId")) or (dataset["id"][8:] if dataset["id"].startswith("catalog_") else None)
            return stage_of.get(key)
        pool = [item for item in sized if stage(item[0]) in {"instructions", "conversation"}] or sized
        dataset, count = max(pool, key=lambda item: item[1])
        cfg = dict(DEFAULT_CONFIG, **SIZE_PRESETS[1][1])
        cfg.update(name="Quick start model", device="auto", tokenizer="byte", validation=0.1, eval_every=25, steps=300)
        cfg = validate_model_config(cfg)
        params = parameter_estimate(cfg, 263)
        kind = device_kind("auto", self.gpu_info)
        scale = (self.workspace.state.get("speedScale") or {}).get(kind, 1.0)
        per_step = step_seconds(cfg, params, kind, scale)
        cfg["steps"] = int(min(1500, max(100, round(QUICK_START_SECONDS / per_step / 50) * 50)))
        name = next((key for key, value in self.dataset_map.items() if value == dataset["id"]), dataset["name"])
        return {"dataset_id": dataset["id"], "dataset_name": name, "examples": count, "limit": min(count, QUICK_START_RECORDS), "config": cfg,
                "params": params, "seconds": cfg["steps"] * per_step, "kind": kind, "device": self.device_description("auto")}

    def quick_start(self, confirm=None):
        """Explain, ask, then start a short first training run. `confirm(title, message)` can be injected (tests)."""
        try:
            plan = self.quick_start_plan()
        except ValueError as exc:
            messagebox.showinfo("Quick start needs data", str(exc), parent=self)
            self.show_page("catalog")
            return False
        cfg = plan["config"]
        message = (f"Quick start will:\n\n"
                   f"\u2022 Learn from \u201c{plan['dataset_name']}\u201d: {plan['examples']:,} examples, using up to {plan['limit']:,} of them\n"
                   f"\u2022 Build a Small model from scratch (about {plan['params']:,} parameters)\n"
                   f"\u2022 Train for {cfg['steps']:,} steps on {plan['device']}: {format_duration(plan['seconds'])}\n\n"
                   "You can pause or stop at any time and nothing you already have is changed. When it finishes, your model is ready to chat with.\n\n"
                   "Start now?")
        ask = confirm or (lambda title, text: messagebox.askyesno(title, text, parent=self))
        if not ask("Quick start", message):
            return False
        for key, value in cfg.items():
            if key in self.config_vars:
                self.config_vars[key].set("" if value is None else str(value))
        self.use_all_records.set(False)
        self.reviewed_only.set(False)
        self.train_limit.set(str(plan["limit"]))
        label_for = next((key for key, value in self.dataset_map.items() if value == plan["dataset_id"]), None)
        if label_for:
            self.dataset_var.set(label_for)
            self.select_dataset()
        return bool(self.start_training(sources=[plan["dataset_id"]]))

    def _build_catalog(self):
        page = self.pages["catalog"].body
        banner = card(page, "Recommended collection")
        banner.pack(fill="x", pady=(0, 14))
        summary = catalog_summary()
        self.catalog_total = tk.StringVar(value=(f"Conversation, coding, reasoning and grounded knowledge. {summary['count']} datasets, about {summary['megabytes']:,.0f} MB "
                                                 f"(roughly {summary['records']:,} examples), about {format_duration(summary['minutes'] * 60).replace('about ', '')} on a {DOWNLOAD_MBPS} Mbps connection."))
        label(banner, textvariable=self.catalog_total, style="CardMuted.TLabel", wraplength=920, justify="left").pack(anchor="w")
        controls = ttk.Frame(banner, style="Card.TFrame")
        controls.pack(fill="x", pady=(14, 0))
        button(controls, "Download", self.download_core, True)
        button(controls, "Stop download", self.stop_job)
        self.catalog_status = tk.StringVar(value="Choose a source below. Completed downloads can be reused.")
        label(banner, textvariable=self.catalog_status, style="CardMuted.TLabel", wraplength=920).pack(anchor="w", pady=(8, 0))
        self.catalog_labels = {}
        chooser = getattr(studio_catalog, "default_catalog_ids", None)
        recommended = set(chooser()) if chooser else set()
        label(page, "Or choose sources", "Muted.TLabel").pack(anchor="w", pady=(0, 4))
        short, rest = ttk.Frame(page), ttk.Frame(page)
        short.pack(fill="x")
        for source in CATALOG:
            self._catalog_row(short if source["id"] in recommended else rest, source)
        links = ttk.Frame(page)
        links.pack(fill="x", pady=(10, 0))
        if rest.winfo_children():
            self.drilldown(links, f"More sources ({len(rest.winfo_children())})", rest)
        button(links, "Import my own files", self.import_files)
        button(links, "Open my data", lambda: self.show_page("datasets"))

    def _catalog_row(self, parent, source):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        name = label(row, source["name"], font=(FONT, 11, "bold"), width=34)
        name.pack(side="left")
        Tooltip(name, f"{source['description']}\n\nLicense: {source['license']}")
        label(row, f"{source.get('size_hint', '')}", "Muted.TLabel", width=18).pack(side="left")
        var = tk.StringVar(value="Not in your library")
        self.catalog_labels[source["id"]] = var
        label(row, textvariable=var, style="Muted.TLabel", width=26).pack(side="left")
        button(row, "Download", lambda key=source["id"]: self.download_sources([key]))
        button(row, "Source & license", lambda url=source["source_url"]: webbrowser.open(url))

    def refresh_catalog(self):
        if not hasattr(self, "catalog_labels"):
            return
        for key, variable in self.catalog_labels.items():
            match = next((d for d in self.workspace.state["datasets"] if d.get("catalog_id", d.get("catalogId")) == key or d["id"] == "catalog_" + key), None)
            variable.set(f"In library · {self.dataset_stats(match['id'])['records']:,} examples" if match else "Available to download")

    def download_core(self):
        chooser = getattr(studio_catalog, "default_catalog_ids", None)
        identifiers = list(chooser()) if chooser else [s["id"] for s in CATALOG if s["id"] in {"squad2", "dolly15k", "codealpaca20k", "gsm8k", "oasst1"}]
        self.download_sources(identifiers)

    def download_sources(self, identifiers):
        if self.busy():
            return
        def work():
            completed = []
            for index, identifier in enumerate(identifiers):
                if self.stop.is_set():
                    break
                self.events.put(("catalog_progress", {"text": f"Dataset {index + 1}/{len(identifiers)} · {identifier}"}))
                result = download_dataset(identifier, root=self.workspace.root, stop=self.stop, emit=lambda event: self.events.put(("catalog_progress", event)))
                if self.stop.is_set():
                    break
                dataset_id = register_download(self.workspace, result)
                completed.append(dataset_id)
                self.events.put(("catalog_registered", dataset_id))
            return completed
        def done(identifiers):
            self.refresh_datasets(identifiers[-1] if identifiers else None)
            self.refresh_catalog()
            self.catalog_status.set(f"{len(identifiers)} collection(s) ready in Curate data.")
        self.start_job("dataset download", work, done)

    def dataset_stats(self, dataset_id):
        if dataset_id not in self._stats_cache:
            self._stats_cache[dataset_id] = self.workspace.audit(dataset_id)
        return self._stats_cache[dataset_id]

    def refresh_datasets(self, selected=None):
        self._stats_cache.clear()
        self.record_page = 0
        super().refresh_datasets(selected)
        if hasattr(self, "target_box"):
            keys = list(self.dataset_map)
            self.target_box.configure(values=["Create a new synthetic collection"] + keys)
            if self.target_dataset.get() not in keys:
                self.target_dataset.set("Create a new synthetic collection")
        if hasattr(self, "mixture_tree"):
            self.refresh_mixture()
        self.refresh_rotation_text()
        self.refresh_home()
        self.refresh_catalog()

    def select_dataset(self):
        self.record_page = 0
        super().select_dataset()
        self.refresh_home()

    def _build_datasets(self):
        super()._build_datasets()
        page = self.pages["datasets"]
        self.search.trace_add("write", lambda *_: self.reset_record_page())
        pager = ttk.Frame(page)
        pager.pack(fill="x", pady=(8, 0), before=self.dataset_links)
        button(pager, "← Previous", lambda: self.page_records(-1))
        button(pager, "Next 100 →", lambda: self.page_records(1))
        self.record_count = tk.StringVar()
        label(pager, textvariable=self.record_count, style="Muted.TLabel").pack(side="left", padx=10)
        details = ttk.Frame(page)
        details.pack(fill="x", pady=(10, 0), before=self.dataset_links)
        self.example_detail = text_box(details, height=5, state="disabled")
        self.example_detail.pack(fill="x")
        self.examples_tree.bind("<<TreeviewSelect>>", lambda _: self.preview_example())

    def reset_record_page(self):
        self.record_page = 0
        if self.search_after:
            self.after_cancel(self.search_after)
        self.search_after = self.after(220, self.refresh_examples)

    def page_records(self, direction):
        count = self.workspace.count(self.selected_dataset(), query=self.search.get().strip())
        self.record_page = max(0, min(max(0, (count - 1) // self.record_page_size), self.record_page + direction))
        self.refresh_examples()

    def refresh_examples(self):
        if not hasattr(self, "examples_tree"):
            return
        self.examples_tree.delete(*self.examples_tree.get_children())
        selected = self.selected_dataset()
        query = self.search.get().strip()
        count = self.workspace.count(selected, query=query)
        self.record_page = min(self.record_page, max(0, (count - 1) // self.record_page_size))
        offset = self.record_page * self.record_page_size
        for row in self.workspace.examples(selected, limit=self.record_page_size, offset=offset, query=query):
            self.examples_tree.insert("", "end", iid=row["id"], values=(row.get("user", "").replace("\n", " ")[:180], row.get("assistant", "").replace("\n", " ")[:180], row.get("intent", "general"), row.get("quality", "Review")))
        stats = self.dataset_stats(selected)
        for key, var in self.audit_vars.items():
            var.set(f"{stats[key]:,}")
        self.dataset_summary.set("\u00b7 " + self.dataset_line())
        if hasattr(self, "record_count"):
            self.record_count.set(f"{offset + 1 if count else 0:,}–{min(offset + self.record_page_size, count):,} of {count:,}")
        if hasattr(self, "example_detail"):
            set_text(self.example_detail, "Select an example to read its full conversation and source details. Double-click to edit.")

    def preview_example(self):
        selection = self.examples_tree.selection()
        if not selection:
            return
        example = self.workspace.get_example(selection[0])
        if not example:
            return
        parts = [f"Source: {example.get('source', 'manual')}    Review: {example.get('quality', 'Review')}"]
        if example.get("text"):
            parts.append(example["text"])
        else:
            from studio_core import messages_of
            parts.extend(f"{turn['role'].upper()}\n{turn['content']}" for turn in messages_of(example))
        if example.get("provenance"):
            parts.append("PROVENANCE\n" + json.dumps(example["provenance"], ensure_ascii=False, indent=2))
        set_text(self.example_detail, "\n\n".join(parts))

    def edit_example(self):
        selection = self.examples_tree.selection()
        if not selection:
            return
        example = self.workspace.get_example(selection[0])
        dialog = ExampleEditor(self, example, "Edit training example")
        self.wait_window(dialog)
        if dialog.result:
            try:
                self.workspace.update_example(dialog.result)
                self.refresh_datasets()
            except ValueError as exc:
                messagebox.showerror("Cannot save example", str(exc), parent=self)

    def mark_reviewed(self):
        self.workspace.mark_reviewed(list(self.examples_tree.selection()))
        self._stats_cache.clear()
        self.refresh_examples()

    def remove_examples(self):
        selected = list(self.examples_tree.selection())
        if selected and messagebox.askyesno("Remove examples", f"Remove {len(selected)} selected examples?", parent=self):
            self.workspace.remove_examples(selected)
            self.refresh_datasets()

    # ------------------------------------------------------------------ synthesis page
    def _build_synthesis(self):
        style = ttk.Style(self)
        for kind in ("Checkbutton", "Radiobutton"):
            style.configure(f"Card.T{kind}", background=PANEL, foreground=TEXT)
            style.map(f"Card.T{kind}", background=[("active", PANEL)], foreground=[("active", ACCENT)])
        page = self.pages["synthesis"].body
        self.synthesis_stage = PaneSwitcher(page)       # 0 = make data, 1 = review drafts, 2 = history (the last two are drill-downs)
        self.synthesis_stage.configure(style="TFrame")
        self.synthesis_stage.pack(fill="both", expand=True)
        configure, review, history = [ttk.Frame(self.synthesis_stage, padding=(0, 8)) for _ in range(3)]
        for frame in (configure, review, history):
            self.synthesis_stage.add(frame)
            button(frame, "\u2190 Back to Make extra data", lambda: self.synthesis_stage.select(0)).pack_configure(side="top", padx=0, pady=(0, 10), anchor="w")
        self.batch_history_page = history
        # shared settings (names are part of the test hooks and saved recipes)
        self.synth_count, self.synth_seed = tk.StringVar(value="200"), tk.StringVar(value="42")
        self.synth_difficulty, self.synth_seed_limit = tk.StringVar(value="curriculum"), tk.StringVar(value="2000")
        self.synth_all_sources = tk.BooleanVar(value=False)
        self.synth_attempts = tk.StringVar(value="6")
        self.synth_goal = tk.StringVar(value="practice")
        self.synth_goal_title = tk.StringVar()
        self.synth_length = tk.StringVar(value=LENGTH_LABELS["short"])
        self.synth_plain = tk.BooleanVar(value=True)
        saved = self.workspace.state.get("synthTeacher", {}) if isinstance(self.workspace.state.get("synthTeacher"), dict) else {}
        self.teacher_url = tk.StringVar(value=saved.get("url") or DEFAULT_TEACHER_URL)
        self.teacher_model = tk.StringVar(value=saved.get("model", ""))
        self.teacher_status = tk.StringVar(value="Click Refresh models to list what Ollama has installed.")
        self.teacher_models_loaded = False
        self.teacher_grade, self.teacher_threshold = tk.BooleanVar(value=False), tk.StringVar(value="6")
        self.teacher_batch, self.teacher_temperature = tk.StringVar(value="5"), tk.StringVar(value="")
        self.teach_kind, self.grow_kind, self.remix_kind = tk.StringVar(value="topic"), tk.StringVar(value="evolve"), tk.StringVar(value="reformat")
        self.grow_filter = tk.StringVar(value=FILTER_LABELS["all"])
        self.teach_audience = tk.StringVar()
        self.quality_min, self.quality_max = tk.StringVar(value="1"), tk.StringVar(value="12000")
        self.quality_repeat, self.quality_near = tk.StringVar(value="0.65"), tk.StringVar(value="0")
        self.quality_json, self.quality_conflicts = tk.BooleanVar(value=False), tk.BooleanVar(value=True)
        self.synth_checkpoint = tk.StringVar()
        self.synth_temperature, self.synth_top_k, self.synth_max_tokens = tk.StringVar(value="0.8"), tk.StringVar(value="40"), tk.StringVar(value="160")

        # ---- the focal block: what to make, how many, go
        goals = card(configure, "What do you want to make?")
        goals.pack(fill="x", pady=(0, 12))
        grid = ttk.Frame(goals, style="Card.TFrame")
        grid.pack(fill="x")
        self.goal_cards = {}
        for i, (key, title, blurb) in enumerate(GOALS):
            grid.columnconfigure(i % 3, weight=1, uniform="goal")
            widget = ModeCard(grid, key, title, blurb, self.select_goal)
            widget.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else 6, 0 if i % 3 == 2 else 6), pady=(0, 10 if i < 3 else 0))
            self.goal_cards[key] = widget
        self.count_row = ttk.Frame(goals, style="Card.TFrame")
        self.count_row.pack(fill="x", pady=(12, 0))
        frame, _ = field(self.count_row, "How many examples", self.synth_count, width=12)
        frame.pack(side="left", padx=(0, 16))
        action = ttk.Frame(self.count_row, style="Card.TFrame")
        action.pack(side="left", pady=(20, 0))
        button(action, "Generate examples \u2192", self.generate_drafts, True)
        button(action, "Try 5 samples first", self.preview_samples)
        button(action, "Stop", self.stop_job)
        self.synth_summary = tk.StringVar(value="Pick what to make, try 5 samples, then generate a full batch.")
        label(configure, textvariable=self.synth_summary, style="Muted.TLabel", wraplength=1000, justify="left").pack(anchor="w", pady=(0, 8))

        # ---- quiet drill-downs
        links = ttk.Frame(configure)
        links.pack(fill="x")
        options = self.synth_options = card(configure)
        label(options, textvariable=self.synth_goal_title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))
        self.synth_tabs = PaneSwitcher(options, self._synth_changed)
        self.synth_tabs.pack(fill="x")
        panes = {key: ttk.Frame(self.synth_tabs, style="Card.TFrame") for key in PANE_ORDER}
        for key in PANE_ORDER:
            self.synth_tabs.add(panes[key])
        self._build_practice_pane(panes["practice"])
        self._build_teach_pane(panes["teach"])
        self._build_grow_pane(panes["grow"])
        self._build_remix_pane(panes["remix"])
        self._build_model_pane(panes["model"])
        self._build_template_pane(panes["templates"])
        self.teacher_box = self._build_teacher_box(options)
        self.review_link = button(links, "Review drafts (0)", lambda: self.synthesis_stage.select(1))
        self.options_button = self.drilldown(links, "Options", options)
        self.advanced_button = ttk.Button(links, text="\u25b8 Quality checks & recipes", command=self.toggle_advanced)
        self.advanced_button.pack(side="left", padx=(0, 8), pady=4)
        button(links, "History", lambda: self.synthesis_stage.select(2))
        self.mark_advanced(self.advanced_button)
        self.sample_preview = text_box(configure, 8, state="disabled")      # packed by show_samples
        set_text(self.sample_preview, "Samples appear here. They are not saved: use them to check the style, then generate the full batch.")
        self.synth_advanced = self._build_advanced(configure)
        self.advanced_anchor = links

        # ---- review tab
        review_controls = card(review, "Choose what joins your training collection")
        review_controls.pack(fill="x", pady=(0, 12))
        target_row = ttk.Frame(review_controls, style="Card.TFrame")
        target_row.pack(fill="x", pady=(0, 8))
        label(target_row, "Add to", "CardMuted.TLabel").pack(side="left", padx=(0, 8))
        self.target_dataset = tk.StringVar(value="Create a new synthetic collection")
        self.target_box = ttk.Combobox(target_row, textvariable=self.target_dataset, state="readonly", width=48)
        self.target_box.pack(side="left")
        actions = ttk.Frame(review_controls, style="Card.TFrame")
        actions.pack(fill="x")
        button(actions, "Add selected to training collection", lambda: self.keep_drafts(False), True)
        self.add_all_button = button(actions, "Add all", lambda: self.keep_drafts(True))
        button(actions, "Edit", self.edit_draft)
        button(actions, "Mark reviewed", self.toggle_reviewed)
        button(actions, "Discard selected", self.discard_drafts)
        button(actions, "Discard all", self.discard_all_drafts)
        self.review_report = tk.StringVar(value="Select drafts and add them, or add everything. Marking reviewed is optional; it lets you train on reviewed examples only.")
        label(review_controls, textvariable=self.review_report, style="CardMuted.TLabel", wraplength=940).pack(anchor="w", pady=(10, 0))
        self.draft_tree = tree(review, [("prompt", "PROMPT", 320), ("answer", "TARGET", 320), ("recipe", "TASK", 90), ("checks", "CHECKS", 120), ("review", "STATUS", 80)], 9)
        self.draft_tree.bind("<Double-1>", lambda _: self.edit_draft())
        self.draft_tree.bind("<<TreeviewSelect>>", lambda _: self.preview_draft())
        pager = ttk.Frame(review)
        pager.pack(fill="x", pady=(8, 0))
        button(pager, "← Previous", lambda: self.page_drafts(-1))
        button(pager, "Next 100 →", lambda: self.page_drafts(1))
        self.draft_count = tk.StringVar()
        label(pager, textvariable=self.draft_count, style="Muted.TLabel").pack(side="left", padx=10)
        button(pager, "Train this collection →", lambda: self.show_page("blueprint"), True)
        self.draft_detail = text_box(review, 5, state="disabled")
        self.draft_detail.pack(fill="x", pady=(10, 0))
        label(history, "Saved batch recipes and quality reports", font=(FONT, 13, "bold")).pack(anchor="w", pady=(0, 12))
        self.batch_tree = tree(history, [("created", "CREATED", 190), ("mode", "RECIPE", 100), ("accepted", "ACCEPTED", 90), ("rejected", "FILTERED", 90)], 9)
        self.batch_tree.bind("<<TreeviewSelect>>", lambda _: self.inspect_batch())
        self.batch_detail = text_box(history, 11, state="disabled")
        self.batch_detail.pack(fill="x", pady=(12, 0))
        self.synth_tabs.select(PANE_ORDER.index("practice"))
        self.refresh_batches()
        self.synthesis_stage.select(0)

    # ---- the six goal panes: only the essential fields live here
    def _build_practice_pane(self, pane):
        label(pane, "Which kinds of problems?", "CardMuted.TLabel").pack(anchor="w", pady=(0, 6))
        grid = ttk.Frame(pane, style="Card.TFrame")
        grid.pack(fill="x")
        self.family_vars, self.family_weights = {}, {}
        for i, (key, title) in enumerate(PRACTICE_KINDS):
            self.family_vars[key] = tk.BooleanVar(value=True)
            self.family_weights[key] = tk.StringVar(value="1")
            grid.columnconfigure(i % 3, weight=1, uniform="kind")
            ttk.Checkbutton(grid, text=title, variable=self.family_vars[key], style="Card.TCheckbutton").grid(row=i // 3, column=i % 3, sticky="w", pady=4)
        row = ttk.Frame(pane, style="Card.TFrame")
        row.pack(fill="x", pady=(10, 0))
        frame, _ = field(row, "Difficulty (curriculum = easy to hard across the batch)", self.synth_difficulty, ["easy", "medium", "hard", "curriculum"], width=18, help="difficulty")
        frame.pack(side="left")

    def _build_teach_pane(self, pane):
        row = ttk.Frame(pane, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(row, text="Questions and answers", value="topic", variable=self.teach_kind, style="Card.TRadiobutton").pack(side="left", padx=(0, 18))
        ttk.Radiobutton(row, text="Short conversations (2–4 turns)", value="dialogue", variable=self.teach_kind, style="Card.TRadiobutton").pack(side="left")
        label(pane, "What should the teacher write about? A goal, or one topic per line.", "CardMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.teach_topics = text_box(pane, 3)
        self.teach_topics.pack(fill="x")
        set_text(self.teach_topics, "basic arithmetic word problems")
        row = ttk.Frame(pane, style="Card.TFrame")
        row.pack(fill="x", pady=(10, 0))
        frame, _ = field(row, "Audience (optional)", self.teach_audience, width=30)
        frame.pack(side="left", padx=(0, 16))
        self._style_fields(row)

    def _style_fields(self, row):
        frame, _ = field(row, "Answer length", self.synth_length, list(LENGTH_LABELS.values()), width=30)
        frame.pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row, text="Plain text only (no markdown or emoji)", variable=self.synth_plain, style="Card.TCheckbutton").pack(side="left", pady=(18, 0))

    def _build_grow_pane(self, pane):
        pane.columnconfigure(0, weight=1)
        kinds = [("evolve", "Evolve prompts: make harder, more specific versions and answer them"),
                 ("paraphrase", "Reword prompts: new wording, the original answers are copied exactly"),
                 ("answer", "Answer prompts: the teacher writes answers (distillation)")]
        for i, (value, text) in enumerate(kinds):
            ttk.Radiobutton(pane, text=text, value=value, variable=self.grow_kind, command=self._grow_changed, style="Card.TRadiobutton").grid(row=i, column=0, sticky="w", pady=2)
        label(pane, "Prompts come from the working dataset chosen at the top of the page. Computed answers are never rewritten.", "CardMuted.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 8))
        self.grow_filter_row = ttk.Frame(pane, style="Card.TFrame")
        self.grow_filter_row.grid(row=4, column=0, sticky="w")
        frame, _ = field(self.grow_filter_row, "Which prompts get answered?", self.grow_filter, list(FILTER_LABELS.values()), width=34)
        frame.pack(side="left")
        label(pane, "Optional: paste more prompts, one per line", "CardMuted.TLabel").grid(row=5, column=0, sticky="w", pady=(10, 5))
        self.grow_prompts = text_box(pane, 3)
        self.grow_prompts.grid(row=6, column=0, sticky="ew")
        self.grow_style_row = ttk.Frame(pane, style="Card.TFrame")
        self.grow_style_row.grid(row=7, column=0, sticky="w", pady=(10, 0))
        self._style_fields(self.grow_style_row)
        self._grow_changed()

    def _grow_changed(self):
        kind = self.grow_kind.get()
        for widget, shown in ((self.grow_filter_row, kind == "answer"), (self.grow_style_row, kind != "paraphrase")):
            widget.grid() if shown else widget.grid_remove()

    def _build_remix_pane(self, pane):
        kinds = [("reformat", "Reformat answers: final answer only, clean worked solutions, JSON answers, conversation chains"),
                 ("wrap", "Reword requests: wrap each request in a different phrasing, every answer kept"),
                 ("copy", "Copy tasks: ask for exact passages or answers taken from the source records")]
        for value, text in kinds:
            ttk.Radiobutton(pane, text=text, value=value, variable=self.remix_kind, command=self._remix_changed, style="Card.TRadiobutton").pack(anchor="w", pady=2)
        self.remix_grid = ttk.Frame(pane, style="Card.TFrame")
        self.remix_grid.pack(fill="x", pady=(8, 0))
        self.remix_vars, self.remix_weights = {}, {}
        for i, (key, title) in enumerate(REMIX_KINDS):
            self.remix_vars[key] = tk.BooleanVar(value=True)
            self.remix_weights[key] = tk.StringVar(value="1")
            self.remix_grid.columnconfigure(i % 3, weight=1, uniform="remix")
            ttk.Checkbutton(self.remix_grid, text=title, variable=self.remix_vars[key], style="Card.TCheckbutton").grid(row=i // 3, column=i % 3, sticky="w", pady=4)
        label(pane, "Records the sampled data cannot support are skipped and reported afterwards.", "CardMuted.TLabel").pack(anchor="w", pady=(8, 0))

    def _remix_changed(self):
        if self.remix_kind.get() == "reformat":
            self.remix_grid.pack(fill="x", pady=(8, 0), after=self.remix_grid.master.winfo_children()[2])
        else:
            self.remix_grid.pack_forget()

    def _build_model_pane(self, pane):
        with_help(pane, "A checkpoint trained in this studio (.npz file)", "checkpoint").pack(anchor="w", pady=(0, 5))
        row = ttk.Frame(pane, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 8))
        ttk.Entry(row, textvariable=self.synth_checkpoint).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Choose .npz", command=lambda: self.choose_checkpoint(self.synth_checkpoint)).pack(side="left")
        label(pane, "Prompts to answer, one per line. Leave empty to use the prompts in the working dataset.", "CardMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.synth_prompts = text_box(pane, 3)
        self.synth_prompts.pack(fill="x")
        label(pane, "Answers come only from your own model and always need review. Sampling controls are under Quality checks & recipes.", "CardMuted.TLabel").pack(anchor="w", pady=(8, 0))

    def _build_template_pane(self, pane):
        pane.columnconfigure((0, 1), weight=1)
        label(pane, "Fact rows (list of objects) or variable lists", "CardMuted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        label(pane, "Prompt template and answer template, with {placeholders}", "CardMuted.TLabel").grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6))
        self.template_variables = text_box(pane, 7)
        self.template_variables.grid(row=1, column=0, rowspan=2, sticky="nsew")
        set_text(self.template_variables, '[\n  {"country": "France", "capital": "Paris"},\n  {"country": "Japan", "capital": "Tokyo"}\n]')
        self.template_prompt, self.template_response = text_box(pane, 3), text_box(pane, 3)
        self.template_prompt.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=(0, 6))
        self.template_response.grid(row=2, column=1, sticky="ew", padx=(12, 0))
        set_text(self.template_prompt, "What is the capital of {country}?")
        set_text(self.template_response, "{capital}")

    def _build_teacher_box(self, parent):
        box = ttk.Frame(parent, style="Card.TFrame")
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x")
        column = ttk.Frame(row, style="Card.TFrame")
        column.pack(side="left", padx=(0, 16))
        with_help(column, "Teacher model (Ollama)", "teacher").pack(anchor="w", pady=(0, 5))
        self.teacher_combo = ttk.Combobox(column, textvariable=self.teacher_model, width=30)
        self.teacher_combo.pack()
        frame, _ = field(row, "Ollama address", self.teacher_url, width=28, help="ollama")
        frame.pack(side="left", padx=(0, 12))
        ttk.Button(row, text="Refresh models", command=self.refresh_teacher_models).pack(side="left", pady=(20, 0))
        label(box, textvariable=self.teacher_status, style="CardMuted.TLabel").pack(anchor="w", pady=(6, 0))
        grade = ttk.Frame(box, style="Card.TFrame")
        grade.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(grade, text="Have the teacher grade each example from 1 to 10 and keep scores of at least", variable=self.teacher_grade, style="Card.TCheckbutton").pack(side="left")
        ttk.Entry(grade, textvariable=self.teacher_threshold, width=4).pack(side="left", padx=6)
        return box

    def _build_advanced(self, parent):
        box = card(parent, "Advanced options")
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x")
        for title, variable, helper in [("Random seed", self.synth_seed, "seed"), ("Source sample size", self.synth_seed_limit, "sample_size"),
                                        ("Attempt multiplier", self.synth_attempts, "attempts"), ("Teacher requests: examples per call", self.teacher_batch, "teacher_batch"),
                                        ("Teacher temperature (blank = default)", self.teacher_temperature, "temperature")]:
            frame, _ = field(row, title, variable, width=17, help=helper)
            frame.pack(side="left", fill="x", expand=True, padx=(0, 12))
        ttk.Checkbutton(box, text="Sample sources from every collection (not only the working dataset)", variable=self.synth_all_sources, style="Card.TCheckbutton").pack(anchor="w", pady=(10, 0))
        with_help(box, "Quality gates", "quality_gates", "CardTitle.TLabel").pack(anchor="w", pady=(16, 8))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x")
        for title, var, helper in [("Min answer characters", self.quality_min, "quality_gates"), ("Max answer characters", self.quality_max, "quality_gates"),
                                   ("Max repetition fraction", self.quality_repeat, "repetition"), ("Near-duplicate cutoff \u00b7 0 off", self.quality_near, "near_duplicate")]:
            frame, _ = field(row, title, var, width=16, help=helper)
            frame.pack(side="left", fill="x", expand=True, padx=(0, 12))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(row, text="Require a valid JSON answer", variable=self.quality_json, style="Card.TCheckbutton").pack(side="left", padx=(0, 20))
        ttk.Checkbutton(row, text="Reject conflicting answers for the same prompt", variable=self.quality_conflicts, style="Card.TCheckbutton").pack(side="left")
        label(box, "Exact duplicates are always removed. Length and similarity checks are heuristics, not fact checks. Teacher answers are capped by the length you chose.", "CardMuted.TLabel", wraplength=900).pack(anchor="w", pady=(8, 0))
        label(box, "System prompt added to generated examples (optional)", "CardMuted.TLabel").pack(anchor="w", pady=(14, 5))
        self.synth_system = text_box(box, 2)
        self.synth_system.pack(fill="x")
        label(box, "Your-model sampling (creativity, top-k, reply length)", "CardTitle.TLabel").pack(anchor="w", pady=(16, 8))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x")
        for title, variable, helper in [("Temperature", self.synth_temperature, "temperature"), ("Top-k", self.synth_top_k, "top_k"), ("New tokens", self.synth_max_tokens, "max_tokens")]:
            with_help(row, title, helper).pack(side="left", padx=(0, 5))
            ttk.Entry(row, textvariable=variable, width=7).pack(side="left", padx=(0, 14))
        ttk.Combobox(row, textvariable=self.inference_device, values=["auto", "cpu", "cuda"], state="readonly", width=8).pack(side="left")
        with_help(box, "Task weights (relative sampling emphasis: 2 = twice as often)", "weight", "CardTitle.TLabel").pack(anchor="w", pady=(16, 8))
        grid = ttk.Frame(box, style="Card.TFrame")
        grid.pack(fill="x")
        entries = [(title, self.family_weights[key]) for key, title in PRACTICE_KINDS] + [(title, self.remix_weights[key]) for key, title in REMIX_KINDS]
        for i, (title, variable) in enumerate(entries):
            grid.columnconfigure(i % 2 * 2, weight=1)
            label(grid, title, "CardMuted.TLabel").grid(row=i // 2, column=i % 2 * 2, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(grid, textvariable=variable, width=5).grid(row=i // 2, column=i % 2 * 2 + 1, padx=(0, 30), pady=3)
        label(box, "Recipes", "CardTitle.TLabel").pack(anchor="w", pady=(16, 8))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x")
        self.preset_names = {item["name"]: item for item in RECIPE_PRESETS}
        self.synth_preset = tk.StringVar(value="")
        chooser = ttk.Combobox(row, textvariable=self.synth_preset, values=list(self.preset_names), state="readonly", width=48)
        chooser.pack(side="left", padx=(0, 10))
        chooser.bind("<<ComboboxSelected>>", lambda _: self.apply_synth_preset())
        button(row, "Load recipe", self.load_recipe)
        button(row, "Save recipe", self.save_recipe)
        self.recipe_description = tk.StringVar(value="Pick a ready-made recipe to fill in the settings above, or save your own.")
        label(box, textvariable=self.recipe_description, style="CardMuted.TLabel", wraplength=900).pack(anchor="w", pady=(8, 0))
        return box

    def toggle_advanced(self):
        if self.synth_advanced.winfo_ismapped():
            self.synth_advanced.pack_forget()
            self.advanced_button.configure(text="▸ Quality checks & recipes")
        else:
            self.synth_advanced.pack(fill="x", pady=(0, 12), after=self.advanced_anchor)
            self.advanced_button.configure(text="▾ Quality checks & recipes")

    # ---- goal selection
    def select_goal(self, key):
        self.synth_tabs.select(PANE_ORDER.index(key))

    def _synth_changed(self, index):
        goal = PANE_ORDER[index]
        self.synth_goal.set(goal)
        for key, widget in self.goal_cards.items():
            widget.set_selected(key == goal)
        self.synth_goal_title.set(next(title for key, title, _ in GOALS if key == goal))
        if hasattr(self, "options_button"):
            self.options_button.base = f"Options for {self.synth_goal_title.get().lower()}"
            if goal != "practice" and not self.synth_options.winfo_manager():     # these goals need input, so open their options
                self.options_button.invoke()
            self.options_button.sync()
        if goal in TEACHER_GOALS:
            self.teacher_box.pack(fill="x", pady=(12, 0))
            if not self.teacher_models_loaded and not self.teacher_model.get().strip():
                self.after(80, lambda: self.refresh_teacher_models(quiet=True))
        else:
            self.teacher_box.pack_forget()

    def refresh_teacher_models(self, quiet=False):
        """List Ollama's models on a helper thread; the UI stays responsive and nothing blocks on a dead server."""
        url = self.teacher_url.get().strip()
        self.teacher_models_loaded = True
        self.teacher_status.set("Looking for models…")
        outbox = queue.Queue()
        def work():
            try:
                outbox.put((list_teacher_models(url), ""))
            except Exception as exc:
                outbox.put((None, str(exc)))
        threading.Thread(target=work, name="studio-teacher-models", daemon=True).start()
        def poll_models():
            if self.closing:
                return
            try:
                models, error = outbox.get_nowait()
            except queue.Empty:
                try:
                    self.after(120, poll_models)
                except tk.TclError:
                    pass
                return
            if models is None:
                self.teacher_status.set(error or "Ollama did not answer.")
                return
            self.teacher_combo.configure(values=models)
            current = self.teacher_model.get().strip()
            if models and current not in models:
                self.teacher_model.set(next((m for m in models if "llama3.2" in m), models[0]))
            self.teacher_status.set(f"{len(models)} model(s) available." if models else "Ollama is running but has no models yet.")
        self.after(120, poll_models)

    def _teacher_settings(self):
        return {"teacher_url": self.teacher_url.get().strip(), "teacher_model": self.teacher_model.get().strip(),
                "self_grade": self.teacher_grade.get(), "grade_threshold": float(self.teacher_threshold.get() or 6),
                "batch_size": int(self.teacher_batch.get() or 5)}

    # ---- reading and writing the form
    def read_quality(self):
        return {"min_answer_chars": int(self.quality_min.get()), "max_answer_chars": int(self.quality_max.get()),
                "max_repeat_fraction": float(self.quality_repeat.get()), "near_duplicate_threshold": float(self.quality_near.get()),
                "require_json": self.quality_json.get(), "check_conflicts": self.quality_conflicts.get()}

    def read_synth_config(self):
        goal = PANE_ORDER[self.synth_tabs.index(self.synth_tabs.select())]
        cfg = {"count": int(self.synth_count.get()), "seed": int(self.synth_seed.get()), "attempt_multiplier": int(self.synth_attempts.get()),
               "quality": self.read_quality()}
        system = content(self.synth_system)
        if system:
            cfg["system_prompt"] = system
        if goal == "practice":
            families = [key for key, var in self.family_vars.items() if var.get()]
            if not families:
                raise ValueError("Choose at least one kind of practice problem.")
            cfg.update(mode="arithmetic" if families == ["arithmetic"] else "mixed", difficulty=self.synth_difficulty.get(),
                       families=families, weights={key: float(self.family_weights[key].get()) for key in families})
        elif goal == "templates":
            values = json.loads(content(self.template_variables))
            cfg.update(mode="templates", difficulty=self.synth_difficulty.get(), prompt=content(self.template_prompt), response=content(self.template_response))
            cfg["fact_rows" if isinstance(values, list) else "variables"] = values
        elif goal == "remix":
            kind = self.remix_kind.get()
            cfg.update(mode={"reformat": "remix", "wrap": "seeds", "copy": "grounded"}[kind], difficulty=self.synth_difficulty.get())
            if kind == "reformat":
                cfg["families"] = [key for key, var in self.remix_vars.items() if var.get()]
                cfg["weights"] = {key: float(self.remix_weights[key].get()) for key in cfg["families"]}
        elif goal == "model":
            cfg.update(mode="model", checkpoint=self.synth_checkpoint.get(), device=self.inference_device.get(), temperature=float(self.synth_temperature.get()),
                       top_k=int(self.synth_top_k.get()), max_tokens=int(self.synth_max_tokens.get()), model_prompts=content(self.synth_prompts).splitlines())
        else:
            mode = self.teach_kind.get() if goal == "teach" else self.grow_kind.get()
            cfg.update(mode=mode, answer_length=next(k for k, v in LENGTH_LABELS.items() if v == self.synth_length.get()),
                       plain_text=self.synth_plain.get(), **self._teacher_settings())
            if goal == "teach":
                cfg.update(topics=content(self.teach_topics), audience=self.teach_audience.get().strip())
            else:
                cfg["prompts"] = content(self.grow_prompts)
                if mode == "answer":
                    cfg["answer_filter"] = next(k for k, v in FILTER_LABELS.items() if v == self.grow_filter.get())
            if self.teacher_temperature.get().strip():
                cfg["teacher_temperature"] = float(self.teacher_temperature.get())
        return cfg

    def apply_synth_config(self, cfg):
        mode = cfg.get("mode")
        goal = ("practice" if mode in {"mixed", "reasoning", "arithmetic", "structured", None} else "templates" if mode == "templates" else
                "teach" if mode in {"topic", "dialogue"} else "grow" if mode in {"evolve", "paraphrase", "answer"} else
                "model" if mode == "model" else "remix")
        for key, variable in [("count", self.synth_count), ("seed", self.synth_seed), ("difficulty", self.synth_difficulty), ("checkpoint", self.synth_checkpoint),
                              ("temperature", self.synth_temperature), ("top_k", self.synth_top_k), ("max_tokens", self.synth_max_tokens), ("device", self.inference_device),
                              ("attempt_multiplier", self.synth_attempts), ("teacher_url", self.teacher_url), ("teacher_model", self.teacher_model),
                              ("audience", self.teach_audience), ("grade_threshold", self.teacher_threshold), ("batch_size", self.teacher_batch)]:
            if key in cfg:
                variable.set(str(cfg[key]))
        if "self_grade" in cfg:
            self.teacher_grade.set(bool(cfg["self_grade"]))
        if "plain_text" in cfg:
            self.synth_plain.set(bool(cfg["plain_text"]))
        if cfg.get("answer_length") in LENGTH_LABELS:
            self.synth_length.set(LENGTH_LABELS[cfg["answer_length"]])
        if cfg.get("answer_filter") in FILTER_LABELS:
            self.grow_filter.set(FILTER_LABELS[cfg["answer_filter"]])
        if "teacher_temperature" in cfg:
            self.teacher_temperature.set(str(cfg["teacher_temperature"]))
        if mode in {"topic", "dialogue"}:
            self.teach_kind.set(mode)
        if mode in {"evolve", "paraphrase", "answer"}:
            self.grow_kind.set(mode)
            self._grow_changed()
        if mode in {"remix", "seeds", "grounded"}:
            self.remix_kind.set({"remix": "reformat", "seeds": "wrap", "grounded": "copy"}[mode])
            self._remix_changed()
        if "topics" in cfg:
            set_text(self.teach_topics, cfg["topics"] if isinstance(cfg["topics"], str) else "\n".join(cfg["topics"]))
        if "prompts" in cfg:
            set_text(self.grow_prompts, cfg["prompts"] if isinstance(cfg["prompts"], str) else "\n".join(cfg["prompts"]))
        if "system_prompt" in cfg:
            set_text(self.synth_system, cfg["system_prompt"])
        if "fact_rows" in cfg or "variables" in cfg:
            set_text(self.template_variables, json.dumps(cfg.get("fact_rows", cfg.get("variables")), ensure_ascii=False, indent=2))
        if "prompt" in cfg:
            set_text(self.template_prompt, cfg["prompt"])
        if "response" in cfg:
            set_text(self.template_response, cfg["response"])
        if "model_prompts" in cfg:
            set_text(self.synth_prompts, "\n".join(cfg["model_prompts"]))
        if "families" in cfg and mode == "remix":
            for key, var in self.remix_vars.items():
                var.set(key in cfg["families"])
                self.remix_weights[key].set(str(cfg.get("weights", {}).get(key, 1)))
        elif "families" in cfg and goal == "practice":
            for key, var in self.family_vars.items():
                var.set(key in cfg["families"])
                self.family_weights[key].set(str(cfg.get("weights", {}).get(key, 1)))
        elif mode == "arithmetic":
            for key, var in self.family_vars.items():
                var.set(key == "arithmetic")
        quality = cfg.get("quality", {})
        for key, var in [("min_answer_chars", self.quality_min), ("max_answer_chars", self.quality_max), ("max_repeat_fraction", self.quality_repeat),
                         ("near_duplicate_threshold", self.quality_near), ("require_json", self.quality_json), ("check_conflicts", self.quality_conflicts)]:
            if key in quality:
                var.set(quality[key])
        self.select_goal(goal)

    def apply_synth_preset(self):
        preset = self.preset_names.get(self.synth_preset.get())
        if preset:
            self.apply_synth_config(preset["config"])
            self.recipe_description.set(preset["description"])

    def save_recipe(self):
        try:
            cfg = self.read_synth_config()
            path = filedialog.asksaveasfilename(parent=self, title="Save generation recipe", defaultextension=".json", filetypes=[("JSON recipe", "*.json")])
            if path:
                atomic_json(path, cfg)
                self.status_var.set(f"Recipe saved · {path}")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Recipe could not be saved", str(exc), parent=self)

    def load_recipe(self):
        path = filedialog.askopenfilename(parent=self, title="Load generation recipe", filetypes=[("JSON recipe", "*.json")])
        if path:
            try:
                cfg = json.loads(Path(path).read_text(encoding="utf-8"))
                if not isinstance(cfg, dict):
                    raise ValueError("The recipe must be a JSON object.")
                self.apply_synth_config(cfg)
                self.recipe_description.set(f"Loaded recipe · {Path(path).name}")
            except (ValueError, OSError, TypeError) as exc:
                messagebox.showerror("Recipe could not be loaded", str(exc), parent=self)

    # ---- generating
    def generate_drafts(self):
        self._start_generation(preview=False)

    def preview_samples(self):
        self._start_generation(preview=True)

    def _start_generation(self, preview):
        if self.busy():
            return
        try:
            cfg = self.read_synth_config()
            if preview:
                cfg["count"] = 5
            validate_config(cfg)
            seed_limit = int(self.synth_seed_limit.get())
            if not 1 <= seed_limit <= 20000:
                raise ValueError("Source sample size must be 1–20,000.")
            dataset_id = self.selected_dataset()
            sources = [d["id"] for d in self.workspace.state["datasets"]] if self.synth_all_sources.get() else [dataset_id] if dataset_id else []
            existing = copy.deepcopy(self.drafts)
            if cfg["mode"] in TEACHER_MODES:
                self.workspace.state["synthTeacher"] = {"url": cfg["teacher_url"], "model": cfg["teacher_model"]}
                self.workspace.save()
            def work():
                # Split the sample budget evenly so a large corpus does not crowd out small ones.
                share = max(1, seed_limit // max(1, len(sources)))
                seeds = [row for source in sources for row in self.workspace.sample(source, limit=share, seed=cfg["seed"])]
                result = generate_batch(cfg, seeds=seeds, existing=existing + seeds, stop=self.stop, emit=lambda event: self.events.put(("synth_progress", event)))
                result["source_dataset"] = dataset_id
                result["source_datasets"] = sources
                result["source_sample_size"] = len(seeds)
                return result
            self.start_job("synthesis", work, self.show_samples if preview else self.accept_drafts)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Check this recipe", str(exc), parent=self)

    def show_samples(self, result):
        """Show a preview batch without saving it as drafts or in the history."""
        parts = []
        for row in result["rows"]:
            parts.append(f"PROMPT\n{row.get('user', '')}\n\nANSWER\n{row.get('assistant', '')}")
        if not parts:
            parts.append("Nothing passed the quality checks. Skipped examples and reasons are listed in the summary above.")
        if not self.sample_preview.winfo_manager():
            self.sample_preview.pack(fill="x", pady=(0, 12))
        set_text(self.sample_preview, "\n\n────────────\n\n".join(parts))
        self.synth_summary.set(summarize_batch(result) + "  These samples were not saved.")

    def accept_drafts(self, result):
        if isinstance(result, list):
            result = {"rows": result, "stats": {"accepted": len(result)}, "batch_id": uid("batch"), "config": {}}
        known = {fingerprint(row) for row in self.drafts}
        added = 0
        for row in result["rows"]:
            key = fingerprint(row)
            if key not in known:
                known.add(key)
                self.drafts.append(row)
                added += 1
        self.last_generation = result
        metadata = {key: value for key, value in result.items() if key != "rows"}
        metadata["createdAt"] = now()
        metadata["new_drafts"] = added
        atomic_json(self.workspace.root / "data" / "synthetic_batches" / f"{result['batch_id']}.json", metadata)
        self.save_drafts()
        summary = summarize_batch(result)
        if added != len(result["rows"]):
            summary += f" {len(result['rows']) - added:,} were already waiting for review."
        summary += f" {len(self.drafts):,} waiting in Review & add."
        self.synth_summary.set(summary)
        self.review_report.set(summary + "\nFilters compare this batch, pending drafts and the sampled source records.")
        self.refresh_batches()
        self.synthesis_stage.select(1)
        self.refresh_home()

    # ---- reviewing
    def refresh_drafts(self):
        if not hasattr(self, "draft_tree"):
            return
        self.draft_tree.delete(*self.draft_tree.get_children())
        self.draft_page = min(self.draft_page, max(0, (len(self.drafts) - 1) // self.draft_page_size))
        offset = self.draft_page * self.draft_page_size
        for row in self.drafts[offset:offset + self.draft_page_size]:
            task = row.get("provenance", {}).get("recipe", row.get("intent", "synthetic"))
            report = row.get("qualityReport", {}) if isinstance(row.get("qualityReport"), dict) else {}
            checks = ("edited" if report.get("edited") else "computed ✓" if row.get("computedVerified") else
                      f"teacher {report['teacher_score']:g}/10" if "teacher_score" in report else "")
            self.draft_tree.insert("", "end", iid=row["id"], values=(row.get("user", "")[:160].replace("\n", " "), row.get("assistant", "")[:160].replace("\n", " "), task,
                                                                        checks, "Reviewed" if row.get("quality") == "Good" else "To review"))
        if hasattr(self, "draft_count"):
            self.draft_count.set(f"{offset + 1 if self.drafts else 0:,}–{min(offset + self.draft_page_size, len(self.drafts)):,} of {len(self.drafts):,}")
        if hasattr(self, "review_link"):
            self.review_link.configure(text=f"Review drafts ({len(self.drafts):,})")
        if hasattr(self, "add_all_button"):
            self.add_all_button.configure(text=f"Add all {len(self.drafts):,}" if self.drafts else "Add all")

    def page_drafts(self, direction):
        self.draft_page = max(0, self.draft_page + direction)
        self.refresh_drafts()

    def preview_draft(self):
        selected = self.draft_tree.selection()
        row = next((r for r in self.drafts if r["id"] == selected[0]), None) if selected else None
        if row is None:
            return
        report = row.get("qualityReport", {}) if isinstance(row.get("qualityReport"), dict) else {}
        origin = row.get("provenance", {})
        how = []
        if row.get("computedVerified"):
            how.append("Computed answer: calculated by the generator, so it is correct for this prompt.")
        if origin.get("teacher_model"):
            how.append(f"Written by teacher {origin['teacher_model']} ({origin.get('mode', '')})" + (f" · graded {report['teacher_score']:g}/10: {report.get('teacher_rationale', '')}" if "teacher_score" in report else ""))
        how.extend(report.get("warnings", []))
        details = json.dumps({k: row[k] for k in ("quality", "computedVerified", "qualityReport", "provenance") if k in row}, ensure_ascii=False, indent=2)
        set_text(self.draft_detail, f"PROMPT\n{row.get('user', '')}\n\nTARGET\n{row.get('assistant', '')}\n\nHOW THIS WAS MADE\n" + "\n".join(how or ["No extra notes."]) + "\n\nDETAILS\n" + details)

    def save_drafts(self):
        # Drafts have their own file so the workspace manifest stays small.
        atomic_json(self.workspace.root / "data" / "drafts.json", self.drafts)
        self.workspace.state["generated"] = []
        self.workspace.state["draftsFile"] = "data/drafts.json"
        self.workspace.save()
        self.refresh_drafts()

    def _selected_drafts(self):
        selected = set(self.draft_tree.selection())
        return selected, [row for row in self.drafts if row["id"] in selected]

    def _reselect(self, identifiers):
        self.draft_tree.selection_set([i for i in identifiers if self.draft_tree.exists(i)])

    def toggle_reviewed(self):
        """One switch: mark the selected drafts reviewed, or clear the mark if they all already are."""
        selected, rows = self._selected_drafts()
        if not rows:
            self.status_var.set("Select drafts to mark as reviewed.")
            return
        mark = any(row.get("quality") != "Good" for row in rows)
        for row in rows:
            row.update(quality="Good" if mark else "Review", updatedAt=now())
        self.save_drafts()
        self._reselect(selected)
        self.status_var.set(f"{len(rows)} draft(s) {'marked reviewed' if mark else 'set back to to-review'}.")

    def keep_drafts(self, all_rows):
        """Add drafts (selected or all) to the training collection. Quality, provenance and verification flags travel with them."""
        selected, picked = self._selected_drafts()
        rows = list(self.drafts) if all_rows else picked
        if not rows:
            self.status_var.set("Select drafts first, or use Add all." if self.drafts else "There are no drafts to add yet.")
            return
        target = self.dataset_map.get(self.target_dataset.get())
        if not target:
            target = self.workspace.create_dataset("Synthetic collection " + now()[:10], "Examples from the synthesis workbench")
        added, duplicate, invalid = self.workspace.add(target, rows, "synthetic")
        kept = {row["id"] for row in rows}
        self.drafts = [row for row in self.drafts if row["id"] not in kept]
        self.save_drafts()
        self.refresh_datasets(target)
        self.target_dataset.set(self.dataset_var.get())
        self.status_var.set(f"Added {added:,} examples to {self.dataset_var.get()} · {duplicate:,} duplicates · {invalid:,} invalid")

    add_drafts_to_collection = keep_drafts

    def edit_draft(self):
        selected = self.draft_tree.selection()
        row = next((r for r in self.drafts if r["id"] == selected[0]), None) if selected else None
        if row is None:
            return
        dialog = ExampleEditor(self, row, "Review synthetic draft")
        self.wait_window(dialog)
        if not dialog.result:
            return
        try:
            quality = self.read_quality()
        except ValueError:
            quality = None
        merged, reason = apply_draft_edit(row, dialog.result, self.drafts, quality)
        if reason:
            messagebox.showwarning("This edit was not saved", f"The edited example would not pass the quality checks: {explain_reason(reason)}.", parent=self)
            return
        row.clear()
        row.update(merged)
        self.save_drafts()
        self._reselect([row["id"]])

    def discard_drafts(self):
        selected, rows = self._selected_drafts()
        if rows:
            self.drafts = [r for r in self.drafts if r["id"] not in selected]
            self.save_drafts()
            self.status_var.set(f"Discarded {len(rows):,} draft(s).")

    def discard_all_drafts(self):
        if self.drafts and messagebox.askyesno("Discard all drafts", f"Discard all {len(self.drafts):,} drafts? They have not been added to any collection.", parent=self):
            self.drafts = []
            self.save_drafts()

    def refresh_batches(self):
        if not hasattr(self, "batch_tree"):
            return
        self.batch_tree.delete(*self.batch_tree.get_children())
        self.batch_records = {}
        folder = self.workspace.root / "data" / "synthetic_batches"
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                stats = record.get("stats", {})
                self.batch_records[path.stem] = record
                self.batch_tree.insert("", "end", iid=path.stem, values=(record.get("createdAt", ""), record.get("config", {}).get("mode", ""), stats.get("accepted", 0), stats.get("rejected", 0)))
            except (ValueError, OSError):
                continue

    def inspect_batch(self):
        selected = self.batch_tree.selection()
        if selected:
            record = self.batch_records[selected[0]]
            set_text(self.batch_detail, summarize_batch(record) + "\n\n" + json.dumps(record, ensure_ascii=False, indent=2))

    def _build_blueprint(self):
        super()._build_blueprint()
        page = self.pages["blueprint"].body
        drawer = Collapsible(self.data_card, "Change or mix")
        drawer.pack(fill="x", pady=(10, 0))
        mixture = drawer.body
        label(mixture, "By default Start training trains on everything: every dataset, stage by stage (language, then instructions, code, "
                       "reasoning, conversation, knowledge), one model throughout. Rotate trains on one dataset per run, cycling through your library; "
                       "Choose by hand lets you pick the datasets for a run. The budget caps how many examples a run uses.",
              "CardMuted.TLabel", wraplength=950, justify="left").pack(anchor="w", pady=(0, 10))
        self.dataset_mode = tk.StringVar(value=self.workspace.state.get("datasetMode", "everything"))
        modes = ttk.Frame(mixture, style="Card.TFrame")
        modes.pack(fill="x")
        ttk.Radiobutton(modes, text="Train on everything", value="everything", variable=self.dataset_mode,
                        command=self.dataset_mode_changed).pack(side="left", padx=(0, 20))
        ttk.Radiobutton(modes, text="Rotate one dataset per run", value="rotate", variable=self.dataset_mode,
                        command=self.dataset_mode_changed).pack(side="left", padx=(0, 20))
        ttk.Radiobutton(modes, text="Choose by hand for this run", value="manual", variable=self.dataset_mode,
                        command=self.dataset_mode_changed).pack(side="left")
        self.rotation_text = tk.StringVar()
        label(mixture, textvariable=self.rotation_text, style="Card.TLabel", wraplength=950, justify="left").pack(anchor="w", pady=(8, 12))
        self.reviewed_only = tk.BooleanVar(value=False)
        self.use_all_records = tk.BooleanVar(value=False)
        self.balance_var = tk.StringVar(value="tokens")
        self.train_limit = tk.StringVar(value="10000")
        row = ttk.Frame(mixture, style="Card.TFrame")
        row.pack(fill="x")
        only =ttk.Checkbutton(row, text="Good examples only", variable=self.reviewed_only)
        only.pack(side="left", padx=(0, 20))
        with_help(row, "Example budget", "record_budget").pack(side="left", padx=(0, 8))
        ttk.Entry(row, textvariable=self.train_limit, width=10).pack(side="left", padx=(0, 12))
        everything = ttk.Checkbutton(row, text="Use all examples", variable=self.use_all_records)
        everything.pack(side="left", padx=(0, 20))
        balance_label = with_help(row, "Balance by", "balance")
        balance_label.pack(side="left", padx=(0, 8))
        balance_box = ttk.Combobox(row, textvariable=self.balance_var, values=["tokens", "records"], state="readonly", width=8)
        balance_box.pack(side="left")
        for widget in (only, everything, balance_label, balance_box):
            self.mark_advanced(widget)
        Tooltip(only, help_text("reviewed_only"))
        detail = label(mixture, "Budget: 1\u20131,000,000 records, or Use all examples for every example in the selected sources. Balance 'tokens' gives each source an equal share of text, so long-answer sets do not dominate; 'records' splits by count.\n"
                       "Good examples only: Good means you marked the example as approved. Downloaded datasets arrive unchecked, so a source with none marked Good is included anyway (with a warning) rather than blocking the run.", "CardMuted.TLabel", wraplength=950, justify="left")
        detail.pack(anchor="w", pady=(6, 8))
        self.mark_advanced(detail)
        self.mixture_weights = {}
        # Only shown in "Choose by hand" mode (see apply_dataset_mode).
        self.manual_frame = ttk.Frame(mixture, style="Card.TFrame")
        self.manual_frame.pack(fill="x")
        label(self.manual_frame, "Select the datasets for this run (Ctrl/Shift to pick several). With none selected, the working dataset from Curate is used.",
              "CardMuted.TLabel", wraplength=950, justify="left").pack(anchor="w", pady=(0, 6))
        self.mixture_tree = tree(self.manual_frame, [("name", "SOURCE \u00b7 CTRL/SHIFT TO SELECT MULTIPLE", 440), ("records", "EXAMPLES", 110), ("weight", "WEIGHT", 90)], 5)
        self.mixture_tree.bind("<<TreeviewSelect>>", lambda _: self.update_estimate())
        weight_row = ttk.Frame(self.manual_frame, style="Card.TFrame")
        weight_row.pack(fill="x", pady=(8, 0))
        self.weight_value = tk.StringVar(value="1")
        with_help(weight_row, "Selected source weight", "weight").pack(side="left", padx=(0, 8))
        ttk.Entry(weight_row, textvariable=self.weight_value, width=7).pack(side="left", padx=(0, 8))
        button(weight_row, "Apply weight", self.set_mixture_weight)
        button(weight_row, "Select all sources", lambda: self.mixture_tree.selection_set(self.mixture_tree.get_children()))
        self.mark_advanced(weight_row)
        self.mixture_report = tk.StringVar(value="Pick a model size below, then press Start training.")
        self.mixture_report_label = label(mixture, textvariable=self.mixture_report, style="CardMuted.TLabel", wraplength=950)
        self.mixture_report_label.pack(anchor="w", pady=(10, 0))
        self.apply_dataset_mode()

    # ---- which datasets a run uses: rotation by default, hand-picked on request
    def dataset_mode_changed(self):
        self.workspace.state["datasetMode"] = self.dataset_mode.get()
        self.workspace.save()
        self.apply_dataset_mode()

    def apply_dataset_mode(self):
        if not hasattr(self, "manual_frame"):
            return
        if self.dataset_mode.get() == "manual":
            self.manual_frame.pack(fill="x", before=self.mixture_report_label)
        else:
            self.manual_frame.pack_forget()
        self.refresh_rotation_text()
        self.update_estimate()

    def rotation_candidates(self):
        """Datasets with examples, in library order. Rotation cycles through these one run at a time."""
        return [d["id"] for d in self.workspace.state["datasets"] if self.dataset_stats(d["id"])["records"] > 0]

    def rotation_pick(self):
        ids = self.rotation_candidates()
        return ids[self.workspace.state.get("rotationIndex", 0) % len(ids)] if ids else None

    def refresh_rotation_text(self):
        if not hasattr(self, "rotation_text"):
            return
        if self.dataset_mode.get() == "manual":
            self.rotation_text.set("Choosing by hand: the next run uses the datasets selected below. Rotation resumes where it left off when you switch back.")
            return
        if self.dataset_mode.get() == "everything":
            count = len(self.rotation_candidates())
            self.rotation_text.set(f"Start training runs all {count} dataset(s) stage by stage; follow the stages in the Train on everything card on the Training page."
                                   if count else "No dataset has examples yet. Download or import one first.")
            return
        ids, pick = self.rotation_candidates(), self.rotation_pick()
        if not pick:
            self.rotation_text.set("No dataset has examples yet. Download or import one first.")
            return
        names = {d["id"]: d["name"] for d in self.workspace.state["datasets"]}
        after = ids[(ids.index(pick) + 1) % len(ids)]
        self.rotation_text.set(f"Next run: {names[pick]}  ({ids.index(pick) + 1} of {len(ids)})"
                               + (f"  ·  then {names[after]}" if len(ids) > 1 else ""))

    def run_sources(self):
        """Dataset ids a single run trains on: every dataset (Train on everything, e.g. a Network-page run),
        the rotation's next dataset, or the hand-picked ones."""
        mode = self.dataset_mode.get() if getattr(self, "dataset_mode", None) is not None else "rotate"
        if mode == "manual":
            selected = list(self.mixture_tree.selection())
            return selected or [key for key in [self.selected_dataset()] if key]
        if mode == "everything":
            return self.rotation_candidates()
        pick = self.rotation_pick()
        return [pick] if pick else []

    def refresh_mixture(self):
        selected = self.mixture_tree.selection()
        self.mixture_tree.delete(*self.mixture_tree.get_children())
        for dataset in self.workspace.state["datasets"]:
            key = dataset["id"]
            self.mixture_tree.insert("", "end", iid=key, values=(dataset["name"], f"{self.dataset_stats(key)['records']:,}", self.mixture_weights.get(key, 1)))
        self.mixture_tree.selection_set([key for key in selected if self.mixture_tree.exists(key)])

    def set_mixture_weight(self):
        try:
            value = float(self.weight_value.get())
            if not math.isfinite(value) or value < 0:
                raise ValueError("Weights must be finite and nonnegative.")
            for key in self.mixture_tree.selection():
                self.mixture_weights[key] = value
            self.refresh_mixture()
        except ValueError as exc:
            messagebox.showerror("Invalid source weight", str(exc), parent=self)

    def vocabulary_estimate(self, cfg, rows=5000):
        """(vocab size, description). Byte is fixed; a character vocabulary is fitted on >= `rows` rows across the sources."""
        if cfg["tokenizer"] == "byte":
            return Tokenizer("byte").vocab_size, "byte vocabulary"
        sources = self.run_sources()
        key = (tuple(sources), tuple(self.dataset_stats(source)["records"] for source in sources))
        cache = self.__dict__.setdefault("_vocab_cache", {})
        if key not in cache:
            share = max(1, -(-rows // max(1, len(sources))))
            sample = [row for source in sources for row in self.workspace.sample(source, limit=share, seed=0)]
            cache[key] = (Tokenizer.fit("character", sample).vocab_size if sample else 263, len(sample))
        size, used = cache[key]
        return size, f"character vocabulary sampled from {used:,} rows"

    def update_estimate(self):
        if not hasattr(self, "estimate_var"):
            return
        try:
            cfg = self.read_config()
            records = sum(self.dataset_stats(source)["records"] for source in self.run_sources())
            self._set_data_summary(records)
            vocab, basis = self.vocabulary_estimate(cfg)
            count = parameter_estimate(cfg, vocab)
            self._set_time_estimate(cfg, count)
            self.estimate_var.set(f"\u2248 {count:,} parameters \u00b7 {vocab:,} tokens ({basis}) \u00b7 {records:,} source examples\n\u2248 {count * 16 / 1024**2:.1f} MB for weights, gradients and Adam state + activations")
        except (ValueError, TypeError) as exc:
            self.estimate_var.set(str(exc))
            self.time_var.set("")

    def start_training(self, config=None, train_options=None, navigate=True, persist_config=True, on_finished=None, sources=None):
        """Train on a bounded mixture. The Network page passes its own config, train() options and a finish hook.
        `sources` overrides the dataset choice for this one run (Quick start); otherwise rotation or the hand-picked set decides."""
        if self.busy():
            return False
        if sources is None and config is None and self.dataset_mode.get() == "everything" and hasattr(self, "start_curriculum"):
            # The Start training button in Train on everything mode runs the stage-by-stage plan.
            if navigate:
                self.show_page("training")
            self.start_curriculum()
            return self.job_kind == "curriculum"
        try:
            cfg = config if config is not None else self.read_config()
            rotating = sources is None and self.dataset_mode.get() == "rotate"
            identifiers = [key for key in (sources if sources is not None else self.run_sources()) if key]
            if not identifiers:
                raise ValueError("No dataset to train on. Download or import one first, or choose datasets by hand.")
            use_all = self.use_all_records.get()
            limit = 1 if use_all else int(self.train_limit.get())
            if not 1 <= limit <= 1000000:
                raise ValueError("Record budget must be 1–1,000,000, or tick All records.")
            reviewed_only = self.reviewed_only.get()
            balance = self.balance_var.get()
            weights = dict(self.mixture_weights)
            options = dict(train_options or {})
            if persist_config:
                self.workspace.state["studioConfig"] = cfg
                self.workspace.save()
            directory = self.workspace.run_dir / uid("run")
            self.history = []
            self._run_cfg = cfg
            self.train_plain.set("Preparing your data\u2026 the chart starts moving in a moment.")
            self.draw_loss()
            self.progress.configure(value=0, maximum=cfg["steps"])
            for var in self.metric_vars.values():
                var.set("\u2014")
            set_text(self.run_log, "Preparing a bounded training mixture\u2026\n")
            if navigate:
                self.show_page("training")
            def work():
                plan = build_training_set(self.workspace, identifiers, max_records=limit, seed=cfg["seed"], reviewed_only=reviewed_only,
                                          weights=weights, balance=balance, use_all=use_all, stop=self.stop, emit=lambda event: self.events.put(("training", event)))
                self.events.put(("mixture_ready", plan["report"]))
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / "mixture.json", plan["report"])
                return train(cfg, plan["examples"], directory, lambda event: self.events.put(("training", event)), self.stop, self.pause, **options)
            def finished(record):
                self.refresh_runs()
                if on_finished:
                    on_finished(directory, record)
            started = self.start_job("training", work, finished)
            if started and rotating:
                self.workspace.state["rotationIndex"] = self.workspace.state.get("rotationIndex", 0) + 1
                self.workspace.save()
                self.refresh_rotation_text()
            return started
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Check your training plan", str(exc), parent=self)
            return False

    @staticmethod
    def mixture_summary(report):
        """One-paragraph description of a prepared mixture: per-source counts, cleaning and warnings."""
        parts = [f"{info['name']} {info['selected']:,}/{info['records']:,}" + (f" ({info['character_share']:.0%} of text)" if "character_share" in info else "")
                 for info in report.get("sources", {}).values()]
        cleaning = report.get("cleaning", {})
        text = (f"Prepared {report.get('selected', 0):,} examples · {report.get('duplicates', 0):,} duplicates removed · "
                f"{cleaning.get('annotated_rows', 0):,} answers cleaned · {cleaning.get('empty_answers_dropped', 0):,} empty dropped · balance {report.get('balance', 'records')}\n"
                + " · ".join(parts))
        if report.get("warnings"):
            text += "\nWarning: " + " ".join(report["warnings"])
        return text + "\nmixture.json saved with this run"

    def poll(self):
        # Dispatch workflow-specific messages, then leave model/chat events to the base loop.
        deferred = []
        try:
            import queue
            for _ in range(200):
                kind, value = self.events.get_nowait()
                if kind in {"synth_progress", "catalog_progress"}:
                    text = value.get("text", "Working…")
                    self.status_var.set(text)
                    if kind == "synth_progress":
                        self.synth_summary.set(text)
                    else:
                        self.catalog_status.set(text)
                    total = value.get("total") or 0
                    completed = value.get("completed", 0)
                    if total:
                        self.task_progress.configure(value=min(100, completed / total * 100))
                elif kind == "catalog_registered":
                    self.refresh_datasets(value)
                elif kind == "mixture_ready":
                    self.mixture_report.set(self.mixture_summary(value))
                    self.log(self.mixture_report.get())
                else:
                    deferred.append((kind, value))
        except queue.Empty:
            pass
        for item in deferred:
            self.events.put(item)
        if self.job_kind is None:
            self.task_progress.configure(value=0)
        super().poll()

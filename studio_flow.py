"""Guided native workflow: acquire, curate, blend, train and improve."""
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


class WorkflowMixin:
    PAGES = [("home", "↗", "Start here", "Four steps, all on this computer. Follow the list; optional steps can be skipped."),
             ("catalog", "01", "Get data", "Download published training datasets. Their licenses and sources stay attached."),
             ("datasets", "02", "Your data", "Read your examples, fix mistakes, and mark the good ones."),
             ("blueprint", "03", "Training setup", "Choose your data and a model size, then start training."),
             ("network", "04", "Network designer", "Advanced: snap together the model's blocks, watch signals flow, and version every change."),
             ("training", "05", "Training", "Watch the model learn. Lower error is better."),
             ("align", "06", "Improve with feedback", "Optional: pick the better reply or write the right one (or let a teacher do it) to polish the model."),
             ("chat", "07", "Chat", "Talk to your model. If a reply is wrong, send it to Improve with feedback.")]

    def __init__(self, workspace=None):
        self.record_page = 0
        self.record_page_size = 100
        self.search_after = None
        self._stats_cache = {}
        super().__init__(workspace)
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
        self.next_action = ttk.Button(self.flow_bar, text="Discover data →", command=self.next_step, style="Accent.TButton")
        self.next_action.pack(side="right")
        self.task_progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.task_progress.pack(side="bottom", fill="x", before=self.page_host, pady=(6, 0))
        self._build_home()
        self._build_catalog()

    NEXT_LABELS = {"home": "", "catalog": "Review your data \u2192", "datasets": "Set up training \u2192",
                   "blueprint": "Design the network \u2192", "network": "View training \u2192", "training": "Improve with feedback \u2192",
                   "align": "Chat with your model \u2192", "chat": "Add more data \u2192"}

    def next_page_of(self, name):
        mapping = {"home": "catalog", "catalog": "datasets", "datasets": "blueprint",
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
        self.home_metrics = {key: tk.StringVar(value="0") for key in ("datasets", "records", "runs")}
        self.home_totals, self.home_runs = tk.StringVar(), tk.StringVar()
        ttk.Button(links, textvariable=self.home_totals, command=lambda: self.show_page("datasets")).pack(side="left", padx=(0, 8))
        ttk.Button(links, textvariable=self.home_runs, command=lambda: self.show_page("training")).pack(side="left")
        self.home_hint = tk.StringVar()
        label(page, textvariable=self.home_hint, style="Muted.TLabel", wraplength=950, justify="left").pack(anchor="w", pady=12)

    @staticmethod
    def home_steps():
        return [{"key": "data", "title": "Get data", "desc": "Download a starter collection or import your own examples.", "page": "catalog", "action": "Get data", "optional": False},
                {"key": "train", "title": "Train", "desc": "Pick a model size and let it learn from your data.", "page": "blueprint", "action": "Set up training", "optional": False},
                {"key": "improve", "title": "Improve with feedback", "desc": "Pick the better reply or write the right one to polish the model.", "page": "align", "action": "Give feedback", "optional": True},
                {"key": "chat", "title": "Chat", "desc": "Talk to your model.", "page": "chat", "action": "Open chat", "optional": False}]

    def home_progress(self):
        """Which checklist steps are done, derived from what exists in the workspace. Returns [{key, done, ...}] in order."""
        datasets = self.workspace.state["datasets"]
        runs = self.workspace.runs()
        done = {"data": any(self.dataset_stats(d["id"])["records"] > 0 for d in datasets),
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
                  "runs": len([r for r in runs if r.get("stage") != "dpo"])}
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
        label(self.data_card, textvariable=self.rotation_text, style="CardMuted.TLabel", wraplength=950, justify="left").pack(anchor="w", pady=(4, 0), before=drawer)
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
            mode = {"everything": "Train on everything", "rotate": "One dataset per run", "manual": "Chosen by hand"}[self.dataset_mode.get()]
            self.data_summary.set(f"{mode}  \u00b7  {records:,} examples")
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
                if kind == "catalog_progress":
                    text = value.get("text", "Working…")
                    self.status_var.set(text)
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

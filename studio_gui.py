"""Native desktop workspace with NumPy CPU and optional CuPy CUDA compute."""
from __future__ import annotations

import copy
import json
import os
import queue
import threading
import traceback
import zipfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from studio_model import DEFAULT_CONFIG, LocalModel, Tokenizer, load_checkpoint, parameter_estimate, train, validate_config
from studio_backend import gpu_status
from studio_core import ROOT, Workspace, fingerprint, messages_of, normalize, now, synthetic_recipe, uid

from studio_widgets import (BG, PANEL, INPUT, EDGE, TEXT, MUTED, ACCENT, BLUE, FONT,
                            label, button, text_box, content, set_text, card, field, tree, ScrollPage, ExampleEditor,
                            Tooltip, help_icon, with_help, help_text)
from studio_align import Collapsible
from studio_ux import SIZE_PRESETS, device_kind, estimate_seconds, format_duration, plain_training_status


class NativeStudio(tk.Tk):
    PAGES = [("datasets", "01", "Dataset library", "Import, write, and review the examples your model will learn from."),
             ("synthesis", "02", "Synthesis lab", "Create reproducible training data. Inspect every draft before keeping it."),
             ("blueprint", "03", "Model blueprint", "Define your own transformer. Every new model starts with random weights."),
             ("training", "04", "Training console", "Watch real optimization steps, compare losses, and save your progress."),
             ("chat", "05", "Playground", "Talk to your trained checkpoint and turn corrections into better data.")]

    def __init__(self, workspace=None):
        super().__init__()
        self.title("Groundwork · AI Studio")
        self.geometry("1420x960")
        self.minsize(1160, 780)
        self.configure(bg=BG)
        self.workspace = workspace or Workspace()
        self.events = queue.Queue()
        self.worker = None
        self.job_kind = None
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.closing = False
        self.drafts = copy.deepcopy(self.workspace.state.get("generated", []))
        self.history = []
        self.chat_messages = []
        self.chat_model = None
        self.last_prompt = ""
        self.last_answer = ""
        self.dataset_var = tk.StringVar()
        self.dataset_map = {}
        self.dataset_boxes = []
        self.status_var = tk.StringVar(value="Ready · local workspace")
        self.inference_device = tk.StringVar(value="auto")
        self.gpu_info = None
        self.compute_status = tk.StringVar(value="Checking CUDA support in the background…")
        self._style()
        self._layout()
        self._build_datasets()
        self._build_synthesis()
        self._build_blueprint()
        self._build_training()
        self._build_chat()
        self.refresh_datasets()
        self.refresh_runs()
        self.refresh_drafts()
        self.show_page("datasets")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(60, self.poll)
        threading.Thread(target=self.probe_gpu, name="studio-gpu-probe", daemon=True).start()

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, font=(FONT, 10), bordercolor=EDGE, focuscolor=ACCENT)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=(FONT, 24, "bold"))
        style.configure("Card.TLabel", background=PANEL)
        style.configure("CardMuted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("CardTitle.TLabel", background=PANEL, font=(FONT, 12, "bold"))
        style.configure("Metric.TLabel", background=PANEL, foreground=ACCENT, font=(FONT, 23, "bold"))
        style.configure("TButton", background=EDGE, foreground=TEXT, borderwidth=0, padding=(13, 9))
        style.map("TButton", background=[("active", "#354763"), ("disabled", PANEL)], foreground=[("disabled", MUTED)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#081c19", font=(FONT, 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#a0f0d8"), ("disabled", EDGE)])
        style.configure("Nav.TButton", background=BG, foreground=MUTED, anchor="w", padding=(12, 11), font=(FONT, 11))
        style.map("Nav.TButton", background=[("active", PANEL)])
        style.configure("ActiveNav.TButton", background=PANEL, foreground=ACCENT, anchor="w", padding=(12, 11), font=(FONT, 11, "bold"))
        style.configure("TEntry", fieldbackground=INPUT, foreground=TEXT, insertcolor=ACCENT, borderwidth=1, padding=7)
        style.configure("TCombobox", fieldbackground=INPUT, background=EDGE, foreground=TEXT, arrowcolor=MUTED, padding=7)
        style.map("TCombobox", fieldbackground=[("readonly", INPUT)], foreground=[("readonly", TEXT)], selectbackground=[("readonly", INPUT)], selectforeground=[("readonly", TEXT)])
        style.configure("Treeview", background=INPUT, fieldbackground=INPUT, foreground=TEXT, rowheight=36, borderwidth=0)
        style.map("Treeview", background=[("selected", "#27463f")], foreground=[("selected", "#dcfff2")])
        style.configure("Treeview.Heading", background=PANEL, foreground=MUTED, font=(FONT, 10, "bold"), padding=9, relief="flat")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(15, 10))
        style.map("TNotebook.Tab", background=[("selected", EDGE)], foreground=[("selected", TEXT)], padding=[("selected", (15, 10))])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=INPUT, borderwidth=0)
        style.configure("TScrollbar", background=EDGE, troughcolor=BG, arrowcolor=MUTED, borderwidth=0)
        self.option_add("*TCombobox*Listbox.background", INPUT)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", EDGE)

    def _layout(self):
        sidebar = self.sidebar = ttk.Frame(self, padding=(16, 22), width=276)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        label(sidebar, "G /", foreground=ACCENT, font=(FONT, 32, "bold")).pack(anchor="w")
        label(sidebar, "GROUNDWORK", font=(FONT, 16, "bold")).pack(anchor="w", pady=(2, 0))
        label(sidebar, "AI STUDIO", "Muted.TLabel", font=(FONT, 10)).pack(anchor="w", pady=(0, 22))
        self.nav = {}
        sections = {"catalog": "DATA", "blueprint": "MODEL", "align": "IMPROVE"}
        for key, number, title, _ in self.PAGES:
            if key in sections:
                label(sidebar, sections[key], "Muted.TLabel", font=(FONT, 9, "bold")).pack(anchor="w", padx=12, pady=(14, 2))
            nav = ttk.Button(sidebar, text=self.nav_label(key), style="Nav.TButton", command=lambda k=key: self.show_page(k))
            nav.pack(fill="x", pady=2)
            self.nav[key] = nav
        footer = self.sidebar_footer = ttk.Frame(sidebar)
        footer.pack(side="bottom", fill="x")
        self._build_sidebar_footer(footer)
        divider = tk.Frame(self, bg=EDGE, width=1)
        divider.pack(side="left", fill="y")
        main = ttk.Frame(self, padding=(26, 23, 22, 12))
        main.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(main)
        header.pack(fill="x", pady=(0, 14))
        self.page_title = label(header, "", "Title.TLabel")
        self.page_title.pack(anchor="w")
        self.page_description = label(header, "", "Muted.TLabel")
        self.page_description.pack(anchor="w", pady=(7, 0))
        # The status line is packed first (at the bottom) so a tall page can never squeeze it out of view.
        label(main, textvariable=self.status_var, style="Muted.TLabel").pack(side="bottom", fill="x", pady=(8, 0))
        self.page_host = ttk.Frame(main)
        self.page_host.pack(fill="both", expand=True)
        self.page_host.rowconfigure(0, weight=1)
        self.page_host.columnconfigure(0, weight=1)
        self.pages = {}
        for key, *_ in self.PAGES:
            frame = ScrollPage(self.page_host) if key in {"home", "catalog", "blueprint", "synthesis", "align", "training"} else ttk.Frame(self.page_host)
            frame.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = frame

    # Beginner/Advanced hooks: UXMixin overrides these; a bare NativeStudio simply shows everything.
    def mark_advanced(self, widget, **info):
        return widget

    def mark_beginner_only(self, widget, **info):
        return widget

    def is_advanced(self):
        return True

    def hidden_pages(self):
        return set()

    def nav_label(self, key):
        """Sidebar text. Numbers count only the pages that are currently available."""
        hidden, count = self.hidden_pages(), 0
        for page, number, title, _ in self.PAGES:
            if page in hidden:
                if page == key:
                    return title
                continue
            if number != "↗":
                count += 1
            if page == key:
                return f"{count:02d}   {title}" if number != "↗" else f"{number}   {title}"
        return key

    def _build_sidebar_footer(self, footer):
        label(footer, "BUILT FROM ZERO", foreground=ACCENT, font=(FONT, 9, "bold")).pack(anchor="w")
        label(footer, "Your data. Your weights.", "Muted.TLabel", justify="left").pack(anchor="w", pady=(4, 0))

    def show_page(self, name):
        self.pages[name].tkraise()
        for key, _, title, description in self.PAGES:
            self.nav[key].configure(style="ActiveNav.TButton" if key == name else "Nav.TButton")
            if key == name:
                self.page_title.configure(text=title)
                self.page_description.configure(text=description)
        if name == "blueprint":
            self.update_estimate()

    def dataset_picker(self, parent):
        widget = ttk.Combobox(parent, textvariable=self.dataset_var, state="readonly", width=35)
        widget.pack(side="left", padx=(0, 10), pady=4)
        widget.bind("<<ComboboxSelected>>", lambda _: self.select_dataset())
        self.dataset_boxes.append(widget)
        return widget

    def selected_dataset(self):
        return self.dataset_map.get(self.dataset_var.get())

    def select_dataset(self):
        self.workspace.state["selectedDatasetId"] = self.selected_dataset()
        self.workspace.save()
        self.refresh_examples()
        self.update_estimate()
        self.refresh_correction_note()

    def refresh_datasets(self, selected=None):
        selected = selected or self.selected_dataset() or self.workspace.state.get("selectedDatasetId")
        self.dataset_map = self.dataset_labels()
        keys = list(self.dataset_map)
        for widget in self.dataset_boxes:
            widget.configure(values=keys)
        self.dataset_var.set(next((key for key, value in self.dataset_map.items() if value == selected), keys[0] if keys else ""))
        self.refresh_examples()
        self.update_estimate()
        self.refresh_correction_note()

    def dataset_labels(self):
        """Display names for the dataset pickers. A short id suffix appears only when two datasets share a name."""
        from collections import Counter
        counts = Counter(d["name"] for d in self.workspace.state["datasets"])
        return {(d["name"] if counts[d["name"]] == 1 else f"{d['name']}  ·  {d['id'][-6:]}"): d["id"] for d in self.workspace.state["datasets"]}

    def drilldown(self, row, text, panel):
        """Quiet button in `row` that shows/hides `panel` (packed right after the row). Set `button.base` to retitle it."""
        def sync():
            button_.configure(text=("\u25be " if panel.winfo_manager() else "\u25b8 ") + button_.base)
        def toggle():
            if panel.winfo_manager():
                panel.pack_forget()
            else:
                panel.pack(fill="x", pady=(8, 0), after=row)
            sync()
        button_ = ttk.Button(row, command=toggle)
        button_.base, button_.sync = text, sync
        button_.pack(side="left", padx=(0, 8), pady=4)
        sync()
        return button_

    def _build_datasets(self):
        page = self.pages["datasets"]
        head = ttk.Frame(page)
        head.pack(fill="x", pady=(0, 10))
        self.dataset_picker(head)
        self.dataset_summary = tk.StringVar()
        label(head, textvariable=self.dataset_summary, style="Muted.TLabel").pack(side="left", padx=4)
        button(head, "Train on this data \u2192", lambda: self.show_page("blueprint"), True).pack_configure(side="right", padx=0)
        tools = ttk.Frame(page)
        tools.pack(fill="x", pady=(0, 10))
        self.search = tk.StringVar()
        label(tools, "Search").pack(side="left", padx=(0, 8))
        ttk.Entry(tools, textvariable=self.search, width=40).pack(side="left", padx=(0, 12))
        self.search.trace_add("write", lambda *_: self.refresh_examples())
        button(tools, "Edit", self.edit_example)
        Tooltip(button(tools, "Mark as Good", self.mark_reviewed), help_text("review_good"))
        button(tools, "Remove", self.remove_examples)
        self.examples_tree = tree(page, [("prompt", "PROMPT (or raw text)", 350), ("answer", "ANSWER", 350), ("intent", "CATEGORY", 100), ("review", "QUALITY", 80)])
        self.examples_tree.bind("<Double-1>", lambda _: self.edit_example())
        # quiet drill-downs: everything else about the dataset lives behind these
        links = self.dataset_links = ttk.Frame(page)
        links.pack(fill="x", pady=(12, 0))
        button(links, "+ Write example", self.new_example)
        button(links, "Import files", self.import_files)
        stats = ttk.Frame(page)
        overview = ttk.Frame(stats)
        overview.pack(fill="x")
        self.audit_vars = {}
        for key, title, helper in [("records", "EXAMPLES", "example"), ("review", "NOT YET CHECKED", "review_good"), ("characters", "CHARACTERS", "token"), ("duplicates", "DUPLICATES", "duplicates")]:
            box = card(overview)
            box.pack(side="left", fill="x", expand=True, padx=(0, 10))
            value = tk.StringVar(value="0")
            self.audit_vars[key] = value
            label(box, textvariable=value, style="Metric.TLabel").pack(anchor="w")
            with_help(box, title, helper, font=(FONT, 9)).pack(anchor="w", pady=(4, 0))
        hint = ttk.Frame(stats)
        hint.pack(fill="x", pady=(8, 0))
        label(hint, "Importing: each example needs a prompt and an answer (JSONL, JSON, CSV, TSV or TXT). Duplicates are skipped.", "Muted.TLabel").pack(side="left")
        help_icon(hint, "import_format").pack(side="left", padx=(4, 18))
        label(hint, "Quality: Review = not checked yet, Good = you approved it.", "Muted.TLabel").pack(side="left")
        help_icon(hint, "review_good").pack(side="left", padx=(4, 0))
        manage = ttk.Frame(page)
        button(manage, "+ New dataset", self.new_dataset)
        self.mark_advanced(button(manage, "Rename", self.rename_dataset))
        self.mark_advanced(button(manage, "Export", self.export_dataset))
        delete = button(manage, "Delete dataset", self.delete_dataset)
        self.mark_advanced(delete)
        self.drilldown(links, "Stats and checks", stats)
        self.drilldown(links, "Rename, export or delete", manage)
        label(links, "Double-click an example to read or edit it.", "Muted.TLabel").pack(side="right")

    def dataset_line(self):
        a = self.audit_vars
        return f"{a['records'].get()} examples \u00b7 {a['review'].get()} not yet checked"

    def refresh_examples(self):
        if not hasattr(self, "examples_tree"):
            return
        self.examples_tree.delete(*self.examples_tree.get_children())
        query = self.search.get().casefold()
        for example in self.workspace.examples(self.selected_dataset()):
            if query and query not in (example.get("user", "") + " " + example.get("assistant", "") + " " + example.get("intent", "")).casefold():
                continue
            self.examples_tree.insert("", "end", iid=example["id"], values=(example.get("user", "").replace("\n", " ")[:180], example.get("assistant", "").replace("\n", " ")[:180], example.get("intent", "general"), example.get("quality", "Review")))
        stats = self.workspace.audit(self.selected_dataset())
        for key, variable in self.audit_vars.items():
            variable.set(f"{stats[key]:,}")
        self.dataset_summary.set("\u00b7 " + self.dataset_line())

    def new_dataset(self):
        name = simpledialog.askstring("Create dataset", "Dataset name", parent=self)
        if name and name.strip():
            self.refresh_datasets(self.workspace.create_dataset(name))

    def rename_dataset(self):
        dataset = next((d for d in self.workspace.state["datasets"] if d["id"] == self.selected_dataset()), None)
        if dataset:
            name = simpledialog.askstring("Rename dataset", "Dataset name", initialvalue=dataset["name"], parent=self)
            if name and name.strip():
                dataset["name"] = name.strip()
                self.workspace.save()
                self.refresh_datasets()

    def delete_dataset(self):
        selected = self.selected_dataset()
        if selected and messagebox.askyesno("Delete dataset", "Delete this dataset and all its examples? Saved run snapshots are kept.", parent=self):
            self.workspace.remove_dataset(selected)
            self.refresh_datasets()

    def import_files(self):
        paths = filedialog.askopenfilenames(parent=self, title="Select training datasets", filetypes=[("Training data", "*.jsonl *.ndjson *.json *.csv *.tsv *.txt"), ("All files", "*.*")])
        if paths:
            report = self.workspace.import_files(paths, self.selected_dataset())
            self.refresh_datasets()
            messagebox.showinfo("Import results", "\n".join(report), parent=self)

    def export_dataset(self):
        if not self.selected_dataset():
            return
        path = filedialog.asksaveasfilename(parent=self, title="Export dataset", defaultextension=".jsonl", filetypes=[("JSON Lines", "*.jsonl"), ("CSV", "*.csv")])
        if path:
            try:
                self.workspace.export(self.selected_dataset(), path)
                self.status_var.set(f"Dataset exported · {path}")
            except (ValueError, OSError) as exc:
                messagebox.showerror("Export failed", str(exc), parent=self)

    def new_example(self):
        if not self.selected_dataset():
            self.new_dataset()
        if not self.selected_dataset():
            return
        dialog = ExampleEditor(self)
        self.wait_window(dialog)
        if dialog.result:
            added, duplicates, _ = self.workspace.add(self.selected_dataset(), [dialog.result])
            self.status_var.set(f"{added} example saved · {duplicates} duplicate skipped")
            self.refresh_datasets()

    def edit_example(self):
        selection = self.examples_tree.selection()
        if not selection:
            return
        example = next(e for e in self.workspace.state["examples"] if e["id"] == selection[0])
        dialog = ExampleEditor(self, example, "Edit training example")
        self.wait_window(dialog)
        if dialog.result:
            others = [e for e in self.workspace.examples(example["datasetId"]) if e["id"] != example["id"]]
            if any(fingerprint(e) == fingerprint(dialog.result) for e in others):
                messagebox.showerror("Duplicate example", "This exact example already exists in the dataset.", parent=self)
                return
            example.clear()
            example.update(dialog.result)
            self.workspace.save()
            self.refresh_datasets()

    def mark_reviewed(self):
        selected = set(self.examples_tree.selection())
        for e in self.workspace.state["examples"]:
            if e["id"] in selected:
                e.update(quality="Good", updatedAt=now())
        self.workspace.save()
        self.refresh_examples()

    def remove_examples(self):
        selected = set(self.examples_tree.selection())
        if selected and messagebox.askyesno("Remove examples", f"Remove {len(selected)} selected examples?", parent=self):
            self.workspace.state["examples"] = [e for e in self.workspace.state["examples"] if e["id"] not in selected]
            self.workspace.save()
            self.refresh_datasets()

    def _build_blueprint(self):
        page = self.pages["blueprint"].body
        cfg = dict(DEFAULT_CONFIG, **{k: v for k, v in self.workspace.state.get("studioConfig", {}).items() if k in DEFAULT_CONFIG})
        self.config_vars = {key: tk.StringVar(value=str(value)) for key, value in cfg.items()}
        self.data_card = card(page, "Training data")
        self.data_card.pack(fill="x", pady=(0, 14))
        self.data_summary = tk.StringVar()
        label(self.data_card, textvariable=self.data_summary, style="Card.TLabel", font=(FONT, 12, "bold")).pack(anchor="w")
        top = card(page, "Model size")
        top.pack(fill="x", pady=(0, 14))
        grid = ttk.Frame(top, style="Card.TFrame")
        grid.pack(fill="x")
        for i, (name, values, blurb) in enumerate(SIZE_PRESETS):
            grid.columnconfigure(i, weight=1, uniform="size")
            cell = ttk.Frame(grid, style="Card.TFrame")
            cell.grid(row=0, column=i, sticky="nsew", padx=(0, 14))
            ttk.Button(cell, text=name, command=lambda v=values: self.apply_preset(v)).pack(anchor="w")
            label(cell, blurb, "CardMuted.TLabel", wraplength=300, justify="left").pack(anchor="w", pady=(8, 0))
        hint = ttk.Frame(top, style="Card.TFrame")
        hint.pack(fill="x", pady=(12, 0))
        label(hint, "Not sure? Pick Small. All of the model's numbers start random and are learned from your data.", "CardMuted.TLabel").pack(side="left")
        help_icon(hint, "preset").pack(side="left", padx=(4, 0))
        launch = card(page)
        launch.pack(fill="x", pady=(0, 14))
        self.estimate_var = tk.StringVar()
        self.time_var = tk.StringVar()
        label(launch, textvariable=self.time_var, style="Card.TLabel", font=(FONT, 13, "bold"), foreground=ACCENT).pack(anchor="w", pady=(0, 6))
        label(launch, textvariable=self.estimate_var, style="CardMuted.TLabel", justify="left").pack(anchor="w")
        action = ttk.Frame(launch, style="Card.TFrame")
        action.pack(fill="x", pady=(10, 0))
        button(action, "Start training", self.start_training, True)
        self.mark_advanced(button(action, "Save blueprint", self.save_blueprint))
        self.mark_advanced(button(action, "Design the layers yourself", lambda: self.show_page("network")))
        unlock = label(launch, "Want to design the layers yourself? Switch to Advanced (bottom-left) to unlock the Network designer.", "Hint.TLabel", background=PANEL)
        unlock.pack(anchor="w", pady=(10, 0))
        self.mark_beginner_only(unlock)

        def drawer(title, advanced=False):
            box = Collapsible(page, title)
            box.pack(fill="x", pady=(0, 8))
            return self.mark_advanced(box) if advanced else box

        def fields(parent, settings, columns=4):
            grid = ttk.Frame(parent, style="Card.TFrame")
            grid.pack(fill="x")
            for col in range(columns):
                grid.columnconfigure(col, weight=1)
            for i, (key, name, options, width, helper) in enumerate(settings):
                frame, widget = field(grid, name, self.config_vars[key], options, width, help=helper)
                frame.grid(row=i // columns, column=i % columns, sticky="ew", padx=(0, 15), pady=(0, 14))
                widget.bind("<FocusOut>", lambda _: self.update_estimate())
                if options:
                    widget.bind("<<ComboboxSelected>>", lambda _: self.update_estimate())

        fields(drawer("Name and training steps").body, [("name", "Model name", None, 30, "model_name"), ("steps", "Training steps (more = learns longer)", None, 14, "steps")])
        shape = [("tokenizer", "Tokenizer (how text is split)", ["byte", "character"]), ("width", "Model width (embedding size)", None),
                 ("layers", "Layers (transformer blocks)", None), ("heads", "Attention heads", None), ("context", "Context length (tokens)", None),
                 ("expansion", "Feed-forward size (MLP expansion)", None), ("dropout", "Dropout (0 to 0.5)", None)]
        learning = [("batch_size", "Batch size (examples per step)", None), ("accumulation", "Gradient accumulation", None), ("learning_rate", "Learning rate (step size)", None),
                    ("weight_decay", "Weight decay", None), ("warmup", "Warmup steps", None), ("eval_every", "Test every N steps (validation)", None),
                    ("validation", "Validation fraction (held-out test data)", None), ("seed", "Random seed", None),
                    ("objective", "Train on (objective)", ["assistant", "all"]), ("windowing", "Windows (how text is cut)", ["anchored", "sequential"])]
        fields(drawer("Model shape", True).body, [(k, n, o, 18, k) for k, n, o in shape])
        box = drawer("Learning settings", True)
        fields(box.body, [(k, n, o, 18, k) for k, n, o in learning])
        label(box.body, "Character vocabulary is fitted to the training split. Byte vocabulary covers UTF-8.\n'assistant' learns responses (and all raw-text tokens); 'all' predicts every token. Anchored windows end at an answer and carry as much of its prompt as fits; sequential cuts independent chunks.", "CardMuted.TLabel").pack(anchor="w")
        box = drawer("Device")
        frame, selector = field(box.body, "Device (auto uses the GPU if it works)", self.config_vars["device"], ["auto", "cpu", "cuda"], 20, help="device")
        frame.pack(anchor="w", pady=(0, 10))
        selector.bind("<<ComboboxSelected>>", lambda _: self.update_estimate())
        label(box.body, "auto prefers a supported NVIDIA GPU and falls back to CPU. cuda requires a working CuPy installation and NVIDIA driver.", "CardMuted.TLabel").pack(anchor="w")
        label(box.body, textvariable=self.compute_status, style="Card.TLabel", justify="left", wraplength=880).pack(anchor="w", pady=(8, 0))

    def _set_data_summary(self, count):
        self.data_summary.set(f"{self.dataset_var.get() or 'No dataset selected'}  \u00b7  {count:,} examples")

    def read_config(self):
        return validate_config({key: variable.get() for key, variable in self.config_vars.items()})

    def probe_gpu(self):
        # CUDA initialization can take several seconds. Only poll() touches Tk.
        try:
            status = gpu_status()
        except Exception as exc:
            status = {"available": False, "name": "", "reason": str(exc)}
        self.events.put(("gpu_status", status))

    def device_description(self, requested):
        if requested == "cpu":
            return "NumPy CPU"
        if self.gpu_info is None:
            return "CUDA detection pending" if requested == "cuda" else "Auto · detecting GPU"
        if self.gpu_info["available"]:
            return f"CUDA · {self.gpu_info['name']}"
        return "CUDA unavailable" if requested == "cuda" else "Auto · NumPy CPU"

    def update_gpu_status(self, status):
        self.gpu_info = status
        if status["available"]:
            self.compute_status.set(f"CUDA ready · {status['name']}\nTraining and inference can both use this GPU.")
        else:
            reason = status.get("reason") or "No supported GPU was detected."
            self.compute_status.set(f"CPU available · CUDA unavailable: {reason}\nOptional GPU dependencies: python -m pip install -r requirements-gpu.txt")
        self.update_estimate()

    def apply_preset(self, values):
        for key, value in values.items():
            self.config_vars[key].set(str(value))
        self.update_estimate()

    def _set_time_estimate(self, cfg, count):
        """Training-time line on the setup page: a heuristic per device, self-calibrated from finished runs."""
        kind = device_kind(cfg["device"], self.gpu_info)
        scale = (self.workspace.state.get("speedScale") or {}).get(kind, 1.0)
        seconds = estimate_seconds(cfg, count, kind, scale)
        if self.gpu_info is None and cfg["device"] != "cpu":
            where = "your computer (GPU check still running)"
        else:
            where = self.device_description(cfg["device"])
        self.time_var.set(f"Estimated training time: {format_duration(seconds)}  \u00b7  {cfg['steps']:,} steps on {where}")
        return seconds

    def update_estimate(self):
        if not hasattr(self, "estimate_var"):
            return
        try:
            cfg = self.read_config()
            examples = self.workspace.examples(self.selected_dataset())
            vocab = Tokenizer.fit(cfg["tokenizer"], examples).vocab_size if examples else 263
            self._set_data_summary(len(examples))
            count = parameter_estimate(cfg, vocab)
            self._set_time_estimate(cfg, count)
            self.estimate_var.set(f"\u2248 {count:,} parameters  \u00b7  {vocab:,} tokens before validation split  \u00b7  {len(examples):,} examples\n\u2248 {count * 16 / 1024**2:.1f} MB for weights, gradients and Adam state + activations")
        except (ValueError, TypeError) as exc:
            self.estimate_var.set(str(exc))
            self.time_var.set("")

    def save_blueprint(self):
        try:
            self.workspace.state["studioConfig"] = self.read_config()
            self.workspace.save()
            self.status_var.set("Blueprint saved")
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Check the blueprint", str(exc), parent=self)

    def _build_training(self):
        page = self.pages["training"].body
        status = card(page)
        status.pack(fill="x", pady=(0, 12))
        self.train_plain = tk.StringVar(value=plain_training_status([]))
        label(status, textvariable=self.train_plain, style="Card.TLabel", wraplength=1000, justify="left", font=(FONT, 12)).pack(anchor="w")
        self.progress = ttk.Progressbar(page, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 12))
        metrics = ttk.Frame(page)
        metrics.pack(fill="x", pady=(0, 12))
        details = Collapsible(page, "Speed and run details")
        extra = ttk.Frame(details.body)
        extra.pack(fill="x", pady=(0, 10))
        self.metric_vars = {}
        for key, title, helper in [("step", "STEP", "step"), ("loss", "TRAINING ERROR (LOSS)", "train_loss"), ("val_loss", "TEST ERROR (VALIDATION)", "val_loss"), ("tokens_per_second", "SPEED (TOKENS / SEC)", "tokens_sec")]:
            box = card(metrics if key in ("loss", "val_loss") else extra)
            box.pack(side="left", fill="x", expand=True, padx=(0, 10))
            variable = tk.StringVar(value="\u2014")
            self.metric_vars[key] = variable
            label(box, textvariable=variable, style="Metric.TLabel").pack(anchor="w")
            with_help(box, title, helper, font=(FONT, 9)).pack(anchor="w", pady=(4, 0))
        self.loss_plot = tk.Canvas(page, bg=PANEL, height=270, highlightthickness=0)
        self.loss_plot.pack(fill="x")
        self.loss_plot.bind("<Configure>", lambda _: self.draw_loss())
        controls = ttk.Frame(page)
        controls.pack(fill="x", pady=10)
        self.pause_button = button(controls, "Pause", self.toggle_pause)
        button(controls, "Stop & save", self.stop_job)
        self.chat_button = ttk.Button(controls, text="Chat with this model", command=self.play_run, style="Accent.TButton")
        label(controls, "Green = training error    Blue = test error (validation)", "Muted.TLabel").pack(side="right")
        reading = self.reading_drawer = Collapsible(page, "Reading the chart")
        reading.pack(fill="x", pady=(0, 8))
        label(reading.body, "\u2022 The green line should fall: the model is getting less wrong. Lower is better; a brand-new model starts near 5.5.\n"
                            "\u2022 The blue line is the honest score. It uses examples the model never studied.\n"
                            "\u2022 If blue stops falling or climbs while green keeps falling, the model is memorizing: stop and save, or add more data.\n"
                            "\u2022 Rough guide: under 3 it knows common letters and words; under 2 it writes recognisable phrases; near 1 it has learned this data well.\n"
                            "\u2022 Stop when both lines flatten out. Extra steps after that mostly cost time.", "CardMuted.TLabel", justify="left", wraplength=1000).pack(anchor="w")
        help_row = ttk.Frame(reading.body, style="Card.TFrame")
        help_row.pack(anchor="w", pady=(8, 0))
        for text_, key in (("What is loss?", "loss"), ("Memorizing", "overfitting"), ("Checkpoint", "checkpoint")):
            with_help(help_row, text_, key).pack(side="left", padx=(0, 18))
        saved = Collapsible(page, "Saved runs")
        saved.pack(fill="x", pady=(0, 8))
        self.runs_tree = tree(saved.body, [("name", "MODEL", 270), ("status", "STATUS", 100), ("step", "STEPS", 80), ("params", "PARAMETERS", 100), ("loss", "BEST ERROR", 100)], 4)
        self.runs_tree.bind("<<TreeviewSelect>>", lambda _: self.inspect_run())
        bottom = ttk.Frame(saved.body)
        bottom.pack(fill="x", pady=(10, 0))
        resume = button(bottom, "Resume checkpoint", self.resume_run)
        Tooltip(resume, help_text("resume"))
        self.mark_advanced(resume)
        button(bottom, "Chat with this model", self.play_run)
        self.mark_advanced(button(bottom, "Open folder", self.open_run_folder))
        self.mark_advanced(button(bottom, "Export run ZIP", self.export_run))
        button(bottom, "Refresh", self.refresh_runs)
        button(bottom, "Set up a new run", lambda: self.show_page("blueprint"))
        details.pack(fill="x", pady=(0, 8))
        self.run_log = text_box(details.body, 3, state="disabled")
        self.run_log.pack(fill="x")
        self.mark_advanced(self.run_log)

    def start_training(self):
        if self.busy():
            return
        try:
            cfg = self.read_config()
            examples = copy.deepcopy(self.workspace.examples(self.selected_dataset()))
            if not examples:
                raise ValueError("Import or write some examples in the selected dataset first.")
            self.workspace.state["studioConfig"] = cfg
            self.workspace.save()
            self.launch_training(cfg, examples)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Cannot start training", str(exc), parent=self)

    def launch_training(self, cfg, examples, resume=None):
        directory = self.workspace.run_dir / uid("run")
        self.history = []
        self._run_cfg = cfg
        self.draw_loss()
        self.progress.configure(value=0, maximum=cfg["steps"])
        for var in self.metric_vars.values():
            var.set("\u2014")
        set_text(self.run_log, f"Starting {cfg['name']} \u00b7 {len(examples):,} examples \u00b7 requested device: {cfg['device']}\n")
        self.show_page("training")
        self.start_job("training", lambda: train(cfg, examples, directory, lambda e: self.events.put(("training", e)), self.stop, self.pause, resume=resume), lambda _: self.refresh_runs())

    def calibrate_speed(self, record):
        """Remember how fast this computer really trains, so the next time estimate is closer."""
        cfg = getattr(self, "_run_cfg", None)
        speeds = [h["tokens_per_second"] for h in self.history if h.get("step", 0) >= 3 and h.get("tokens_per_second")]
        if not cfg or len(speeds) < 3 or not record.get("parameters"):
            return
        from studio_ux import step_seconds
        kind = "cuda" if record.get("device") == "cuda" else "cpu"
        try:
            tokens = int(cfg["batch_size"]) * int(cfg["context"]) * int(cfg.get("accumulation", 1))
            predicted = tokens / step_seconds(cfg, record["parameters"], kind)
            observed = sorted(speeds)[len(speeds) // 2]
            table = self.workspace.state.setdefault("speedScale", {})
            table[kind] = round(min(5.0, max(0.2, predicted / observed)), 3)
            self.workspace.save()
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            pass

    def training_event(self, event):
        kind = event["type"]
        if kind == "metric":
            if self.chat_button.winfo_manager():
                self.chat_button.pack_forget()
            self.history.append(event)
            for key, variable in self.metric_vars.items():
                value = event.get(key)
                variable.set("\u2014" if value is None else f"{value:,}" if key == "step" else f"{value:,.0f}" if key == "tokens_per_second" else f"{value:.4f}")
            self.progress.configure(value=event["step"])
            self.status_var.set(f"Training \u00b7 step {event['step']}/{event.get('total', '?')} \u00b7 learning rate {event.get('lr', 0):.6f}")
            self.train_plain.set(plain_training_status(self.history, event.get("total")))
            self.draw_loss()
        elif kind == "history":
            self.history = [dict(row) for row in event["history"]]
            self.train_plain.set(plain_training_status(self.history))
            self.draw_loss()
        elif kind == "info":
            self.log(event["text"])
        elif kind == "done":
            self.log(f"{event['run']['status'].capitalize()} \u00b7 {event['directory']}")
            self.train_plain.set(plain_training_status(self.history, event["run"].get("steps"), finished=True))
            self.calibrate_speed(event["run"])
            self.refresh_runs()
            if self.run_map:
                self.runs_tree.selection_set("0")
            self.chat_button.pack(side="left", padx=(0, 8), pady=4)

    def log(self, text):
        self.run_log.configure(state="normal")
        self.run_log.insert("end", text + "\n")
        self.run_log.see("end")
        self.run_log.configure(state="disabled")

    def draw_loss(self):
        if not hasattr(self, "loss_plot"):
            return
        canvas = self.loss_plot
        canvas.delete("all")
        w, h = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 150)
        canvas.create_text(20, 18, anchor="w", text="ERROR (LOSS) \u00b7 LOWER IS BETTER", fill=MUTED, font=(FONT, 9, "bold"))
        values = [float(item[key]) for item in self.history for key in ("loss", "val_loss") if item.get(key) is not None]
        if not values:
            canvas.create_text(w / 2, h / 2, text="Your learning curve will appear here once training starts", fill=MUTED, font=(FONT, 12))
            return
        low, high = min(values), max(values)
        gap = max((high - low) * .12, .05)
        low, high = max(0, low - gap), high + gap
        max_step = max(1, max(item["step"] for item in self.history))
        left, right, top, bottom = 58, w - 24, 42, h - 30
        for i in range(4):
            y = top + (bottom - top) * i / 3
            canvas.create_line(left, y, right, y, fill=EDGE)
            canvas.create_text(left - 10, y, text=f"{high - (high - low) * i / 3:.2f}", fill=MUTED, anchor="e", font=(FONT, 9))
        for key, color in (("loss", ACCENT), ("val_loss", BLUE)):
            points = [(left + item["step"] / max_step * (right - left), bottom - (item[key] - low) / (high - low) * (bottom - top)) for item in self.history if item.get(key) is not None]
            if len(points) > 1:
                canvas.create_line(*[v for point in points for v in point], fill=color, width=2)
            elif points:
                x, y = points[0]
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")
        canvas.create_text(left, h - 12, text="0", fill=MUTED, font=(FONT, 9))
        canvas.create_text(right, h - 12, text=f"{max_step} steps", fill=MUTED, anchor="e", font=(FONT, 9))

    def refresh_runs(self):
        if not hasattr(self, "runs_tree"):
            return
        selected = self.runs_tree.selection()
        self.runs_tree.delete(*self.runs_tree.get_children())
        self.run_map = {}
        for i, record in enumerate(self.workspace.runs()):
            key = str(i)
            self.run_map[key] = record
            best = record.get("best_loss")
            self.runs_tree.insert("", "end", iid=key, values=(record.get("name", "Untitled"), record.get("status", "unknown"), record.get("step", 0), f"{record.get('parameters', 0):,}", f"{best:.4f}" if isinstance(best, (int, float)) else "—"))
        if selected and selected[0] in self.run_map:
            self.runs_tree.selection_set(selected[0])

    def selected_run(self):
        selected = self.runs_tree.selection()
        return self.run_map.get(selected[0]) if selected else None

    def inspect_run(self):
        record = self.selected_run()
        if not record or self.job_kind == "training":
            return
        path = Path(record["directory"]) / "metrics.jsonl"
        try:
            self.history = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.draw_loss()
            if self.history:
                last = self.history[-1]
                for key, var in self.metric_vars.items():
                    value = last.get(key)
                    var.set("—" if value is None else str(value) if key == "step" else f"{value:.4f}" if "loss" in key else f"{value:,.0f}")
        except (OSError, ValueError):
            pass

    def resume_run(self):
        if self.busy():
            return
        selected = self.selected_run()
        initial = selected["directory"] if selected else str(self.workspace.run_dir)
        path = filedialog.askopenfilename(parent=self, title="Resume your checkpoint", initialdir=initial, filetypes=[("Groundwork checkpoint", "*.npz")])
        if not path:
            return
        try:
            checkpoint = load_checkpoint(path)
            additional = simpledialog.askinteger("Continue training", "Additional optimizer steps", initialvalue=300, minvalue=1, maxvalue=1000000 - checkpoint["step"], parent=self)
            if additional is None:
                return
            cfg = dict(checkpoint["config"], steps=checkpoint["step"] + additional, device=self.config_vars["device"].get())
            snapshot = Path(path).parent / "dataset.jsonl"
            examples = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.launch_training(cfg, examples, path)
        except (ValueError, KeyError, OSError) as exc:
            messagebox.showerror("Cannot resume checkpoint", str(exc), parent=self)

    def open_run_folder(self):
        record = self.selected_run()
        if record:
            os.startfile(record["directory"])

    def export_run(self):
        record = self.selected_run()
        if not record:
            return
        if record.get("status") == "training":
            messagebox.showinfo("Run still active", "Stop and save this run before exporting it.", parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, title="Export model and training snapshot", defaultextension=".zip", filetypes=[("ZIP archive", "*.zip")])
        if path:
            try:
                directory = Path(record["directory"])
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                    for file in directory.iterdir():
                        if file.is_file() and file.suffix in {".npz", ".json", ".jsonl"}:
                            archive.write(file, arcname=file.name)
                self.status_var.set(f"Run exported · {path}")
            except OSError as exc:
                messagebox.showerror("Export failed", str(exc), parent=self)

    def play_run(self):
        record = self.selected_run()
        if record:
            directory = Path(record["directory"])
            path = directory / "best.npz"
            if not path.exists():
                path = directory / "last.npz"
            self.chat_checkpoint.set(str(path))
            self.show_page("chat")
            self.load_chat_model()

    def _build_chat(self):
        from studio_align import Collapsible
        page = self.pages["chat"]
        self.chat_checkpoint = tk.StringVar()
        self.chat_info = tk.StringVar(value="Press Use latest run to chat with your newest model, or choose a saved .npz file.")
        label(page, textvariable=self.chat_info, style="Muted.TLabel", wraplength=1020, justify="left").pack(anchor="w", pady=(0, 6))
        model = Collapsible(card(page), "Change model")
        model.master.pack(fill="x", pady=(0, 8))
        model.pack(fill="x")
        bar = ttk.Frame(model.body, style="Card.TFrame")
        bar.pack(fill="x")
        ttk.Entry(bar, textvariable=self.chat_checkpoint).pack(side="left", fill="x", expand=True, padx=(0, 8))
        button(bar, "Use latest run", self.use_latest_run, True)
        button(bar, "Choose file\u2026", lambda: self.choose_checkpoint(self.chat_checkpoint))
        button(bar, "Load model", self.load_chat_model)
        help_icon(bar, "npz").pack(side="left", padx=(2, 0))
        self.transcript = text_box(page, 16, state="disabled")
        self.transcript.pack(fill="both", expand=True)
        self.transcript.tag_configure("speaker", foreground=ACCENT, font=(FONT, 10, "bold"))
        self.chat_input = text_box(page, 3)
        self.chat_input.pack(fill="x", pady=(12, 0))
        self.chat_input.bind("<Control-Return>", lambda _: self.send_chat())
        controls = ttk.Frame(page)
        controls.pack(fill="x", pady=(8, 0))
        button(controls, "Send  Ctrl+Enter", self.send_chat, True)
        button(controls, "Stop", self.stop_job)
        Tooltip(button(controls, "Teach a correction", self.teach_correction), help_text("correction"))
        self.correction_note = tk.StringVar()
        label(page, textvariable=self.correction_note, style="Muted.TLabel", wraplength=1020, justify="left").pack(anchor="w", pady=(6, 0))
        self.temperature = tk.StringVar(value="0.8")
        self.top_k = tk.StringVar(value="40")
        self.max_tokens = tk.StringVar(value="160")
        self.system_prompt = tk.StringVar()
        more = Collapsible(card(page), "Reply settings, system prompt, device")
        more.master.pack(fill="x", pady=(8, 0))
        more.pack(fill="x")
        config = ttk.Frame(more.body, style="Card.TFrame")
        config.pack(fill="x")
        for title, var, width, helper in [("Creativity (temperature) \u00b7 0 = safest", self.temperature, 14, "temperature"), ("Top-k \u00b7 0 = all", self.top_k, 12, "top_k"),
                                          ("Reply length (new tokens)", self.max_tokens, 12, "max_tokens"), ("System prompt \u00b7 optional", self.system_prompt, 35, "system_prompt")]:
            frame, _ = field(config, title, var, width=width, help=helper)
            frame.pack(side="left", fill="x", expand=title.startswith("System"), padx=(0, 15))
        device_row = ttk.Frame(more.body, style="Card.TFrame")
        device_row.pack(fill="x", pady=(10, 0))
        label(device_row, "Run on", "Card.TLabel").pack(side="left", padx=(0, 8))
        ttk.Combobox(device_row, textvariable=self.inference_device, values=("auto", "cpu", "cuda"), state="readonly", width=12).pack(side="left", padx=(0, 12))
        label(device_row, "Applies when loading a model \u00b7 auto prefers the GPU \u00b7 shared with synthesis", "CardMuted.TLabel").pack(side="left")
        help_icon(device_row, "inference").pack(side="left", padx=(4, 0))
        button(device_row, "Clear conversation", self.clear_chat).pack_configure(side="right", padx=0)
        self.refresh_correction_note()

    def refresh_correction_note(self):
        if not hasattr(self, "correction_note"):
            return
        name = self.dataset_var.get() or "your working dataset"
        self.correction_note.set(f"Teach a correction saves your better answer as a new example in \u201c{name}\u201d (the working dataset picked at the top). "
                                 "Train again to make the model learn it.")

    def latest_checkpoint(self):
        """Newest run that left a checkpoint behind, or None."""
        for record in self.workspace.runs():
            directory = Path(record["directory"])
            for name in ("best.npz", "last.npz"):
                if (directory / name).exists():
                    return directory / name
        return None

    def use_latest_run(self, load=True):
        path = self.latest_checkpoint()
        if path is None:
            self.chat_info.set("No trained model yet. Train one first (Training setup), or choose a saved .npz file.")
            return None
        self.chat_checkpoint.set(str(path))
        if load:
            self.load_chat_model()
        return path

    def choose_checkpoint(self, variable):
        path = filedialog.askopenfilename(parent=self, title="Select your trained model", initialdir=self.workspace.run_dir, filetypes=[("Groundwork checkpoint", "*.npz")])
        if path:
            variable.set(path)

    def load_chat_model(self):
        if self.busy():
            return
        path = self.chat_checkpoint.get().strip()
        if not path:
            self.choose_checkpoint(self.chat_checkpoint)
            path = self.chat_checkpoint.get().strip()
        if path:
            device = self.inference_device.get()
            def loaded(model):
                self.chat_model = model
                self.clear_chat()
                backend = model.model.backend
                fallback = f"\n{backend.reason}" if backend.reason else ""
                self.chat_info.set(f"Loaded {model.config['name']} · step {model.step:,} · {model.model.parameter_count:,} parameters · {backend.name}{fallback}\nType a message below and press Send.")
                if hasattr(self, "refresh_home"):
                    self.refresh_home()
            self.start_job("loading model", lambda: LocalModel(path, device=device), loaded)

    def clear_chat(self):
        if self.job_kind == "chat":
            return
        self.chat_messages = []
        self.last_prompt = self.last_answer = ""
        set_text(self.transcript, "")

    def render_chat(self, partial=None):
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        for message in self.chat_messages:
            if message["role"] == "system":
                continue
            self.transcript.insert("end", ("YOU" if message["role"] == "user" else "MODEL") + "\n", "speaker")
            self.transcript.insert("end", message["content"] + "\n\n")
        if partial is not None:
            self.transcript.insert("end", "MODEL\n", "speaker")
            self.transcript.insert("end", partial or "Thinking…")
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def send_chat(self):
        if self.busy():
            return
        if not self.chat_model:
            messagebox.showinfo("Load a model", "Train your model, then load its .npz checkpoint here.", parent=self)
            return
        prompt = content(self.chat_input)
        if not prompt:
            return
        try:
            temperature, top_k, max_tokens = float(self.temperature.get()), int(self.top_k.get()), int(self.max_tokens.get())
            if not 0 <= temperature <= 3 or not 0 <= top_k <= 100000 or not 1 <= max_tokens <= 4096:
                raise ValueError("Use temperature 0–3, top-k ≥ 0, and 1–4096 new tokens.")
        except ValueError as exc:
            messagebox.showerror("Sampling settings", str(exc), parent=self)
            return
        self.chat_messages = [m for m in self.chat_messages if m["role"] != "system"]
        if self.system_prompt.get().strip():
            self.chat_messages.insert(0, {"role": "system", "content": self.system_prompt.get().strip()})
        self.chat_messages.append({"role": "user", "content": prompt})
        self.last_prompt, self.last_answer = prompt, ""
        set_text(self.chat_input, "")
        self.render_chat("")
        messages = copy.deepcopy(self.chat_messages)
        def completed(answer):
            self.last_answer = answer
            if answer:
                self.chat_messages.append({"role": "assistant", "content": answer})
            self.render_chat()
            if not answer:
                self.chat_info.set("The model ended without text. Try more training or a different sampling temperature.")
        self.start_job("chat", lambda: self.chat_model.generate(messages, max_tokens, temperature, top_k, self.stop, lambda text: self.events.put(("chat_text", text))), completed)

    def teach_correction(self):
        if self.job_kind == "chat" or not self.last_prompt:
            return
        if not self.selected_dataset():
            self.new_dataset()
        if not self.selected_dataset():
            return
        turns = copy.deepcopy(self.chat_messages)
        if not turns or turns[-1]["role"] != "assistant":
            turns.append({"role": "assistant", "content": "Write the correct response here."})
        example = normalize({"messages": turns, "quality": "Good", "intent": "correction"})
        dialog = ExampleEditor(self, example, "Teach the correct response")
        self.wait_window(dialog)
        if dialog.result:
            added, duplicates, _ = self.workspace.add(self.selected_dataset(), [dialog.result])
            self.refresh_datasets()
            self.status_var.set(f"{added} correction saved to {self.dataset_var.get()} · train a new run to learn the updated dataset")

    def busy(self):
        if self.job_kind is not None:
            self.status_var.set(f"{self.job_kind.capitalize()} is active. Finish or stop it before starting another compute task.")
            return True
        return False

    def start_job(self, kind, function, callback):
        if self.busy():
            return False
        self.job_kind = kind
        self.result_callback = callback
        self.stop.clear()
        self.pause.clear()
        self.pause_button.configure(text="Pause")
        self.status_var.set(f"{kind.capitalize()} · running locally")
        def run():
            try:
                self.events.put(("result", function()))
            except Exception as exc:
                self.events.put(("error", (str(exc), traceback.format_exc())))
        self.worker = threading.Thread(target=run, name=f"studio-{kind}", daemon=False)
        self.worker.start()
        return True

    def poll(self):
        try:
            for _ in range(100):
                kind, value = self.events.get_nowait()
                if kind == "training":
                    self.training_event(value)
                elif kind == "gpu_status":
                    self.update_gpu_status(value)
                elif kind == "status":
                    self.status_var.set(value)
                elif kind == "chat_text":
                    self.render_chat(value)
                elif kind == "result":
                    previous = self.job_kind
                    self.job_kind = None
                    self.pause.clear()
                    self.pause_button.configure(text="Pause")
                    self.status_var.set(f"{previous.capitalize()} · {'stopped' if self.stop.is_set() else 'complete'}")
                    self.result_callback(value)
                elif kind == "error":
                    self.job_kind = None
                    self.log(value[1])
                    self.refresh_runs()
                    self.status_var.set(f"Task failed · {value[0]}")
                    if not self.closing:
                        messagebox.showerror("Task failed", value[0], parent=self)
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.workspace.save()
            self.destroy()
            return
        self.after(60, self.poll)

    def toggle_pause(self):
        if self.job_kind != "training":
            return
        if self.pause.is_set():
            self.pause.clear()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Training resumed")
        else:
            self.pause.set()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Pausing after the current step")

    def stop_job(self):
        if self.job_kind:
            self.stop.set()
            self.pause.clear()
            self.status_var.set("Stopping after the current operation · training progress will be saved")

    def close(self):
        if self.job_kind:
            if not messagebox.askyesno("Save and close", "Stop the current task and close? Active training will save a checkpoint first.", parent=self):
                return
            self.closing = True
            self.stop_job()
        else:
            self.workspace.save()
            self.destroy()


from studio_flow import WorkflowMixin
from studio_align import AlignMixin
from studio_network import NetworkMixin
from studio_curriculum_page import CurriculumMixin
from studio_ux import UXMixin


class Studio(UXMixin, CurriculumMixin, AlignMixin, NetworkMixin, WorkflowMixin, NativeStudio):
    """Guided studio workflow with a scalable library and synthesis workbench."""


def main():
    try:
        app = Studio()
        app.mainloop()
        return 0
    except Exception:
        report = traceback.format_exc()
        (ROOT / "studio-error.log").write_text(report, encoding="utf-8")
        messagebox.showerror("Groundwork could not start", "See studio-error.log for details.\n\n" + report.splitlines()[-1])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

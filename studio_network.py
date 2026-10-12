"""Network page: a neural-circuit editor with versioned architectures, live telemetry and a 3D link."""
from __future__ import annotations

import copy
import re
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from studio_arch import config_from_spec, default_spec, new_block, parameter_breakdown, spec_from_config, validate_spec
from studio_archdb import ArchitectureStore
from studio_circuit import ACCENT as CHIP_COLORS, CircuitCanvas, ERR
from studio_model import Tokenizer, load_checkpoint, validate_config
from studio_vizbridge import VizBridge
from studio_widgets import (BG, PANEL, INPUT, EDGE, TEXT, MUTED, ACCENT, FONT, label, button, card,
                            Tooltip, help_icon, help_text, with_help)

TEXT_WIDGETS = {"TEntry", "Entry", "TSpinbox", "Text", "TCombobox", "Spinbox"}
PARTS = [("transformer", "Transformer block", "attention + MLP", "att"),
         ("attention", "Attention-only block", "attention, no MLP", "att"),
         ("mlp", "MLP-only block", "feed-forward, no attention", "mlp")]
PART_HELP = {"transformer": "Transformer block: one layer that first lets each character look back at earlier ones (attention), then thinks about it (MLP). Drag it onto the board to add a layer.",
             "attention": "Attention-only block: a layer that only looks back at earlier characters, with no thinking step after it.",
             "mlp": "MLP-only block: a layer that only does the thinking step (feed-forward network), with no looking back."}
# glossary key for each inspector field title
INSPECTOR_HELP = {"Tokenizer": "tokenizer", "Width": "width", "Context": "context", "Dropout": "dropout", "Heads": "heads", "Activation": "activation",
                  "Expansion ×": "expansion", "Hidden layers": "hidden_layers", "Steps": "steps", "Batch size": "batch_size", "Accumulation": "accumulation",
                  "Learning rate": "learning_rate", "Warmup steps": "warmup", "Weight decay": "weight_decay", "Evaluate every": "eval_every",
                  "Validation split": "validation", "Seed": "seed", "Objective": "objective", "Device": "device"}
TRAIN_FIELDS = [("steps", "Steps", "spin", (1, 1000000, 50), int), ("batch_size", "Batch size", "spin", (1, 128, 1), int),
                ("accumulation", "Accumulation", "spin", (1, 64, 1), int), ("learning_rate", "Learning rate", "entry", None, float),
                ("warmup", "Warmup steps", "spin", (0, 100000, 10), int), ("weight_decay", "Weight decay", "entry", None, float),
                ("eval_every", "Evaluate every", "spin", (1, 100000, 5), int), ("validation", "Validation split", "entry", None, float),
                ("seed", "Seed", "spin", (0, 2147483647, 1), int), ("objective", "Objective", "choice", ["all", "assistant"], str),
                ("device", "Device", "choice", ["auto", "cpu", "cuda"], str)]
SUB_KINDS = {"att": "attention", "mlp": "mlp"}


def valid_heads(width):
    """Head counts that divide the embedding width (1–16)."""
    try:
        width = int(width)
    except (TypeError, ValueError):
        return [1]
    return [h for h in range(1, 17) if width > 0 and width % h == 0] or [1]


class NetworkMixin:
    # ------------------------------------------------------------------ construction
    def _layout(self):
        super()._layout()
        self.arch_store = ArchitectureStore(self.workspace.root)
        self.arch_id = None
        self.net_spec = None
        self.net_error = ""
        self.net_undo_stack, self.net_redo_stack = [], []
        self.net_selected = None
        self.net_commit_job, self.net_notes = None, []
        self.net_init_version = None
        self.net_map = {}
        self._net_run, self._net_live_pending = None, False
        self.net_tele, self.net_tele_count = None, 0
        self._net_paint_job, self._net_last_paint = None, 0.0
        self.net_ins = []
        self._pal = None
        self.viz = None
        self._build_network()
        for sequence, action in (("<Control-z>", self.net_undo), ("<Control-y>", self.net_redo), ("<Control-Shift-Z>", self.net_redo),
                                 ("<Delete>", self.net_delete_selected), ("<BackSpace>", self.net_delete_selected)):
            self.bind_all(sequence, lambda event, a=action: self._net_key(event, a), add="+")

    def _net_style(self):
        style = ttk.Style(self)
        style.configure("Net.Treeview", background=INPUT, fieldbackground=INPUT, foreground=TEXT, rowheight=24, borderwidth=0)
        style.map("Net.Treeview", background=[("selected", "#27463f")], foreground=[("selected", "#dcfff2")])
        style.configure("Net.Treeview.Heading", background=PANEL, foreground=MUTED, font=(FONT, 9, "bold"), padding=5, relief="flat")
        style.configure("TSpinbox", fieldbackground=INPUT, background=EDGE, foreground=TEXT, arrowcolor=MUTED, padding=6, insertcolor=ACCENT)
        style.configure("Card.TCheckbutton", background=PANEL, foreground=TEXT)
        style.map("Card.TCheckbutton", background=[("active", PANEL)])
        style.configure("Err.TLabel", foreground=ERR, background=BG)
        style.configure("Ok.TLabel", foreground=ACCENT, background=BG)
        style.configure("Small.TButton", padding=(10, 6))

    def _build_network(self):
        page = self.pages["network"]
        self._net_style()
        bar = ttk.Frame(page)
        bar.pack(fill="x", pady=(0, 6))
        self.net_arch_var = tk.StringVar()
        self.net_picker = ttk.Combobox(bar, textvariable=self.net_arch_var, state="readonly", width=34)
        self.net_picker.pack(side="left", padx=(0, 8), pady=3)
        self.net_picker.bind("<<ComboboxSelected>>", lambda _: self.net_pick())
        for text, command in (("+ New", self.net_new), ("Rename", self.net_rename), ("Delete", self.net_delete),
                              ("Import", self.net_import), ("Export", self.net_export)):
            ttk.Button(bar, text=text, command=command, style="Small.TButton").pack(side="left", padx=(0, 6), pady=3)
        self.net_summary = tk.StringVar()
        label(bar, textvariable=self.net_summary, style="Muted.TLabel").pack(side="right")
        strip = ttk.Frame(page)
        strip.pack(fill="x", pady=(0, 6))
        label(strip, "PARTS", "Muted.TLabel", font=(FONT, 9, "bold")).pack(side="left", padx=(0, 10))
        for kind, title, sub, color in PARTS:
            chip = tk.Label(strip, text=f"▣  {title}", bg="#0c1626", fg=CHIP_COLORS[color], font=(FONT, 10, "bold"), padx=12, pady=6,
                            highlightthickness=1, highlightbackground=CHIP_COLORS[color], cursor="hand2")
            chip.pack(side="left", padx=(0, 8))
            chip.bind("<ButtonPress-1>", lambda e, k=kind, t=title: self._pal_press(e, k, t))
            chip.bind("<B1-Motion>", self._pal_motion)
            chip.bind("<ButtonRelease-1>", self._pal_release)
            Tooltip(chip, PART_HELP[kind])
        label(strip, "drag to place", "Muted.TLabel").pack(side="left", padx=(2, 0))
        help_icon(strip, "block").pack(side="left", padx=(2, 0))
        for text, command in (("Panel", self.net_toggle_panel), ("Fit", lambda: self.net_canvas.fit()), ("−", lambda: self.net_canvas.zoom(1 / 1.25)), ("+", lambda: self.net_canvas.zoom(1.25)),
                              ("Redo", lambda: self.net_redo()), ("Undo", lambda: self.net_undo())):
            ttk.Button(strip, text=text, command=command, style="Small.TButton", width=3 if len(text) < 2 else 4 if len(text) < 4 else 6).pack(side="right", padx=(5, 0))
        body = ttk.Frame(page)
        body.pack(fill="both", expand=True)
        side = self.net_side = ttk.Frame(body, width=360)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)
        self.net_canvas = CircuitCanvas(body, self)
        self.net_canvas.pack(side="left", fill="both", expand=True)
        notebook = self.net_tabs = ttk.Notebook(side)
        notebook.pack(fill="both", expand=True)
        inspect, history, weights, backups = [ttk.Frame(notebook, padding=10) for _ in range(4)]
        for frame, title in zip((inspect, history, weights, backups), ("Inspector", "History", "Weights", "Backups")):
            notebook.add(frame, text=title)
        self.ins_frame = inspect
        self._build_history(history)
        self._build_weights(weights)
        self._build_backups(backups)
        foot = ttk.Frame(page)
        foot.pack(fill="x", pady=(8, 0))
        self.net_msg = ttk.Label(foot, text="", style="Ok.TLabel")
        self.net_msg.pack(side="left", fill="x", expand=True)
        self.net_run_var = tk.StringVar()
        self.net_init_var = tk.StringVar(value="Starting weights: random (brand-new model)")
        actions = ttk.Frame(foot)
        actions.pack(side="right")
        self.net_train_button = ttk.Button(actions, text="Train this architecture", command=self.net_train, style="Accent.TButton")
        self.net_train_button.pack(side="left", padx=(8, 0))
        Tooltip(self.net_train_button, "Trains this design using the training settings stored with it (Training tab in the inspector). "
                                       "Your data and example budget come from the Training setup page.")
        self.net_live_button = ttk.Button(actions, text="Apply live edit", command=self.net_apply_live)
        self.net_live_button.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Stop", command=self.stop_job).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Open 3D brain", command=self.net_open_viewer).pack(side="left", padx=(8, 0))
        meta = ttk.Frame(page)
        meta.pack(fill="x", pady=(4, 0))
        label(meta, textvariable=self.net_init_var, style="Muted.TLabel").pack(side="left")
        label(meta, textvariable=self.net_run_var, style="Muted.TLabel").pack(side="right")

    def net_toggle_panel(self):
        """Hide or show the inspector dock to give the circuit the whole page."""
        if self.net_side.winfo_manager():
            self.net_side.pack_forget()
        else:
            self.net_side.pack(side="right", fill="y", padx=(10, 0), before=self.net_canvas)

    def _build_history(self, page):
        note = ttk.Frame(page)
        note.pack(anchor="w", pady=(0, 6))
        label(note, "Every valid edit is saved as a revision (a snapshot).", "Muted.TLabel").pack(side="left")
        help_icon(note, "revision").pack(side="left", padx=(4, 0))
        self.rev_tree = self._small_tree(page, [("rev", "REV", 46), ("note", "CHANGE", 170), ("when", "SAVED", 90)])
        row = ttk.Frame(page)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Restore this revision", command=self.net_restore_revision, style="Small.TButton").pack(side="left")

    def _build_weights(self, page):
        label(page, "Trained weights stay attached to this architecture.", "Muted.TLabel").pack(anchor="w", pady=(0, 6))
        self.weights_tree = self._small_tree(page, [("v", "VER", 40), ("step", "STEP", 54), ("loss", "LOSS", 60), ("source", "SOURCE", 66), ("rev", "REV", 44)])
        grid = ttk.Frame(page)
        grid.pack(fill="x", pady=(8, 0))
        for i, (text, command) in enumerate((("Register checkpoint…", self.net_register_checkpoint), ("Use as initial weights", self.net_use_weights),
                                              ("Open in playground", self.net_open_weights))):
            ttk.Button(grid, text=text, command=command, style="Small.TButton").grid(row=i // 2, column=i % 2, sticky="ew", padx=(0, 6), pady=3)
        grid.columnconfigure((0, 1), weight=1)

    def _build_backups(self, page):
        label(page, "Snapshots of every architecture, revision and weight file.", "Muted.TLabel").pack(anchor="w", pady=(0, 6))
        self.backup_tree = self._small_tree(page, [("when", "CREATED", 128), ("label", "LABEL", 80), ("size", "SIZE", 70)])
        row = ttk.Frame(page)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Create backup", command=self.net_backup, style="Small.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(row, text="Restore backup…", command=self.net_restore_backup, style="Small.TButton").pack(side="left")

    def _small_tree(self, parent, columns):
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        widget = ttk.Treeview(container, columns=[c[0] for c in columns], show="headings", height=9, selectmode="browse", style="Net.Treeview")
        for name, title, width in columns:
            widget.heading(name, text=title)
            widget.column(name, width=width, minwidth=36, stretch=name in ("note", "label"))
        scroll = ttk.Scrollbar(container, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=scroll.set)
        widget.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return widget

    # ------------------------------------------------------------------ page lifecycle
    def show_page(self, name):
        super().show_page(name)
        if name == "network":
            self.net_on_show()

    def net_on_show(self):
        if self.net_spec is None:
            self.net_reload()
        self.net_canvas.update_idletasks()
        if self.net_tele:
            self.net_canvas.set_telemetry(self.net_tele)
        self.net_canvas.focus_set()

    def net_active(self):
        return getattr(self, "active_page", None) == "network"

    def net_visible(self):
        return self.net_active() and self.winfo_viewable()

    def net_is_training(self):
        return self.job_kind == "training" and self._net_run is not None

    def net_hover(self, text):
        self.net_msg.configure(text=text or self._net_status_text(), style="TLabel" if text else "Err.TLabel" if self.net_error else "Ok.TLabel")

    def _net_status_text(self):
        return self.net_error or "Circuit is valid · every edit is saved as a revision"

    def _net_key(self, event, action):
        if not self.net_active():
            return None
        try:
            widget = self.focus_get()
        except (KeyError, tk.TclError):
            widget = None
        if widget is not None and widget.winfo_class() in TEXT_WIDGETS:
            return None
        action()
        return "break"

    def close(self):
        self.net_flush()
        super().close()

    def destroy(self):
        for job in (self.net_commit_job, self._net_paint_job):
            if job:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        try:
            if self.net_spec is not None:
                self.net_commit_job = None
                self.net_commit_now()
        except Exception:
            pass
        if self.viz is not None:
            try:
                self.viz.stop()
            except Exception:
                pass
        try:  # leave no timers behind: a later Tk root in this thread would run them against deleted commands
            for job in self.tk.splitlist(self.tk.call("after", "info")):
                self.tk.call("after", "cancel", job)
        except tk.TclError:
            pass
        super().destroy()

    # ------------------------------------------------------------------ architectures
    def _net_seed_spec(self):
        saved = self.workspace.state.get("studioConfig") or {}
        if saved:
            try:
                cfg = validate_config(saved)
                return validate_spec(cfg["architecture"]) if cfg.get("architecture") else spec_from_config(cfg)
            except (ValueError, TypeError, KeyError):
                pass
        return default_spec()

    def net_reload(self, select=None):
        archs = self.arch_store.list()
        if not archs:
            spec = self._net_seed_spec()
            self.arch_store.create(spec["name"], spec, "created from the blueprint" if self.workspace.state.get("studioConfig") else "created")
            archs = self.arch_store.list()
        self.net_map = {f"{a['name']}  ·  {a['id'][-6:]}": a["id"] for a in archs}
        self.net_picker.configure(values=list(self.net_map))
        wanted = select or self.workspace.state.get("networkArchitecture")
        self.net_open(wanted if wanted in self.net_map.values() else archs[0]["id"])

    def net_open(self, arch_id):
        self.net_flush()
        record = self.arch_store.get(arch_id)
        self.arch_id, self.net_spec = arch_id, record["spec"]
        self.net_undo_stack, self.net_redo_stack = [], []
        self.net_selected, self.net_init_version = None, None
        self.net_tele = None
        self.net_canvas.clear_telemetry()
        self.net_canvas.set_selected(None)
        self.net_arch_var.set(next((k for k, v in self.net_map.items() if v == arch_id), ""))
        self.workspace.state["networkArchitecture"] = arch_id
        self.workspace.save()
        self.net_refresh_all()
        self.net_send_spec()

    def net_pick(self):
        target = self.net_map.get(self.net_arch_var.get())
        if target and target != self.arch_id:
            self.net_open(target)

    def net_new(self, name=None):
        name = name or simpledialog.askstring("New architecture", "Architecture name", parent=self)
        if name and name.strip():
            self.net_flush()
            spec = default_spec(name.strip())
            self.net_reload(self.arch_store.create(name.strip(), spec, "created"))
            return self.arch_id

    def net_rename(self, name=None):
        self.net_ensure()
        current = self.arch_store.get(self.arch_id)["name"]
        name = name or simpledialog.askstring("Rename architecture", "Architecture name", initialvalue=current, parent=self)
        if name and name.strip():
            self.arch_store.rename(self.arch_id, name.strip())
            self.net_edit(lambda spec: spec.__setitem__("name", name.strip()), f"rename to {name.strip()}")
            self.net_reload(self.arch_id)

    def net_delete(self, confirm=True):
        self.net_ensure()
        if confirm and not messagebox.askyesno("Delete architecture", "Delete this architecture with all its revisions and weights? Create a backup first if unsure.", parent=self):
            return False
        self.net_commit_job = self._cancel(self.net_commit_job)
        self.arch_store.delete(self.arch_id)
        self.arch_id = self.net_spec = None
        self.net_reload()
        return True

    def net_import(self, path=None):
        path = path or filedialog.askopenfilename(parent=self, title="Import architecture", filetypes=[("Architecture", "*.json"), ("All files", "*.*")])
        if path:
            try:
                self.net_reload(self.arch_store.import_spec(path))
            except ValueError as exc:
                messagebox.showerror("Cannot import", str(exc), parent=self)

    def net_export(self, path=None):
        self.net_ensure()
        self.net_flush()
        path = path or filedialog.asksaveasfilename(parent=self, title="Export architecture", defaultextension=".json", filetypes=[("Architecture", "*.json")])
        if path:
            self.arch_store.export_spec(self.arch_id, path)
            self.status_var.set(f"Architecture exported · {path}")

    def net_ensure(self):
        if self.net_spec is None:
            self.net_reload()

    @staticmethod
    def _cancel_job(widget, job):
        if job:
            try:
                widget.after_cancel(job)
            except tk.TclError:
                pass

    def _cancel(self, job):
        self._cancel_job(self, job)
        return None

    # ------------------------------------------------------------------ editing core
    def net_edit(self, mutate, note, strict=False, record=True):
        """Apply `mutate(draft)`; a valid result is committed (debounced), an invalid one is shown in red."""
        self.net_ensure()
        draft = copy.deepcopy(self.net_spec)
        try:
            mutate(draft)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self.status_var.set(f"Edit not applied · {exc}")
            return False
        try:
            draft = validate_spec(draft)
        except ValueError as exc:
            if strict:
                self.status_var.set(f"Edit rejected · {exc}")
                return False
        if draft == self.net_spec:
            return False
        if record:
            self.net_undo_stack.append(copy.deepcopy(self.net_spec))
            del self.net_undo_stack[:-100]
            self.net_redo_stack.clear()
        self.net_spec = draft
        self.net_refresh_view()
        if not self.net_error:
            self.net_schedule_commit(note)
            self.net_send_spec()
        return True

    def net_undo(self):
        self.net_ensure()
        if not self.net_undo_stack:
            self.status_var.set("Nothing to undo")
            return False
        self.net_redo_stack.append(copy.deepcopy(self.net_spec))
        self.net_spec = self.net_undo_stack.pop()
        self.net_refresh_view()
        if not self.net_error:
            self.net_schedule_commit("undo")
            self.net_send_spec()
        return True

    def net_redo(self):
        self.net_ensure()
        if not self.net_redo_stack:
            self.status_var.set("Nothing to redo")
            return False
        self.net_undo_stack.append(copy.deepcopy(self.net_spec))
        self.net_spec = self.net_redo_stack.pop()
        self.net_refresh_view()
        if not self.net_error:
            self.net_schedule_commit("redo")
            self.net_send_spec()
        return True

    def net_schedule_commit(self, note):
        self.net_notes.append(note)
        self._cancel(self.net_commit_job)
        self.net_commit_job = self.after(600, self.net_commit_now)

    def net_flush(self):
        """Commit any debounced edit immediately."""
        if self.net_spec is None:
            return
        if self.net_commit_job:
            self.net_commit_job = self._cancel(self.net_commit_job)
        if self.net_notes:
            self.net_commit_now()

    def net_commit_now(self):
        self.net_commit_job = None
        if self.net_spec is None or self.arch_id is None or self.net_error or not self.net_notes:
            return None
        notes, self.net_notes = self.net_notes, []
        unique = list(dict.fromkeys(notes))
        note = "; ".join(unique[-3:]) + (f" (+{len(unique) - 3} more)" if len(unique) > 3 else "")
        try:
            number = self.arch_store.commit(self.arch_id, validate_spec(self.net_spec), note)
        except ValueError as exc:
            self.status_var.set(f"Could not save revision · {exc}")
            return None
        self.net_refresh_history()
        self.net_refresh_picker()
        return number

    # ------------------------------------------------------------------ editing operations
    @staticmethod
    def _relabel(spec):
        for i, block in enumerate(spec["blocks"]):
            if re.fullmatch(r"Block( \d+)?", str(block.get("label", ""))):
                block["label"] = f"Block {i + 1}"

    def _find(self, spec, block_id):
        return next((i for i, b in enumerate(spec["blocks"]) if b.get("id") == block_id), None)

    def _block_id(self, key=None):
        key = key or self.net_selected
        return key.split(":")[1] if key and key.startswith("b:") else None

    def _default_heads(self, spec):
        choices = valid_heads(spec.get("embedding", {}).get("width", 64))
        return max([h for h in choices if h <= 4] or [1])

    def net_insert_block(self, kind="transformer", index=None):
        self.net_ensure()
        result = {}
        def mutate(spec):
            if index is None:
                selected = self._find(spec, self._block_id()) if self._block_id() else None
                at = len(spec["blocks"]) if selected is None else selected + 1
            else:
                at = max(0, min(len(spec["blocks"]), index))
            block = new_block(heads=self._default_heads(spec), attention=kind != "mlp", mlp=kind != "attention")
            spec["blocks"].insert(at, block)
            self._relabel(spec)
            result.update(id=block["id"], at=at)
        title = next(t for k, t, *_ in PARTS if k == kind)
        if self.net_edit(mutate, "insert " + title.lower()):
            self.net_select(f"b:{result['id']}")
            return result["id"]
        return None

    def net_move_block(self, block_id, to_index):
        self.net_ensure()
        result = {}
        def mutate(spec):
            i = self._find(spec, block_id)
            if i is None:
                raise KeyError("unknown block")
            block = spec["blocks"].pop(i)
            to = max(0, min(len(spec["blocks"]), to_index))
            spec["blocks"].insert(to, block)
            result.update(src=i, dst=to)
            self._relabel(spec)
        if self.net_edit(mutate, "move block"):
            self.net_history_note_fix(f"move block {result['src'] + 1} → {result['dst'] + 1}")
            self.net_select(f"b:{block_id}")
            return True
        return False

    def net_history_note_fix(self, note):
        if self.net_notes:
            self.net_notes[-1] = note

    def net_remove_block(self, block_id=None):
        self.net_ensure()
        block_id = block_id or self._block_id()
        if not block_id or self._find(self.net_spec, block_id) is None:
            return False
        if len(self.net_spec["blocks"]) <= 1:
            self.status_var.set("A network needs at least one block.")
            return False
        index = self._find(self.net_spec, block_id)
        def mutate(spec):
            spec["blocks"].pop(self._find(spec, block_id))
            self._relabel(spec)
        if self.net_edit(mutate, f"remove block {index + 1}"):
            blocks = self.net_spec["blocks"]
            self.net_select(f"b:{blocks[min(index, len(blocks) - 1)]['id']}")
            return True
        return False

    def net_duplicate_block(self, block_id=None):
        self.net_ensure()
        block_id = block_id or self._block_id()
        if not block_id or self._find(self.net_spec, block_id) is None:
            return None
        result = {}
        def mutate(spec):
            i = self._find(spec, block_id)
            clone = copy.deepcopy(spec["blocks"][i])
            clone["id"] = new_block()["id"]
            if not re.fullmatch(r"Block( \d+)?", clone["label"]):
                clone["label"] += " copy"
            spec["blocks"].insert(i + 1, clone)
            self._relabel(spec)
            result["id"], result["i"] = clone["id"], i
        if self.net_edit(mutate, f"duplicate block {self._find(self.net_spec, block_id) + 1}"):
            self.net_select(f"b:{result['id']}")
            return result["id"]
        return None

    def net_set_block(self, block_id, path, value, strict=False):
        """Set one block field: label, attention.enabled/heads, mlp.enabled/expansion/layers/activation."""
        self.net_ensure()
        def mutate(spec):
            i = self._find(spec, block_id)
            if i is None:
                raise KeyError("unknown block")
            block = spec["blocks"][i]
            if path == "label":
                block["label"] = str(value)
            else:
                part, field = path.split(".")
                if part not in ("attention", "mlp"):
                    raise KeyError(path)
                block[part][field] = value
        index = self._find(self.net_spec, block_id)
        old = None
        if index is not None:
            part, _, field = path.partition(".")
            old = self.net_spec["blocks"][index].get(part) if not field else self.net_spec["blocks"][index].get(part, {}).get(field)
        label_text = f"block {index + 1}" if index is not None else "block"
        return self.net_edit(mutate, f"{label_text} {path} {old}→{value}", strict=strict)

    def net_set_embedding(self, key, value):
        self.net_ensure()
        old = self.net_spec["embedding"].get(key)
        notes = []
        def mutate(spec):
            spec["embedding"][key] = value
            if key == "width":
                for i, block in enumerate(spec["blocks"]):
                    heads = block["attention"].get("heads", 4)
                    if block["attention"].get("enabled") and isinstance(value, int) and value > 0 and isinstance(heads, int) and value % heads:
                        block["attention"]["heads"] = max(h for h in valid_heads(value) if h <= heads)
                        notes.append(f"block {i + 1} heads → {block['attention']['heads']}")
        applied = self.net_edit(mutate, f"{key} {old}→{value}")
        if applied and notes:
            self.status_var.set("Heads adjusted to divide the new width · " + ", ".join(notes))
        return applied

    def net_set_training(self, key, value):
        self.net_ensure()
        old = self.net_spec["training"].get(key)
        return self.net_edit(lambda spec: spec["training"].__setitem__(key, value), f"{key} {old}→{value}")

    # ------------------------------------------------------------------ refresh
    def net_vocab(self, tokenizer):
        try:
            examples = self.workspace.examples(self.selected_dataset(), limit=250) if self.selected_dataset() else []
            return Tokenizer.fit(tokenizer, examples).vocab_size if examples else 263
        except Exception:
            return 263

    def _err_keys(self, spec, message):
        if not message:
            return set()
        lowered = message.lower()
        match = re.search(r"Block (\d+)", message)
        blocks = spec.get("blocks") if isinstance(spec, dict) and isinstance(spec.get("blocks"), list) else []
        if match and 0 < int(match.group(1)) <= len(blocks) and isinstance(blocks[int(match.group(1)) - 1], dict):
            return {"b:" + str(blocks[int(match.group(1)) - 1].get("id"))}
        if any(w in lowered for w in ("steps", "batch", "accumulation", "learning", "warmup", "weight decay", "eval", "validation", "seed", "objective", "device")):
            return {"train"}
        return {"emb"}

    def net_refresh_all(self):
        self.net_refresh_view()
        self.net_refresh_history()
        self.net_refresh_weights()
        self.net_refresh_backups()
        self.net_refresh_picker()

    def net_refresh_view(self):
        parts, total, stale = None, None, False
        try:
            valid = validate_spec(self.net_spec)
            breakdown = parameter_breakdown(valid, self.net_vocab(valid["embedding"]["tokenizer"]))
            parts, total = {c["id"]: c["params"] for c in breakdown["components"]}, breakdown["total"]
            self.net_error = ""
        except ValueError as exc:
            self.net_error, stale = str(exc), True
        keys = self._err_keys(self.net_spec, self.net_error)
        self.net_canvas.set_spec(self.net_spec, self.net_error, keys, parts, stale)
        if total is not None:
            blocks = valid["blocks"]
            heads = sum(b["attention"]["heads"] for b in blocks if b["attention"]["enabled"])
            self.net_summary.set(f"≈ {total:,} parameters · {len(blocks)} blocks · {heads} heads · {valid['embedding']['width']} wide")
        else:
            self.net_summary.set("Fix the highlighted chip to see parameter counts")
        self.net_msg.configure(text=self._net_status_text(), style="Err.TLabel" if self.net_error else "Ok.TLabel")
        self.net_train_button.configure(state="disabled" if self.net_error else "normal")
        self.net_live_button.configure(state="normal" if self.net_is_training() and not self.net_error else "disabled")
        self.net_refresh_inspector()

    def net_refresh_picker(self):
        if self.arch_id is None:
            return
        archs = self.arch_store.list()
        self.net_map = {f"{a['name']}  ·  {a['id'][-6:]}": a["id"] for a in archs}
        self.net_picker.configure(values=list(self.net_map))
        self.net_arch_var.set(next((k for k, v in self.net_map.items() if v == self.arch_id), ""))

    def net_refresh_history(self):
        if self.arch_id is None:
            return
        self.rev_tree.delete(*self.rev_tree.get_children())
        head = None
        for row in self.arch_store.revisions(self.arch_id):
            head = row["number"] if head is None else head
            self.rev_tree.insert("", "end", iid=str(row["number"]), values=(f"{'● ' if row['number'] == head else ''}r{row['number']}", row["note"] or "edit", row["created_at"][5:16].replace("T", " ")))

    def net_refresh_weights(self):
        if self.arch_id is None:
            return
        self.weights_tree.delete(*self.weights_tree.get_children())
        for row in self.arch_store.weights(self.arch_id):
            loss = "—" if row["loss"] is None else f"{row['loss']:.3f}"
            self.weights_tree.insert("", "end", iid=str(row["version"]), values=(f"v{row['version']}{' ◆' if row['version'] == self.net_init_version else ''}", row["step"] if row["step"] is not None else "—", loss, row["source"], f"r{row['revision']}"))
        self._net_init_label()

    def _net_init_label(self):
        if self.net_init_version is None:
            self.net_init_var.set("Initial weights: random")
            return
        row = next((w for w in self.arch_store.weights(self.arch_id) if w["version"] == self.net_init_version), None)
        self.net_init_var.set("Initial weights: " + (f"v{row['version']} · step {row['step']} · loss {row['loss']:.3f}" if row and row["loss"] is not None else f"v{self.net_init_version}"))

    def net_refresh_backups(self):
        self.backup_tree.delete(*self.backup_tree.get_children())
        for row in self.arch_store.backups():
            self.backup_tree.insert("", "end", iid=row["path"], values=(row["created_at"][:16].replace("T", " "), row["label"], f"{row['bytes'] / 1024:.0f} KB"))

    # ------------------------------------------------------------------ selection and canvas callbacks
    def net_select(self, key, from_viewer=False, reveal=True):
        self.net_ensure()
        if key and key.startswith("b:") and self._find(self.net_spec, self._block_id(key)) is None:
            key = None
        if reveal and key and key.startswith("b:"):
            self.net_canvas.reveal(self._block_id(key))
        previous, self.net_selected = self.net_selected, key
        self.net_canvas.set_selected(key)
        if key != previous:
            self.net_build_inspector()
            if not from_viewer:
                self.net_send({"type": "select", "block_id": self._block_id(key)})

    def net_canvas_select(self, key):
        self.net_select(key, reveal=False)

    def net_canvas_activate(self, key):
        if key and key.count(":") == 2 and key.startswith("b:"):
            _, bid, kind = key.split(":")
            index = self._find(self.net_spec, bid)
            if index is not None and not self.net_spec["blocks"][index][SUB_KINDS[kind]].get("enabled"):
                self.net_set_block(bid, f"{SUB_KINDS[kind]}.enabled", True)

    def net_delete_selected(self):
        key = self.net_selected
        bid = self._block_id(key)
        if not bid:
            self.status_var.set("Select a block to delete it.")
            return False
        if key.count(":") == 2:
            part = SUB_KINDS[key.split(":")[2]]
            other = "mlp" if part == "attention" else "attention"
            index = self._find(self.net_spec, bid)
            if index is not None and self.net_spec["blocks"][index][other].get("enabled"):
                return self.net_set_block(bid, f"{part}.enabled", False)
        return self.net_remove_block(bid)

    # ------------------------------------------------------------------ palette drag and drop
    def _pal_press(self, event, kind, title):
        self._pal = {"kind": kind, "title": title, "x": event.x_root, "y": event.y_root, "moved": False, "slot": None}

    def _pal_motion(self, event):
        pal = self._pal
        if not pal or (not pal["moved"] and abs(event.x_root - pal["x"]) + abs(event.y_root - pal["y"]) < 6):
            return
        pal["moved"] = True
        canvas = self.net_canvas
        cx, cy = event.x_root - canvas.winfo_rootx(), event.y_root - canvas.winfo_rooty()
        if 0 <= cx < canvas.winfo_width() and 0 <= cy < canvas.winfo_height() and canvas.geo:
            pal["slot"] = canvas.slot_at(cx)
            canvas.show_marker(pal["slot"])
            canvas.show_ghost(cx, cy, pal["title"].split()[0].upper())
        else:
            pal["slot"] = None
            canvas.hide_marker()
            canvas.hide_ghost()

    def _pal_release(self, event):
        pal, self._pal = self._pal, None
        if not pal:
            return
        self.net_canvas.hide_marker()
        self.net_canvas.hide_ghost()
        if not pal["moved"]:
            self.net_insert_block(pal["kind"])
        elif pal["slot"] is not None:
            self.net_insert_block(pal["kind"], pal["slot"])

    # ------------------------------------------------------------------ inspector
    def net_refresh_inspector(self):
        if self.net_selected and self.net_selected.startswith("b:") and self._find(self.net_spec, self._block_id()) is None:
            self.net_selected = None
            self.net_canvas.set_selected(None)
            self.net_build_inspector()
        elif not self.ins_frame.winfo_children():
            self.net_build_inspector()
        else:
            self.net_sync_inspector()

    def net_build_inspector(self):
        for child in self.ins_frame.winfo_children():
            child.destroy()
        self.net_ins = []
        key = self.net_selected
        box = card(self.ins_frame)
        box.pack(fill="x")
        bid = self._block_id(key)
        if key is None:
            label(box, "NETWORK", "CardTitle.TLabel").pack(anchor="w")
            label(box, "Select a chip on the circuit to edit it.\n\n• Drag parts from the palette onto the board\n• Drag a block sideways to reorder\n• Mouse wheel zooms, drag empty space pans\n• Delete removes · Ctrl+Z / Ctrl+Y undo and redo\n• Double-click a dashed slot to switch it on",
                  "CardMuted.TLabel", justify="left", wraplength=290).pack(anchor="w", pady=(8, 0))
        elif key == "emb":
            head = ttk.Frame(box, style="Card.TFrame")
            head.pack(anchor="w")
            label(head, "TOKEN EMBEDDING", "CardTitle.TLabel").pack(side="left")
            help_icon(head, "embedding").pack(side="left", padx=(4, 0))
            label(box, "Turns tokens into vectors and adds positions. The output head reuses this table.", "CardMuted.TLabel", wraplength=290, justify="left").pack(anchor="w", pady=(4, 10))
            self._ins_grid(box, [("Tokenizer", "choice", ["character", "byte"], str, lambda: self._dig("embedding.tokenizer"), lambda v: self.net_set_embedding("tokenizer", v), None),
                                 ("Width", "spin", (16, 512, 16), int, lambda: self._dig("embedding.width"), lambda v: self.net_set_embedding("width", v), None),
                                 ("Context", "spin", (32, 1024, 32), int, lambda: self._dig("embedding.context"), lambda v: self.net_set_embedding("context", v), None),
                                 ("Dropout", "entry", None, float, lambda: self._dig("embedding.dropout"), lambda v: self.net_set_embedding("dropout", v), None)])
        elif key == "train":
            head = ttk.Frame(box, style="Card.TFrame")
            head.pack(anchor="w")
            label(head, "TRAINING (THIS DESIGN)", "CardTitle.TLabel").pack(side="left")
            help_icon(head, "net_training").pack(side="left", padx=(4, 0))
            label(box, "These are used by Train this architecture. They are stored with this design and are separate from the Training setup page, "
                       "which only supplies the data.", "CardMuted.TLabel", wraplength=250, justify="left").pack(anchor="w", pady=(4, 10))
            self._ins_grid(box, [(title, kind, values, cast, (lambda k=k: self._dig("training." + k)), (lambda v, k=k: self.net_set_training(k, v)), None)
                                 for k, title, kind, values, cast in TRAIN_FIELDS])
        elif key in ("final", "out"):
            title, text = (("FINAL NORM", "A LayerNorm over the residual stream before the output head. It has a gain and bias per channel and no tunable options.")
                           if key == "final" else ("OUTPUT HEAD", "Projects the final vectors back onto the vocabulary with the embedding table's weights (weight tying), so it adds no parameters."))
            label(box, title, "CardTitle.TLabel").pack(anchor="w")
            label(box, text, "CardMuted.TLabel", wraplength=290, justify="left").pack(anchor="w", pady=(6, 0))
        elif bid and self._find(self.net_spec, bid) is not None:
            self._block_inspector(box, bid, key)
        self.net_sync_inspector()

    def _block_inspector(self, box, bid, key):
        index = self._find(self.net_spec, bid)
        focus = {"att": "ATTENTION", "mlp": "MLP"}.get(key.split(":")[2], "") if key.count(":") == 2 else ""
        label(box, f"BLOCK {index + 1}" + (f"  ·  {focus}" if focus else ""), "CardTitle.TLabel").pack(anchor="w")
        flow = ttk.Frame(box, style="Card.TFrame")
        flow.pack(anchor="w", pady=(4, 10))
        label(flow, "Residual stream in → (norm → attention) ⊕ → (norm → MLP) ⊕ → out", "CardMuted.TLabel", wraplength=260, justify="left").pack(side="left")
        help_icon(flow, "residual").pack(side="left", padx=(4, 0), anchor="n")
        def other_on(part):
            i = self._find(self.net_spec, bid)
            return i is not None and self.net_spec["blocks"][i]["mlp" if part == "attention" else "attention"].get("enabled")
        def get(path):
            i = self._find(self.net_spec, bid)
            if i is None:
                return None
            node = self.net_spec["blocks"][i]
            for part in path.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            return node
        def setter(path):
            return lambda v: self.net_set_block(bid, path, v)
        self._ins_grid(box, [("Label", "entry", None, str, lambda: get("label"), setter("label"), None)], columns=1)
        self._ins_grid(box, [("Attention on", "bool", None, bool, lambda: get("attention.enabled"), setter("attention.enabled"), lambda: bool(other_on("attention")) or not get("attention.enabled")),
                             ("MLP on", "bool", None, bool, lambda: get("mlp.enabled"), setter("mlp.enabled"), lambda: bool(other_on("mlp")) or not get("mlp.enabled"))], columns=2)
        self._ins_grid(box, [("Heads", "choice", lambda: valid_heads(self._dig("embedding.width")), int, lambda: get("attention.heads"), setter("attention.heads"), lambda: bool(get("attention.enabled"))),
                             ("Activation", "choice", ["gelu", "relu"], str, lambda: get("mlp.activation"), setter("mlp.activation"), lambda: bool(get("mlp.enabled"))),
                             ("Expansion ×", "choice", list(range(1, 9)), int, lambda: get("mlp.expansion"), setter("mlp.expansion"), lambda: bool(get("mlp.enabled"))),
                             ("Hidden layers", "choice", list(range(1, 5)), int, lambda: get("mlp.layers"), setter("mlp.layers"), lambda: bool(get("mlp.enabled")))], columns=2)
        label(box, "Heads offered divide the embedding width.", "CardMuted.TLabel").pack(anchor="w", pady=(2, 8))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x", pady=(4, 0))
        for text, command in (("Duplicate", lambda: self.net_duplicate_block(bid)), ("Delete", lambda: self.net_remove_block(bid)),
                              ("◀", lambda: self._nudge(bid, -1)), ("▶", lambda: self._nudge(bid, 1))):
            ttk.Button(row, text=text, command=command, style="Small.TButton", width=9 if len(text) > 1 else 3).pack(side="left", padx=(0, 6))

    def _nudge(self, bid, step):
        index = self._find(self.net_spec, bid)
        if index is not None and 0 <= index + step < len(self.net_spec["blocks"]):
            self.net_move_block(bid, index + step)

    def _dig(self, path, default=None):
        node = self.net_spec
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        return default if node is None else node

    def _ins_grid(self, parent, fields, columns=2):
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(fill="x", pady=(0, 4))
        for c in range(columns):
            grid.columnconfigure(c, weight=1)
        for i, (title, kind, values, cast, get, set_, enabled) in enumerate(fields):
            box = ttk.Frame(grid, style="Card.TFrame")
            box.grid(row=i // columns, column=i % columns, sticky="ew", padx=(0, 10), pady=(0, 8))
            var = tk.BooleanVar() if kind == "bool" else tk.StringVar()
            entry = {"var": var, "get": get, "kind": kind, "values": values, "enabled": enabled, "cast": cast}
            def apply(*_, entry=entry, set_=set_):
                try:
                    raw = entry["var"].get()
                    value = bool(raw) if entry["kind"] == "bool" else entry["cast"](float(raw) if entry["cast"] is int else raw)
                except (TypeError, ValueError):
                    self.net_sync_inspector(force=True)
                    return
                set_(value)
                self.net_sync_inspector(force=True)
            if kind == "bool":
                widget = ttk.Checkbutton(box, text=title, variable=var, style="Card.TCheckbutton", command=apply)
                widget.pack(anchor="w")
                Tooltip(widget, help_text("attention" if title.startswith("Attention") else "mlp") if title.split()[0] in {"Attention", "MLP"} else title)
            else:
                if title in INSPECTOR_HELP:
                    with_help(box, title, INSPECTOR_HELP[title]).pack(anchor="w", pady=(0, 3))
                else:
                    label(box, title, "CardMuted.TLabel").pack(anchor="w", pady=(0, 3))
                if kind == "choice":
                    widget = ttk.Combobox(box, textvariable=var, state="readonly", width=12)
                    widget.bind("<<ComboboxSelected>>", apply)
                elif kind == "spin":
                    low, high, step = values
                    widget = ttk.Spinbox(box, textvariable=var, from_=low, to=high, increment=step, width=12, command=apply)
                    widget.bind("<Return>", apply)
                    widget.bind("<FocusOut>", apply)
                else:
                    widget = ttk.Entry(box, textvariable=var, width=14)
                    widget.bind("<Return>", apply)
                    widget.bind("<FocusOut>", apply)
                widget.pack(fill="x")
            entry["widget"] = widget
            self.net_ins.append(entry)

    def net_sync_inspector(self, force=False):
        try:
            focused = self.focus_get()
        except (KeyError, tk.TclError):
            focused = None
        for entry in self.net_ins:
            widget = entry["widget"]
            try:
                value = entry["get"]()
                if entry["kind"] == "choice":
                    values = entry["values"]() if callable(entry["values"]) else entry["values"]
                    widget.configure(values=[str(v) for v in values])
                if entry["kind"] == "bool":
                    entry["var"].set(bool(value))
                elif (force or widget is not focused) and str(value) != entry["var"].get():
                    entry["var"].set("" if value is None else str(value))
                if entry["enabled"]:
                    state = "normal" if entry["enabled"]() else "disabled"
                    widget.configure(state=state if entry["kind"] != "choice" or state == "disabled" else "readonly")
            except tk.TclError:
                continue

    # ------------------------------------------------------------------ revisions, weights, backups
    @staticmethod
    def _picked(tree):
        selected = tree.selection()
        return selected[0] if selected else None

    def net_restore_revision(self, number=None):
        self.net_ensure()
        self.net_flush()
        number = number if number is not None else self._picked(self.rev_tree)
        if number is None:
            self.status_var.set("Select a revision to restore.")
            return None
        previous = copy.deepcopy(self.net_spec)
        head = self.arch_store.checkout(self.arch_id, int(number))
        self.net_undo_stack.append(previous)
        self.net_redo_stack.clear()
        self.net_spec = self.arch_store.get(self.arch_id)["spec"]
        self.net_refresh_all()
        self.net_send_spec()
        self.status_var.set(f"Restored revision {number} as r{head}")
        return head

    def net_register_checkpoint(self, path=None):
        self.net_ensure()
        path = path or filedialog.askopenfilename(parent=self, title="Register a checkpoint", filetypes=[("Groundwork checkpoint", "*.npz")])
        if not path:
            return None
        try:
            data = load_checkpoint(path)
            self.net_flush()
            version = self.arch_store.add_weights(self.arch_id, path, step=data.get("step"), loss=data.get("best"), source="imported", note=Path(path).name)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Cannot register checkpoint", str(exc), parent=self)
            return None
        self.net_refresh_weights()
        return version

    def net_use_weights(self, version=None):
        self.net_ensure()
        version = version if version is not None else self._picked(self.weights_tree)
        if version is None:
            self.status_var.set("Select a weight version first.")
            return None
        version = int(version)
        self.net_init_version = None if self.net_init_version == version else version
        self.net_refresh_weights()
        return self.net_init_version

    def net_open_weights(self, version=None):
        self.net_ensure()
        version = version if version is not None else self._picked(self.weights_tree)
        if version is None:
            self.status_var.set("Select a weight version first.")
            return False
        try:
            self.chat_checkpoint.set(self.arch_store.weight_path(self.arch_id, int(version)))
        except ValueError as exc:
            messagebox.showerror("Cannot open weights", str(exc), parent=self)
            return False
        self.show_page("chat")
        self.load_chat_model()
        return True

    def net_backup(self, label_text="manual"):
        try:
            path = self.arch_store.backup(label_text)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Backup failed", str(exc), parent=self)
            return None
        self.net_refresh_backups()
        self.status_var.set(f"Backup created · {path}")
        return path

    def net_restore_backup(self, path=None):
        if self.job_kind:
            self.status_var.set("Stop the running task before restoring a backup.")
            return None
        path = path or self._picked(self.backup_tree)
        if not path:
            path = filedialog.askopenfilename(parent=self, title="Restore backup", initialdir=str(self.arch_store.backup_dir), filetypes=[("Architecture backup", "*.zip")])
        if not path:
            return None
        if not messagebox.askyesno("Restore backup", "Replace every architecture, revision and weight file with this backup?\nA pre-restore backup is created first.", parent=self):
            return None
        self.net_commit_job = self._cancel(self.net_commit_job)
        self.net_notes = []
        try:
            report = self.arch_store.restore(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Restore failed", str(exc), parent=self)
            return None
        self.net_spec = None
        self.net_reload(self.arch_id)
        self.status_var.set(f"Backup restored · {report['architectures']} architecture(s)")
        return report

    # ------------------------------------------------------------------ training
    def net_train(self, init_from=None, parent=None):
        """Train the active architecture through the normal mixture flow."""
        self.net_ensure()
        self.net_flush()
        if self.net_error:
            messagebox.showerror("Fix the circuit first", self.net_error, parent=self)
            return False
        try:
            spec = validate_spec(self.net_spec)
            cfg = config_from_spec(spec)
            if init_from is None and self.net_init_version is not None:
                init_from = self.arch_store.weight_path(self.arch_id, self.net_init_version)
                parent = self.net_init_version
        except ValueError as exc:
            messagebox.showerror("Cannot start training", str(exc), parent=self)
            return False
        head = self.arch_store.get(self.arch_id)["head_revision"]
        run = {"arch": self.arch_id, "rev": head, "parent": parent}
        options = {"telemetry_every": 5}
        if init_from:
            options["init_from"] = str(init_from)
        self.net_tele = None
        self.net_canvas.clear_telemetry()
        started = self.start_training(config=cfg, train_options=options, navigate=False, persist_config=False,
                                      on_finished=lambda directory, record: self._net_run_finished(run, directory, record))
        if started:
            self._net_run = run
            self.net_live_button.configure(state="normal")
            self.net_run_var.set("Starting…")
        return bool(started)

    def _net_run_finished(self, run, directory, record):
        self._net_run = None
        directory, version = Path(directory), None
        try:
            best, last = directory / "best.npz", directory / "last.npz"
            path = best if best.exists() else last
            if path.exists() and isinstance(record, dict):
                loss = record.get("best_loss") if path == best else record.get("loss")
                version = self.arch_store.add_weights(run["arch"], path, revision=run["rev"], step=record.get("step"), loss=loss,
                                                      source="train", run_dir=str(directory), note=str(record.get("status", "")), parent=run.get("parent"))
        except ValueError as exc:
            self.status_var.set(f"Trained, but the weights could not be registered · {exc}")
        if run["arch"] == self.arch_id:
            self.net_refresh_weights()
        self.net_live_button.configure(state="disabled")
        if version is not None:
            self.status_var.set(f"Training {record.get('status', 'finished')} · weights registered as v{version}")
        if self._net_live_pending:
            self._net_live_pending = False
            last = directory / "last.npz"
            if last.exists() and run["arch"] == self.arch_id:
                self.net_train(init_from=str(last), parent=version)

    def net_apply_live(self):
        """Stop the running network training, then continue with the edited spec from the saved weights."""
        if not self.net_is_training():
            self.status_var.set("Start training from the Network page first.")
            return False
        self.net_flush()
        if self.net_error:
            messagebox.showerror("Fix the circuit first", self.net_error, parent=self)
            return False
        self._net_live_pending = True
        self.stop_job()
        self.status_var.set("Saving weights, then restarting with your edit…")
        return True

    def training_event(self, event):
        if event.get("type") == "telemetry":
            self.net_telemetry(event)
            return
        super().training_event(event)

    def net_telemetry(self, event):
        self.net_tele = event
        self.net_tele_count += 1
        self.net_run_var.set(f"step {event.get('step', 0)}/{event.get('total', '?')} · loss {event.get('loss', 0):.3f}")
        if self.viz is not None and self.viz.running:
            self.viz.send(event)
        if self._net_paint_job is None:
            wait = max(0, int((0.1 - (time.monotonic() - self._net_last_paint)) * 1000))
            self._net_paint_job = self.after(wait, self._net_paint)

    def _net_paint(self):
        self._net_paint_job = None
        self._net_last_paint = time.monotonic()
        if self.net_active() and self.net_tele:
            self.net_canvas.set_telemetry(self.net_tele)

    # ------------------------------------------------------------------ 3D viewer
    def net_send(self, message):
        if self.viz is not None and self.viz.running:
            self.viz.send(message)

    def net_send_spec(self):
        if self.viz is not None and self.viz.running and not self.net_error and self.net_spec is not None:
            self.viz.send({"type": "spec", "spec": validate_spec(self.net_spec), "selected": self._block_id()})

    def net_open_viewer(self):
        script = Path(__file__).with_name("studio_brain3d.py")
        if not script.exists():
            self.status_var.set("The 3D viewer (studio_brain3d.py) is not installed yet.")
            self.net_msg.configure(text="The 3D viewer (studio_brain3d.py) is not installed yet.", style="Err.TLabel")
            return False
        self.net_ensure()
        if self.net_error:
            messagebox.showerror("Fix the circuit first", self.net_error, parent=self)
            return False
        if self.viz is None:
            self.viz = VizBridge(script)
        try:
            self.viz.start()
        except OSError as exc:
            messagebox.showerror("Cannot open the 3D viewer", str(exc), parent=self)
            return False
        self.net_send_spec()
        if self.net_tele:
            self.viz.send(self.net_tele)
        self.status_var.set("3D brain opened · edits and telemetry are mirrored live")
        return True

    def net_handle_viewer(self, message):
        """Apply one viewer→GUI message through the same edit path as the inspector."""
        self.net_ensure()
        kind = message.get("type")
        if kind == "select":
            bid = message.get("block_id")
            self.net_select(f"b:{bid}" if bid else None, from_viewer=True)
            return True
        if kind == "closed":
            self.status_var.set("3D viewer closed")
            return True
        applied = False
        if kind == "edit":
            allowed = {"attention.heads", "mlp.expansion", "mlp.layers", "attention.enabled", "mlp.enabled"}
            if message.get("path") in allowed:
                applied = self.net_set_block(message.get("block_id"), message["path"], message.get("value"), strict=True)
        elif kind == "add_block":
            after = message.get("after")
            index = self._find(self.net_spec, after) if after else None
            applied = self.net_insert_block("transformer", len(self.net_spec["blocks"]) if index is None else index + 1) is not None
        elif kind == "remove_block":
            applied = self.net_remove_block(message.get("block_id"))
        else:
            return False
        if applied:
            self.net_flush()
        self.net_send_spec()
        if kind in ("add_block", "remove_block") and applied:
            self.net_send({"type": "select", "block_id": self._block_id()})
        return applied

    def net_poll_viewer(self):
        if self.viz is None:
            return
        for message in self.viz.poll():
            try:
                self.net_handle_viewer(message)
            except Exception as exc:  # a bad viewer message must never stop the GUI poll loop
                self.status_var.set(f"Viewer message ignored · {exc}")

    def poll(self):
        self.net_poll_viewer()
        super().poll()

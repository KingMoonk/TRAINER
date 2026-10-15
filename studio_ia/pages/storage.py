"""Storage (in Resources): how much history the app keeps on disk, and deleting it by age, rendered from services.storage()."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

from studio_ia.copy import EXPORT as X
from studio_ia.copy import STORAGE as T
from studio_ia.pages.base import Page
from studio_ia.services import plain_error
from studio_ia.theme import C, GUTTER, tk_label
from studio_ia.widgets import info_icon, tip

CHOICES = {"Older than 1 day": (1, False), "Older than 7 days": (7, False), "Older than 30 days": (30, False), "Older than 90 days": (90, False), "All history": (None, False)}
ALL = "All history"


def size_text(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def storage_state(services, choice, data=False):
    """Window-free state of the Storage card: (days, data, checkbox enabled, line text, button enabled).
    `data` only counts with "All history"."""
    days = CHOICES[choice][0]
    data = bool(data) and choice == ALL
    items, size = services.storage(days, data)
    line = T["storage_frees"].format(size=size_text(size), n=items) if items else T["storage_nothing"]
    return days, data, choice == ALL, line, items > 0


def export_rows(versions):
    """[(number, name, why GGUF is unavailable or "")] for each version that has trained, in order."""
    import studio_export
    from studio_model import effective_spec, validate_config
    return [(v.number, v.name, studio_export.gguf_problem(effective_spec(validate_config(v.spec)))) for v in versions if v.best_checkpoint]


def export_default_name(name):
    """A file name stem for a version: its name with anything a file system dislikes turned into a dash."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name.strip()).strip("-") or "model"


def export_run(services, number, name, kind, path):
    """The line to show after exporting version `number` to `path`: what was written, or the plain reason it was not."""
    try:
        report = services.export_version(number, kind, path)
    except (ValueError, OSError) as error:
        return plain_error(error, name)
    return X["done"].format(name=name, path=report["path"], note=report.get("note", "")).strip()


class StoragePage(Page):
    title = T["h1"]
    lead = T["lead"]

    choice, data = "Older than 30 days", False

    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        card = ttk.Frame(self.body, padding=GUTTER)
        card.pack(fill="x", pady=(0, GUTTER))
        head = tk.Frame(card, bg=C["bg"])
        head.pack(anchor="w")
        tk_label(head, T["kicker"], 12, "muted", bg="bg").pack(side="left")
        info_icon(head, T["desc"], "bg").pack(side="left", padx=(4, 0))
        tk_label(card, T["storage_free"].format(size=size_text(self.services.free_space())), 14, "text2").pack(anchor="w", pady=(6, 0))
        choice = tk.StringVar(value=self.choice)
        data = tk.BooleanVar(value=self.data)
        self.line = tk_label(card, "", 14, "text2")
        self.delete_btn = ttk.Button(card, text=T["storage_delete"], command=self._delete)
        self.data_box = ttk.Checkbutton(card, text=T["storage_data"], variable=data, command=lambda: self._pick(choice, data))
        for label in CHOICES:
            ttk.Radiobutton(card, text=T["storage_all"] if label == ALL else label,
                            value=label, variable=choice, command=lambda: self._pick(choice, data)).pack(anchor="w")
        self.data_box.pack(anchor="w", pady=(6, 0))
        self.line.pack(anchor="w", pady=6)
        self.delete_btn.pack(anchor="w")
        self._vars = (choice, data)
        self._paint_storage()
        self._export_card()

    def _pick(self, choice, data):
        self.choice, self.data = choice.get(), data.get()
        self._paint_storage()

    def _paint_storage(self, text=None):
        days, self.data, box, line, ok = storage_state(self.services, self.choice, self.data)
        if not box:
            self._vars[1].set(False)
        self.data_box.state(["!disabled" if box else "disabled"])
        self.delete_btn.state(["!disabled" if ok else "disabled"])
        self.line.configure(text=text or line)

    def _delete(self):
        from tkinter import messagebox
        days, data, _box, _line, _ok = storage_state(self.services, self.choice, self.data)
        items, size = self.services.storage(days, data)
        if not messagebox.askyesno(T["h1"], T["storage_confirm"].format(n=items, size=size_text(size))):
            return
        try:
            _n, freed = self.services.clean_storage(days, data)
        except RuntimeError as error:
            return self._paint_storage(str(error))
        self.refresh()
        self._paint_storage(T["storage_freed"].format(size=size_text(freed)))

    exported = ""

    def _export_card(self):
        card = ttk.Frame(self.body, padding=GUTTER)
        card.pack(fill="x", pady=(0, GUTTER))
        head = tk.Frame(card, bg=C["bg"])
        head.pack(anchor="w")
        tk_label(head, X["kicker"], 12, "muted", bg="bg").pack(side="left")
        info_icon(head, X["desc"], "bg").pack(side="left", padx=(4, 0))
        rows = export_rows(self.services.versions())
        if not rows:
            tk_label(card, X["none"], 14, "text2").pack(anchor="w", pady=(6, 0))
        for number, name, no_gguf in rows:
            row = ttk.Frame(card)
            row.pack(anchor="w", pady=(6, 0))
            tk_label(row, name, 14, "text2").pack(side="left", padx=(0, GUTTER))
            ttk.Button(row, text=X["hf"], command=lambda n=number, v=name: self._export(n, v, "hf")).pack(side="left")
            gguf = ttk.Button(row, text=X["gguf"], command=lambda n=number, v=name: self._export(n, v, "gguf"))
            gguf.pack(side="left", padx=(6, 0))
            if no_gguf:
                gguf.state(["disabled"])
                tip(gguf, no_gguf)
        self.export_line = tk_label(card, self.exported, 14, "text2", wraplength=720)
        self.export_line.pack(anchor="w", pady=(6, 0))

    def _export(self, number, name, kind):
        from tkinter import filedialog
        stem = export_default_name(name)
        if kind == "hf":
            parent = filedialog.askdirectory(parent=self, title=X["pick_hf"].format(name=name))
            path = str(Path(parent) / stem) if parent else ""   # the model gets a folder of its own inside the one chosen
        else:
            path = filedialog.asksaveasfilename(parent=self, title=X["pick_gguf"].format(name=name), initialfile=f"{stem}.gguf",
                                                defaultextension=".gguf", filetypes=[("GGUF model", "*.gguf")])
        if not path:
            return

        def done(line):
            self.exported = line
            if self.export_line.winfo_exists():
                self.export_line.configure(text=line)
        self.background(lambda: export_run(self.services, number, name, kind, path), done, X["working"])

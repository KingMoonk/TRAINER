"""Storage (in Resources): how much history the app keeps on disk, and deleting it by age, rendered from services.storage()."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from studio_ia.copy import STORAGE as T
from studio_ia.pages.base import Page
from studio_ia.theme import C, GUTTER, tk_label
from studio_ia.widgets import info_icon

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

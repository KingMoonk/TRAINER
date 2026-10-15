"""Your data: three tabs (text, conversations, problems) with counts, search, a table with STATUS, review and actions.

Reads and writes only through services (data, add_data, edit_data, import_data, review_data). Decision logic is in plain
functions at the top so it runs without Tk.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import studio_rules as rules  # names and labels of the rules only
from studio_ia.copy import YOURDATA as T
from studio_ia.pages.base import Page
from studio_ia.pages.pretraining import source_action, source_status
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import _activate, button

KINDS = ("text", "conversations", "problems")
STATUS_COLOR = {"In use": "text2", "Held back": "blue", "Near-duplicate": "amber", "Good": "violet", "Review": "muted",
                "From Chat": "pink", "Ready": "text2"}


def back_link(kind):
    """(label, route) of the back link: text trains in Pretraining, the other two in Post-training."""
    return (T["back_pretraining"], "pretraining") if kind == "text" else (T["back_post"], "post_training")


def visible_rows(rows, only_dups):
    return [r for r in rows if r["status"] == "Near-duplicate"] if only_dups else list(rows)


def dup_label(found, only_dups):
    return T["near_duplicates"].replace("[N]", f"{found:,}") + (" · showing" if only_dups else "")


def shown_line(summary, query):
    """'Showing the first N of M.' when the table holds fewer rows than exist and nothing is filtered, else ''."""
    return T["shown"].format(shown=len(summary.rows), count=summary.count) if not query.strip() and len(summary.rows) < summary.count else ""


def parse_rule(text):
    """A rule key or its plain label (either case) -> key, else None."""
    word = (text or "").strip().casefold()
    return next((k for k in rules.CHECKABLE if word in (k, k.replace("_", " "), rules.LABELS[k].casefold())), None)


def collect_fields(kind, ask, say, initial=None):
    """Ask for the fields of one item (initial = its current fields when editing). ask(prompt, initial) -> str | None;
    say(message) reports a problem. None = cancelled."""
    was = initial or {}
    if kind == "text":
        text = ask(T["ask_text"], was.get("text", ""))
        return {"text": text} if text and text.strip() else None
    if kind == "conversations":
        prompt = ask(T["ask_prompt"], was.get("user", ""))
        answer = ask(T["ask_answer"], was.get("assistant", "")) if prompt else None
        return {"user": prompt, "assistant": answer} if prompt and answer else None
    prompt = ask(T["ask_problem"], was.get("prompt", ""))
    if not prompt:
        return None
    allowed = rules.CHECKABLE if was.get("rule") in rules.KEYED else rules.RULES  # a keyed rule needs the key an import stored
    names = ", ".join(allowed)
    rule = parse_rule(ask(T["ask_rule"].format(rules=names), was.get("rule", "")))
    if rule not in allowed:
        say(T["unknown_rule"].format(rules=names))
        return None
    return {"prompt": prompt, "rule": rule}


class YourDataPage(Page):
    title = T["h1"]
    lead = T["lead"]

    def __init__(self, parent, services, navigate=None, **kw):
        super().__init__(parent, services, navigate, **kw)
        self.kind, self.only_dups, self.summaries, self.selected = "conversations", False, {}, None
        self.query = tk.StringVar()
        style = ttk.Style(self)
        style.configure("YD.Treeview", background=C["panel"], fieldbackground=C["panel"], foreground=C["text"], rowheight=28,
                        borderwidth=0, font=font(14))
        style.configure("YD.Treeview.Heading", background=C["bg"], foreground=C["muted"], font=font(12, bold=True))
        style.map("YD.Treeview", background=[("selected", C["violet_bg"])], foreground=[("selected", C["text"])])
        body = self.body
        self.back = tk_label(body, "", 14, "violet", cursor="hand2", takefocus=1)
        self.back.pack(anchor="w")
        self.tabs = {}
        row = tk.Frame(body, bg=C["bg"])
        row.pack(fill="x", pady=GUTTER)
        for k in KINDS:
            tab = tk.Frame(row, bg=C["panel"], takefocus=1, cursor="hand2", highlightthickness=1, highlightbackground=C["border"],
                           highlightcolor=C["violet"], padx=GUTTER, pady=10)
            tab.pack(side="left", fill="x", expand=True, padx=(0, 8))
            name, used, count = (tk_label(tab, "", s, c, "panel", font=font(s, bold=b)) for s, c, b in ((15, "text", True), (12, "muted", False), (13, "text2", False)))
            for w in (name, used, count):
                w.pack(fill="x")
            for w in (tab, name, used, count):
                _activate(w, lambda k=k: self.show(k))
            self.tabs[k] = (tab, name, used, count)
        self.lead_line = tk_label(body, "", 15, "text")
        self.lead_line.pack(fill="x")
        self.sets_line = tk_label(body, "", 13, "muted")
        self.sets_line.pack(fill="x", pady=(2, 0))
        bar = tk.Frame(body, bg=C["bg"])
        bar.pack(fill="x", pady=GUTTER)
        tk_label(bar, T["search"], 13, "muted").pack(side="left")
        self.entry = tk.Entry(bar, textvariable=self.query, width=40, bg=C["panel2"], fg=C["text"], insertbackground=C["text"],
                              relief="flat", highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["violet"])
        self.entry.pack(side="left", padx=8, ipady=6)
        self.entry.bind("<KeyRelease>", lambda _e: self._search())
        self.table = ttk.Treeview(body, columns=("a", "b", "status"), show="headings", style="YD.Treeview", height=9)
        self.table.pack(fill="x")
        self.table.column("a", width=340)
        self.table.column("b", width=340)
        self.table.column("status", width=130, anchor="w")
        self.table.heading("status", text=T["status"])
        for status, color in STATUS_COLOR.items():
            self.table.tag_configure(status, foreground=C[color])
        self.table.tag_configure("mono", font=font(13, mono=True))
        self.table.bind("<<TreeviewSelect>>", lambda _e: self._select())
        self.shown = tk_label(body, "", 12, "dim")
        self.shown.pack(fill="x")
        self.detail = tk.Frame(body, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=GUTTER, pady=10)
        self.detail.pack(fill="x", pady=(8, 0))
        self.detail_text = tk_label(self.detail, T["pick_row"], 14, "text2", "panel", wraplength=760)
        self.detail_text.pack(fill="x")
        self.review_row = tk.Frame(self.detail, bg=C["panel"])
        ttk.Button(self.review_row, text=T["mark_good"], style="Primary.TButton", command=lambda: self._review(True)).pack(side="left")
        ttk.Button(self.review_row, text=T["discard"], command=lambda: self._review(False)).pack(side="left", padx=8)
        self.edit_button = ttk.Button(self.detail, text=T["edit"], command=self.edit)
        self.note = tk_label(body, "", 13, "muted", wraplength=760)
        self.note.pack(fill="x", pady=GUTTER)
        buttons = tk.Frame(body, bg=C["bg"])
        buttons.pack(fill="x")
        self.add_button = ttk.Button(buttons, style="Primary.TButton", command=self.add)
        self.add_button.pack(side="left")
        ttk.Button(buttons, text=T["import"], command=self.import_files).pack(side="left", padx=8)
        self.dup_button = ttk.Button(buttons, command=self.toggle_duplicates)
        self.dup_button.pack(side="left")
        self.make_button = ttk.Button(buttons, command=self.make)
        self.make_button.pack(side="left", padx=8)
        card = tk.Frame(body, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=GUTTER, pady=10)
        card.pack(fill="x", pady=(GUTTER, 0), before=self.entry.master)  # above search and the table, so it is in view
        tk_label(card, T["collection"], 15, "text", "panel", font=font(15, bold=True)).pack(anchor="w")
        self.collection_desc = tk_label(card, "", 13, "muted", "panel", wraplength=760)
        self.collection_desc.pack(anchor="w")
        self.collection_rows = tk.Frame(card, bg=C["panel"])
        self.collection_rows.pack(fill="x", pady=(6, 0))
        self.notes = {}  # source id -> latest progress line
        _activate(self.back, self.go_back)

    # navigation ---------------------------------------------------------------------
    def show(self, kind):
        """Open on one tab (links from other pages call this after navigating here)."""
        self.kind, self.only_dups, self.selected = kind, False, None
        self.query.set("")
        self.reload()

    def go_back(self):
        self.navigate(back_link(self.kind)[1])

    def make(self):
        page = self.navigate("make_data")
        if page is not None:
            page.show(self.kind)

    # data ---------------------------------------------------------------------------
    def _load(self, kind):
        self.summaries[kind] = self.services.data(kind, self.query.get().strip() if kind == self.kind else "")

    def preload(self):
        for k in KINDS:
            self.services.data(k)

    def refresh(self):
        for k in KINDS:
            self._load(k)
        self._render()

    def _search(self):
        self.selected = None
        self._load(self.kind)
        self._render()

    def _render(self):
        kind, s = self.kind, self.summaries[self.kind]
        for k, (tab, name, used, count) in self.tabs.items():
            on, sk = k == kind, self.summaries[k]
            bg = C["panel2"] if on else C["panel"]
            tab.configure(bg=bg, highlightbackground=C["violet"] if on else C["border"])
            name.configure(text=T[f"{k}_tab"], bg=bg)
            used.configure(text=sk.used_by, bg=bg)
            count.configure(text=sk.count_line, bg=bg)
        self.back.configure(text="‹ " + back_link(kind)[0])
        self.lead_line.configure(text=T[f"{kind}_lead"])
        self.sets_line.configure(text=" · ".join(f"{x['name']} ({x['count']:,})" for x in s.sets) or T["no_sets"])
        self.table.heading("a", text=T[f"{kind}_col_a"])
        self.table.heading("b", text=T[f"{kind}_col_b"])
        self.table.delete(*self.table.get_children())
        for r in visible_rows(s.rows, self.only_dups):
            self.table.insert("", "end", iid=r["id"], values=(r["a"], r["b"], r["status"]),
                              tags=(r["status"], "mono") if kind == "problems" else (r["status"],))
        if self.selected and self.table.exists(self.selected):
            self.table.selection_set(self.selected)
        self.shown.configure(text=shown_line(s, self.query.get()))
        self.note.configure(text=T[f"{kind}_note"])
        self.add_button.configure(text=T[f"{kind}_add"])
        self.make_button.configure(text=T[f"{kind}_make"])
        self.dup_button.configure(text=dup_label(s.near_duplicates, self.only_dups), state="normal" if s.near_duplicates else "disabled")
        self._detail()
        self.paint_collection()

    # recommended collection of the open kind ------------------------------------------
    def paint_collection(self):
        kind = self.kind
        try:
            rows = self.services.collection(kind)
        except NotImplementedError:
            rows = []
        self.collection_desc.configure(text=T[f"{kind}_collection_desc"])
        for child in self.collection_rows.winfo_children():
            child.destroy()
        for row in rows:
            line = tk.Frame(self.collection_rows, bg=C["panel"])
            line.pack(fill="x", pady=1)
            tk_label(line, row["name"], 14, "text", "panel").pack(side="left")
            tk_label(line, f"{row['size']} · {row['records']}", 12, "dim", "panel").pack(side="left", padx=10)
            label, action = source_action(row)
            if action:
                label = T["download_add"] if action == "download" and kind != "text" else label
                button(line, label, lambda r=row["id"], a=action: self.fetch(r, a)).pack(side="right")
            tk_label(line, source_status(row, self.notes.get(row["id"], "")), 12,
                     "coral" if row.get("error") else "muted", "panel").pack(side="right", padx=10)
        if any(source_action(r)[1] == "download" for r in rows):
            button(self.collection_rows, T["download_add_all"] if kind != "text" else T["download_all"],
                   lambda: self.fetch(None, "download"), primary=True).pack(anchor="w", pady=(8, 0))

    def fetch(self, source_id, action):
        """Download (and, outside text, add) one source or the whole collection; import a downloaded one."""
        ids = None if source_id is None else [source_id]
        started = (self.services.download_sources(ids, kind=self.kind, install=self.kind != "text") if action == "download"
                   else self.services.import_source(source_id))
        if started:
            self.paint_collection()

    def on_event(self, kind, payload):
        if kind == "download":
            self.notes[payload["id"]] = payload["text"]
            self.paint_collection()
        elif kind in ("download_done", "import_done", "download_failed"):
            self.paint_collection()  # cheap: manifests and the set list only
        elif kind == "downloads_finished":
            self.quiet_reload()  # re-counting every row is slow; the page stays usable while it runs

    def _row(self):
        return next((r for r in self.summaries[self.kind].rows if r["id"] == self.selected), None)

    def _select(self):
        picked = self.table.selection()
        self.selected = picked[0] if picked else None
        self._detail()

    def _detail(self):
        row = self._row()
        self.detail_text.configure(text=T["pick_row"] if row is None else f"{row['a']}\n\n{row['b']}")
        if row is not None and row["status"] == "Review":
            self.review_row.pack(anchor="w", pady=(8, 0))
        else:
            self.review_row.pack_forget()
        self.edit_button.pack_forget()
        if row is not None:
            self.edit_button.pack(anchor="w", pady=(8, 0))

    def _review(self, keep):
        if self.selected:
            self.services.review_data(self.selected, keep)
            self.selected = None
            self.reload()

    # actions ------------------------------------------------------------------------
    def toggle_duplicates(self):
        self.only_dups = not self.only_dups
        self._render()

    def _ask(self, prompt, initial=""):
        return simpledialog.askstring(T["dialog"], prompt, parent=self, initialvalue=initial)

    def edit(self):
        if not self.selected:
            return
        fields = collect_fields(self.kind, self._ask, lambda m: messagebox.showinfo(T["dialog"], m, parent=self),
                                self.services.data_fields(self.selected))
        if fields is not None:
            try:
                self.services.edit_data(self.selected, fields)
            except ValueError as error:
                messagebox.showinfo(T["dialog"], T["edit_failed"].format(why=error), parent=self)
            self.reload()

    def add(self):
        ask = self._ask
        fields = collect_fields(self.kind, ask, lambda m: messagebox.showinfo(T["dialog"], m, parent=self))
        if fields is not None and not self.services.add_data(self.kind, fields):
            messagebox.showinfo(T["dialog"], T["already"], parent=self)
        self.reload()

    def import_files(self):
        paths = filedialog.askopenfilenames(parent=self)
        if paths:
            kind = self.kind

            def imported(lines):
                messagebox.showinfo(T["dialog"], "\n".join(lines), parent=self)
                self.reload()
            self.background(lambda: self.services.import_data(kind, paths), imported, "Importing your files")

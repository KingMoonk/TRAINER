"""Your data: three tabs (text, conversations, problems) with counts, search, a table with STATUS, review and actions.

Reads and writes only through services (data, data_page, add_data, edit_data, import_data, review_data). Decision logic is in
plain functions and classes at the top so it runs without Tk. The summaries give the counts; the table is read a page at a time
(data_page) on the shell's worker, so a search reaches every row and a typing burst costs one read.
"""
from __future__ import annotations

import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk

import studio_rules as rules  # names and labels of the rules only
from studio_ia.copy import YOURDATA as T
from studio_ia.pages.base import Page, Throttle
from studio_ia.pages.pretraining import collection_row, source_action, source_status
from studio_ia.theme import C, GUTTER, entry, font, tk_label
from studio_ia.widgets import _activate, button, tip

KINDS = ("text", "conversations", "problems")
PAGE = 200  # rows the table gains per read
SEARCH_DELAY = 250  # ms after the last key before a search runs
PROGRESS_EVERY = 1.0  # seconds between repaints of a download's status line
CELL_PAD = 16  # px of a table cell that text cannot use
CELL_READ = 2000  # characters of a cell looked at; the rest could never fit
STATUS_COLOR = {"In use": "text2", "Held back": "blue", "Near-duplicate": "amber", "Good": "violet", "Review": "muted",
                "From Chat": "pink", "Ready": "text2"}


def fit_cell(text, width, measure):
    """One line for a table cell: every run of whitespace (newlines too) becomes one space, and the line is cut with an
    ellipsis at the last whole character that fits `width` pixels as measure(text) reports them. The full text is in the detail."""
    text = str(text)
    line = " ".join(text[:CELL_READ].split())
    if len(text) <= CELL_READ and measure(line) <= width:
        return line
    low, high = 0, len(line)  # the longest prefix that still fits with the ellipsis
    while low < high:
        middle = (low + high + 1) // 2
        low, high = (middle, high) if measure(line[:middle].rstrip() + "…") <= width else (low, middle - 1)
    return line[:low].rstrip() + "…"


def back_link(kind):
    """(label, route) of the back link: text trains in Pretraining, the other two in Post-training."""
    return (T["back_pretraining"], "pretraining") if kind == "text" else (T["back_post"], "post_training")


def visible_rows(rows, only_dups):
    return [r for r in rows if r["status"] == "Near-duplicate"] if only_dups else list(rows)


def dup_label(found, only_dups):
    return T["near_duplicates"].replace("[N]", f"{found:,}") + (" · showing" if only_dups else "")


def shown_line(loaded, count, query, more):
    """'Showing the first N of M.' (of the matches, when searching) while more rows wait behind Load more, else ''."""
    if not more:
        return ""
    return T["shown_matches"].format(shown=loaded) if query.strip() else T["shown"].format(shown=loaded, count=count)


class Search:
    """Debounce for the search box plus a token that drops stale results. The page hands in its own after/after_cancel, so this
    runs without Tk. run(token) starts a read; its result is used only while wanted(token)."""

    def __init__(self, after, cancel, run, delay=SEARCH_DELAY):
        self.after, self.cancel, self.run, self.delay = after, cancel, run, delay
        self.token, self.timer = 0, None

    def type(self):
        """A key was pressed: search `delay` ms after the last one, and drop any read still in flight."""
        self.claim()
        self.timer = self.after(self.delay, self._fire)

    def _fire(self):
        self.timer = None
        self.run(self.claim())

    def claim(self):
        """A new token, now: the pending search is cancelled and every read made with an older token is stale."""
        if self.timer is not None:
            self.cancel(self.timer)
            self.timer = None
        self.token += 1
        return self.token

    def wanted(self, token):
        return token == self.token


def collection_shown(summary):
    """The recommended collection is first-run guidance: it shows while the open kind has no sets."""
    return not summary.sets


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
        self.rows, self.cursor, self.applied, self.loading_more = [], None, "", False  # the table: loaded rows, where the next page starts, the query they answer
        self.search = Search(self.after, self.after_cancel, lambda token: self._request(token, None))
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
            tip(tab, T[f"{k}_lead"])
            _activate(tab, lambda k=k: self.show(k))
            for w in (name, used, count):
                w.bind("<Button-1>", lambda _e, k=k: self.show(k))
            self.tabs[k] = (tab, name, used, count)
        self.sets_line = tk_label(body, "", 13, "muted", wraplength=760)
        self.sets_line.pack(fill="x", pady=(GUTTER, 0))
        bar = tk.Frame(body, bg=C["bg"])
        bar.pack(fill="x", pady=GUTTER)
        tk_label(bar, T["search"], 13, "muted").pack(side="left")
        self.entry = entry(bar, 14, textvariable=self.query, width=40)
        self.entry.pack(side="left", padx=8)
        self.entry.bind("<KeyRelease>", lambda _e: self._typed())
        self.table = ttk.Treeview(body, columns=("a", "b", "status"), show="headings", style="YD.Treeview", height=9)
        self.table.pack(fill="x")
        self.table.column("a", width=340)
        self.table.column("b", width=340)
        self.table.column("status", width=110, anchor="w", stretch=False)
        for column in ("a", "b", "status"):
            self.table.heading(column, anchor="w")  # the cells are left-aligned
        self.table.heading("status", text=T["status"])
        self.cell_fonts, self.fitted = (tkfont.Font(root=self, font=font(14)), tkfont.Font(root=self, font=font(13, mono=True))), None
        self.table.bind("<Configure>", lambda _e: self._resized())
        for status, color in STATUS_COLOR.items():
            self.table.tag_configure(status, foreground=C[color])
        self.table.tag_configure("mono", font=font(13, mono=True))
        self.table.bind("<<TreeviewSelect>>", lambda _e: self._select())
        self.shown = tk_label(body, "", 12, "dim")
        self.shown.pack(fill="x")
        self.more_button = ttk.Button(body, text=T["load_more"], command=self.load_more)
        self.detail = tk.Frame(body, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=GUTTER, pady=10)
        self.detail.pack(fill="x", pady=(8, 0))
        self.detail_text = tk_label(self.detail, T["pick_row"], 14, "text2", "panel", wraplength=760)
        self.detail_text.pack(fill="x")
        self.review_row = tk.Frame(self.detail, bg=C["panel"])
        ttk.Button(self.review_row, text=T["mark_good"], style="Primary.TButton", command=lambda: self._review(True)).pack(side="left")
        ttk.Button(self.review_row, text=T["discard"], command=lambda: self._review(False)).pack(side="left", padx=8)
        self.edit_button = ttk.Button(self.detail, text=T["edit"], command=self.edit)
        tip(self.detail, lambda: T[f"{self.kind}_note"])
        buttons = self.buttons = tk.Frame(body, bg=C["bg"])
        buttons.pack(fill="x", pady=(GUTTER, 0))
        self.add_button = ttk.Button(buttons, style="Primary.TButton", command=self.add)
        self.add_button.pack(side="left")
        ttk.Button(buttons, text=T["import"], command=self.import_files).pack(side="left", padx=8)
        self.dup_button = ttk.Button(buttons, command=self.toggle_duplicates)
        self.dup_button.pack(side="left")
        self.make_button = ttk.Button(buttons, command=self.make)
        self.make_button.pack(side="left", padx=8)
        card = self.collection_card = tk.Frame(body, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=GUTTER, pady=10)
        tip(tk_label(card, T["collection"], 15, "text", "panel", font=font(15, bold=True)), lambda: T[f"{self.kind}_collection_desc"]).pack(anchor="w")
        self.collection_rows = tk.Frame(card, bg=C["panel"])
        self.collection_rows.pack(fill="x", pady=(6, 0))
        self.notes, self.collections, self.status_labels, self.painted = {}, {}, {}, None  # notes: source id -> latest progress line
        self.throttle, self.dirty, self.flush_timer, self.resize_timer = Throttle(PROGRESS_EVERY), set(), None, None
        _activate(self.back, self.go_back)

    # navigation ---------------------------------------------------------------------
    def show(self, kind):
        """Open on one tab (links from other pages call this after navigating here)."""
        self.kind, self.only_dups, self.selected = kind, False, None
        self.rows, self.cursor, self.applied = [], None, ""
        self.query.set("")
        self._paint_rows()
        self.reload()

    def go_back(self):
        self.navigate(back_link(self.kind)[1])

    def make(self):
        page = self.navigate("make_data")
        if page is not None:
            page.show(self.kind)

    # data ---------------------------------------------------------------------------
    def preload(self):
        for k in KINDS:
            self.summaries[k] = self.services.data(k)

    def refresh(self):
        for k in KINDS:
            if k not in self.summaries:
                self.summaries[k] = self.services.data(k)
        self._render()
        self._collect()
        self._request(self.search.claim(), None)

    def _typed(self):
        if self.query.get().strip() != self.applied:
            self.search.type()

    def _request(self, token, cursor):
        """Read a page of the open kind on the worker: the first page (cursor None) or the one after `cursor`."""
        kind = self.kind
        if cursor is None:
            self.applied, self.loading_more = self.query.get().strip(), False
        query = self.applied  # a later page answers the query the table shows, not what is being typed
        self.quietly(lambda: self.services.data_page(kind, query, cursor, PAGE), lambda result: self._arrived(token, kind, cursor, result))

    def _arrived(self, token, kind, cursor, result):
        if not self.search.wanted(token) or kind != self.kind:  # typed on, or moved to another tab, while it was being read
            return
        rows, self.cursor = result
        self.loading_more = False
        self.more_button.configure(state="normal")
        if cursor is not None:
            self.rows += rows
            self._paint_rows(len(self.rows) - len(rows))
        elif rows != self.rows:
            self.rows = rows
            self._paint_rows()
        else:
            self._paint_rows(len(rows))  # the same rows again (a reload): leave the table as it is

    def _resized(self):
        """The columns got wider or narrower: cut the cells again once the dragging stops."""
        if self.fitted != tuple(self.table.column(c, "width") - CELL_PAD for c in ("a", "b")):
            if self.resize_timer is not None:
                self.after_cancel(self.resize_timer)
            self.resize_timer = self.after(150, self._refit)

    def _refit(self):
        self.resize_timer = None
        self._paint_rows()

    def load_more(self):
        if self.cursor is not None and not self.loading_more:
            self.loading_more = True
            self.more_button.configure(state="disabled")
            self._request(self.search.token, self.cursor)

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
        self.sets_line.configure(text=" · ".join(f"{x['name']} ({x['count']:,})" for x in s.sets) or T["no_sets"])
        self.table.heading("a", text=T[f"{kind}_col_a"])
        self.table.heading("b", text=T[f"{kind}_col_b"])
        self.add_button.configure(text=T[f"{kind}_add"])
        self.make_button.configure(text=T[f"{kind}_make"])
        self.dup_button.configure(text=dup_label(s.near_duplicates, self.only_dups), state="normal" if s.near_duplicates else "disabled")
        self.paint_collection()

    def _paint_rows(self, keep=0):
        """Show self.rows; the first `keep` of them are already in the table (a page was appended, or nothing changed)."""
        if not keep:
            self.table.delete(*self.table.get_children())
        mono = self.kind == "problems"
        measure = self.cell_fonts[mono].measure
        wa, wb = self.fitted = tuple(self.table.column(c, "width") - CELL_PAD for c in ("a", "b"))
        for r in visible_rows(self.rows[keep:], self.only_dups):
            self.table.insert("", "end", iid=r["id"], values=(fit_cell(r["a"], wa, measure), fit_cell(r["b"], wb, measure), r["status"]),
                              tags=(r["status"], "mono") if mono else (r["status"],))
        if self.selected and self.table.exists(self.selected):
            self.table.selection_set(self.selected)
        else:
            self.selected = None
        summary = self.summaries.get(self.kind)
        self.shown.configure(text=shown_line(len(self.rows), summary.count if summary else 0, self.applied, self.cursor is not None))
        if self.cursor is None:
            self.more_button.pack_forget()
        else:
            self.more_button.pack(anchor="w", pady=(4, 0), after=self.shown)
        self._detail()

    # recommended collection of the open kind ------------------------------------------
    def paint_collection(self):
        kind = self.kind
        if kind not in self.summaries or collection_shown(self.summaries[kind]):  # below the table, and only while the kind is empty: Next says what to do first
            self.collection_card.pack(fill="x", pady=(GUTTER, 0), after=self.buttons)
        else:
            self.collection_card.pack_forget()
        rows = self.collections.get(kind, [])
        if self.painted == (kind, rows):  # only the status lines move while a download runs; they are updated in place
            return
        self.painted = (kind, rows)
        self.status_labels = {}
        for child in self.collection_rows.winfo_children():
            child.destroy()
        for row in rows:
            self.status_labels[row["id"]] = collection_row(self.collection_rows, row, self.notes.get(row["id"], ""), self.fetch)
        if any(source_action(r)[1] == "download" for r in rows):
            button(self.collection_rows, T["download_add_all"],
                   lambda: self.fetch(None, "download"), primary=True).pack(anchor="w", pady=(8, 0))

    def _collect(self):
        """Read the open kind's collection (manifests on disk) on the worker and repaint it when it differs. Only a download
        finishing or failing, a download starting or the page opening asks again; progress never does."""
        kind = self.kind

        def work():
            try:
                return self.services.collection(kind)
            except NotImplementedError:
                return []
        self.quietly(work, lambda rows: self._collected(kind, rows))

    def _collected(self, kind, rows):
        self.collections[kind] = rows
        if kind == self.kind:
            self.paint_collection()

    def fetch(self, source_id, action):
        """Download and add one source or the whole collection; add one downloaded earlier."""
        ids = None if source_id is None else [source_id]
        started = (self.services.download_sources(ids, kind=self.kind, install=True) if action == "download"
                   else self.services.import_source(source_id))
        if started:
            self._collect()

    def on_event(self, kind, payload):
        if kind == "download":
            self.notes[payload["id"]] = payload["text"]
            self.dirty.add(payload["id"])
            self._flush_soon()
        elif kind in ("download_done", "import_done", "download_failed"):
            self._collect()
        elif kind == "downloads_finished":
            self.quiet_reload()  # re-counting every row is slow; the page stays usable while it runs

    def _flush_soon(self):
        """Progress arrives several times a second: repaint the changed status lines at most once per PROGRESS_EVERY."""
        if self.flush_timer is not None:
            return
        now = time.monotonic()
        if self.throttle.ready(now):
            self._flush_notes()
        else:
            self.flush_timer = self.after(int(self.throttle.wait(now) * 1000) + 20, self._flush_later)

    def _flush_later(self):
        self.flush_timer = None
        self.throttle.ready(time.monotonic())
        self._flush_notes()

    def _flush_notes(self):
        rows = {r["id"]: r for r in self.collections.get(self.kind, [])}
        for source_id in self.dirty:
            label, row = self.status_labels.get(source_id), rows.get(source_id)
            if label is not None and row is not None:  # a progress line means the source is being fetched, whatever the rows read last said
                label.configure(text=source_status({**row, "state": "downloading", "error": ""}, self.notes[source_id]), fg=C["muted"])
        self.dirty.clear()

    def _row(self):
        return next((r for r in self.rows if r["id"] == self.selected), None)

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
        self._paint_rows()

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

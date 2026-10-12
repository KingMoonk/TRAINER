"""Shared native widgets and the training-example editor."""
import json
import tkinter as tk
from tkinter import messagebox, ttk
from studio_core import messages_of, normalize
from studio_glossary import GLOSSARY, glossary_text

BG = "#0b101a"
PANEL = "#121b2b"
INPUT = "#0e1624"
EDGE = "#25344b"
TEXT = "#e8edf5"
MUTED = "#92a2b9"
ACCENT = "#65e0bd"
BLUE = "#78a9ff"
FONT = "Segoe UI"


def normalize_scroll_delta(delta):
    """Convert mouse wheel delta to scroll units. Windows uses 120 per unit; macOS uses 1/-1."""
    if delta == 0:
        return 0
    # Windows: delta is a multiple of 120; macOS: delta is 1 or -1
    # Scale by 120 on Windows (int(delta / 120) rounds to 0 for delta=1)
    # On macOS, delta is already ±1, so multiply by 1; on Windows, divide by 120
    if abs(delta) >= 120:
        return delta // 120
    else:
        return 1 if delta > 0 else -1


def label(parent, text="", style="TLabel", **kwargs):
    return ttk.Label(parent, text=text, style=style, **kwargs)


def button(parent, text, command, primary=False):
    widget = ttk.Button(parent, text=text, command=command, style="Accent.TButton" if primary else "TButton")
    widget.pack(side="left", padx=(0, 8), pady=4)
    return widget


def text_box(parent, height=5, **kwargs):
    return tk.Text(parent, height=height, bg=INPUT, fg=TEXT, insertbackground=ACCENT,
                   selectbackground="#355473", relief="flat", borderwidth=0, padx=12,
                   pady=10, font=(FONT, 10), wrap="word", undo=True, **kwargs)


def content(widget):
    return widget.get("1.0", "end-1c").strip()


def set_text(widget, value):
    state = str(widget.cget("state"))
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", value)
    widget.configure(state=state)


def card(parent, title=None):
    frame = ttk.Frame(parent, style="Card.TFrame", padding=16)
    if title:
        label(frame, title, "CardTitle.TLabel").pack(anchor="w", pady=(0, 12))
    return frame


def field(parent, title, variable, values=None, width=20, help=None):
    """Titled input. `help` is a GLOSSARY key: it adds a "?" icon and a tooltip on the input itself."""
    frame = ttk.Frame(parent, style="Card.TFrame")
    if help:
        with_help(frame, title, help).pack(anchor="w", pady=(0, 5))
    else:
        label(frame, title, "CardMuted.TLabel").pack(anchor="w", pady=(0, 5))
    widget = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=width) if values else ttk.Entry(frame, textvariable=variable, width=width)
    widget.pack(fill="x")
    if help:
        Tooltip(widget, help_text(help))
    return frame, widget


def tree(parent, columns, heights=12):
    container = ttk.Frame(parent)
    widget = ttk.Treeview(container, columns=[c[0] for c in columns], show="headings", height=heights, selectmode="extended")
    for name, title, width in columns:
        widget.heading(name, text=title)
        widget.column(name, width=width, minwidth=60, stretch=True)
    scroll = ttk.Scrollbar(container, orient="vertical", command=widget.yview)
    widget.configure(yscrollcommand=scroll.set)
    widget.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    container.pack(fill="both", expand=True)
    return widget


class ScrollPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas)
        window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width))
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.bind_all("<MouseWheel>", self.scroll, add="+")

    def scroll(self, event):
        # The pointer may be over a nested card; text and tables scroll independently.
        if event.widget.winfo_class() in {"Text", "Treeview", "TCombobox"}:
            return
        widget = event.widget
        while widget is not None:
            if widget is self:
                self.canvas.yview_scroll(-normalize_scroll_delta(event.delta), "units")
                return "break"
            widget = widget.master


class ExampleEditor(tk.Toplevel):
    def __init__(self, parent, example=None, title="New training example"):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=BG)
        self.geometry("780x690")
        self.minsize(600, 560)
        self.transient(parent)
        self.result = None
        self.original = example or {}
        outer = ttk.Frame(self, padding=22)
        outer.pack(fill="both", expand=True)
        label(outer, title, "Title.TLabel").pack(anchor="w")
        label(outer, "Teach one response, preserve a full conversation, or add a text corpus.", "Muted.TLabel").pack(anchor="w", pady=(6, 18))
        tabs = self.tabs = ttk.Notebook(outer)
        tabs.pack(fill="both", expand=True)
        pair, turns, raw = [ttk.Frame(tabs, padding=12) for _ in range(3)]
        for frame, name in zip((pair, turns, raw), ("Prompt + response", "Conversation JSON", "Raw text")):
            tabs.add(frame, text=name)
        label(pair, "User prompt").pack(anchor="w", pady=(0, 5))
        self.prompt = text_box(pair, 6)
        self.prompt.pack(fill="both", expand=True)
        label(pair, "Target assistant response").pack(anchor="w", pady=(12, 5))
        self.answer = text_box(pair, 8)
        self.answer.pack(fill="both", expand=True)
        self.turns = text_box(turns, 14)
        self.turns.pack(fill="both", expand=True)
        self.raw = text_box(raw, 14)
        self.raw.pack(fill="both", expand=True)
        set_text(self.prompt, self.original.get("user", ""))
        set_text(self.answer, self.original.get("assistant", ""))
        messages = messages_of(self.original) if example and not example.get("text") else [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
        set_text(self.turns, json.dumps(messages, ensure_ascii=False, indent=2))
        set_text(self.raw, self.original.get("text", ""))
        if self.original.get("text"):
            tabs.select(raw)
        elif len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            tabs.select(turns)
        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(16, 0))
        self.category = tk.StringVar(value=self.original.get("intent", "general"))
        self.review = tk.StringVar(value=self.original.get("quality", "Review"))
        label(bottom, "Category").pack(side="left", padx=(0, 6))
        ttk.Entry(bottom, textvariable=self.category, width=18).pack(side="left", padx=(0, 12))
        ttk.Combobox(bottom, textvariable=self.review, values=("Review", "Good"), state="readonly", width=10).pack(side="left", padx=(0, 15))
        button(bottom, "Save example", self.save, True)
        button(bottom, "Cancel", self.destroy)
        self.grab_set()

    def save(self):
        try:
            index = self.tabs.index(self.tabs.select())
            row = {"user": content(self.prompt), "assistant": content(self.answer)} if index == 0 else {"messages": json.loads(content(self.turns))} if index == 1 else {"text": content(self.raw)}
            row.update(intent=self.category.get().strip() or "general", quality=self.review.get())
            result = normalize(row, self.original.get("datasetId", ""), self.original.get("source", "manual"))
            if not result:
                raise ValueError("Add both a user prompt and an assistant response, or non-empty raw text.")
            if self.original.get("id"):
                result["id"] = self.original["id"]
            self.result = result
            self.destroy()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Check this example", str(exc), parent=self)



# --------------------------------------------------------------------------- help system
USED_HELP_KEYS = set()   # every glossary key referenced by a tooltip or "?" icon (checked by the tests)
TOOLTIP_BG = "#1b2a42"


def help_text(key):
    """Tooltip text for a glossary key. Unknown keys fail loudly so typos are caught in the tests."""
    USED_HELP_KEYS.add(key)
    return glossary_text(key)


class Tooltip:
    """Small themed hover box. Shows after `delay` ms, wraps at `wrap` px, hides on leave/click."""

    def __init__(self, widget, text, delay=450, wrap=320):
        self.widget, self.text, self.delay, self.wrap = widget, text, delay, wrap
        self.window, self._job = None, None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")
        widget.bind("<Destroy>", self.hide, add="+")
        widget._tooltip = self

    def _enter(self, _event=None):
        self._cancel()
        try:
            self._job = self.widget.after(self.delay, self.show)
        except tk.TclError:
            self._job = None

    def _cancel(self):
        if self._job is not None:
            try:
                self.widget.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def show(self, _event=None):
        self._cancel()
        text = self.text() if callable(self.text) else self.text
        if self.window is not None or not text:
            return
        try:
            window = self.window = tk.Toplevel(self.widget)
            window.wm_overrideredirect(True)
            window.configure(bg=EDGE)
            box = tk.Label(window, text=text, justify="left", wraplength=self.wrap, bg=TOOLTIP_BG, fg=TEXT,
                           font=(FONT, 10), padx=12, pady=9)
            box.pack(padx=1, pady=1)
            window.update_idletasks()
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            x = max(0, min(x, window.winfo_screenwidth() - window.winfo_reqwidth() - 8))
            if y + window.winfo_reqheight() > window.winfo_screenheight() - 40:
                y = max(0, self.widget.winfo_rooty() - window.winfo_reqheight() - 6)
            window.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            self.window = None

    def hide(self, _event=None):
        self._cancel()
        window, self.window = self.window, None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass


def _background_of(widget):
    try:
        return widget.cget("bg")
    except (tk.TclError, AttributeError):
        pass
    try:
        style = str(widget.cget("style")) or widget.winfo_class()
        return ttk.Style(widget).lookup(style, "background") or BG
    except (tk.TclError, AttributeError):
        return BG


def help_icon(parent, key):
    """A small "?" that shows the glossary entry on hover and toggles it on click."""
    text = help_text(key)
    icon = tk.Label(parent, text="?", bg=_background_of(parent), fg=ACCENT, font=(FONT, 9, "bold"), cursor="question_arrow", padx=3)
    tip = Tooltip(icon, text)
    icon.bind("<Button-1>", lambda _e: tip.hide() if tip.window else tip.show(), add="+")
    icon.help_key = key
    return icon


def with_help(parent, text, key, style="CardMuted.TLabel", **kwargs):
    """A label followed by its "?" icon, inside a frame matching the parent's background."""
    frame = ttk.Frame(parent, style="Card.TFrame" if style.startswith("Card") else "TFrame")
    label(frame, text, style, **kwargs).pack(side="left")
    help_icon(frame, key).pack(side="left", padx=(4, 0))
    return frame


class GlossaryDialog(tk.Toplevel):
    """Searchable list of every term used in the studio."""

    def __init__(self, parent, term=None):
        super().__init__(parent)
        self.title("Glossary")
        self.configure(bg=BG)
        self.geometry("760x520")
        self.minsize(560, 380)
        self.transient(parent)
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)
        label(outer, "Glossary", "CardTitle.TLabel", background=BG, font=(FONT, 16, "bold")).pack(anchor="w")
        label(outer, "Plain-English meanings for every term in the studio.", "Muted.TLabel").pack(anchor="w", pady=(2, 10))
        self.query = tk.StringVar()
        search = ttk.Entry(outer, textvariable=self.query)
        search.pack(fill="x", pady=(0, 10))
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self.listbox = tk.Listbox(left, width=30, exportselection=False, bg=INPUT, fg=TEXT, selectbackground="#355473", relief="flat",
                                  highlightthickness=0, font=(FONT, 10), activestyle="none")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side="left", fill="y")
        scroll.pack(side="left", fill="y")
        self.detail = text_box(body, 10, state="disabled")
        self.detail.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.detail.tag_configure("term", foreground=ACCENT, font=(FONT, 13, "bold"))
        self.keys = []
        self.query.trace_add("write", lambda *_: self.refresh())
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self.show())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.refresh(term)
        search.focus_set()

    def refresh(self, select=None):
        needle = self.query.get().strip().casefold()
        self.keys = sorted((k for k, (t, d) in GLOSSARY.items() if not needle or needle in t.casefold() or needle in d.casefold() or needle in k),
                           key=lambda k: GLOSSARY[k][0].casefold())
        self.listbox.delete(0, "end")
        for key in self.keys:
            self.listbox.insert("end", GLOSSARY[key][0])
        if self.keys:
            index = self.keys.index(select) if select in self.keys else 0
            self.listbox.selection_set(index)
            self.listbox.see(index)
        self.show()

    def show(self):
        selection = self.listbox.curselection()
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        if selection:
            term, text = GLOSSARY[self.keys[selection[0]]]
            self.detail.insert("end", term + "\n", "term")
            self.detail.insert("end", "\n" + text)
        else:
            self.detail.insert("end", "No matching terms.")
        self.detail.configure(state="disabled")

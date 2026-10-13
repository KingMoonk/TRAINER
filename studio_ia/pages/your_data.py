"""Your data: three tabs (text, conversations, problems) with counts, search, a table with STATUS and actions.

Plain ttk.Frame with refresh(); reads through services.data(kind) when given, else studio_data_kinds.summarize.
ponytail: stand-alone until studio_ia.pages.base / widgets land; swap the Frame base for the shared Page then.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import studio_data_kinds as dk
import studio_rules as rules

BG, PANEL, EDGE, TEXT, MUTED, VIOLET = "#0e1014", "#15181e", "#2a2f3a", "#e6e8ec", "#8f97a6", "#9f8cff"
STATUS_COLOR = {"In use": "#c3c8d1", "Held back": "#6cb4ee", "Near-duplicate": "#e5b567", "Good": VIOLET,
                "Review": "#8f97a6", "From Chat": "#e58fb0", "Ready": "#c3c8d1"}
TABS = {
    "text": dict(name="Text it reads", back="Back to Pretraining", lead="Plain text. It learns by predicting the next piece.",
                 cols=("DOCUMENT", "FIRST LINE"), search="Search text", add="Add a document", make="Generate prose",
                 note="Held-back text is split off by document, after exact duplicates are removed, and never trained on."),
    "conversations": dict(name="Conversations it practises", back="Back to Post-training",
                          lead="A prompt and a good answer. Only the answer is trained on.", cols=("PROMPT", "ANSWER"),
                          search="Search conversations", add="Write a conversation", make="Make extra conversations",
                          note="Open one to read, edit or mark it Good. Corrections you teach in Chat appear here."),
    "problems": dict(name="Problems with a checkable answer", back="Back to Post-training",
                     lead="A prompt and a rule that decides if an answer is right. No answer key needed.", cols=("PROMPT", "RULE"),
                     search="Search problems", add="Write a problem", make="Generate problems a rule can check",
                     note="A rule decides what is right, so these can be generated safely in any amount."),
}


class YourDataPage(ttk.Frame):
    def __init__(self, master, workspace=None, services=None, kind="conversations", on_navigate=None):
        super().__init__(master)
        self.services = services
        self.workspace = workspace or services.workspace
        self.kind = kind
        self.on_navigate = on_navigate or (lambda page, **kw: None)
        self.only_dups = False
        self.summaries = {}
        self.configure(style="YD.TFrame")
        style = ttk.Style(self)
        style.configure("YD.TFrame", background=BG)
        style.configure("YD.Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=26, borderwidth=0)
        style.configure("YD.Treeview.Heading", background=BG, foreground=MUTED)
        tk.Label(self, text="Your data", bg=BG, fg=TEXT, font=("TkDefaultFont", 24, "bold")).pack(anchor="w", padx=16, pady=(14, 0))
        tk.Label(self, text="Three kinds, because each one trains a different stage.", bg=BG, fg=MUTED).pack(anchor="w", padx=16)
        self.back = tk.Button(self, command=lambda: self.on_navigate("pretraining" if self.kind == "text" else "post_training"),
                              bd=0, bg=BG, fg=VIOLET, activeforeground=TEXT)
        self.back.pack(anchor="w", padx=16, pady=4)
        tabs = tk.Frame(self, bg=BG)
        tabs.pack(fill="x", padx=16, pady=6)
        self.tab_buttons = {}
        for k in dk.KINDS:
            b = tk.Button(tabs, justify="left", anchor="w", bd=0, padx=10, pady=6, command=lambda k=k: self.show(k),
                          highlightthickness=1, highlightbackground=EDGE, fg=TEXT)
            b.pack(side="left", expand=True, fill="x", padx=3)
            self.tab_buttons[k] = b
        self.lead = tk.Label(self, bg=BG, fg=TEXT, anchor="w")
        self.lead.pack(fill="x", padx=16)
        self.sets = tk.Label(self, bg=BG, fg=MUTED, anchor="w")
        self.sets.pack(fill="x", padx=16)
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=6)
        tk.Label(bar, text="Search", bg=BG, fg=MUTED).pack(side="left")
        self.query = tk.StringVar()
        entry = tk.Entry(bar, textvariable=self.query, width=40)
        entry.pack(side="left", padx=6)
        entry.bind("<KeyRelease>", lambda _e: self.refresh())
        self.table = ttk.Treeview(self, columns=("a", "b", "status"), show="headings", style="YD.Treeview", height=10)
        self.table.pack(fill="both", expand=True, padx=16)
        self.table.column("a", width=320)
        self.table.column("b", width=320)
        self.table.column("status", width=120, anchor="w")
        self.table.heading("status", text="STATUS")
        for status, color in STATUS_COLOR.items():
            self.table.tag_configure(status, foreground=color)
        self.note = tk.Label(self, bg=BG, fg=MUTED, anchor="w", wraplength=900, justify="left")
        self.note.pack(fill="x", padx=16, pady=6)
        buttons = tk.Frame(self, bg=BG)
        buttons.pack(fill="x", padx=16, pady=(0, 14))
        self.add_button = tk.Button(buttons, command=self.add)
        self.add_button.pack(side="left", padx=3)
        tk.Button(buttons, text="Import files", command=self.import_files).pack(side="left", padx=3)
        self.dup_button = tk.Button(buttons, command=self.toggle_duplicates)
        self.dup_button.pack(side="left", padx=3)
        self.make_button = tk.Button(buttons, command=lambda: self.on_navigate("make_data", kind=self.kind))
        self.make_button.pack(side="left", padx=3)
        self.refresh()

    def show(self, kind):
        self.kind, self.only_dups = kind, False
        self.refresh()

    def _summary(self, kind, query=""):
        if self.services is not None and not query:
            return self.services.data(kind)
        return dk.summarize(self.workspace, kind, query=query)

    def refresh(self):
        tab = TABS[self.kind]
        self.summaries = {k: self._summary(k, self.query.get() if k == self.kind else "") for k in dk.KINDS}
        for k, button in self.tab_buttons.items():
            s, on = self.summaries[k], k == self.kind
            button.configure(text=f"{TABS[k]['name']}\n{dk.USED_BY[k]}\n{s.count_line}", bg="#1b1f27" if on else PANEL,
                             highlightbackground=VIOLET if on else EDGE, activebackground="#1b1f27")
        s = self.summaries[self.kind]
        self.back.configure(text=tab["back"])
        self.lead.configure(text=tab["lead"])
        self.sets.configure(text=" · ".join(f"{x['name']} ({x['count']:,})" for x in s.sets) or "No sets yet")
        self.table.heading("a", text=tab["cols"][0])
        self.table.heading("b", text=tab["cols"][1])
        self.table.delete(*self.table.get_children())
        for r in s.rows:
            if self.only_dups and r["status"] != "Near-duplicate":
                continue
            self.table.insert("", "end", iid=r["id"], values=(r["a"], r["b"], r["status"]), tags=(r["status"],))
        self.note.configure(text=tab["note"])
        self.add_button.configure(text=tab["add"])
        self.dup_button.configure(text=f"Near-duplicates ({s.near_duplicates} found)" + (" · showing" if self.only_dups else ""),
                                  state="disabled" if self.kind == "problems" else "normal")
        self.make_button.configure(text=tab["make"])

    def toggle_duplicates(self):
        self.only_dups = not self.only_dups
        self.refresh()

    def _target(self, name):
        return dk.ensure_dataset(self.workspace, name)

    def add(self):
        ask = lambda t: simpledialog.askstring("Your data", t, parent=self)  # noqa: E731
        if self.kind == "text":
            text = ask("Document text")
            row = {"text": text} if text else None
            name = "Your files"
        elif self.kind == "conversations":
            prompt, answer = ask("Prompt"), None
            answer = ask("Good answer") if prompt else None
            row = {"user": prompt, "assistant": answer, "quality": "Good"} if prompt and answer else None
            name = "Your conversations"
        else:
            prompt = ask("Problem prompt")
            rule = ask("Rule (" + ", ".join(rules.RULES) + ")") if prompt else None
            if rule not in rules.RULES:
                row = None
            else:
                row = dict(dk.make_problem({"prompt": prompt, "rule": rule, "params": {}}), source="manual")
            name = "Your problems"
        if row is None:
            return
        added = dk.add_problems(self.workspace, self._target(name), [{"prompt": row["prompt"], "rule": row["rule"], "params": {}}])[0] \
            if self.kind == "problems" else self.workspace.add(self._target(name), [row], "manual")[0]
        if not added:
            messagebox.showinfo("Your data", "That is already in this set (or empty).", parent=self)
        self.refresh()

    def import_files(self):
        paths = filedialog.askopenfilenames(parent=self)
        if paths:
            self.workspace.import_files(list(paths), self._target({"text": "Your files", "conversations": "Your conversations",
                                                                  "problems": "Your problems"}[self.kind]))
            self.refresh()

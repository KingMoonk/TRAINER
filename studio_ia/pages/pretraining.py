"""Pretraining: Data -> Batches -> Train for the current design version."""
from __future__ import annotations

import tkinter as tk
import uuid

from studio_ia.copy import PRETRAINING as T
from studio_ia.pages.base import Page
from studio_ia.services import Job
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import Card, Kicker, Segmented, button

ROWS = ("8", "16", "32")
HOLD = ("2%", "5%", "10%")
LENGTHS = {"Short": 1, "Medium": 2, "Long": 4}  # passes over the chosen text
RULE = 20  # word pieces per parameter, a rule of thumb only
PALETTE = ("#3a3560", "#2a4558", "#4a3a2a", "#2f4a36", "#4a2f45")
END = "#0a0b0e"


def word_list_changed(v, parent):
    """A different word list (tokenizer) makes a warm start invalid."""
    return parent is not None and v is not None and v.spec.get("tokenizer") != parent.spec.get("tokenizer")


def can_continue(v, parent):
    """Continue needs a parent version with a trained checkpoint and the same word list."""
    return parent is not None and bool(parent.best_checkpoint) and not word_list_changed(v, parent)


def forced_mode(v, parent, requested):
    """The mode a run may use: catch_up only when continuing is valid, otherwise from_scratch."""
    return requested if requested == "from_scratch" or can_continue(v, parent) else "from_scratch"


def steps_for(pieces, per_step, passes):
    """Steps for `passes` passes over `pieces` word pieces at `per_step` pieces a step; None when the size is unknown."""
    return max(1, -(-int(pieces) * passes // per_step)) if pieces and per_step else None


def pretrain_config(spec, steps, rows, hold):
    """Engine config for a plain-text run of `spec`: loss on every piece, `hold` of the text held back for testing."""
    from studio_spec import config_from_spec
    return {**config_from_spec(spec), "steps": steps, "batch_size": rows, "validation": hold, "objective": "all"}


def source_action(row):
    """(button label, service to call) for a collection row; (None, None) while it works or once it is in your sources."""
    return {"not downloaded": (T["download"], "download"), "downloaded": (T["import"], "import")}.get(row["state"], (None, None))


def source_status(row, note=""):
    """The plain-words line under a collection row: a failure first, then live progress, then the state."""
    if row.get("error"):
        return T["failed"].format(error=row["error"].rstrip(".") + ".")
    if row["state"] == "downloading":
        return note or T["state_downloading"]
    return T["state_" + row["state"]]


def sets_of(summary):
    return (summary["sets"] if isinstance(summary, dict) else summary.sets) or []


class PretrainingPage(Page):
    title = T["h1"]
    lead = T["lead"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.use, self.mode = {}, "catch_up"
        self.preview, self.notes = {}, {}  # notes: latest progress line per source being fetched
        self.build()

    def build(self):
        b = self.body
        tk_label(b, T["model_label"], 12, "muted").pack(anchor="w")
        self.model_line = tk_label(b, "", 15)
        self.model_line.pack(anchor="w", pady=(2, GUTTER))
        # data
        self.data_card = self._card(T["data"], T["what_reads"])
        self.tiles = tk.Frame(self.data_card, bg=C["panel"])
        self.tiles.pack(fill="x")
        self.data_line = tk_label(self.data_card, "", 13, "muted", "panel", wraplength=720)
        self.data_line.pack(anchor="w", pady=(8, 0))
        links = tk.Frame(self.data_card, bg=C["panel"])
        links.pack(anchor="w", pady=(8, 0))
        for text, route in (("Browse and clean", "your_data"), ("Make extra data", "make_data")):
            self._link(links, text, route).pack(side="left", padx=(0, GUTTER))
        row = tk.Frame(self.data_card, bg=C["panel"])
        row.pack(anchor="w", pady=(10, 0))
        tk_label(row, T["text_only"] + " ", 13, "muted", "panel").pack(side="left")
        self._link(row, T["post_training_r1"], "post_training").pack(side="left")
        tk_label(row, T["post_training_r1_end"], 13, "muted", "panel").pack(side="left")
        # recommended collection
        self.collection_card = self._card(T["collection"], T["collection_desc"])
        self.collection_rows = tk.Frame(self.collection_card, bg=C["panel"])
        self.collection_rows.pack(fill="x")
        # enough text
        card = self._card(T["enough_text"])
        self.enough = tk_label(card, "", 17, "text", "panel", font=font(17, bold=True))
        self.enough.pack(anchor="w")
        self.enough_note = tk_label(card, "", 13, "muted", "panel", wraplength=720)
        self.enough_note.pack(anchor="w", pady=(4, 0))
        # batches
        card = self._card(T["batches"], T["batches_desc"])
        tk_label(card, T["rebuilt_auto"], 12, "dim", "panel").pack(anchor="w")
        self.step_line = tk_label(card, "", 13, "text2", "panel", wraplength=720)
        self.step_line.pack(anchor="w", pady=(4, 6))
        self.grid = tk.Canvas(card, height=20, bg=C["panel"], highlightthickness=0)
        self.grid.pack(fill="x")
        self.grid.bind("<Configure>", lambda _e: self.draw_rows())
        tk_label(card, T["rows_per_step"], 14, "text", "panel").pack(anchor="w", pady=(10, 0))
        tk_label(card, T["rows_desc"], 12, "dim", "panel").pack(anchor="w")
        self.rows = Segmented(card, ROWS, "16", command=lambda _v: self.update_numbers())
        self.rows.pack(anchor="w", pady=4)
        tk_label(card, T["held_back"], 14, "text", "panel").pack(anchor="w", pady=(10, 0))
        tk_label(card, T["held_back_desc"], 12, "dim", "panel").pack(anchor="w")
        self.hold = Segmented(card, HOLD, "5%")
        self.hold.pack(anchor="w", pady=4)
        self.per_step = tk_label(card, "", 13, "muted", "panel")
        self.per_step.pack(anchor="w", pady=(6, 0))
        # train
        card = self._card(T["train_label"], T["continue_start"])
        self.need = tk_label(card, "", 13, "muted", "panel")
        self.need.pack(anchor="w", pady=(0, 6))
        self.mode_boxes = {}
        for key, name, note in (("catch_up", "", ""), ("from_scratch", "Start from scratch", "")):
            box = tk.Frame(card, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=12, pady=8, cursor="hand2", takefocus=1)
            box.pack(fill="x", pady=3)
            title = tk_label(box, name, 15, "text", "panel")
            title.pack(anchor="w")
            sub = tk_label(box, "", 13, "muted", "panel", wraplength=700)
            sub.pack(anchor="w")
            for w in (box, title, sub):
                w.bind("<Button-1>", lambda _e, k=key: self.pick_mode(k))
            box.bind("<Return>", lambda _e, k=key: self.pick_mode(k))
            self.mode_boxes[key] = (box, title, sub)
        tk_label(card, T["how_long"], 14, "text", "panel").pack(anchor="w", pady=(10, 0))
        self.length = Segmented(card, list(LENGTHS), "Medium", command=lambda _v: self.update_numbers())
        self.length.pack(anchor="w", pady=4)
        self.steps_line = tk_label(card, "", 13, "muted", "panel")
        self.steps_line.pack(anchor="w")
        tk_label(card, f"{T['learning_rate']} {T['lr_auto']} {T['lr_info']}", 13, "text2", "panel").pack(anchor="w", pady=(8, 0))
        tk_label(card, f"{T['schedule_label']} {T['schedule_value']} {T['schedule_note']}", 13, "text2", "panel").pack(anchor="w")
        tk_label(card, T["schedule_desc"], 12, "dim", "panel", wraplength=720).pack(anchor="w", pady=(2, 10))
        self.add_btn = button(card, T["add_queue"], self.add_to_queue, primary=True)
        self.add_btn.pack(anchor="w")
        self.footer = tk_label(card, "", 13, "muted", "panel", wraplength=720)
        self.footer.pack(anchor="w", pady=(8, 0))

    # helpers ------------------------------------------------------------
    def _card(self, kicker, gloss=""):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        head = tk.Frame(card, bg=C["panel"])
        head.pack(fill="x", pady=(0, 8))
        Kicker(head, kicker, bg="panel").pack(side="left")
        if gloss:
            tk_label(head, "  " + gloss, 13, "muted", "panel").pack(side="left")
        return card

    def _link(self, parent, text, route):
        link = tk_label(parent, text, 13, "violet", "panel", cursor="hand2", takefocus=1)
        for seq in ("<Button-1>", "<Return>", "<space>"):
            link.bind(seq, lambda _e: self.navigate(route))
        return link

    def version(self):
        try:
            return self.services.current_version()
        except NotImplementedError:
            return None

    def parent(self, v):
        if v is None or v.parent is None:
            return None
        return next((p for p in self.services.versions() if p.number == v.parent), None)

    def chosen(self):
        return [i for i, on in self.use.items() if on]

    def user_text(self):
        """True when a chosen set is the person's own text (not the bundled library): the queue then adds a restore job."""
        return any(not s.get("library") for s in sets_of(self.services.data("text")) if s["id"] in self.chosen())

    def steps(self):
        p = self.preview
        return steps_for(p.get("pieces"), self.rows_n() * p.get("context", 256), LENGTHS[self.length.value])

    def rows_n(self):
        return int(self.rows.value)

    # state ---------------------------------------------------------------
    def preload(self):
        for s in sets_of(self.services.data("text")):
            self.use.setdefault(s["id"], True)
        self._preview()

    def refresh(self):
        v = self.version()
        parent = self.parent(v)
        self.v, self.parent_v = v, parent
        n = v.number if v else "[N]"
        self.model_line.configure(text=f"Version {n} · from the Network designer" if v else "No design saved yet")
        for child in self.tiles.winfo_children():
            child.destroy()
        sets = sets_of(self.services.data("text"))
        for s in sets:
            self.use.setdefault(s["id"], True)
            on = self.use[s["id"]]
            tile = tk.Frame(self.tiles, bg=C["panel2"], highlightthickness=1, highlightbackground=C["violet"] if on else C["border"],
                            padx=12, pady=8, cursor="hand2", takefocus=1)
            tile.pack(fill="x", pady=3)
            mark = tk_label(tile, "✓" if on else " ", 14, "bg" if on else "text", "violet" if on else "panel2", width=2)
            mark.pack(side="left", padx=(0, 10))
            name = tk_label(tile, s["name"], 15, "text", "panel2")
            name.pack(side="left")
            note = tk_label(tile, f"{s['count']:,} examples", 13, "muted", "panel2")
            note.pack(side="left", padx=10)
            for w in (tile, mark, name, note):
                w.bind("<Button-1>", lambda _e, i=s["id"]: self.toggle(i))
            tile.bind("<Return>", lambda _e, i=s["id"]: self.toggle(i))
        if not sets:
            tk_label(self.tiles, "No text yet.", 14, "text2", "panel").pack(anchor="w")
        k = len(self.chosen())
        self.data_card_status = f"{k} {'source' if k == 1 else 'sources'} selected" if k else "nothing selected"
        self.data_line.configure(text=self.data_card_status)
        self.preview = self._preview()
        self.update_numbers()
        self.paint_modes(v, parent)
        self.paint_collection()

    def paint_collection(self):
        try:
            rows = self.services.collection()
        except NotImplementedError:
            rows = []
        for child in self.collection_rows.winfo_children():
            child.destroy()
        for row in rows:
            line = tk.Frame(self.collection_rows, bg=C["panel"])
            line.pack(fill="x", pady=3)
            tk_label(line, row["name"], 14, "text", "panel").pack(side="left")
            tk_label(line, f"{row['size']} · {row['records']}", 12, "dim", "panel").pack(side="left", padx=10)
            label, action = source_action(row)
            if action:
                button(line, label, lambda r=row["id"], a=action: self.fetch(r, a)).pack(side="right")
            tk_label(self.collection_rows, source_status(row, self.notes.get(row["id"], "")), 12,
                     "coral" if row.get("error") else "muted", "panel", wraplength=720).pack(anchor="w")
        if any(source_action(r)[1] == "download" for r in rows):
            button(self.collection_rows, T["download_all"], lambda: self.fetch(None, "download"), primary=True).pack(anchor="w", pady=(8, 0))

    def fetch(self, source_id, action):
        """Start a background download (all sources when source_id is None) or import; events repaint the card."""
        started = self.services.download_sources(None if source_id is None else [source_id]) if action == "download" \
            else self.services.import_source(source_id)
        if started:
            self.refresh()

    def on_event(self, kind, payload):
        if kind == "download":
            self.notes[payload["id"]] = payload["text"]
            self.paint_collection()
        elif kind in ("download_done", "import_done", "download_failed"):
            self.reload()

    def toggle(self, dataset_id):
        self.use[dataset_id] = not self.use.get(dataset_id)
        self.reload()

    def _preview(self):
        ids = self.chosen()
        if not ids:
            return {}
        try:
            return self.services.pretrain_preview(ids, rows=6)
        except NotImplementedError:
            return {}

    def update_numbers(self):
        p = self.preview
        pieces, params, ctx = p.get("pieces"), p.get("params"), p.get("context", 256)
        per_step = self.rows_n() * ctx
        self.step_line.configure(text=T["one_step"].format(rows=self.rows_n()).replace("256", str(ctx)))
        passes = f"{max(1, round(pieces / per_step)):,}" if pieces else "[N]"
        self.per_step.configure(text=T["per_step_pieces"].format(pieces=f"{per_step:,}").replace("[N]", passes))
        if pieces and params:
            ratio = pieces / params
            self.enough.configure(text=T["text_estimate"].replace("[N]", f"{max(RULE / ratio, 0):.1f}") if ratio < RULE
                                  else "You have about as much text as the rule of thumb asks for")
            self.enough_note.configure(text=T["word_pieces_rule"].replace("[N]", f"{ratio:.2f}", 1))
        else:
            self.enough.configure(text=T["text_estimate"])
            self.enough_note.configure(text=T["word_pieces_rule"])
        steps, tps = self.steps(), p.get("tokens_per_second")
        minutes = f"{max(1, round(steps * per_step / tps / 60)):,}" if steps and tps else "[N]"
        self.steps_line.configure(text=T["steps_time"].replace("[N]", f"{steps:,}" if steps else "[N]", 1).replace("[N]", minutes, 1))
        self.draw_rows()

    def draw_rows(self):
        c, rows = self.grid, self.preview.get("first_rows") or []
        c.delete("all")
        cell = 14
        c.configure(height=max(20, len(rows) * (cell + 6)))
        for r, row in enumerate(rows):
            c.create_text(0, r * (cell + 6) + cell / 2, text=f"row {r + 1}", anchor="w", fill=C["muted"], font=font(11))
            for i, d in enumerate(row):
                x = 52 + i * (cell + 2)
                c.create_rectangle(x, r * (cell + 6), x + cell, r * (cell + 6) + cell, width=0,
                                   fill=END if d < 0 else PALETTE[d % len(PALETTE)])
        if not rows:
            c.create_text(0, 10, text="[N]", anchor="w", fill=C["dim"], font=font(14, mono=True))

    def paint_modes(self, v, parent):
        self.mode = forced_mode(v, parent, self.mode)
        ok = can_continue(v, parent)
        pn = parent.number if parent else "[N]"
        texts = {"catch_up": (f"Continue from version {pn}", "Recommended. Keeps what version %s learned; parts that are new in version %s start empty, so the curve starts at version %s’s level." % (pn, v.number if v else "[N]", pn)),
                 "from_scratch": ("Start from scratch", "New random weights. Only required when the word list changes. Slower to reach the same error.")}
        for key, (box, title, sub) in self.mode_boxes.items():
            on, blocked = self.mode == key, key == "catch_up" and not ok
            title.configure(text=texts[key][0])
            why = "The word list changed" if word_list_changed(v, parent) else "There is no trained earlier version to continue from"
            sub.configure(text="%s, so version %s must start from scratch." % (why, v.number if v else "[N]") if blocked else texts[key][1])
            box.configure(highlightbackground=C["violet"] if on else C["border"], highlightthickness=2 if on else 1)
        self.need.configure(text=f"version {v.number} needs catch-up training" if v else "")
        self.footer.configure(text=T["queue_note"].replace("version 4", f"version {pn}").replace("version 5", f"version {v.number if v else '[N]'}"))
        self.add_btn.state(["!disabled" if v and self.chosen() and self.steps() else "disabled"])

    def pick_mode(self, key):
        v, parent = self.v, self.parent_v
        self.mode = forced_mode(v, parent, key)
        self.paint_modes(v, parent)

    def add_to_queue(self):
        v = self.v
        if v is None or not self.chosen():
            return
        steps = self.steps()
        if not steps:
            return
        cfg = pretrain_config(v.spec, steps, self.rows_n(), int(self.hold.value[:-1]) / 100)
        self.mode = forced_mode(v, self.parent_v, self.mode)
        init = self.parent_v.best_checkpoint if self.mode == "catch_up" else None
        label = f"Version {v.number} {'catch-up' if self.mode == 'catch_up' else 'from scratch'}"
        self.services.enqueue(Job(id=uuid.uuid4().hex[:8], kind="pretrain", label=label, params={
            "config": cfg, "datasets": self.chosen(), "mode": self.mode, "init_from": init, "output_dir": None,
            "user_text": self.user_text(), "previous_best": self.parent_v.best_checkpoint if self.parent_v else None,
            "version": v.number, "heldout_fraction": cfg["validation"],
            "pieces_per_pass": self.preview.get("pieces")}))

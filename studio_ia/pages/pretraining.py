"""Pretraining: Data -> Batches -> Train for the current design version."""
from __future__ import annotations

import time
import tkinter as tk
import uuid

from studio_ia.copy import PRETRAINING as T
from studio_ia.pages.base import Page, Throttle
from studio_ia.pretraining_plan import (
    HOLD, LENGTHS, ROWS, can_continue, collection_items, enough_text, forced_mode, held_out, job_params, ratio_text, sets_of,
    source_action, source_status, steps_for, tile_items, version_expectation, word_list_changed,
)
from studio_ia.pretraining_plan import user_text as has_user_text
from studio_ia.services import Job, if_added, queued_outlook
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import Card, KeyedRows, Kicker, Segmented, button, info_icon, tip

PROGRESS_EVERY = 0.5  # seconds between repaints of a download's status line
PALETTE = ("#3a3560", "#2a4558", "#4a3a2a", "#2f4a36", "#4a2f45")
END = "#0a0b0e"


def queue_line(services, job_id):
    """The line under "Add to the queue": when the job this page added starts, or, before one is added (or once it has
    left the queue), what adding one now would do. The same services.queue_outlook text as every other add-to-queue surface."""
    when = queued_outlook(services, job_id) if job_id else ""
    return f"Queued. {when}." if when else f"{if_added(services.queue_outlook())}."


def collection_row(parent, fetch):
    """One recommended source on one line, the same on Pretraining and Your data: its name, size, status and the button for
    its next step. fetch(source id, action) runs the step. Builds the line, unpacked; fill_collection_row shows a row in it
    each time the row changes, so the line is never rebuilt. `.status` is the status label, which progress updates in place."""
    line = tk.Frame(parent, bg=C["panel"])
    line.source = None
    line.name_label = tk_label(line, "", 14, "text", "panel")
    line.name_label.pack(side="left")
    line.size_label = tk_label(line, "", 12, "dim", "panel")
    line.size_label.pack(side="left", padx=10)
    slot = tk.Frame(line, bg=C["panel"])  # keeps the button right of the status line while it comes and goes
    slot.pack(side="right")
    line.action = button(slot, "", lambda: fetch(line.source["id"], source_action(line.source)[1]))
    line.status = tk_label(line, "", 12, "muted", "panel", wraplength=320, justify="right")
    line.status.pack(side="right", padx=10)
    return line


def fill_collection_row(line, row, note=""):
    """Show `row` (with its latest progress `note`) in a line collection_row built."""
    line.source = row
    line.name_label.configure(text=row["name"])
    line.size_label.configure(text=f"{row['size']} · {row['records']}")
    label, action = source_action(row)
    line.action.configure(text=label or "")
    if action:
        line.action.pack()
    else:
        line.action.pack_forget()
    line.status.configure(text=source_status(row, note), fg=C["coral" if row.get("error") else "muted"])


class PretrainingPage(Page):
    title = T["h1"]
    lead = T["lead"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.use, self.mode = {}, "catch_up"
        self.preview, self.notes = {}, {}  # notes: latest progress line per source being fetched
        self.throttle, self.dirty, self.flush_timer = Throttle(PROGRESS_EVERY), set(), None
        self.job_id = None  # the last job "Add to the queue" added, for the line under the button
        self.build()

    def build(self):
        b = self.body
        tk_label(b, T["model_label"], 12, "muted").pack(anchor="w")
        self.model_line = tk_label(b, "", 15)
        self.model_line.pack(anchor="w", pady=(2, GUTTER))
        # data
        self.data_card = self._card(T["data"], T["what_reads"])
        tiles = tk.Frame(self.data_card, bg=C["panel"])
        tiles.pack(fill="x")
        self.tile_list = KeyedRows(tiles, self._make_tile, self._fill_tile, fill="x", pady=3)
        self.tiles_note = tk_label(self.data_card, "", 14, "muted", "panel")
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
        rows = tk.Frame(self.collection_card, bg=C["panel"])
        rows.pack(fill="x")
        self.collection_list = KeyedRows(rows, lambda parent, _data: collection_row(parent, self.fetch),
                                         lambda line, data: fill_collection_row(line, *data), fill="x", pady=1)
        self.download_all = button(self.collection_card, "", lambda: self.fetch(None, "download"), primary=True)
        # enough text
        card = self._card(T["enough_text"])
        self.enough = tk_label(card, "", 17, "text", "panel", font=font(17, bold=True))
        self.enough.pack(anchor="w")
        self.enough_tip = ""
        tip(self.enough, lambda: self.enough_tip)
        row = tk.Frame(card, bg=C["panel"])
        row.pack(anchor="w", pady=(8, 0))
        self.expect_line = tk_label(row, "", 13, "text2", "panel")
        self.expect_line.pack(side="left")
        self.expect_details = ""
        self.expect_info = info_icon(row, lambda: self.expect_details)
        # batches
        card = self._card(T["batches"], f"{T['batches_desc'].capitalize()}; {T['rebuilt_auto']}.")
        self.step_line = tk_label(card, "", 13, "text2", "panel", wraplength=720)
        self.step_line.pack(anchor="w", pady=(4, 6))
        self.grid = tk.Canvas(card, height=20, bg=C["panel"], highlightthickness=0)
        self.grid.pack(fill="x")
        self.grid.bind("<Configure>", lambda _e: self.draw_rows())
        tip(tk_label(card, T["rows_per_step"], 14, "text", "panel"), T["rows_desc"].capitalize() + ".").pack(anchor="w", pady=(10, 0))
        self.rows = Segmented(card, ROWS, "16", command=lambda _v: self.update_numbers())
        self.rows.pack(anchor="w", pady=4)
        tip(tk_label(card, T["held_back"], 14, "text", "panel"), T["held_back_desc"].capitalize() + ".").pack(anchor="w", pady=(10, 0))
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
        self.lr_line = tk_label(card, "", 13, "text2", "panel")
        self.lr_line.pack(anchor="w", pady=(8, 0))
        tip(tk_label(card, f"{T['schedule_label']} {T['schedule_value']} {T['schedule_note']}", 13, "text2", "panel"), T["schedule_desc"]).pack(anchor="w", pady=(0, 10))
        self.footer_tip = ""
        self.add_btn = tip(button(card, T["add_queue"], self.add_to_queue, primary=True), lambda: self.footer_tip)
        self.add_btn.pack(anchor="w")
        self.queue_line = tk_label(card, "", 13, "muted", "panel")
        self.queue_line.pack(anchor="w", pady=(6, 0))

    # helpers ------------------------------------------------------------
    def _card(self, kicker, gloss=""):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        head = tk.Frame(card, bg=C["panel"])
        head.pack(fill="x", pady=(0, 8))
        Kicker(head, kicker, bg="panel").pack(side="left")
        if gloss:
            info_icon(head, gloss).pack(side="left", padx=(4, 0))
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
        return has_user_text(sets_of(self.services.data("text")), self.chosen())

    def steps(self):
        p = self.preview
        return steps_for(p.get("pieces"), self.rows_n() * p.get("context", 256), LENGTHS[self.length.value])

    def rows_n(self):
        return int(self.rows.value)

    # state ---------------------------------------------------------------
    opens_at_once = True

    def preload(self):
        """Only the set list: the size preview is read after the page shows (see _ask_preview)."""
        self.services.data("text")

    def refresh(self):
        v = self.version()
        parent = self.parent(v)
        self.v, self.parent_v = v, parent
        n = v.number if v else None
        self.model_line.configure(text=f"Version {n} · from the Network designer" if v else "No design saved yet")
        if not self.warmed():  # the set list is being read: the page shows now, adding to the queue waits for it
            if not len(self.tile_list):  # tiles already shown stay until the new list is in
                self._tiles_note(T["reading_text"], "muted")
                self.data_line.configure(text="")
            self.preview, self.estimating = {}, True
            self.update_numbers()
            self.paint_modes(v, parent)
            self.add_btn.state(["disabled"])
            self.paint_collection()
            return
        sets = sets_of(self.services.data("text"))
        for s in sets:
            self.use.setdefault(s["id"], not held_out(s["id"]))
        self.tile_list.set(tile_items(sets, self.use))
        self._tiles_note("" if sets else "No text yet.", "text2")
        k = len(self.chosen())
        self.data_card_status = f"{k} {'source' if k == 1 else 'sources'} selected" if k else "nothing selected"
        self.data_line.configure(text=self.data_card_status)
        self._ask_preview()
        self.update_numbers()
        self.paint_modes(v, parent)
        self.paint_collection()

    def _make_tile(self, parent, data):
        tile = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, padx=12, pady=8, cursor="hand2", takefocus=1)
        tile.mark = tk_label(tile, "", 14, "text", "panel2", width=2)
        tile.mark.pack(side="left", padx=(0, 10))
        tile.name_label = tk_label(tile, "", 15, "text", "panel2")
        tile.name_label.pack(side="left")
        tile.note = tk_label(tile, "", 13, "muted", "panel2")
        tile.note.pack(side="left", padx=10)
        for w in (tile, tile.mark, tile.name_label, tile.note):
            w.bind("<Button-1>", lambda _e, i=data["id"]: self.toggle(i))
        tile.bind("<Return>", lambda _e, i=data["id"]: self.toggle(i))
        return tile

    @staticmethod
    def _fill_tile(tile, data):
        on = data["on"]
        tile.configure(highlightbackground=C["violet"] if on else C["border"])
        tile.mark.configure(text="✓" if on else " ", fg=C["bg"] if on else C["text"], bg=C["violet"] if on else C["panel2"])
        tile.name_label.configure(text=data["name"])
        tile.note.configure(text=f"{data['count']:,} examples" + (f" · {T['held_out']}" if data["held"] else ""))

    def _tiles_note(self, text, color):
        """The line in place of the tiles while the set list is read or when there is none; hidden otherwise."""
        self.tiles_note.configure(text=text, fg=C[color])
        if text:
            self.tiles_note.pack(anchor="w", before=self.data_line)
        else:
            self.tiles_note.pack_forget()

    def _ask_preview(self):
        """Use the preview if it is already known; otherwise show "estimating" and work it out in the background (it
        samples the chosen text and builds a word list: seconds, longer while a run trains)."""
        ids = tuple(self.chosen())
        try:
            known = self.services.cached_preview(list(ids), rows=6) if ids else {}
        except NotImplementedError:
            known = {}
        self.estimating = known is None
        self.preview = known or {}
        if self.estimating and getattr(self, "_asked", None) != ids:
            self._asked = ids
            self.quietly(lambda: self.services.pretrain_preview(list(ids), rows=6), lambda p: self._got_preview(ids, p))

    def _got_preview(self, ids, preview):
        self._asked = None
        if ids != tuple(self.chosen()) or not self.winfo_exists():
            return  # the choice changed meanwhile; that choice asked for its own preview
        self.preview, self.estimating = preview or {}, False
        self.update_numbers()
        self.paint_modes(self.v, self.parent_v)

    def paint_collection(self):
        """Read the collection (a manifest on disk per source) on the worker, then update its rows in place."""
        def work():
            try:
                return self.services.collection()
            except NotImplementedError:
                return []
        self.quietly(work, self._collected)

    def _collected(self, rows):
        if not self.winfo_exists():
            return
        self.collection_list.set(collection_items(rows, self.notes))
        if any(source_action(r)[1] == "download" for r in rows):
            self.download_all.configure(text=T["download_all"])
            self.download_all.pack(anchor="w", pady=(8, 0))
        else:
            self.download_all.configure(text="")
            self.download_all.pack_forget()

    def fetch(self, source_id, action):
        """Start a background download (all sources when source_id is None) or import; events repaint the card."""
        started = self.services.download_sources(None if source_id is None else [source_id], install=True) if action == "download" \
            else self.services.import_source(source_id)
        if started:
            self.paint_collection()

    def on_event(self, kind, payload):
        if kind == "download":  # several a second: only the status line changes, at most PROGRESS_EVERY apart
            self.notes[payload["id"]] = payload["text"]
            self.dirty.add(payload["id"])
            self._flush_soon()
        elif kind in ("download_done", "import_done", "download_failed"):
            self.paint_collection()
        elif kind == "downloads_finished":
            self.quiet_reload()  # re-counting every row is slow; the page stays usable while it runs
        elif kind in ("queue", "job", "blocked"):
            self.paint_queue_line()

    def paint_queue_line(self):
        self.queue_line.configure(text=queue_line(self.services, self.job_id))

    def _flush_soon(self):
        if self.flush_timer is not None:
            return
        now = time.monotonic()
        if self.throttle.ready(now):  # the first change of a burst is drawn at once; the rest wait for the next tick
            self._flush_notes()
        else:
            self.flush_timer = self.after(int(self.throttle.wait(now) * 1000) + 20, self._flush_later)

    def _flush_later(self):
        self.flush_timer = None
        self.throttle.ready(time.monotonic())
        self._flush_notes()

    def _flush_notes(self):
        for source_id in self.dirty:
            line = self.collection_list.get(source_id)
            if line is not None:  # a progress line means it is being fetched, whatever the rows say
                line.status.configure(text=self.notes[source_id], fg=C["muted"])
                self.collection_list.touched(source_id)
        self.dirty.clear()

    def toggle(self, dataset_id):
        self.use[dataset_id] = not self.use.get(dataset_id)
        self.refresh()  # the set list is cached; a new preview comes in the background

    def update_numbers(self):
        p = self.preview
        pieces, params, ctx = p.get("pieces"), p.get("params"), p.get("context", 256)
        per_step = self.rows_n() * ctx
        self.step_line.configure(text=T["one_step"].format(rows=self.rows_n()).replace("256", str(ctx)))
        about = "about " if p.get("estimated") else ""
        self.per_step.configure(text=T["per_step_pieces"].format(pieces=f"{per_step:,}", passes=f"{about}{max(1, round(pieces / per_step)):,}")
                                if pieces else T["per_step_only"].format(pieces=f"{per_step:,}"))
        if getattr(self, "estimating", False):
            self.per_step.configure(text=T["per_step_only"].format(pieces=f"{per_step:,}"))
            self.enough.configure(text=T["estimating"])
            self.enough_tip = T["word_pieces_plain"]
        elif pieces and params:
            ratio = pieces / params
            headline, verdict = enough_text(ratio)
            self.enough.configure(text=headline)
            self.enough_tip = (T["word_pieces_plenty"] if verdict == "plenty" else T["word_pieces_rule"]).format(ratio=ratio_text(ratio))
        else:
            self.enough.configure(text=T["text_unknown"])
            self.enough_tip = T["word_pieces_plain"]
        line, self.expect_details = version_expectation(getattr(self, "v", None) and self.v.spec, p, LENGTHS[self.length.value])
        self.expect_line.configure(text=line)
        self.expect_info.pack_forget()
        if line:
            self.expect_info.pack(side="left", padx=(6, 0))
        steps, tps = self.steps(), p.get("tokens_per_second")
        if steps and tps:
            self.steps_line.configure(text=T["steps_time"].format(steps=f"{steps:,}", minutes=f"{max(1, round(steps * per_step / tps / 60)):,}"))
        else:
            self.steps_line.configure(text=T["steps_only"].format(steps=f"{steps:,}") if steps
                                      else T["steps_estimating"] if getattr(self, "estimating", False) else T["steps_none"])
        lr = ((getattr(self, "v", None) and self.v.spec or {}).get("train", {}).get("optimizer", {}).get("config", {}) or {}).get("lr")
        self.lr_line.configure(text=f"{T['learning_rate']} {T['lr_auto']}" + (" " + T["lr_info"].format(lr=f"{lr:.2g}") if lr else ""))
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
            c.create_text(0, 10, text="Choose some text to see how it is cut into rows.", anchor="w", fill=C["dim"], font=font(12))

    def paint_modes(self, v, parent):
        self.mode = forced_mode(v, parent, self.mode)
        ok = can_continue(v, parent)
        pn = parent.number if parent else None
        vn = f"version {v.number}" if v else "this version"
        texts = {"catch_up": (f"Continue from version {pn}" if pn else "Continue from an earlier version",
                              f"Recommended. Keeps what version {pn} learned; parts that are new in {vn} start empty, so the curve starts at version {pn}’s level."
                              if pn else "Keeps what an earlier trained version learned. There is none yet."),
                 "from_scratch": ("Start from scratch", "New random weights. Only required when the word list changes. Slower to reach the same error.")}
        for key, (box, title, sub) in self.mode_boxes.items():
            on, blocked = self.mode == key, key == "catch_up" and not ok
            title.configure(text=texts[key][0])
            why = "The word list changed" if word_list_changed(v, parent) else "There is no trained earlier version to continue from"
            sub.configure(text=f"{why}, so {vn} must start from scratch." if blocked else texts[key][1])
            box.configure(highlightbackground=C["violet"] if on else C["border"], highlightthickness=2 if on else 1)
        self.need.configure(text=f"version {v.number} needs catch-up training" if v else "")
        self.footer_tip = (T["queue_note"].format(parent=f"version {pn}" if pn else "its current version", version=vn))
        self.add_btn.state(["!disabled" if v and self.chosen() and self.steps() else "disabled"])
        self.paint_queue_line()

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
        self.mode = forced_mode(v, self.parent_v, self.mode)
        label, params = job_params(v, self.parent_v, self.chosen(), self.mode, self.rows_n(), int(self.hold.value[:-1]) / 100, steps,
                                   self.preview.get("pieces"), self.user_text())
        self.job_id = uuid.uuid4().hex[:8]
        self.services.enqueue(Job(id=self.job_id, kind="pretrain", label=label, params=params))
        self.paint_queue_line()

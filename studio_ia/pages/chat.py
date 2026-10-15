"""Chat: talk to any trained version, compare two, teach a correction."""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk

from studio_ia import threads
from studio_ia.copy import BENCH, CHAT as T
from studio_ia.next_step import empty_line
from studio_ia.pages.base import Page
from studio_ia.services import plain_error
from studio_ia.theme import C, GUTTER, entry, font, text_box, tk_label
from studio_ia.widgets import Card, Kicker, Toggle, _activate, button, info_icon, tip

A_COLOR, B_COLOR = "violet", "blue"
TONE = {"better": "violet", "baseline": "muted", "within the spread": "muted", "worse": "coral", "catching up": "coral"}


def name(v):
    return f"Version {v.number}"


def score_rows(v):
    s = v.scorecard or {}

    def kn(key):
        k = s.get(key)
        return T["val_kofn"] if not k else f"{k[0]} of {k[1]}"
    bpc, size, speed = s.get("bpc"), s.get("params_m"), s.get("pieces_per_s")
    known = [f"{size}M" if size is not None else "", f"{speed} pieces/s" if speed is not None else ""]
    return [(T["error_held_back"], T["val_bpc"] if bpc is None else f"{bpc:.2f} bits/char"),
            (T["answers_prompts"], "yes" if s.get("answers") in (True, "yes") else "not yet"),
            (T["checked_problems"], kn("problems")), (T["judge_preferred"], kn("judge")),
            (T["size_speed"], " · ".join(k for k in known if k) or T["val_size_speed"])]


def against_chance(correct, items, chance):
    """better | about the same | worse than guessing: outside two standard deviations of a guesser's count."""
    spread = 2 * math.sqrt(items * chance * (1 - chance))
    return T["few_shot_above"] if correct > items * chance + spread else T["few_shot_below"] if correct < items * chance - spread \
        else T["few_shot_near"]


def few_shot_rows(table):
    """(examples shown, right, score, guessing gets, against guessing) per row of services.few_shot; [] when nothing was asked."""
    if not table or not table.get("items"):
        return []
    items, chance = table["items"], table["chance"]
    out = []
    for r in table["rows"]:
        n = r["shots"]
        shown = T["few_shot_zero"] if n == 0 else T["few_shot_one"] if n == 1 else T["few_shot_some"].format(n=n)
        if r["used"] < n:
            shown += T["few_shot_fitted"].format(used=f"{r['used']:.1f}")
        out.append((shown, f"{r['correct']} of {items}", f"{r['accuracy']:.0%}", f"{chance:.0%}", against_chance(r["correct"], items, chance)))
    return out


def few_shot_note(table):
    """The plain lines under the table: what was asked, where it came from, what guessing scores."""
    lines = [T["few_shot_asked"].format(items=table["items"], tasks=table.get("tasks", 1))]
    if not table.get("held_back", True):
        lines.append(T["few_shot_training"])
    return " ".join(lines + [T["few_shot_chance"].replace("[N]", f"{100 * table['chance']:.0f}")])


def answering_version(numbers, best, picked, follows=True, current=None, trained=None):
    """Which version answers: the one the person picked; otherwise the best checkpoint's (when the plan says Chat follows
    it), otherwise the one already answering; the newest as a last resort. With `trained` (the numbers that have a
    checkpoint) only those can answer, and None means there is nothing to chat with yet."""
    if trained is not None:
        numbers = [n for n in numbers if n in trained]
    if picked in numbers:
        return picked
    if follows and best in numbers:
        return best
    return current if current in numbers else (best if best in numbers else max(numbers, default=None))


def verdict_view(a, b, result=None):
    """(line, sub, comparable) for two versions. No verdict while either run is mid-schedule or the runs differ in length or
    schedule; the comparison is training_saved's result for (older, newer)."""
    older, newer = (a, b) if a.number < b.number else (b, a)
    unfinished = [v for v in (a, b) if v.status in ("queued", "training")]
    if unfinished:
        return T["no_verdict_yet"].format(version=name(unfinished[0]).lower()), T["no_verdict_text"], False
    result = result or {}
    if result.get("comparable") and result.get("percent") is not None:
        line = T["verdict_line"].replace("[N]", f"{100 - result['percent']:.0f}").format(newer=name(newer), older=name(older))
        return line, ("Under 100% means the change saves training. Size and seed checks: not run yet (Compare runs) · "
                      "same length and schedule · error per character on held-back text"), True
    return "No verdict yet: " + result.get("reason", "the runs are not comparable."), T["no_verdict_text"], False


def stream_reply(chat, version, messages, stop, put):
    """Run one reply. chat(version, messages, on_text, stop) reports the reply so far (cumulative text) through on_text;
    put(text) receives each cumulative text, then put(None) when the reply is over. A failure becomes a sentence in the reply."""
    seen = []

    def on_text(text):
        seen.append(text)
        put(text)
    try:
        final = chat(version, messages, on_text, stop)
        if final and not seen:
            put(final)
    except Exception as error:  # show it in the pane instead of dying silently
        put(plain_error(error))
    put(None)


class ChatPage(Page):
    title, lead = T["h1"], T["lead"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.use = self.b = None  # version numbers: left/answering, right (compare)
        self.compare = self.how = self.picked = False
        self.messages, self.replies, self.base = [], {}, {}
        self.pump_q, self.stop, self.pending = queue.Queue(), threading.Event(), 0
        self.verdict_line = ""
        self._build()

    # state ------------------------------------------------------------------
    def _versions(self):
        """The versions Chat can talk to: those with a trained checkpoint. An untrained one is never offered or picked."""
        return {v.number: v for v in self.services.versions() if v.best_checkpoint}

    @staticmethod
    def _other(vs, number):
        """The trained version beside `number` for the compare slot: the one just before it, else the one just after, else None."""
        before = [n for n in vs if n < (number or 0)]
        after = [n for n in vs if n > (number or 0)]
        return max(before) if before else min(after) if after else None

    def _best(self):
        best = self.services.best_version()
        return best.number if best else None

    def refresh(self):
        vs = self._versions()
        follows = (self.services.queue().plan or {}).get("chat_follows_best", True)
        self.use = answering_version(vs, self._best(), self.use if self.picked else None, follows, self.use, trained=vs)
        self.picked = self.picked and self.use != self._best() if follows else self.picked
        if self.b not in vs or self.b == self.use:
            self.b = self._other(vs, self.use)
        if self.b is None:
            self.compare = False
        self.render()

    def on_event(self, kind, payload):
        if kind in ("versions", "queue"):
            self.refresh()

    def pick(self, number):
        if self.compare:
            if number != self.use:
                self.b = number
        else:
            self.use, self.picked = number, number != self._best()
            self.b = self._other(self._versions(), number)
        self.render()

    def toggle_compare(self):
        if self.b is None:
            return
        self.compare, self.how = not self.compare, False
        self.render()

    def _toggle_how(self):
        self.how = not self.how
        self.render()

    # layout -----------------------------------------------------------------
    def _build(self):
        body = self.body
        head = Card(body)
        head.pack(fill="x", pady=(0, GUTTER))
        row = tk.Frame(head, bg=C["panel"])
        row.pack(fill="x")
        tk_label(row, T["live"], 12, "blue", "panel", font=font(12, bold=True)).pack(side="left")
        self.using = tk_label(row, "", 15, "text", "panel", font=font(15, bold=True))
        self.using.pack(side="left", padx=10)
        self.mode = tk_label(row, "", 13, "muted", "panel")
        self.mode.pack(side="left")
        self.build_link = tk_label(row, T["no_trained_link"], 13, "violet", "panel", cursor="hand2", takefocus=1)
        _activate(self.build_link, lambda: self.navigate("builder"))
        self.cmp_btn = button(row, "", self.toggle_compare)
        self.cmp_btn.pack(side="right")
        self.empty = tk_label(head, "", 13, "muted", "panel", wraplength=720)  # what to do when no version has trained
        self.panes_row = tk.Frame(body, bg=C["bg"])
        self.panes_row.pack(fill="x", pady=(0, GUTTER))
        self.pane_frames, self.pane_tags, self.pane_texts, self.pane_better = [], [], [], []
        for i in range(2):
            f = Card(self.panes_row)
            tag = tk_label(f, "", 13, A_COLOR if i == 0 else B_COLOR, "panel", font=font(13, bold=True))
            tag.pack(anchor="w")
            text = text_box(f, height=7, width=40, state="disabled")
            text.tag_configure("placeholder", foreground=C["muted"])
            text.pack(fill="x", pady=(8, 0))
            self.pane_frames.append(f)
            self.pane_tags.append(tag)
            self.pane_texts.append(text)
            self.pane_better.append(tip(button(f, T["this_one_better"], lambda k=i: self.better(k)), T["picking_reply_desc"]))
        self.send_row = tk.Frame(body, bg=C["bg"])
        self.send_row.pack(fill="x", pady=(0, GUTTER))
        self.entry = entry(self.send_row)
        self.entry.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self.send())
        self.send_btn = button(self.send_row, T["send"], self.send, True)
        self.send_btn.pack(side="left")
        teach = tk.Frame(body, bg=C["bg"])
        teach.pack(fill="x", pady=(0, GUTTER))
        link = tip(tk_label(teach, T["teach_correction"], 14, "violet", cursor="hand2", takefocus=1), T["teach_correction_desc"])
        link.pack(anchor="w")
        _activate(link, lambda: self.teach_row.pack(anchor="w", fill="x", pady=4))
        self.teach_row = tk.Frame(teach, bg=C["bg"])
        self.correction = entry(self.teach_row, 14)
        self.correction.pack(side="left", fill="both", expand=True, padx=(0, 8))
        button(self.teach_row, "Save", self.teach).pack(side="left")
        self.whole = Toggle(self.teach_row, "It is a whole new answer (round 1)", False, bg="bg")
        self.whole.pack(side="left", padx=8)
        self.status_label = tk_label(teach, "", 12, "muted")
        self.status_label.pack(anchor="w")
        self.verdict_card = Card(body)
        self.shelf = Card(body)
        self.shelf.pack(fill="x", pady=(0, GUTTER))
        self.score_card = Card(body)
        self.score_card.pack(fill="x", pady=(0, GUTTER))
        self.shot_card = Card(body)
        self.shots, self.shot_busy = {}, set()  # few-shot tables and the checks running, by version number
        row = tk.Frame(body, bg=C["bg"])
        row.pack(fill="x")
        button(row, T["test_change"], lambda: self.navigate("compare"), True).pack(side="left", padx=(0, 8))
        button(row, T["open_designer"], lambda: self.navigate("builder")).pack(side="left")
        button(row, BENCH["open"], lambda: self.navigate("benchmark")).pack(side="left", padx=(8, 0))

    # render -----------------------------------------------------------------
    def panes(self):
        """Reply panes currently on screen."""
        return [t for t, f in zip(self.pane_texts, self.pane_frames) if f.winfo_manager()]

    def render(self):
        vs = self._versions()
        cur, best = vs.get(self.use), self._best()
        self.using.configure(text=T["head_using"].format(version=name(cur)) if cur else T["no_trained"])
        (self.build_link.pack_forget if cur else lambda: self.build_link.pack(side="left", padx=10))()
        line = "" if cur else empty_line(self.services)
        self.empty.configure(text=line)
        if line:
            self.empty.pack(anchor="w", pady=(6, 0))
        else:
            self.empty.pack_forget()
        ready = cur is not None  # nothing to talk to until a version has trained
        self.entry.configure(state="normal" if ready else "disabled")
        self.send_btn.state(["!disabled" if ready else "disabled"])
        self.cmp_btn.state(["!disabled" if self.b is not None else "disabled"])
        self.mode.configure(text="" if not cur else T["head_mode_follows"] if not self.picked
                            else T["head_mode_picked"].format(best_version=f"Version {best}"))
        self.cmp_btn.configure(text=T["back_to_one"] if self.compare else T["compare_btn"])
        for f in self.pane_frames:
            f.pack_forget()
        keys = (0, 1) if self.compare else (0,)
        for i in keys:
            self.pane_frames[i].pack(side="left", fill="both", expand=True, padx=(0, GUTTER if i == 0 and self.compare else 0))
        for i, num in zip(keys, (self.use, self.b)):
            v = vs.get(num)
            step = self.services.checkpoint_step(v) if v else None
            tag = (name(v) + (f" · checkpoint at step {step}" if step else "")) if v else ""
            if v and not self.compare and (v.scorecard or {}).get("answers") not in (True, "yes"):
                tag += " · hasn’t practised answering yet"
            self.pane_tags[i].configure(text=tag)
            self.pane_better[i].pack_forget()
            if self.compare:
                self.pane_better[i].pack(anchor="w", pady=(8, 0))
            self._set_text(i, self.replies.get(i, T["its_reply"]), () if i in self.replies else "placeholder")
        self._verdict(vs)
        self._shelf(vs, best)
        self._score(cur)
        self._few_shot(cur)

    def _set_text(self, i, text, tags=()):
        t = self.pane_texts[i]
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("1.0", text, tags)
        t.configure(state="disabled")

    def _verdict(self, vs):
        card = self.verdict_card
        for w in card.winfo_children():
            w.destroy()
        card.pack_forget()
        a, b = vs.get(self.use), vs.get(self.b)
        self.verdict_line = ""
        if not (self.compare and a and b):
            return
        card.pack(fill="x", pady=(0, GUTTER), before=self.shelf)
        Kicker(card, T["verdict"], bg="panel").pack(anchor="w")
        line, sub, ok = verdict_view(a, b, None if any(v.status in ("queued", "training") for v in (a, b))
                                     else self.services.verdict(min(a.number, b.number), max(a.number, b.number)))
        color = "text" if ok else "amber"
        self.verdict_line = line
        tk_label(card, line, 17, color, "panel", wraplength=780, font=font(17, bold=True)).pack(anchor="w", pady=4)
        tk_label(card, sub, 13, "muted", "panel", wraplength=780).pack(anchor="w")
        how = tk_label(card, "Hide how this is measured" if self.how else T["how_measured"], 13, "violet", "panel", cursor="hand2", takefocus=1)
        how.pack(anchor="w", pady=(8, 0))
        _activate(how, self._toggle_how)
        if self.how:
            for key in ("training_saved", "verdict_text", "verdict_checked", "verdict_large_change", "verdict_error_unit", "verdict_error_unit_desc"):
                tk_label(card, T[key], 12, "muted", "panel", wraplength=780).pack(anchor="w", pady=1)
            size = tk_label(card, T["size_check"], 13, "violet", "panel", cursor="hand2", takefocus=1)
            size.pack(anchor="w", pady=4)
            _activate(size, lambda: self.navigate("compare"))

    def _shelf(self, vs, best):
        for w in self.shelf.winfo_children():
            w.destroy()
        head = tk.Frame(self.shelf, bg=C["panel"])
        head.pack(anchor="w", pady=(0, 6))
        Kicker(head, T["versions"], bg="panel").pack(side="left")
        info_icon(head, T["shelf_hint_compare" if self.compare else "shelf_hint_single"]).pack(side="left", padx=(4, 0))
        self.rows = {}
        for num, v in sorted(vs.items()):
            is_a, is_b = num == self.use, self.compare and num == self.b
            chip = "catching up" if v.status == "training" else v.verdict or ("baseline" if v.parent is None else "not compared yet")
            slot = ("left reply" if self.compare else "answering now") if is_a else "right reply" if is_b else \
                "best on held-back text" if num == best else \
                "still training · answers from its best checkpoint so far" if v.status == "training" else ""
            on = is_a or is_b
            row = tk.Frame(self.shelf, bg=C["panel2"] if on else C["panel"], highlightthickness=2 if on else 1, takefocus=1, cursor="hand2",
                           highlightbackground=C[A_COLOR] if is_a else C[B_COLOR] if is_b else C["border"])
            row.pack(fill="x", pady=2)
            bg = row["bg"]
            tk_label(row, name(v), 14, "text", bg, font=font(14, bold=True), width=10).pack(side="left", padx=8, pady=8)
            tk_label(row, chip, 12, TONE.get(chip, "muted"), bg, width=16).pack(side="left")
            tk_label(row, v.change, 13, "text2", bg).pack(side="left")
            tk_label(row, slot, 12, A_COLOR if is_a else B_COLOR if is_b else "muted", bg).pack(side="right", padx=8)
            _activate(row, lambda n=num: self.pick(n))
            for w in row.winfo_children():
                w.bind("<Button-1>", lambda _e, n=num: self.pick(n))
            self.rows[num] = row
        steps = []
        for num, v in sorted(vs.items()):
            step = self.services.checkpoint_step(v) if v.best_checkpoint else None
            if step:
                steps.append(f"{name(v).lower()} at step {step:,}")
        tk_label(self.shelf, T["checkpoints"].format(checkpoints=" · ".join(steps) or "none saved yet"), 12, "dim", "panel").pack(anchor="w", pady=(6, 0))

    def _score(self, cur):
        for w in self.score_card.winfo_children():
            w.destroy()
        self.score = score_rows(cur) if cur else []
        for k, val in self.score:
            row = tk.Frame(self.score_card, bg=C["panel"])
            row.pack(fill="x", pady=1)
            tk_label(row, k, 14, "text2", "panel", width=34).pack(side="left")
            tk_label(row, val, 14, "text", "panel", font=font(14, mono=True)).pack(side="left")

    def _few_shot(self, cur):
        card = self.shot_card
        for w in card.winfo_children():
            w.destroy()
        card.pack_forget()
        if cur is None:
            return
        card.pack(fill="x", pady=(0, GUTTER), after=self.score_card)
        head = tk.Frame(card, bg=C["panel"])
        head.pack(fill="x")
        Kicker(head, T["few_shot"], bg="panel").pack(side="left")
        info_icon(head, T["few_shot_hint"]).pack(side="left", padx=(4, 0))
        table = self.shots.get(cur.number)
        run = button(head, T["few_shot_run" if table is None else "few_shot_again"], lambda n=cur.number: self._check_few_shot(n))
        run.pack(side="right")
        run.state(["disabled" if cur.number in self.shot_busy else "!disabled"])
        rows = few_shot_rows(table)
        if not rows:
            line = T["few_shot_not_run"] if table is None else table.get("error") or T["few_shot_none"]
            tk_label(card, line, 13, "muted", "panel", wraplength=780).pack(anchor="w", pady=(6, 0))
            return
        grid = tk.Frame(card, bg=C["panel"])
        grid.pack(anchor="w", pady=(6, 0))
        for c, text in enumerate(T["few_shot_head"]):
            tk_label(grid, text, 12, "muted", "panel").grid(row=0, column=c, sticky="w", padx=(0, 18))
        for r, cells in enumerate(rows, 1):
            for c, text in enumerate(cells):
                tk_label(grid, text, 14, "text2" if c in (0, 4) else "text", "panel", font=font(14, mono=c in (1, 2, 3))).grid(
                    row=r, column=c, sticky="w", padx=(0, 18))
        tk_label(card, few_shot_note(table), 12, "muted", "panel", wraplength=780).pack(anchor="w", pady=(6, 0))

    def _check_few_shot(self, number):
        def work():
            try:
                return self.services.few_shot(number)
            except Exception as error:  # say it in the card; the button comes back
                return {"error": plain_error(error)}

        def done(table):
            self.shot_busy.discard(number)
            self.shots[number] = table
            self._few_shot(self._versions().get(self.use))
        self.shot_busy.add(number)
        self._few_shot(self._versions().get(self.use))
        self.background(work, done, T["few_shot_running"])

    # actions ----------------------------------------------------------------
    def send(self):
        text = self.entry.get().strip()
        if not text or self.use is None:
            return
        self.entry.delete(0, "end")
        self.stop.clear()
        msg = {"role": "user", "content": text}
        self.messages = [msg] if self.compare else self.messages + [msg]
        for i, num in enumerate([self.use, self.b] if self.compare else [self.use]):
            self.base[i] = "" if self.compare else self.replies.get(i, "") + f"You: {text}\n\n"
            self.replies[i] = self.base[i]
            self._set_text(i, self.replies[i])
            self.pending += 1
            threads.start(self._work, i, num, list(self.messages))
        self.after(30, self._pump)

    def _work(self, i, num, messages):
        stream_reply(self.services.chat, num, messages, self.stop, lambda text: self.pump_q.put((i, text)))

    def _pump(self):
        while True:
            try:
                i, text = self.pump_q.get_nowait()
            except queue.Empty:
                break
            if text is None:
                self.pending -= 1
                continue
            self.replies[i] = self.base.get(i, "") + text
            self._set_text(i, self.replies[i])
        if self.pending > 0:
            self.after(30, self._pump)

    def better(self, i):
        chosen, other = (self.use, self.b) if i == 0 else (self.b, self.use)
        if self.pending or not self.replies.get(0) or not self.replies.get(1) or self.replies[0] == self.replies[1]:
            return self.status_label.configure(text="Wait for both replies, and they must differ, before picking one.")
        self.services.rate_pair(self.messages, {"version": chosen, "reply": self.replies.get(i, "")},
                                {"version": other, "reply": self.replies.get(1 - i, "")})
        self.status_label.configure(text=f"Saved: Version {chosen} was better. Used in Post-training round 4.")

    def teach(self):
        text = self.correction.get().strip()
        if not text or self.use is None:
            return
        if not self.messages:
            return self.status_label.configure(text="Send a message first; a correction belongs to a reply.")
        whole = self.whole.value
        try:
            if whole:
                self.services.teach_correction(self.use, self.messages, text, whole_answer=True)
            else:
                self.services.teach_correction(self.use, self.messages, text)
        except NotImplementedError:
            return self.status_label.configure(text="Saving corrections is not available yet.")
        self.status_label.configure(text="Saved for Post-training round 1 as a whole new answer." if whole else "Saved for Post-training round 3.")
        self.correction.delete(0, "end")

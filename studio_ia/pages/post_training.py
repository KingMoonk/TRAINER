"""Post-training: four rounds in a fixed order, one card with the round's visual, and the gate that decides if Chat moves.

Reads services.post_training(version) -> {"rounds", "counts", "gates", "example"}; live train events (group, heat, moved,
gate) from the running round replace the illustrative samples. "Add to the queue" enqueues post_round jobs.
"""
from __future__ import annotations

import tkinter as tk
import uuid

from studio_ia.copy import FEEDBACK as T
from studio_ia.next_step import empty_line
from studio_ia.pages.base import Page
from studio_ia.services import Job, if_added
from studio_ia.theme import C, GUTTER, font, text_box, tk_label
from studio_ia.widgets import Card, Chip, HeatGrid, Kicker, KofN, Segmented, _activate, button, info_icon, tip

ROUNDS = (("Practice answers", T["round_1"], T["r1_method"], T["r1_lead"], T["answers_now"]),
          ("Practice with checks", T["round_2"], T["r2_method"], T["r2_lead"], T["passed_became"]),
          ("Learn from a teacher", T["round_3"], T["r3_method"], T["r3_lead"], T["moved_toward_what"]),
          ("Your votes", T["round_4"], T["r4_method"], T["r4_lead"], T["widened_gap_what"]))
COUNTED = (2, 3)  # rounds whose verdict is a count, known only once the round has run
STATUS = {"done": "done", "running": "running", "queued": "in the queue", "idle": "not started"}
LEASH = (("Close", .5, T["close_note"]), ("Medium", .1, T["medium_note"]), ("Far", .02, T["far_note"]))  # label, beta, note
SAMPLE_GROUP = ("Sort these letters: d b a c", "sort", [("a b c d", 1.), ("a b d c", 0.), ("a b c d", 1.), ("a c b d", 0.), ("a b c d", 1.), ("d c b a", 0.)])
SAMPLE_HEAT = (["The", "capital", "of", "Australia", "is", "Sydney", ",", "which", "is", "on", "the", "coast"],
               [0, 0, 0, .1, 0, .95, .3, .2, .1, .1, 0, .4])
ADD = "Add to the queue"


def round_outlook(services, i, status):
    """The line under round i+1's button: for a round already in the queue, when it starts; otherwise what adding it now would do."""
    q = services.queue()
    if status == "queued":
        waiting = next((j for j in q.up_next if j.kind in ("post_round", "restore_answers")
                        and (3 if j.kind == "restore_answers" else int(j.params.get("round", 0))) == i + 1), None)
        return services.queue_outlook(waiting)
    first = next((j for j in q.up_next if j.status != "held"), None) if i == 2 else None  # "Move to the front": starts when the head would
    return if_added(services.queue_outlook(first))


def blocker(round_index, counts, teacher=0, has_earlier=False, checkpoint=True):
    """Why a round cannot be queued yet (plain words), or None. counts = post_training()["counts"]; teacher 0 = previous best."""
    need = {0: ("conversations", 1, "Add conversations in Your data first."),
            1: ("problems", 1, "Make problems a rule can check first."),
            3: ("pairs", 2, "Rate at least 2 pairs in Chat first.")}
    if not checkpoint:
        return "Train this version in Pretraining first."
    if round_index == 2:
        if teacher == 0:
            return None if has_earlier else "There is no earlier version to learn from yet."
        return None if counts.get("corrections") else "Teach a correction in Chat first."
    key, least, why = need[round_index]
    return None if (counts.get(key) or 0) >= least else why


def step_line(q):
    """'step k of n · about m min left' from the running queue state; only the parts that are known."""
    parts = [f"step {q.step:,} of {q.total:,}" if q.total else "starting"]
    if q.eta_s is not None:
        parts.append(f"about {max(1, round(q.eta_s / 60))} min left")
    return " · ".join(parts)


def live_view(selected, group, heat, moved, gates):
    """What the live events change on screen for the open round: the same value means the page already shows it."""
    visual = (group,) if selected == 1 else (heat, moved) if selected == 2 else ()
    return (selected, visual, gates.get(selected + 1))


def count(value):
    return "–" if value is None else f"{value:,}"


def checkpoint_name(services, version):
    """'version 4 · checkpoint at step 1,200' (the parts that are known), or 'its current checkpoint'."""
    if version is None:
        return "its current checkpoint"
    try:
        step = services.checkpoint_step(version)
    except NotImplementedError:
        step = None
    return f"version {version.number}" + (f" · checkpoint at step {step:,}" if step else "")


class PostTrainingPage(Page):
    title = T["h1"]
    lead = T["lead"]

    def __init__(self, parent, services, navigate=None, **kw):
        super().__init__(parent, services, navigate, **kw)
        self.selected = None      # round index; None = the first round that is not done
        self.teacher = 0          # 0 previous best version, 1 this model with a correction in front
        self.leash = 1
        self.pick = ""            # round 4 vote
        self.group = self.heat = self.moved = None   # live visuals from the running round
        self.live_gate = {}
        self._drawn, self._ticking, self._pending, self._step = None, False, False, None   # live_view on screen; in a burst; events since the last draw; the "step k of n" label

    # data -------------------------------------------------------------------
    def _version(self):
        try:
            return self.services.best_version() or self.services.current_version()
        except NotImplementedError:
            return None

    def _state(self, version):
        try:
            return self.services.post_training(version.number if version else None) or {}
        except NotImplementedError:
            return {}

    def preload(self):
        self._state(self._version())

    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        version = self._version()
        state = self._state(version)
        self.version, self.version_no = version, version.number if version else None
        rounds, self.counts = state.get("rounds", {}), state.get("counts", {})
        self.gates = {**state.get("gates", {}), **self.live_gate}
        self.example = state.get("example", {})
        status = [(rounds.get(i + 1) or {}).get("status", "idle") for i in range(4)]
        notes = [(rounds.get(i + 1) or {}).get("note", "") for i in range(4)]
        if self.selected is None:
            self.selected = next((i for i, s in enumerate(status) if s != "done"), 0)
        tk_label(self.body, T["starts_from"], 12, "muted").pack(anchor="w")
        starts = T["version_best"].replace("Version 4", f"Version {version.number}") if version else "No trained version yet"
        tk_label(self.body, starts, 16, "text").pack(anchor="w", pady=(2, 0 if version else GUTTER))
        if not version and (line := empty_line(self.services)):
            tk_label(self.body, line, 13, "muted", wraplength=760).pack(anchor="w", pady=(2, GUTTER))
        self._stages(status, notes)
        row = tk.Frame(self.body, bg=C["bg"])
        row.pack(fill="x", pady=GUTTER)
        tip(Chip(row, T["at_this_size"], "neutral"), T["click_round"]).pack(anchor="w")
        self._card(version, status)
        self._gate_card(status)
        self._drawn = live_view(self.selected, self.group, self.heat, self.moved, self.gates)

    def on_event(self, kind, payload):
        """Live visuals arrive in bursts (a group per prompt, every step). The first is drawn at once; the rest of a burst
        is drawn at most twice a second, and only if the open round shows something new. Rebuilding the page for each one
        made it flash. A gate comes once per round and is drawn at once."""
        if kind == "train" and isinstance(payload, dict) and payload.get("type") in ("group", "heat", "moved", "gate"):
            t = payload["type"]
            if t == "gate":
                self.live_gate[payload["round"]] = payload
                return self._draw()
            setattr(self, t, payload)
            if self._ticking:
                self._pending = True
            else:
                self._draw()
                self._ticking = True
                self.after(500, self._tick)
        elif kind in ("queue", "versions", "job"):
            self.reload()

    def _tick(self):
        self._ticking = False
        if self._pending:
            self._pending = False
            self._draw()
            self._ticking = True
            self.after(500, self._tick)

    def _draw(self):
        if live_view(self.selected, self.group, self.heat, self.moved, {**getattr(self, "gates", {}), **self.live_gate}) != self._drawn:
            return self.refresh()
        if self._step is not None:  # nothing new to draw: only the step count moves, in place
            self._step.configure(text=step_line(self.services.queue()))

    # stage line ---------------------------------------------------------------
    def _stages(self, status, notes):
        line = tk.Frame(self.body, bg=C["bg"])
        line.pack(fill="x")
        for i, spec in enumerate(ROUNDS):
            name = spec[0]
            s, on = status[i], i == self.selected
            bg = C["panel2"] if on else C["bg"]
            cell = tk.Frame(line, bg=bg, highlightthickness=2, highlightbackground=C["violet"] if on else C["border"],
                            highlightcolor=C["violet"], takefocus=1, cursor="hand2")
            cell.pack(side="left", fill="x", expand=True, padx=(0, 6))
            fill = C["blue"] if s == "done" else C["violet"] if s == "running" else bg
            mark = tk_label(cell, "✓" if s == "done" else str(i + 1), 14, "bg" if s in ("done", "running") else "muted", bg, font=font(14, bold=True))
            mark.configure(bg=fill, width=2)
            mark.pack(side="left", padx=GUTTER, pady=10)
            text = tk.Frame(cell, bg=bg)
            text.pack(side="left", fill="x", expand=True, pady=8)
            label = STATUS[s] + (f" · {notes[i]}" if notes[i] else "")
            if i == 3 and s == "idle":
                label = f"optional · {count(self.counts.get('pairs'))} pairs rated"
            tk_label(text, name, 15, "text", bg, font=font(15, bold=True)).pack(fill="x")
            tk_label(text, label, 12, "blue" if s == "done" else "violet" if s == "running" else "muted", bg).pack(fill="x")
            _activate(cell, lambda i=i: self._select(i))
            for w in (mark, text, *text.winfo_children()):  # only the tile takes focus; a ring on every part would widen the row
                w.bind("<Button-1>", lambda _e, i=i: self._select(i))

    def _select(self, i):
        self.selected = i
        self.refresh()

    # round card ---------------------------------------------------------------
    def _card(self, version, status):
        i = self.selected
        name, kicker, method, lead, _verdict = ROUNDS[i]
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        self._help = [method]  # the round's explanations, in a tooltip beside its name
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w")
        Kicker(head, kicker, "violet", "panel").pack(side="left")
        info_icon(head, lambda: "\n\n".join(self._help)).pack(side="left", padx=(4, 0))
        tk_label(card, lead, 15, "text2", "panel", wraplength=760).pack(anchor="w", pady=(6, GUTTER))
        [self._r1, self._r2, self._r3, self._r4][i](card)
        actions = tk.Frame(card, bg=C["panel"])
        actions.pack(fill="x", pady=(GUTTER, 0))
        label = {2: T["move_front"], 3: T["add_votes"]}.get(i, ADD)
        why = blocker(i, self.counts, self.teacher, bool(self._previous_best(version)), bool(version and version.best_checkpoint))
        add = button(actions, label, lambda: self._enqueue(version, i), primary=True)
        add.pack(side="left")
        if why:
            add.state(["disabled"])
            tk_label(actions, why, 13, "amber", "panel").pack(side="left", padx=GUTTER)
        if not why and status[i] != "running":
            tk_label(card, round_outlook(self.services, i, status[i]), 13, "muted", "panel").pack(anchor="w", pady=(6, 0))
        link = {0: (T["add_conversations"], "your_data"), 1: (T["make_problems"], "make_data"), 3: (T["rate_pairs"], "chat")}.get(i)
        if link:
            go = tk_label(actions, link[0] + " ›", 14, "violet", "panel", cursor="hand2", takefocus=1)
            go.pack(side="left", padx=GUTTER)
            _activate(go, lambda route=link[1]: self.navigate(route))
        foot = {0: "Corrections you teach in Chat that need a whole new answer also land here.", 1: "Prompts where all answers pass, or all fail, teach nothing and are skipped. Each rule gets the room its answers need, up to 96 word pieces, so a step takes longer than it would with every answer cut at 32.",
                2: "Also runs on its own after a night of training on new text: new text can erode how it answers, and a short lesson from the previous best version restores it without losing the new text.",
                3: f"{count(self.counts.get('pairs'))} pairs rated. 1 in 5 is held back to check the result."}[i]
        self._help.append(foot)

    def _box(self, parent, kicker, text, grow=True):
        box = tk.Frame(parent, bg=C["panel2"], padx=12, pady=10)
        box.pack(side="left", fill="both", expand=grow, padx=(0, 8))
        tk_label(box, kicker, 12, "muted", "panel2").pack(anchor="w")
        tk_label(box, text, 15, "text", "panel2", wraplength=300).pack(anchor="w", pady=(4, 0))

    def _r1(self, card):
        pair = tk.Frame(card, bg=C["panel"])
        pair.pack(fill="x")
        self._box(pair, T["prompt_scored"], self.example.get("prompt") or "No conversations yet: add some in Your data.")
        self._box(pair, T["answer_learns"], self.example.get("answer") or "The answer from the same conversation appears here.")
        nums = tk.Frame(card, bg=C["panel"])
        nums.pack(fill="x", pady=(GUTTER, 6))
        for key, label in (("conversations", T["conversations_label"]), ("corrections", T["corrections_label"]), ("problems", T["problems_label"])):
            cell = tk.Frame(nums, bg=C["panel"])
            cell.pack(side="left", padx=(0, 2 * GUTTER))
            tk_label(cell, count(self.counts.get(key)), 22, "text", "panel", font=font(22, mono=True)).pack(anchor="w")
            tk_label(cell, label, 12, "muted", "panel").pack(anchor="w")
        self._help.append(T["few_thousand"])

    def _r2(self, card):
        live = self.group
        prompt, rule, answers = (live["prompt"], live["rule"], [(a["text"], a["score"], a["push"]) for a in live["answers"]]) if live else \
            (SAMPLE_GROUP[0], SAMPLE_GROUP[1], [(t, s, s - .5) for t, s in SAMPLE_GROUP[2]])
        mean = live["mean"] if live else .5
        head = tk.Frame(card, bg=C["panel"])
        head.pack(fill="x")
        tk_label(head, T["prompt_rule"].format(rule=rule.replace("_", " ").upper()) + f": {prompt}", 12, "muted", "panel").pack(side="left")
        if not live:
            Chip(head, T["illustrative"], "neutral").pack(side="left", padx=8)
        grid = tk.Frame(card, bg=C["panel"])
        grid.pack(fill="x", pady=6)
        for c, h in enumerate((T["answer_output"], T["rule_label"], T["score_label"], T["group_avg"].replace("(0.5)", f"({mean:.2f})"))):
            tk_label(grid, h, 12, "muted", "panel").grid(row=0, column=c, sticky="w", padx=(0, GUTTER))
        for r, (text, score, push) in enumerate(answers, 1):
            ok = score >= 1
            color = "blue" if ok else "coral"
            push_text = "no push" if abs(push) < 1e-9 else (f"↑ {push:+.2f} · pushed up" if push > 0 else f"↓ {push:+.2f} · pushed down")
            for c, (value, col) in enumerate(((text, "text"), ("pass" if ok else "fail", color), (f"{score:.1f}", color), (push_text, color))):
                tk_label(grid, value, 14, col, "panel", font=font(14, mono=c == 0)).grid(row=r, column=c, sticky="w", padx=(0, GUTTER), pady=2)
        if not live or live.get("skipped"):
            grey = tk.Frame(card, bg=C["panel"], highlightthickness=1, highlightbackground=C["border2"], padx=10, pady=6)
            grey.pack(fill="x", pady=6)
            n = len(answers)
            first = f"{prompt}" if live else T["copy_this"]
            passed = f"{sum(1 for _t, sc, _p in answers if sc >= 1)} of {n} passed" if live else T["passed_count"]
            for text in (first, passed, T["nothing_learn"]):
                tk_label(grey, text, 13, "dim", "panel").pack(side="left", padx=(0, GUTTER))
        self._help.append(T["rules_check"])

    def _r3(self, card):
        live = self.heat
        words, surprise = (live["words"], live["surprise"]) if live else SAMPLE_HEAT
        tk_label(card, T["prompt_label"] + (": " + live["prompt"] if live else ": What is the capital of Australia?"), 12, "muted", "panel").pack(anchor="w")
        tk_label(card, T["answer_shaded"], 12, "muted", "panel").pack(anchor="w", pady=(6, 4))
        text = text_box(card, height=3, size=16)
        top = max(1.0, max(surprise, default=0))
        for n, (w, s) in enumerate(zip(words, surprise)):
            text.tag_configure(f"w{n}", background=HeatGrid._mix(C["panel2"], C["coral"], min(1., s / top)))
            text.insert("end", w + " ", f"w{n}")
        text.configure(state="disabled")
        text.pack(fill="x")
        legend = tk.Frame(card, bg=C["panel"])
        legend.pack(fill="x", pady=4)
        tk_label(legend, T["not_surprised"], 12, "muted", "panel").pack(side="left")
        tk_label(legend, " → ", 12, "muted", "panel").pack(side="left")
        tk_label(legend, T["very_surprised"], 12, "coral", "panel").pack(side="left")
        if not live:
            Chip(legend, T["illustrative"], "neutral").pack(side="left", padx=8)
        if self.moved:
            KofN(card, self.moved["k"], self.moved["n"], "words moved toward the teacher").pack(anchor="w", pady=4)
        tk_label(card, T["teacher_label"], 12, "muted", "panel").pack(anchor="w", pady=(GUTTER, 4))
        labels = (T["previous_best"], T["this_model_correction"])
        earlier = self._previous_version(self.version)
        previous = T["previous_desc"].format(version=checkpoint_name(self.services, earlier).capitalize()) if earlier else T["no_previous"]
        tip(Segmented(card, labels, labels[self.teacher], lambda v: self._set("teacher", labels.index(v))), (previous, T["correction_desc"])[self.teacher]).pack(anchor="w")
        self._help.append(T["teacher_note"])

    def _r4(self, card):
        pair = tk.Frame(card, bg=C["panel"])
        pair.pack(fill="x")
        for k in "AB":
            on = self.pick == k
            box = tk.Frame(pair, bg=C["panel2"], padx=12, pady=10, highlightthickness=2 if on else 1,
                           highlightbackground=C["violet"] if on else C["border"])
            box.pack(side="left", fill="both", expand=True, padx=(0, 8))
            tk_label(box, f"answer {k}", 12, "muted", "panel2").pack(anchor="w")
            tk_label(box, f"Answer {k} to the same prompt", 15, "text", "panel2").pack(anchor="w", pady=4)
            b = tk.Label(box, text=f"Picked: {k} is better" if on else f"{k} is better", font=font(14, bold=True), padx=12, pady=8, takefocus=1,
                         cursor="hand2", bg=C["violet"] if on else C["panel2"], fg=C["bg"] if on else C["violet"],
                         highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["violet"])
            b.pack(anchor="w")
            _activate(b, lambda k=k: self._set("pick", k))
        self._gap(card)
        self._help.append(T["for_each_pair"])
        tk_label(card, T["how_far"], 12, "muted", "panel").pack(anchor="w", pady=(0, 4))
        names = [name for name, _beta, _note in LEASH]
        Segmented(card, names, names[self.leash], lambda v: self._set("leash", names.index(v))).pack(anchor="w")
        tk_label(card, f"{LEASH[self.leash][2]}  (β = {LEASH[self.leash][1]})", 13, "text2", "panel", wraplength=760).pack(anchor="w", pady=(6, 0))
        self._help.append(T["leash_note"])

    def _gap(self, card):
        """Frozen starting copy vs learning copy: how likely your pick and the other answer are. Heights are illustrative."""
        canvas = tk.Canvas(card, width=460, height=150, bg=C["panel"], highlightthickness=0)
        canvas.pack(anchor="w", pady=GUTTER)
        base = 120
        for x, name, pick, other, fill in ((40, "frozen copy", 70, 60, C["muted"]), (260, "learning copy", 45, 15, C["violet"])):
            for dx, h, label in ((0, pick, "your pick"), (70, other, "the other")):
                canvas.create_rectangle(x + dx, base - h, x + dx + 50, base, fill=fill, width=0)
                canvas.create_text(x + dx + 25, base + 12, text=label, fill=C["muted"], font=font(11))
            canvas.create_text(x + 60, 8, text=name, fill=C["text2"], font=font(12, bold=True))
        canvas.create_line(205, base - 45, 235, base - 45, fill=C["amber"], dash=(3, 3))
        canvas.create_line(415, base - 45, 415, base - 15, fill=C["amber"], arrow="both")
        canvas.create_text(440, base - 30, text="gap", fill=C["amber"], font=font(12, bold=True))
        canvas.create_text(450, 140, text=T["illustrative"], fill=C["dim"], font=font(11), anchor="e")

    def _set(self, name, value):
        setattr(self, name, value)
        self.refresh()

    # gate card ------------------------------------------------------------------
    def _gate_card(self, status):
        i = self.selected
        verdict_text = ROUNDS[i][4]
        s = status[i]
        gate = self.gates.get(i + 1)
        self._step = None
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        if gate:
            kicker = f"RESULT · ROUND {i + 1}"
        else:
            kicker = f"TRAINING · ROUND {i + 1}" if s == "running" else "WHEN THIS ROUND ENDS"
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w")
        Kicker(head, kicker, "violet", "panel").pack(side="left")
        info_icon(head, f"{T['chat_moves']}\n\n{T['judge_note']}").pack(side="left", padx=(4, 0))
        if s == "running":
            self._step = tk_label(card, step_line(self.services.queue()), 13, "muted", "panel")
            self._step.pack(anchor="w")
        tk_label(card, T["verdict_title"], 12, "muted", "panel").pack(anchor="w", pady=(GUTTER, 0))
        if gate:
            verdict = gate.get("verdict") or ("" if i in COUNTED else verdict_text)
        elif i in COUNTED:
            verdict = (T["so_far_count"] if s == "running" else T["will_count"]).format(what=verdict_text)
        elif s == "running":
            verdict = T["so_far"].format(verdict_start=verdict_text[0].lower() + verdict_text[1:]) + " (on training prompts; held-back check at the end)"
        else:
            verdict = T["will_read"].format(verdict_full=verdict_text)
        tk_label(card, verdict, 17, "blue" if gate else "text2", "panel", wraplength=760).pack(anchor="w", pady=(2, GUTTER))
        checks = gate["checks"] if gate else [{"name": T["held_back_check"], "detail": T["held_back_measured"], "ok": None},
                                              {"name": T["problems_check"], "detail": T["problems_measured"], "ok": None}]
        for c in checks:
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", pady=3)
            ok = c["ok"]
            tk_label(row, "…" if ok is None else "✓" if ok else "✕", 16, "muted" if ok is None else "blue" if ok else "coral", "panel",
                     font=font(16, bold=True)).pack(side="left", padx=(0, 10))
            text = tk.Frame(row, bg=C["panel"])
            text.pack(side="left")
            tk_label(text, c["name"], 14, "text", "panel").pack(anchor="w")
            tk_label(text, c["detail"], 12, "muted", "panel").pack(anchor="w")
        judge = (self.gates.get(i + 1) or {}).get("judge")
        tk_label(card, "·", 12, "dim", "panel").pack(anchor="w")
        if judge:
            KofN(card, judge[0], judge[1], "prompts preferred by the judge").pack(anchor="w")
        else:
            tk_label(card, T["judge_none"], 14, "text2", "panel").pack(anchor="w")
        if gate:
            result = T["result_live"] if gate["ok"] else T["result_kept"]
        else:
            result = T["chat_keeps"] if s == "running" else T["until_then"]
        tk_label(card, result.format(checkpoint=checkpoint_name(self.services, self.version)), 14, "text2", "panel", wraplength=760).pack(anchor="w", pady=(GUTTER, 0))

    # actions ----------------------------------------------------------------------
    def _previous_version(self, version):
        earlier = [v for v in self.services.versions() if version and v.number < version.number and v.best_checkpoint]
        return earlier[-1] if earlier else None

    def _previous_best(self, version):
        earlier = self._previous_version(version)
        return earlier.best_checkpoint if earlier else None

    def _enqueue(self, version, i):
        name = ROUNDS[i][0]
        params = {"round": i + 1, "version": version.number if version else None, "checkpoint": version.best_checkpoint if version else None}
        if i == 2:
            params["teacher"] = self._previous_best(version) if self.teacher == 0 else None
            params["teacher_choice"] = ("previous best", "correction")[self.teacher]
        if i == 3:
            params["beta"] = LEASH[self.leash][1]
        job = Job(uuid.uuid4().hex[:8], "post_round", f"Round {i + 1} · {name}", f"post-training round {i + 1}", params=params)
        self.services.enqueue(job)
        if i == 2:  # "Move to the front of the queue"
            for _ in range(len(self.services.queue().up_next)):
                self.services.move_up(job.id)

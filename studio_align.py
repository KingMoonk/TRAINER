"""Align page: check the model → collect preferences → improve with feedback → compare results."""
from __future__ import annotations

import json
import queue
import random
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from studio_core import uid
from studio_model import LocalModel
from studio_rlhf import (AUTO_DEFAULTS, DEFAULT_RUBRIC, DPO_DEFAULTS, LOSS_TYPES, MIN_PAIRS_TO_TRAIN, OllamaTeacher,
                         PreferenceStore, auto_rlhf, compare_checkpoints, dpo_train, evaluate_checkpoint, load_settings,
                         normalize_rubric, prompts_from, save_settings)
from studio_widgets import (ACCENT, BG, FONT, MUTED, TEXT, Tooltip, help_icon, help_text, with_help,
                            label, button, text_box, content, set_text, card, field, tree)

WARN = "#ffb86b"
SAME_AS_TEACHER = "(same as teacher)"
STEPS = [("A", "Check your model", "Is it trained enough to be worth polishing?"),
         ("B", "Collect preferences", "Say which replies are better, or let an Ollama teacher do it."),
         ("C", "Improve with feedback", "Train the model toward the replies that were preferred."),
         ("D", "Compare results", "Check that the new model really beats the old one.")]
TAB_TO_STEP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3}
TABS = ("A · Check model", "B · Rate replies", "B · Auto teacher", "B · Saved preferences", "C · Improve model", "D · Compare")
TAB_CHECK, TAB_RATE, TAB_AUTO, TAB_SAVED, TAB_IMPROVE, TAB_COMPARE = range(6)
KEEP_NOTE = "Only keep a new checkpoint if it beats the current one head-to-head."


def short(text, size=70):
    text = " ".join(str(text).split())
    return text if len(text) <= size else text[:size - 1] + "…"


def prompt_text(messages):
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


class Collapsible(ttk.Frame):
    """A card section that stays closed until asked: rarely-needed knobs live here."""

    def __init__(self, parent, title):
        super().__init__(parent, style="Card.TFrame")
        self.title, self.opened = title, False
        self.toggle_button = ttk.Button(self, text=f"▸  {title}", command=self.toggle)
        self.toggle_button.pack(anchor="w")
        self.body = ttk.Frame(self, style="Card.TFrame")

    def toggle(self, open_=None):
        self.opened = (not self.opened) if open_ is None else open_
        self.toggle_button.configure(text=("▾  " if self.opened else "▸  ") + self.title)
        if self.opened:
            self.body.pack(fill="x", pady=(10, 0))
        else:
            self.body.pack_forget()


def setting_rows(parent, rows):
    """Aligned rows of `label | input | plain-English note`. A row is (title, variable, note[, values])."""
    parent.columnconfigure(2, weight=1)
    for index, (title, variable, note, *rest) in enumerate(rows):
        values = rest[0] if rest else None
        helper = rest[1] if len(rest) > 1 else None
        if isinstance(variable, tk.BooleanVar):
            ttk.Checkbutton(parent, text=title, variable=variable).grid(row=index, column=0, columnspan=2, sticky="w", pady=4)
        else:
            if helper:
                with_help(parent, title, helper, "Card.TLabel").grid(row=index, column=0, sticky="w", padx=(0, 16), pady=4)
            else:
                label(parent, title, "Card.TLabel").grid(row=index, column=0, sticky="w", padx=(0, 16), pady=4)
            widget = (ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=12) if values
                      else ttk.Entry(parent, textvariable=variable, width=13))
            widget.grid(row=index, column=1, sticky="w", pady=4)
        label(parent, note, "CardMuted.TLabel", wraplength=640, justify="left").grid(row=index, column=2, sticky="w", padx=(16, 0))


class AlignMixin:
    def _layout(self):
        super()._layout()
        self.preferences = PreferenceStore(self.workspace.root)
        self.align_settings = load_settings(self.workspace.root)
        self.align_model = None
        self.align_model_path = None
        self.comparison = None
        self.vote_log = []
        self.align_prompts = []
        self.rubric = normalize_rubric(self.align_settings["rubric"])
        self.arena_examples = []
        self._build_align()

    # ------------------------------------------------------------------ page

    def _build_align(self):
        page = self.pages["align"].body
        top = card(page)
        top.pack(fill="x", pady=(0, 8))
        head = ttk.Frame(top, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 8))
        label(head, "Model to improve", "CardTitle.TLabel").pack(side="left")
        help_icon(head, "rlhf").pack(side="left", padx=(6, 0))
        label(head, "Polishes a model that already writes sensible text. The original file is never overwritten.", "CardMuted.TLabel").pack(side="left", padx=(14, 0))
        row = ttk.Frame(top, style="Card.TFrame")
        row.pack(fill="x")
        self.align_checkpoint = tk.StringVar()
        ttk.Entry(row, textvariable=self.align_checkpoint).pack(side="left", fill="x", expand=True, padx=(0, 8))
        button(row, "Choose file\u2026", lambda: self.choose_checkpoint(self.align_checkpoint))
        button(row, "Use latest run", self.use_latest_checkpoint, True)
        device = ttk.Combobox(row, textvariable=self.inference_device, values=("auto", "cpu", "cuda"), state="readonly", width=8)
        device.pack(side="left")
        Tooltip(device, help_text("inference"))
        self.mark_advanced(device)


        steps = ttk.Frame(page)
        steps.pack(fill="x", pady=(0, 8))
        self.step_widgets = []
        for index, (number, title, text) in enumerate(STEPS):
            frame = ttk.Frame(steps, style="Card.TFrame", padding=(14, 8))
            frame.pack(side="left", fill="both", expand=True, padx=(0 if index == 0 else 8, 0))
            head = ttk.Frame(frame, style="Card.TFrame")
            head.pack(anchor="w")
            badge = label(head, number, "Card.TLabel", font=(FONT, 16, "bold"), foreground=MUTED)
            badge.pack(side="left", padx=(0, 10))
            name = label(head, title, "Card.TLabel", font=(FONT, 11, "bold"))
            name.pack(side="left")
            detail = frame     # the one-line description is a tooltip so the vote buttons stay on screen
            Tooltip(frame, text)
            Tooltip(name, text)
            target = {0: TAB_CHECK, 1: TAB_RATE, 2: TAB_IMPROVE, 3: TAB_COMPARE}[index]
            for widget in (frame, head, badge, name, detail):
                widget.bind("<Button-1>", lambda _e, t=target: self.align_tabs.select(t))
            self.step_widgets.append((badge, name))

        self.align_tabs = ttk.Notebook(page)
        self.align_tabs.pack(fill="both", expand=True)
        frames = [ttk.Frame(self.align_tabs, padding=10) for _ in TABS]
        for frame, title in zip(frames, TABS):
            self.align_tabs.add(frame, text=title)
        self._build_readiness(frames[TAB_CHECK])
        self._build_human(frames[TAB_RATE])
        self._build_teacher(frames[TAB_AUTO])
        self._build_saved(frames[TAB_SAVED])
        self._build_dpo(frames[TAB_IMPROVE])
        self._build_arena(frames[TAB_COMPARE])
        self.align_tabs.bind("<<NotebookTabChanged>>", lambda _e: self.highlight_step())
        self.highlight_step()
        self.bind_all("<KeyPress>", self.align_hotkey, add="+")

    def highlight_step(self):
        try:
            active = TAB_TO_STEP[self.align_tabs.index(self.align_tabs.select())]
        except (tk.TclError, KeyError):
            active = 0
        for index, (badge, name) in enumerate(self.step_widgets):
            badge.configure(foreground=ACCENT if index == active else MUTED)
            name.configure(foreground=ACCENT if index == active else TEXT)

    def _build_readiness(self, page):
        box = card(page, "Is the model trained enough to polish?")
        box.pack(fill="x", pady=(0, 12))
        label(box, "Preference training fine-tunes behaviour; it cannot teach a model language from scratch. This check measures how often the model guesses the wrong next word "
                   "inside correct answers from your working dataset (these examples may overlap with the training data).",
              "CardMuted.TLabel", wraplength=960, justify="left").pack(anchor="w", pady=(0, 10))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x")
        self.target_error, self.eval_size = tk.StringVar(value="45"), tk.StringVar(value="300")
        for title, var, helper in [("Good enough when wrong-word rate is below (%)", self.target_error, "token_error"), ("Examples to check", self.eval_size, "example")]:
            frame, _ = field(row, title, var, width=14, help=helper)
            frame.pack(side="left", padx=(0, 14))
        label(box, "A brand-new model guesses wrong almost every time (90%+). Around 45% or lower, a small model usually writes recognisable replies and is worth polishing; "
                   "lower is better.", "CardMuted.TLabel", wraplength=960, justify="left").pack(anchor="w", pady=(10, 0))
        actions = ttk.Frame(box, style="Card.TFrame")
        actions.pack(fill="x", pady=(12, 0))
        button(actions, "Check the model", self.evaluate_readiness, True)
        button(actions, "Train it more first", lambda: self.show_page("training"))
        button(actions, "Next: collect preferences →", lambda: self.align_tabs.select(TAB_RATE))
        metrics = ttk.Frame(page)
        metrics.pack(fill="x", pady=(0, 12))
        self.ready_vars = {}
        for key, title, helper in [("token_error", "WRONG-WORD RATE", "token_error"), ("perplexity", "PERPLEXITY (SURPRISE)", "perplexity"),
                                   ("loss", "ERROR (LOSS)", "loss"), ("val_loss", "TEST ERROR OF THE RUN", "val_loss")]:
            tile = card(metrics)
            tile.pack(side="left", fill="x", expand=True, padx=(0, 10))
            var = tk.StringVar(value="\u2014")
            self.ready_vars[key] = var
            label(tile, textvariable=var, style="Metric.TLabel").pack(anchor="w")
            with_help(tile, title, helper, font=(FONT, 9)).pack(anchor="w", pady=(4, 0))
        self.ready_verdict = tk.StringVar(value="Choose a checkpoint and run the check.")
        self.ready_label = label(page, textvariable=self.ready_verdict, font=(FONT, 14, "bold"))
        self.ready_label.pack(anchor="w", pady=(4, 0))

    def _build_human(self, page):
        controls = card(page, "Pick the better of two replies")
        controls.pack(fill="x", pady=(0, 8))
        intro = ttk.Frame(controls, style="Card.TFrame")
        intro.pack(fill="x", pady=(0, 6))
        label(intro, "The model answers a prompt twice. Say which reply is better, or write the right answer. Each choice is one preference pair.",
              "CardMuted.TLabel").pack(side="left")
        help_icon(intro, "preference").pack(side="left", padx=(4, 0))
        actions = ttk.Frame(controls, style="Card.TFrame")
        actions.pack(fill="x")
        button(actions, "Next prompt from dataset", lambda: self.next_comparison(None), True)
        button(actions, "Use my prompt below", lambda: self.next_comparison(content(self.own_prompt)))
        self.pref_count = tk.StringVar()
        self.pref_hint = tk.StringVar()
        label(actions, textvariable=self.pref_count, style="CardMuted.TLabel").pack(side="left", padx=12)
        self.hint_label = label(controls, textvariable=self.pref_hint, style="Card.TLabel", foreground=ACCENT, font=(FONT, 10, "bold"))
        self.hint_label.pack(anchor="w")
        advanced = Collapsible(controls, "More options: your own prompt, reply settings")
        advanced.pack(fill="x", pady=(8, 0))
        self.mark_advanced(advanced)
        label(advanced.body, "Type your own prompt, then press “Use my prompt below”:", "CardMuted.TLabel").pack(anchor="w")
        self.own_prompt = text_box(advanced.body, 2)
        self.own_prompt.pack(fill="x", pady=(4, 10))
        grid = ttk.Frame(advanced.body, style="Card.TFrame")
        grid.pack(fill="x")
        self.temp_a, self.temp_b, self.compare_tokens = tk.StringVar(value="0.7"), tk.StringVar(value="1.1"), tk.StringVar(value="120")
        setting_rows(grid, [
            ("Creativity of reply A", self.temp_a, "Sampling temperature: low = safe and repetitive, high = varied but riskier.", None, "temperature"),
            ("Creativity of reply B", self.temp_b, "Keep it different from A so you have something to compare.", None, "temperature"),
            ("Longest reply (tokens)", self.compare_tokens, "Replies are cut off after this many tokens (about one character each here).", None, "max_tokens")])
        label(page, "PROMPT", "Muted.TLabel", font=(FONT, 9, "bold")).pack(anchor="w")
        self.compare_prompt = text_box(page, 2, state="disabled")
        self.compare_prompt.pack(fill="x", pady=(4, 8))
        pair = ttk.Frame(page)
        pair.pack(fill="x")
        pair.columnconfigure((0, 1), weight=1, uniform="replies")
        self.reply_boxes = []
        for column, name in enumerate("AB"):
            label(pair, f"REPLY {name}   [press {column + 1}]", "Muted.TLabel", font=(FONT, 9, "bold")).grid(row=0, column=column, sticky="w", padx=(0 if column == 0 else 8, 0))
            box = text_box(pair, 5, state="disabled")
            box.grid(row=1, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 8 if column == 0 else 0), pady=(4, 0))
            self.reply_boxes.append(box)
        votes = ttk.Frame(page)
        votes.pack(fill="x", pady=(8, 0))
        button(votes, "A is better  [1]", lambda: self.vote("A"), True)
        button(votes, "B is better  [2]", lambda: self.vote("B"), True)
        button(votes, "Both wrong: I'll write it  [3]", self.write_answer)
        button(votes, "Skip  [4]", lambda: self.next_comparison(None))
        self.undo_button = button(votes, "Undo  [U]", self.undo_vote)
        button(votes, "Train on these →", lambda: self.align_tabs.select(TAB_IMPROVE))
        self.reference_text = tk.StringVar()
        label(page, textvariable=self.reference_text, style="Muted.TLabel", wraplength=980, justify="left").pack(anchor="w", pady=(8, 0))

    def _build_teacher(self, page):
        box = card(page, "Let a local Ollama model teach")
        box.pack(fill="x", pady=(0, 12))
        label(box, "Each round: your model writes several answers per prompt → the teacher grades them against the rules below → the best and worst become a preference pair "
                   "(if every answer is poor the teacher writes a better one, which must also score higher) → the model trains → the new version is kept only if "
                   "it beats the previous one in a head-to-head test.", "CardMuted.TLabel", wraplength=980, justify="left").pack(anchor="w", pady=(0, 10))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x")
        stored = self.align_settings
        self.ollama_url, self.ollama_model = tk.StringVar(value=stored.get("url", "")), tk.StringVar(value=stored.get("teacher_model", ""))
        self.judge_model = tk.StringVar(value=stored.get("judge_model") or SAME_AS_TEACHER)
        frame, _ = field(row, "Ollama address", self.ollama_url, width=26, help="ollama")
        frame.pack(side="left", padx=(0, 14))
        frame, self.ollama_box = field(row, "Teacher model (writes fixes)", self.ollama_model, values=[stored.get("teacher_model", "")], width=24, help="teacher")
        frame.pack(side="left", padx=(0, 14))
        frame, self.judge_box = field(row, "Judge model (optional, grades answers)", self.judge_model, values=[SAME_AS_TEACHER], width=26, help="judge")
        frame.pack(side="left", padx=(0, 14))
        refresh = ttk.Frame(row, style="Card.TFrame")
        refresh.pack(side="left", anchor="s")
        button(refresh, "Find models", self.refresh_ollama)
        saved = stored.get("auto", {})

        def number(key, default):
            return tk.StringVar(value=str(saved.get(key, default)))
        self.auto_vars = {"rounds": number("rounds", AUTO_DEFAULTS["rounds"]), "prompts_per_round": number("prompts_per_round", AUTO_DEFAULTS["prompts_per_round"]),
                          "candidates": number("candidates", AUTO_DEFAULTS["candidates"]), "dpo_steps": number("dpo_steps", AUTO_DEFAULTS["dpo_steps"]),
                          "eval_prompts": number("eval_prompts", AUTO_DEFAULTS["eval_prompts"]), "win_pct": number("win_pct", 55),
                          "gate_samples": number("gate_samples", AUTO_DEFAULTS["gate_samples"]), "rewrite_below": number("rewrite_below", AUTO_DEFAULTS["rewrite_below"]),
                          "min_margin": number("min_margin", AUTO_DEFAULTS["min_margin"]), "replay": number("replay", AUTO_DEFAULTS["replay"]),
                          "temp_low": number("temp_low", .5), "temp_high": number("temp_high", 1.2)}
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x", pady=(12, 0))
        for key, title, helper in [("rounds", "Rounds", "rounds"), ("prompts_per_round", "Prompts per round", "prompt_count"), ("candidates", "Answers per prompt (2–8)", "candidates"),
                                   ("dpo_steps", "Training steps per round", "steps"), ("eval_prompts", "Test prompts", "arena"), ("win_pct", "Must win at least (%)", "must_win")]:
            frame, _ = field(row, title, self.auto_vars[key], width=12, help=helper)
            frame.pack(side="left", padx=(0, 12))
        label(box, "“Must win at least” is the share of head-to-head comparisons (new vs current model, on test prompts the model never trained on) the new version has to win before it replaces the current one. "
                   "55% means a small but real edge; ties count as half a win.", "CardMuted.TLabel", wraplength=980, justify="left").pack(anchor="w", pady=(8, 0))

        advanced = Collapsible(box, "Advanced: teacher behaviour")
        advanced.pack(fill="x", pady=(10, 0))
        self.mark_advanced(advanced)
        self.both_orders = tk.BooleanVar(value=saved.get("both_orders", True))
        self.keep_improvements = tk.BooleanVar(value=saved.get("keep_improvements", True))
        self.extra_pairs = tk.BooleanVar(value=saved.get("extra_pairs", False))
        setting_rows(advanced.body, [
            ("Rewrite when best answer scores below", self.auto_vars["rewrite_below"], "If even the best of the model's answers scores under this (out of 10), the teacher writes a better one."),
            ("Smallest useful score gap", self.auto_vars["min_margin"], "Best and worst answers must differ by at least this many points, otherwise the prompt teaches nothing and is skipped."),
            ("Comparison samples per test prompt", self.auto_vars["gate_samples"], "More samples make the win-rate test less noisy, but slower."),
            ("Coolest answer temperature", self.auto_vars["temp_low"], "The N answers are spread between the cool and warm temperature so they differ."),
            ("Warmest answer temperature", self.auto_vars["temp_high"], "Higher = more random wording; too high produces gibberish."),
            ("Replayed old pairs", self.auto_vars["replay"], "Older pairs mixed into each round so the model does not forget earlier lessons."),
            ("Judge both orders", self.both_orders, "Grade A-vs-B and B-vs-A and average; cancels the judge's habit of favouring the first answer."),
            ("Also pair the best with every weaker answer", self.extra_pairs, "More training pairs per prompt, but they are less independent of each other."),
            ("Only keep a checkpoint that wins", self.keep_improvements, KEEP_NOTE)])

        rules = card(page, "Rules the teacher grades by (rubric)")
        rules.pack(fill="x", pady=(0, 12))
        label(rules, "Plain-language criteria. A bigger weight (1–10) means it matters more. Changing these changes what the model is steered toward, so write what you actually want.",
              "CardMuted.TLabel", wraplength=960, justify="left").pack(anchor="w", pady=(0, 8))
        self.rubric_tree = tree(rules, [("weight", "WEIGHT", 80), ("text", "CRITERION", 800)], 5)
        self.rubric_tree.column("weight", width=80, stretch=False, anchor="w")
        self.rubric_tree.heading("weight", anchor="w")
        self.rubric_tree.heading("text", anchor="w")
        form = ttk.Frame(rules, style="Card.TFrame")
        form.pack(fill="x", pady=(8, 0))
        self.rule_text, self.rule_weight = tk.StringVar(), tk.StringVar(value="2")
        ttk.Entry(form, textvariable=self.rule_text).pack(side="left", fill="x", expand=True, padx=(0, 8))
        label(form, "weight", "CardMuted.TLabel").pack(side="left", padx=(0, 4))
        ttk.Entry(form, textvariable=self.rule_weight, width=5).pack(side="left", padx=(0, 8))
        button(form, "Add rule", self.add_rule, True)
        button(form, "Set weight of selected", self.set_rule_weight)
        button(form, "Remove selected", self.remove_rule)
        button(form, "Reset to defaults", self.reset_rubric)
        self.refresh_rubric()

        actions = card(page)
        actions.pack(fill="x", pady=(0, 12))
        bar = ttk.Frame(actions, style="Card.TFrame")
        bar.pack(fill="x")
        button(bar, "Start automated improvement", self.start_auto_rlhf, True)
        button(bar, "Stop", self.stop_job)
        self.auto_status = tk.StringVar(value="Uses prompts from the working dataset; its own answers are shown to the teacher as hints.")
        label(bar, textvariable=self.auto_status, style="CardMuted.TLabel", wraplength=700).pack(side="left", padx=10)
        self.rounds_tree = tree(page, [("round", "ROUND", 60), ("win", "WIN RATE", 90), ("wtl", "WIN / TIE / LOSS", 120), ("pairs", "NEW PAIRS", 90),
                                       ("rewrites", "REWRITES", 90), ("skipped", "SKIPPED", 80), ("kept", "KEPT?", 130), ("checkpoint", "CHECKPOINT", 300)], 4)
        self.teacher_log = text_box(page, 7, state="disabled")
        self.teacher_log.pack(fill="x", pady=(10, 0))

    def _build_saved(self, page):
        box = card(page, "Every preference collected so far")
        box.pack(fill="x", pady=(0, 12))
        label(box, "Each row says “for this prompt, this reply was better than that one”. Fix mistakes here: edit the preferred reply, swap the two, or delete a pair. "
                   "Export to share or back up, import to merge a file (identical pairs are skipped).", "CardMuted.TLabel", wraplength=960, justify="left").pack(anchor="w", pady=(0, 8))
        bar = ttk.Frame(box, style="Card.TFrame")
        bar.pack(fill="x")
        self.pref_filter = tk.StringVar(value="All")
        label(bar, "Show", "CardMuted.TLabel").pack(side="left", padx=(0, 6))
        chooser = ttk.Combobox(bar, textvariable=self.pref_filter, values=("All", "Your votes", "Teacher"), state="readonly", width=12)
        chooser.pack(side="left", padx=(0, 10))
        chooser.bind("<<ComboboxSelected>>", lambda _e: self.refresh_pref_browser())
        self.pref_browser_count = tk.StringVar()
        label(bar, textvariable=self.pref_browser_count, style="CardMuted.TLabel").pack(side="left", padx=6)
        self.pref_tree = tree(page, [("source", "WHO", 130), ("kind", "TYPE", 100), ("prompt", "PROMPT", 420), ("chosen", "PREFERRED REPLY", 380)], 7)
        self.pref_tree.bind("<<TreeviewSelect>>", lambda _e: self.show_pref_detail())
        detail = card(page, "Selected pair")
        detail.pack(fill="x", pady=(12, 0))
        label(detail, "PROMPT", "CardMuted.TLabel", font=(FONT, 9, "bold")).pack(anchor="w")
        self.pref_prompt = text_box(detail, 3, state="disabled")
        self.pref_prompt.pack(fill="x", pady=(3, 8))
        label(detail, "PREFERRED REPLY (you can edit this)", "CardMuted.TLabel", font=(FONT, 9, "bold")).pack(anchor="w")
        self.pref_chosen = text_box(detail, 4)
        self.pref_chosen.pack(fill="x", pady=(3, 8))
        label(detail, "REJECTED REPLY", "CardMuted.TLabel", font=(FONT, 9, "bold")).pack(anchor="w")
        self.pref_rejected = text_box(detail, 4, state="disabled")
        self.pref_rejected.pack(fill="x", pady=(3, 6))
        self.pref_meta = tk.StringVar()
        label(detail, textvariable=self.pref_meta, style="CardMuted.TLabel", wraplength=940, justify="left").pack(anchor="w")
        actions = ttk.Frame(detail, style="Card.TFrame")
        actions.pack(fill="x", pady=(8, 0))
        button(actions, "Save edit", self.save_pref_edit, True)
        button(actions, "Swap preferred / rejected", self.flip_pref)
        button(actions, "Delete selected", self.delete_prefs)
        button(actions, "Export…", self.export_preferences)
        button(actions, "Import…", self.import_preferences)

    def _build_dpo(self, page):
        box = card(page, "Train the model on your preferences (DPO)")
        box.pack(fill="x", pady=(0, 12))
        label(box, "The model learns to make preferred replies more likely and rejected replies less likely, while staying close to the version you started from. "
                   "A little ordinary learning on the preferred replies is mixed in to keep small models stable. The result is saved as a new checkpoint.",
              "CardMuted.TLabel", wraplength=980, justify="left").pack(anchor="w", pady=(0, 10))
        self.pref_stats = tk.StringVar()
        label(box, textvariable=self.pref_stats, style="Card.TLabel").pack(anchor="w", pady=(0, 10))
        self.dpo_vars = {key: tk.StringVar(value=str(DPO_DEFAULTS[key])) for key in ("beta", "learning_rate", "steps", "batch_size", "sft_weight", "validation", "seed")}
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x")
        for key, title, helper in [("steps", "Training steps", "steps"), ("batch_size", "Pairs per step", "pairs_per_step"), ("learning_rate", "Learning speed", "learning_speed")]:
            frame, _ = field(row, title, self.dpo_vars[key], width=12, help=helper)
            frame.pack(side="left", padx=(0, 12))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x", pady=(10, 0))
        self.use_human, self.use_teacher = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Use my votes", variable=self.use_human).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(row, text="Use the teacher's pairs", variable=self.use_teacher).pack(side="left")
        advanced = Collapsible(box, "Advanced: how the model is trained")
        advanced.pack(fill="x", pady=(10, 0))
        self.mark_advanced(advanced)
        self.loss_type, self.length_normalize = tk.StringVar(value=DPO_DEFAULTS["loss_type"]), tk.BooleanVar(value=DPO_DEFAULTS["length_normalize"])
        setting_rows(advanced.body, [
            ("β (caution)", self.dpo_vars["beta"], "How far the model may drift from where it started. Higher = more cautious, smaller changes.", None, "beta"),
            ("Mix-in of ordinary learning", self.dpo_vars["sft_weight"], "Extra weight on simply copying preferred replies (SFT). Helps small models stay coherent; 0 turns it off.", None, "sft"),
            ("Loss type", self.loss_type, "dpo = standard; ipo = steadier, less likely to over-push when pairs are noisy.", LOSS_TYPES, "loss_type"),
            ("Judge by words, not totals", self.length_normalize, "Average over reply length so long replies are not favoured or punished just for being long (helps against rambling)."),
            ("Share kept for checking", self.dpo_vars["validation"], "Fraction of pairs held back to measure progress (used once you have 10+ pairs).", None, "validation"),
            ("Random seed", self.dpo_vars["seed"], "Same seed + same data = same result.", None, "seed")])
        actions = ttk.Frame(box, style="Card.TFrame")
        actions.pack(fill="x", pady=(12, 0))
        button(actions, "Train on preferences", self.start_dpo, True)
        button(actions, "Open result in playground", self.play_aligned)
        button(actions, "Compare with the original →", lambda: self.align_tabs.select(TAB_COMPARE))
        note = ttk.Frame(page)
        note.pack(anchor="w")
        label(note, "Training curves appear on the Training page. Reward accuracy is how often the model now ranks the preferred reply above the rejected one; watch it on held-out pairs.",
              "Muted.TLabel", wraplength=960, justify="left").pack(side="left")
        help_icon(note, "reward_accuracy").pack(side="left", padx=(4, 0))

    def _build_arena(self, page):
        box = card(page, "Compare two checkpoints")
        box.pack(fill="x", pady=(0, 12))
        label(box, "Both models answer the same test prompts. The judge picks the better reply each time (in both orders, so position does not matter). "
                   "A win rate well above 50% means A really is better; near 50% means no real difference. Uses the Ollama settings from “B · Auto teacher”.",
              "CardMuted.TLabel", wraplength=980, justify="left").pack(anchor="w", pady=(0, 10))
        self.arena_a, self.arena_b = tk.StringVar(), tk.StringVar()
        for name, variable, note in [("A (new / challenger)", self.arena_a, ""), ("B (old / baseline)", self.arena_b, "")]:
            row = ttk.Frame(box, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            label(row, name, "Card.TLabel", width=22).pack(side="left")
            ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=(0, 8))
            button(row, "Choose .npz", lambda v=variable: self.choose_checkpoint(v))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        self.arena_prompts, self.arena_samples = tk.StringVar(value="8"), tk.StringVar(value="2")
        for title, var, helper in [("Test prompts", self.arena_prompts, "arena"), ("Answers per prompt", self.arena_samples, "candidates")]:
            frame, _ = field(row, title, var, width=10, help=helper)
            frame.pack(side="left", padx=(0, 12))
        actions = ttk.Frame(box, style="Card.TFrame")
        actions.pack(fill="x", pady=(12, 0))
        button(actions, "Compare", self.start_arena, True)
        button(actions, "Stop", self.stop_job)
        button(actions, "Use current as A, original as B", self.arena_defaults)
        self.arena_summary = tk.StringVar(value="Pick two checkpoints and press Compare.")
        self.arena_label = label(page, textvariable=self.arena_summary, font=(FONT, 14, "bold"), wraplength=980, justify="left")
        self.arena_label.pack(anchor="w", pady=(2, 8))
        self.arena_tree = tree(page, [("outcome", "RESULT FOR A", 110), ("prompt", "PROMPT", 700), ("scores", "SCORES A / B", 120)], 5)
        self.arena_tree.bind("<<TreeviewSelect>>", lambda _e: self.show_arena_detail())
        pair = ttk.Frame(page)
        pair.pack(fill="x", pady=(10, 0))
        pair.columnconfigure((0, 1), weight=1, uniform="arena")
        self.arena_boxes = []
        for column, name in enumerate(("REPLY FROM A", "REPLY FROM B")):
            label(pair, name, "Muted.TLabel", font=(FONT, 9, "bold")).grid(row=0, column=column, sticky="w", padx=(0 if column == 0 else 8, 0))
            view = text_box(pair, 5, state="disabled")
            view.grid(row=1, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 8 if column == 0 else 0), pady=(4, 0))
            self.arena_boxes.append(view)
        self.arena_reason = tk.StringVar()
        label(page, textvariable=self.arena_reason, style="Muted.TLabel", wraplength=980, justify="left").pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------ shared helpers

    def show_page(self, name):
        super().show_page(name)
        if name == "align":
            if not self.align_checkpoint.get():
                self.use_latest_checkpoint()
            self.refresh_pref_stats()

    def use_latest_checkpoint(self):
        for record in self.workspace.runs():
            directory = Path(record["directory"])
            for filename in ("best.npz", "last.npz"):
                if (directory / filename).exists():
                    self.align_checkpoint.set(str(directory / filename))
                    return
        self.status_var.set("No checkpoints yet · train a model first.")

    def refresh_pref_stats(self):
        stats = self.preferences.stats()
        total = stats["total"]
        self.pref_count.set(f"{stats['human']:,} of your votes saved · {total:,} pairs in total")
        if total >= MIN_PAIRS_TO_TRAIN:
            self.pref_hint.set(f"Enough to start training ({total} pairs). Open “C · Improve model” when you are ready.")
            self.hint_label.configure(foreground=ACCENT)
        else:
            self.pref_hint.set(f"Aim for at least {MIN_PAIRS_TO_TRAIN} pairs before training: {MIN_PAIRS_TO_TRAIN - total} more to go.")
            self.hint_label.configure(foreground=WARN)
        self.pref_stats.set(f"Collected: {total:,} pairs · {stats['human']:,} from you · {stats['teacher']:,} from the teacher · {stats['rewrites']:,} with a written-out answer"
                            + ("" if total >= MIN_PAIRS_TO_TRAIN else f"  (aim for {MIN_PAIRS_TO_TRAIN}+)"))
        self.undo_button.configure(state="normal" if self.vote_log else "disabled")
        self.refresh_pref_browser()

    def checkpoint_path(self):
        path = self.align_checkpoint.get().strip()
        if not path or not Path(path).exists():
            raise ValueError("Choose an existing checkpoint (.npz) to align.")
        return path

    def dataset_prompts(self, limit=2000):
        dataset = self.selected_dataset()
        if not dataset:
            raise ValueError("Choose a working dataset with user → assistant conversations.")
        prompts = prompts_from(self.workspace.sample(dataset, limit=limit, seed=random.randrange(1 << 30)))
        if not prompts:
            raise ValueError("The working dataset has no user → assistant conversations to use as prompts.")
        return prompts

    def align_log(self, text):
        self.teacher_log.configure(state="normal")
        self.teacher_log.insert("end", text + "\n")
        self.teacher_log.see("end")
        self.teacher_log.configure(state="disabled")

    # ------------------------------------------------------------------ step 1: readiness

    def evaluate_readiness(self):
        if self.busy():
            return
        try:
            path, size = self.checkpoint_path(), int(self.eval_size.get())
            float(self.target_error.get())
            examples = self.workspace.sample(self.selected_dataset(), limit=size, seed=1)
            if not examples:
                raise ValueError("Choose a working dataset to check against.")
        except ValueError as exc:
            messagebox.showerror("Cannot check the model", str(exc), parent=self)
            return
        device = self.inference_device.get()
        val_loss = None
        metrics = Path(path).parent / "metrics.jsonl"
        if metrics.exists():
            rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
            val_loss = next((r["val_loss"] for r in reversed(rows) if r.get("val_loss") is not None), None)

        def done(report):
            self.ready_vars["token_error"].set(f"{report['token_error'] * 100:.1f}%")
            self.ready_vars["perplexity"].set(f"{report['perplexity']:.2f}")
            self.ready_vars["loss"].set(f"{report['loss']:.3f}")
            self.ready_vars["val_loss"].set("—" if val_loss is None else f"{val_loss:.3f}")
            target = float(self.target_error.get()) / 100
            ready = report["token_error"] <= target
            unknown = f" · {report['unknown_tokens']:,} characters unseen by the tokenizer" if report["unknown_tokens"] else ""
            self.ready_verdict.set(("READY TO POLISH" if ready else "TRAIN MORE FIRST") +
                                   f"  ·  wrong {report['token_error'] * 100:.1f}% of the time vs {target * 100:.0f}% goal, over {report['tokens']:,} words{unknown}")
            self.ready_label.configure(foreground=ACCENT if ready else WARN)
        self.start_job("evaluation", lambda: evaluate_checkpoint(path, examples, device, size, self.stop), done)

    # ------------------------------------------------------------------ step 2a: human feedback

    def ensure_align_model(self, path, device):
        # Runs on a worker thread: never touch Tk variables here.
        if self.align_model is None or self.align_model_path != (path, device):
            self.align_model = LocalModel(path, seed=random.randrange(1 << 30), device=device)
            self.align_model_path = (path, device)
        return self.align_model

    def show_comparison(self):
        comparison = self.comparison
        if not comparison:
            return
        set_text(self.compare_prompt, prompt_text(comparison["prompt"]["messages"]))
        for box, reply in zip(self.reply_boxes, comparison["replies"]):
            set_text(box, reply or "(empty reply)")
        reference = comparison["prompt"].get("reference")
        self.reference_text.set(f"Dataset reference answer: {reference[:400]}" if reference else "")

    def next_comparison(self, own_prompt):
        if self.busy():
            return
        try:
            path = self.checkpoint_path()
            temps = float(self.temp_a.get()), float(self.temp_b.get())
            tokens = int(self.compare_tokens.get())
            device = self.inference_device.get()
            if own_prompt:
                prompt = {"messages": [{"role": "user", "content": own_prompt}], "reference": None}
            else:
                if not self.align_prompts:
                    self.align_prompts = self.dataset_prompts()
                prompt = self.align_prompts.pop()
        except ValueError as exc:
            messagebox.showerror("Cannot compare", str(exc), parent=self)
            return
        self.comparison = None
        for box in self.reply_boxes:
            set_text(box, "Generating…")
        set_text(self.compare_prompt, prompt_text(prompt["messages"]))
        self.reference_text.set(f"Dataset reference answer: {prompt['reference'][:400]}" if prompt.get("reference") else "")

        def work():
            model = self.ensure_align_model(path, device)
            first = model.generate(prompt["messages"], tokens, temps[0], 40, self.stop)
            second = model.generate(prompt["messages"], tokens, temps[1], 40, self.stop)
            if first.strip() == second.strip():
                second = model.generate(prompt["messages"], tokens, max(1.2, temps[1] + .3), 0, self.stop)
            return first, second

        def done(replies):
            self.comparison = {"prompt": prompt, "replies": replies, "checkpoint": path}
            self.show_comparison()
        self.start_job("comparison", work, done)

    def vote(self, winner):
        if not self.comparison:
            return
        a, b = self.comparison["replies"]
        chosen, rejected = (a, b) if winner == "A" else (b, a)
        try:
            record = self.preferences.add(self.comparison["prompt"]["messages"], chosen, rejected, "human", kind="comparison",
                                          checkpoint=self.comparison.get("checkpoint", ""), reference=self.comparison["prompt"].get("reference"),
                                          source_id=self.comparison["prompt"].get("source_id"))
            self.vote_log.append({"ids": [record["id"]], "comparison": self.comparison})
        except ValueError as exc:
            self.status_var.set(str(exc) + " · skipped")
        self.refresh_pref_stats()
        self.next_comparison(None)

    def undo_vote(self):
        """Remove the most recent vote and show that prompt again."""
        if self.job_kind is not None:
            self.status_var.set("Wait for the replies to finish loading, then undo.")
            return
        if not self.vote_log:
            self.status_var.set("Nothing to undo.")
            return
        entry = self.vote_log.pop()
        removed = self.preferences.delete(entry["ids"])
        self.comparison = entry["comparison"]
        self.show_comparison()
        self.refresh_pref_stats()
        self.status_var.set(f"Undid your last vote ({removed} pair removed) · vote again on this prompt.")

    def write_answer(self):
        if not self.comparison:
            return
        comparison = self.comparison
        dialog = tk.Toplevel(self)
        dialog.title("Write the correct answer")
        dialog.configure(bg=BG)
        dialog.geometry("720x420")
        dialog.transient(self)
        outer = ttk.Frame(dialog, padding=18)
        outer.pack(fill="both", expand=True)
        label(outer, "The answer you write is preferred over both model replies.", "Muted.TLabel").pack(anchor="w", pady=(0, 8))
        box = text_box(outer, 12)
        box.pack(fill="both", expand=True)
        set_text(box, comparison["prompt"].get("reference") or "")

        def save():
            answer = content(box)
            if not answer:
                return
            ids = []
            for reply in comparison["replies"]:
                if reply.strip() != answer.strip():
                    record = self.preferences.add(comparison["prompt"]["messages"], answer, reply, "human", kind="edit",
                                                  checkpoint=comparison.get("checkpoint", ""), reference=comparison["prompt"].get("reference"),
                                                  source_id=comparison["prompt"].get("source_id"))
                    ids.append(record["id"])
            if ids:
                self.vote_log.append({"ids": ids, "comparison": comparison})
            dialog.destroy()
            self.refresh_pref_stats()
            self.next_comparison(None)
        row = ttk.Frame(outer)
        row.pack(fill="x", pady=(10, 0))
        button(row, "Save preference", save, True)
        button(row, "Cancel", dialog.destroy)
        dialog.grab_set()
        box.focus_set()

    def align_hotkey(self, event):
        if getattr(self, "active_page", None) != "align" or self.align_tabs.index(self.align_tabs.select()) != TAB_RATE:
            return
        if event.widget.winfo_class() in {"Text", "TEntry", "Entry", "TCombobox"} or event.widget.winfo_toplevel() is not self:
            return
        action = {"1": lambda: self.vote("A"), "2": lambda: self.vote("B"), "3": self.write_answer, "4": lambda: self.next_comparison(None),
                  "u": self.undo_vote, "U": self.undo_vote}.get(event.char)
        if action and (self.job_kind is None or event.char in "uU"):
            action()

    # ------------------------------------------------------------------ step 2c: preference browser

    def pref_rows(self):
        mode = self.pref_filter.get()
        sources = {"Your votes": {"human"}, "Teacher": {"teacher"}}.get(mode)
        return self.preferences.all(sources)

    def refresh_pref_browser(self):
        if not hasattr(self, "pref_tree"):
            return
        selected = set(self.pref_tree.selection())
        self.pref_tree.delete(*self.pref_tree.get_children())
        rows = self.pref_rows()
        for row in reversed(rows):  # newest first
            who = "you" if row.get("source") == "human" else str(row.get("source", "?")).replace("teacher:", "teacher · ")
            prompt = next((m["content"] for m in reversed(row["messages"]) if m["role"] == "user"), row["messages"][-1]["content"])
            self.pref_tree.insert("", "end", iid=row["id"], values=(who, row.get("kind", "comparison"), short(prompt, 60), short(row["chosen"], 54)))
        keep = [i for i in selected if self.pref_tree.exists(i)]
        if keep:
            self.pref_tree.selection_set(keep)
        self.pref_browser_count.set(f"{len(rows):,} pairs shown")
        if not keep:
            self.show_pref_detail()

    def selected_pref_ids(self):
        return list(self.pref_tree.selection())

    def show_pref_detail(self):
        ids = self.selected_pref_ids()
        row = self.preferences.get(ids[0]) if len(ids) == 1 else None
        for box in (self.pref_prompt, self.pref_chosen, self.pref_rejected):
            set_text(box, "")
        if row is None:
            self.pref_meta.set("Select one row to read or edit it." if not ids else f"{len(ids)} rows selected · use Delete selected to remove them.")
            return
        set_text(self.pref_prompt, prompt_text(row["messages"]))
        set_text(self.pref_chosen, row["chosen"])
        set_text(self.pref_rejected, row["rejected"])
        bits = [f"{row.get('source', '?')}", f"{row.get('kind', 'comparison')}", row.get("createdAt", "")]
        if row.get("scores"):
            bits.append("teacher scores " + " vs ".join(f"{s:.1f}" for s in row["scores"]))
        if row.get("reason"):
            bits.append(f"“{short(row['reason'], 140)}”")
        if row.get("flipped"):
            bits.append("swapped by you")
        self.pref_meta.set(" · ".join(b for b in bits if b))

    def save_pref_edit(self):
        ids = self.selected_pref_ids()
        if len(ids) != 1:
            self.status_var.set("Select exactly one pair to edit.")
            return
        try:
            self.preferences.update(ids[0], chosen=content(self.pref_chosen), kind="edit")
        except ValueError as exc:
            messagebox.showerror("Cannot save", str(exc), parent=self)
            return
        self.refresh_pref_stats()
        self.status_var.set("Saved your edit.")

    def flip_pref(self):
        ids = self.selected_pref_ids()
        if len(ids) != 1:
            self.status_var.set("Select exactly one pair to swap.")
            return
        self.preferences.flip(ids[0])
        self.refresh_pref_stats()
        self.pref_tree.selection_set(ids[0])
        self.status_var.set("Swapped: the rejected reply is now the preferred one.")

    def delete_prefs(self, confirm=True):
        ids = self.selected_pref_ids()
        if not ids:
            self.status_var.set("Select the pairs to delete first.")
            return
        if confirm and not messagebox.askyesno("Delete preferences", f"Delete {len(ids)} preference pair(s)? This cannot be undone.", parent=self):
            return
        removed = self.preferences.delete(ids)
        self.vote_log = [e for e in self.vote_log if not set(e["ids"]) & set(ids)]
        self.refresh_pref_stats()
        self.status_var.set(f"Deleted {removed} pair(s).")

    def export_preferences(self, path=None):
        ids = self.selected_pref_ids() or None
        path = path or filedialog.asksaveasfilename(parent=self, title="Export preferences", defaultextension=".jsonl",
                                                    initialfile="preferences-export.jsonl", filetypes=[("JSON lines", "*.jsonl")])
        if path:
            count = self.preferences.export(path, ids)
            self.status_var.set(f"Exported {count} pair(s) to {path}")

    def import_preferences(self, path=None):
        path = path or filedialog.askopenfilename(parent=self, title="Import preferences", filetypes=[("JSON lines", "*.jsonl"), ("All files", "*.*")])
        if path:
            try:
                result = self.preferences.import_file(path)
            except (OSError, UnicodeDecodeError) as exc:
                messagebox.showerror("Cannot import", str(exc), parent=self)
                return
            self.refresh_pref_stats()
            self.status_var.set(f"Imported {result['added']} new · {result['duplicates']} duplicates skipped · {result['invalid']} unreadable")

    # ------------------------------------------------------------------ step 2b: teacher loop

    def refresh_rubric(self):
        self.rubric_tree.delete(*self.rubric_tree.get_children())
        for index, item in enumerate(self.rubric):
            self.rubric_tree.insert("", "end", iid=str(index), values=(item["weight"], item["text"]))

    def persist_rubric(self):
        self.rubric = normalize_rubric(self.rubric) or normalize_rubric(DEFAULT_RUBRIC)
        self.align_settings["rubric"] = self.rubric
        save_settings(self.workspace.root, self.align_settings)
        self.refresh_rubric()

    def add_rule(self):
        text = self.rule_text.get().strip()
        if not text:
            return
        try:
            weight = float(self.rule_weight.get())
        except ValueError:
            weight = 1
        self.rubric.append({"text": text, "weight": weight})
        self.rule_text.set("")
        self.persist_rubric()

    def set_rule_weight(self):
        picked = [int(i) for i in self.rubric_tree.selection()]
        try:
            weight = float(self.rule_weight.get())
        except ValueError:
            self.status_var.set("Type a weight (1–10) first.")
            return
        for index in picked:
            self.rubric[index]["weight"] = weight
        self.persist_rubric()

    def remove_rule(self):
        picked = {int(i) for i in self.rubric_tree.selection()}
        self.rubric = [r for i, r in enumerate(self.rubric) if i not in picked]
        self.persist_rubric()

    def reset_rubric(self):
        self.rubric = normalize_rubric(DEFAULT_RUBRIC)
        self.persist_rubric()

    def refresh_ollama(self):
        try:
            models = OllamaTeacher(self.ollama_url.get()).models()
        except RuntimeError as exc:
            messagebox.showerror("Ollama", str(exc), parent=self)
            return
        self.ollama_box.configure(values=models)
        self.judge_box.configure(values=[SAME_AS_TEACHER] + models)
        if models and self.ollama_model.get() not in models:
            preferred = next((m for m in models if m.startswith(("qwen3", "llama3", "gemma"))), models[0])
            self.ollama_model.set(preferred)
        self.auto_status.set(f"{len(models)} Ollama model(s) found.")

    def make_teacher(self):
        judge = self.judge_model.get().strip()
        return OllamaTeacher(self.ollama_url.get(), self.ollama_model.get(), rubric=self.rubric,
                             judge_model=None if judge in {"", SAME_AS_TEACHER} else judge)

    def dpo_config(self):
        cfg = {key: float(var.get()) for key, var in self.dpo_vars.items()}
        cfg["steps"], cfg["batch_size"], cfg["seed"] = int(cfg["steps"]), int(cfg["batch_size"]), int(cfg["seed"])
        cfg["device"] = self.inference_device.get()
        cfg["loss_type"], cfg["length_normalize"] = self.loss_type.get(), self.length_normalize.get()
        return cfg

    def auto_config(self):
        v = self.auto_vars
        win = float(v["win_pct"].get())
        if not 50 <= win <= 100:
            raise ValueError("“Must win at least” is a percentage between 50 and 100.")
        low, high = float(v["temp_low"].get()), float(v["temp_high"].get())
        if not 0 < low <= high <= 3:
            raise ValueError("Temperatures must satisfy 0 < coolest ≤ warmest ≤ 3.")
        candidates = int(v["candidates"].get())
        if not 2 <= candidates <= 8:
            raise ValueError("Use between 2 and 8 answers per prompt.")
        cfg = {"rounds": int(v["rounds"].get()), "prompts_per_round": int(v["prompts_per_round"].get()), "dpo_steps": int(v["dpo_steps"].get()),
               "eval_prompts": int(v["eval_prompts"].get()), "candidates": candidates, "gate_samples": int(v["gate_samples"].get()),
               "rewrite_below": float(v["rewrite_below"].get()), "min_margin": float(v["min_margin"].get()), "replay": int(v["replay"].get()),
               "win_margin": win / 100 - .5, "temperatures": [low, high], "both_orders": self.both_orders.get(),
               "keep_only_improvements": self.keep_improvements.get(), "extra_pairs": self.extra_pairs.get()}
        if min(cfg["rounds"], cfg["prompts_per_round"], cfg["dpo_steps"], cfg["eval_prompts"], cfg["gate_samples"]) < 1:
            raise ValueError("Rounds, prompts, steps, test prompts and samples must all be at least 1.")
        return cfg

    def save_align_settings(self):
        self.align_settings.update(url=self.ollama_url.get(), teacher_model=self.ollama_model.get(),
                                   judge_model="" if self.judge_model.get() == SAME_AS_TEACHER else self.judge_model.get(), rubric=self.rubric,
                                   auto={**{k: var.get() for k, var in self.auto_vars.items()}, "both_orders": self.both_orders.get(),
                                         "keep_improvements": self.keep_improvements.get(), "extra_pairs": self.extra_pairs.get()})
        try:
            save_settings(self.workspace.root, self.align_settings)
        except OSError:
            pass

    def start_auto_rlhf(self):
        if self.busy():
            return
        try:
            path = self.checkpoint_path()
            if not self.ollama_model.get():
                self.refresh_ollama()
            if not self.ollama_model.get():
                raise ValueError("Choose an Ollama teacher model (press “Find models”).")
            cfg = self.auto_config()
            dpo = self.dpo_config()
            prompts = self.dataset_prompts()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Cannot start", str(exc), parent=self)
            return
        self.save_align_settings()
        teacher = self.make_teacher()
        self.rounds_tree.delete(*self.rounds_tree.get_children())
        set_text(self.teacher_log, "")
        self.history = []
        self.draw_loss()
        original = path

        def emit(event):
            self.events.put(("rlhf", event) if event["type"].startswith("rlhf") else ("training", event))

        def done(result):
            self.align_checkpoint.set(result["checkpoint"])
            self.align_model = None
            self.arena_a.set(result["checkpoint"])
            self.arena_b.set(original)
            self.refresh_runs()
            self.refresh_pref_stats()
            kept = sum(1 for r in result["rounds"] if r["accepted"])
            rate = "" if result["win_rate"] is None else f" · latest win rate {result['win_rate'] * 100:.0f}%"
            skipped = ", ".join(f"{k} ×{v}" for k, v in result["skipped"].items())
            text = (f"Finished · {result['pairs']} new pairs · {kept}/{len(result['rounds'])} rounds kept{rate}"
                    + (f" · skipped prompts: {skipped}" if skipped else "")
                    + (" · the current checkpoint was updated." if result["checkpoint"] != original else " · no round beat the current model, so nothing changed."))
            if result.get("aborted"):
                text = result["aborted"] + " " + text
            self.auto_status.set(text)
            self.align_log(text)
        self.start_job("automated improvement", lambda: auto_rlhf(path, prompts, teacher, self.preferences, self.workspace.run_dir, cfg, dpo, emit,
                                                                  self.stop, self.pause, teacher.judge_teacher()), done)

    def rlhf_event(self, event):
        kind = event["type"]
        if kind == "rlhf_log":
            self.align_log(event["text"])
        elif kind == "rlhf_progress":
            reasons = ", ".join(f"{k} ×{v}" for k, v in event.get("skip_reasons", {}).items())
            self.auto_status.set(f"Round {event['round']} · prompt {event['done']}/{event['total']} · {event['pairs']} pairs made"
                                 + (f" · skipped {event['skipped']} ({reasons})" if event["skipped"] else ""))
            self.task_progress.configure(value=event["done"] / max(1, event["total"]) * 100)
        elif kind == "rlhf_gate":
            wtl = f" · wins {event['wins']} / ties {event['ties']} / losses {event['losses']}" if "wins" in event else ""
            self.auto_status.set(f"Head-to-head test {event['done']}/{event['total']}{wtl}")
            if self.job_kind == "checkpoint comparison":
                self.arena_summary.set(f"Comparing… {event['done']}/{event['total']}{wtl}")
            self.task_progress.configure(value=event["done"] / max(1, event["total"]) * 100)
        elif kind == "rlhf_round":
            rate = event.get("win_rate")
            first = event["round"] == 0
            self.rounds_tree.insert("", "end", values=(
                "start" if first else event["round"], "—" if rate is None else f"{rate * 100:.0f}%",
                "—" if rate is None else f"{event.get('wins', 0)} / {event.get('ties', 0)} / {event.get('losses', 0)}",
                "—" if first else event["pairs"], "—" if first else event["rewrites"], "—" if first else sum(event.get("skipped", {}).values()),
                "starting point" if first else "yes" if event["accepted"] else "no, rolled back", Path(event["checkpoint"]).parent.name + "/" + Path(event["checkpoint"]).name))
            self.refresh_pref_stats()

    # ------------------------------------------------------------------ step 3: DPO

    def start_dpo(self):
        if self.busy():
            return
        try:
            path = self.checkpoint_path()
            sources = {name for name, var in (("human", self.use_human), ("teacher", self.use_teacher)) if var.get()}
            pairs = self.preferences.all(sources)
            if len(pairs) < 2:
                raise ValueError("Collect at least two preferences first (Rate replies, or Auto teacher).")
            cfg = self.dpo_config()
        except ValueError as exc:
            messagebox.showerror("Cannot start training", str(exc), parent=self)
            return
        if len(pairs) < MIN_PAIRS_TO_TRAIN and not messagebox.askyesno(
                "Only a few preferences", f"You have {len(pairs)} pairs; about {MIN_PAIRS_TO_TRAIN} or more work much better. Train anyway?", parent=self):
            return
        directory = self.workspace.run_dir / uid("dpo")
        self.history = []
        self.draw_loss()
        self.progress.configure(value=0, maximum=cfg["steps"])
        set_text(self.run_log, f"Training on {len(pairs):,} preference pairs ({cfg['loss_type'].upper()}"
                               f"{', length-normalized' if cfg['length_normalize'] else ''})\n")
        self.show_page("training")

        def done(result):
            self.align_checkpoint.set(result["checkpoint"])
            self.align_model = None
            self.arena_a.set(result["checkpoint"])
            self.arena_b.set(path)
            self.refresh_runs()
            self.auto_status.set("Training finished. Open “D · Compare” to check the new model against the original.")
            self.show_page("align")
            self.align_tabs.select(TAB_COMPARE)
        self.start_job("post-training", lambda: dpo_train(path, pairs, directory, cfg, lambda e: self.events.put(("training", e)), self.stop, self.pause), done)

    def play_aligned(self):
        try:
            self.chat_checkpoint.set(self.checkpoint_path())
        except ValueError as exc:
            messagebox.showerror("No checkpoint", str(exc), parent=self)
            return
        self.show_page("chat")
        self.load_chat_model()

    # ------------------------------------------------------------------ step 4: arena

    def arena_defaults(self):
        self.arena_a.set(self.align_checkpoint.get())
        runs = self.workspace.runs()
        base = ""
        for record in runs:  # the run this one was trained from, if recorded
            if str(Path(record["directory"]) / "last.npz") == self.arena_a.get() and record.get("base_checkpoint"):
                base = record["base_checkpoint"]
                break
        if base:
            self.arena_b.set(base)
        elif not self.arena_b.get():
            self.status_var.set("Choose the older checkpoint for B.")

    def start_arena(self):
        if self.busy():
            return
        try:
            a, b = self.arena_a.get().strip(), self.arena_b.get().strip()
            if not (a and b and Path(a).exists() and Path(b).exists()):
                raise ValueError("Choose two existing checkpoints (.npz) to compare.")
            if not self.ollama_model.get():
                self.refresh_ollama()
            if not self.ollama_model.get():
                raise ValueError("Choose an Ollama model on the “B · Auto teacher” tab: it acts as the judge.")
            count, samples = int(self.arena_prompts.get()), int(self.arena_samples.get())
            if count < 1 or samples < 1:
                raise ValueError("Use at least 1 test prompt and 1 answer per prompt.")
            pool = self.dataset_prompts()
            random.shuffle(pool)
            prompts = pool[:count]
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Cannot compare", str(exc), parent=self)
            return
        self.save_align_settings()
        judge = self.make_teacher().judge_teacher()
        device, both = self.inference_device.get(), self.both_orders.get()
        set_text(self.arena_boxes[0], "")
        set_text(self.arena_boxes[1], "")
        self.arena_tree.delete(*self.arena_tree.get_children())
        self.arena_summary.set("Comparing…")
        self.arena_label.configure(foreground=TEXT)
        emit = lambda event: self.events.put(("rlhf", event))

        def done(result):
            self.show_arena_result(result)
        self.start_job("checkpoint comparison", lambda: compare_checkpoints(a, b, prompts, judge, samples, both_orders=both, device=device,
                                                                           stop=self.stop, emit=emit), done)

    def show_arena_result(self, result):
        n = result["n"]
        self.arena_examples = result["examples"]
        self.arena_tree.delete(*self.arena_tree.get_children())
        for index, example in enumerate(self.arena_examples):
            scores = "—" if not example["scores"] else f"{example['scores'][0]:.1f} / {example['scores'][1]:.1f}"
            outcome = {"win": "A better", "loss": "B better", "tie": "tie"}[example["outcome"]]
            self.arena_tree.insert("", "end", iid=str(index), values=(outcome, short(example["prompt"].replace("\n", " "), 100), scores))
        if not n:
            self.arena_summary.set("Nothing could be judged" + (f" ({result['errors']} judge errors)." if result["errors"] else "."))
            self.arena_label.configure(foreground=WARN)
            return
        rate = result["win_rate"]
        verdict = "A is clearly better" if rate >= .6 else "A is slightly better" if rate >= .53 else "about the same" if rate > .47 else "A is worse"
        self.arena_summary.set(f"{verdict}: A wins {result['wins']}, ties {result['ties']}, B wins {result['losses']}  ·  A's win rate {rate * 100:.0f}% over {n} comparisons"
                               + (f" ({result['errors']} could not be judged)" if result["errors"] else ""))
        self.arena_label.configure(foreground=ACCENT if rate >= .53 else WARN if rate <= .47 else TEXT)

    def show_arena_detail(self):
        picked = self.arena_tree.selection()
        if not picked:
            return
        example = self.arena_examples[int(picked[0])]
        set_text(self.arena_boxes[0], example["a"] or "(empty reply)")
        set_text(self.arena_boxes[1], example["b"] or "(empty reply)")
        self.arena_reason.set(f"Prompt: {short(example['prompt'].replace(chr(10), ' '), 160)}" + (f"  ·  Judge: {short(example['reason'], 160)}" if example.get("reason") else ""))

    # ------------------------------------------------------------------ plumbing

    def toggle_pause(self):
        # Post-training jobs honor pause the same way pretraining does.
        if self.job_kind in {"post-training", "automated improvement"}:
            kind, self.job_kind = self.job_kind, "training"
            super().toggle_pause()
            self.job_kind = kind
        else:
            super().toggle_pause()

    def poll(self):
        deferred = []
        try:
            for _ in range(200):
                kind, value = self.events.get_nowait()
                if kind == "rlhf":
                    self.rlhf_event(value)
                else:
                    deferred.append((kind, value))
        except queue.Empty:
            pass
        for item in deferred:
            self.events.put(item)
        super().poll()

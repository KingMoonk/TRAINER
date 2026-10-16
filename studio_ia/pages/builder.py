"""Build your model: the picture is the editor. Draft edits live on this page until saved."""
from __future__ import annotations

import copy
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox
import uuid
from pathlib import Path

import studio_registry as registry
from studio_compute import MAX_EPOCHS, TOKENS_PER_PARAM
from studio_glossary import glossary_text
from studio_spec import default_spec, validate_spec
from studio_validate import errors
from studio_warmstart import NOTES
from studio_ia import threads
from studio_ia.copy import BUILDER as T
from studio_ia.design import (HEADS, WIDTHS, WORDS, WORD_TERMS, block_parts,
                              blocks_change_pair, change_chips, choose, compute_lines, design_expectation,
                              expectations, fields_of, fits, gpt2_words_ready, gpu_kind, knob_applies, lr_text,
                              module_name, module_tip, new_issues, own_slots, part_options, part_tip, preset_cost,
                              preset_specs, preset_term, read_spec, recipe_text, resize, say, sizes, warm_kind,
                              warm_outlook, write_spec)
from studio_ia.design import (
    GUESSES as GUESSES,
    SAME as SAME,
    _positive as _positive,
    train_memory as train_memory,
    prediction_questions as prediction_questions,
    direction as direction,
    scored as scored,
    trained as trained,
    results as results,
    seconds_text as seconds_text,
    ratio_text as ratio_text,
    outcome as outcome,
    result_sentence as result_sentence,
    guess_check as guess_check,
    page_said as page_said,
    check_text as check_text,
    split_steps as split_steps,
    split_plan as split_plan,
    split_jobs as split_jobs,
    notebook as notebook,
    guesses_from as guesses_from,
    notebook_lines as notebook_lines,
    notebook_line as notebook_line,
    MISSING as MISSING,
    pieces_for as pieces_for,
    pill_text as pill_text,
    catchup_job as catchup_job,
)
from studio_ia.pages.base import Page
from studio_ia.services import if_added, plain_error, queued_outlook
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, Chip, GuessChips, Kicker, Progress, Segmented, StateTray, _activate, button, info_icon, tip


def bar(x, y):
    m = max(x, y)
    return round(x / m * 100), round(y / m * 100)


def preset_draft(spec):
    """(draft choices, anchor) that load `spec` into the Builder. The draft is written onto the preset itself, so what the picture
    has no control for (width, reading window, the training recipe) comes with it, and the old version stays the comparison."""
    s = read_spec(spec)
    return dict(s), (validate_spec(spec), s)


PROMPT = "The largest planet in our solar system is"
LINKS = (("trace", "trace"), ("head_zoom", "inside_head"), ("value_map", "value_map"), ("loss", "loss"),
         ("head_lab", "head_lab"), ("inside3d", "3d"), ("compare", "compare_runs"), ("quicktest", "quick_test"))
TRAINING_TEXT = "Chat switches to {new} only when it beats version {n} on held-back text, or when you pick it."
NO_TEXT = "There is no plain text to train on yet. Add some in Your data, then queue it."
NO_PARENT = "Version {n} has no trained weights to start from, so the next version trains from scratch."
NO_CURVE = "No error curve yet: version {n} has not trained on this computer. It appears here once it has."


# ---------------------------------------------------------------------------------------------- guess, train, check (#83)


# ---------------------------------------------------------------------------------------------- one change at a time (#83)


# ---------------------------------------------------------------------------------------------- lab notebook (#83)


def chart_args(old, new, refs):
    """BpcPlot keywords: the current version's curve bold, the new version's faint and continuing from the old one's end.
    `old`/`new` are run_curve dicts (or None); refs = references() dict. Missing numbers stay missing."""
    old, new = old or {}, new or {}
    base_x = old.get("total") or (old["steps"][-1][0] if old.get("steps") else 0)
    shift = lambda pts: [(x + base_x, y) for x, y in pts or []]  # noqa: E731
    named = (("uniform guess", "uniform_bpc"), ("counting word frequencies", "unigram_bpc"), ("best so far", "best_so_far_bpc"))
    return dict(smooth=list(old.get("smooth") or []), heldout=list(old.get("heldout") or []), steps=shift(new.get("steps")),
                refs=[(name, refs[key]) for name, key in named if refs and refs.get(key) is not None],
                total=base_x + (new.get("total") or 0) or None,
                band=refs.get("noise_band") if refs else None)


def version_name(n):
    """'version 3', or 'no version yet' for 0: the number before the first design is saved."""
    return f"version {n}" if n else "no version yet"


def curve_note(n, phase):
    """The one line shown in place of the chart while there is no curve to draw."""
    if phase == "training":
        return T["no_curve_starting"]
    if n:
        return NO_CURVE.format(n=n)
    return T["no_curve_first"] if phase == "edit" else T["no_curve_waiting"]


def answer_rows(n, trained, phase):
    """[(version number, why it cannot answer or "")] for the 'when answering' card; [] while neither version has trained.
    `trained` = the numbers among n and n + 1 that have trained weights. Version 0 is not a version, so it never gets a row."""
    if not trained:
        return []
    return [(k, "" if k in trained else T["compare_later"] if phase == "edit" and k == n + 1 else T["not_trained"])
            for k in (n, n + 1) if k]


def text_sets(summary):
    """Ids of the plain-text datasets from services.data('text') (a dict or a DataSummary)."""
    sets = summary.get("sets", []) if isinstance(summary, dict) else getattr(summary, "sets", [])
    return [s["id"] for s in sets]


class BuilderPage(Page):
    title, lead = T["h1"], T["lead"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.default = default_spec()
        self.phase, self.base, self.saved, self.job_id = "edit", None, None, None
        self.cur = read_spec(self.default)
        self.draft = dict(self.cur)
        self.anchor = None  # (spec, choices) when a design from another page was reopened here
        self._staged, self.dismissed = None, None
        self.tray = self.chips = self.costs = self.expect = None
        self.pieces_box = None
        self.note = None
        self.chip_texts, self.cost_say, self.expect_texts, self.actions = [], {}, [], {}
        self.guide_text = self.guide_details = self.hover_text = ""
        self.old_curve = self.new_curve = self.refs = None
        self.lists = None  # services.word_lists(), read on the slow lane
        self.attr_cache, self.attr_fresh, self.attr_busy = {}, set(), set()  # run facts and Compare runs per pair of versions
        self.painting_expect = False
        self.plot_args, self.plot = {}, None
        self._chart_token = None
        self.answers, self.answer_q, self.answer_stop, self.answer_labels, self.workers = {}, queue.Queue(), threading.Event(), {}, []
        self.pill_bar = self.pill_label = None
        self._curve_at = 0.
        self.open, self.scope = set(), {}  # blocks shown with their parts; blocks whose picks apply to them only
        self.draft_spec = None
        self.presets = preset_specs(gpt2_words_ready())
        self.compute_preview, self._asking = None, False  # services.pretrain_preview: the size of your text, None until counted
        self.lr_stop, self.lr_result, self.lr_for = None, None, None  # the learning-rate test: running (its stop event), its answer, the design it ran on
        self.preset_key = None  # the ready-made design being looked at
        self.asking, self.guesses, self._predict_for = None, [], None  # ("one" | "split", the draft spec) while the save-time guesses show
        self.picks = {}  # {(version index, dimension): label} picked so far, kept when the page is rebuilt
        self.chain = set()  # ids of split jobs that each start from the one before: their order is not to be changed here
        self._split_key = self._split_plan = None  # the split of the draft into one-change versions, kept until the draft changes
        self.speeds, self.notebook_rows, self._notebook_token = {}, None, None  # measured step times and the lab notebook (slow lane)
        self.check_shown, self.notebook_texts = None, []

    # state ------------------------------------------------------------------
    @property
    def n(self):
        return self.base.number if self.base else 0

    @property
    def nv(self):
        return f"version {self.n + 1}"

    def version(self, number):
        return next((v for v in self.services.versions() if v.number == number), None)

    def base_spec(self):
        return validate_spec(self.base.spec if self.base and self.base.spec else self.default)

    def job_for(self, number):
        """The queued or running pretrain job of a version, with its phase."""
        q = self.services.queue()
        if q.job and q.job.params.get("version") == number:
            return q.job, "training"
        return next(((j, "queued") for j in q.up_next if j.params.get("version") == number), (None, None))

    def refresh(self):
        current = self.services.current_version()
        if self.phase == "edit":
            self._restore_saved(current)
        if self.phase == "edit":
            changed = (current.number if current else 0) != self.n
            self.base = current
            self.cur = read_spec(current.spec) if current and current.spec else read_spec(self.default)
            if changed:
                self.draft, self.anchor, self._staged = dict(self.cur), None, None
                self.services.stage_draft(None)
            staged = self.services.draft_spec()
            if staged is not None and staged != self._staged:  # a design reopened from Compare runs
                try:
                    self.draft, self.anchor = read_spec(staged), (staged, read_spec(staged))
                except ValueError:
                    staged = None
            elif staged is None and self._staged is not None:  # discarded from Quick test
                self.draft, self.anchor = dict(self.cur), None
            self._staged = staged
        self.attr_fresh.clear()  # read the run files again (they may have changed); the last result shows meanwhile
        self.load_chart()
        self.render()
        self.load_lists()
        self.load_notebook()

    def _restore_saved(self, current):
        """A saved version nobody has trained yet shows its coral tray, wherever it was saved (e.g. from Quick test)."""
        if not (current and current.status == "saved" and current.parent and current.spec and current.number != self.dismissed):
            return
        parent = self.version(current.parent)
        if not (parent and parent.spec):
            return
        job, phase = self.job_for(current.number)
        self.base, self.saved = parent, current
        self.cur, self.draft, self.anchor = read_spec(parent.spec), read_spec(current.spec), None
        self.job_id, self.phase = (job.id, phase) if job else (None, "saved")

    def load_chart(self):
        """Curves and reference lines from the real run files; nothing invented when there are none. Scoring the held-back text
        can take a while the first time: that read runs on the shell's worker and the page redraws when it arrives."""
        base = (self.version(self.base.number) or self.base) if self.base else None  # as stored now (a split step may have trained since)
        newer = self.version(self.n + 1) if self.phase != "edit" else None
        self._chart_token = token = object()  # a later load_chart supersedes one still on the worker

        def work():
            try:
                refs = self.services.references(base) if base and base.best_checkpoint else None
            except (OSError, ValueError, KeyError, StopIteration):
                refs = None
            return self.curve_of(base), self.curve_of(newer), refs

        def apply(chart, late):
            if self._chart_token is not token:
                return
            self.old_curve, self.new_curve, self.refs = chart
            self.plot_args = chart_args(self.old_curve, self.new_curve, self.refs)
            if late:
                self.render()
        if self.services.ready(work):
            apply(work(), False)
        else:
            self.quietly(work, lambda chart: apply(chart, True))

    def load_lists(self):
        """The four word lists fitted on your text, for the pieces card. Fitting reads a sample of the text (and fetches
        GPT-2's list once), so it runs on the slow lane; services keep the result until the stored data changes."""
        def work():
            try:
                return self.services.word_lists()
            except (OSError, ValueError):
                return None

        def apply(lists):
            if lists is not None and lists is not self.lists:
                self.lists = lists
                if self.pieces_box is not None and self.pieces_box.winfo_exists():
                    self._paint_pieces()
        self.quietly(work, apply, lane="slow")

    def load_notebook(self):
        """The lab notebook and the measured step times behind the guess check. Reading every version's run files (metrics,
        and run facts that hash the data file once) runs on the slow lane; the card and the guess tray repaint when it arrives."""
        services = self.services
        self._notebook_token = token = object()

        def work():  # no widgets here
            versions = services.versions()
            speeds, facts = {}, {}
            for v in versions:
                try:
                    speeds[v.number] = services.run_speed(v.number)
                except Exception:  # an unreadable run is a missing number, never a stuck card
                    speeds[v.number] = None
                try:
                    facts[v.number] = services.run_facts(v.number)
                except Exception:
                    facts[v.number] = {}
            return speeds, notebook(versions, speeds, facts)

        def apply(value):
            if self._notebook_token is not token:
                return
            self.speeds, self.notebook_rows = value
            self._paint_notebook()
            if self.tray is not None and self.tray.winfo_exists():
                self._paint_tray()
        self.quietly(work, apply, lane="slow")

    def curve_of(self, version):
        runs = getattr(version, "runs", None) or ([str(Path(version.best_checkpoint).parent)] if version and version.best_checkpoint else [])
        if not runs:
            return None
        try:
            return self.services.run_curve(runs[-1])
        except (OSError, ValueError, KeyError, StopIteration):
            return None

    def on_event(self, kind, payload):
        if kind == "versions":
            return self.refresh()
        if kind == "answer":
            return
        if kind != "queue":
            return
        q = self.services.queue()
        if self.phase == "queued" and q.job and q.job.id == self.job_id:
            self.phase = "training"
            return self.render()
        if self.phase in ("queued", "training") and self.job_id and not (q.job and q.job.id == self.job_id) \
                and not any(j.id == self.job_id for j in q.up_next):
            self.load_chart()  # our job left the queue: it finished or was removed
            return self.render()
        if self.phase == "queued" and self.outlook() not in self.tray_text:  # paused, held, or the job ahead finished
            return self.render()
        if self.pill_bar is not None:  # progress only: do not rebuild the page for every step
            self._paint_pill(q)
            if self.phase == "training" and time.time() - self._curve_at > 2:
                self._curve_at = time.time()
                job = q.job
                out = job.params.get("output_dir") if job else None
                if out:
                    try:
                        self.new_curve = self.services.run_curve(out)
                    except (OSError, ValueError, KeyError, StopIteration):
                        return
                    self.plot_args = chart_args(self.old_curve, self.new_curve, self.refs)
                    if self.plot:
                        self.plot.set(**self.plot_args)
                    elif self.plot_args["steps"] or self.plot_args["smooth"]:
                        self.render()  # the first points have arrived: the note becomes a chart

    def edit(self, **patch):
        if self.phase == "edit":
            self.draft.update(patch)
            self.stage()
            self._paint_design()

    def spec_base(self):
        """(spec, its choices) the draft is written onto: a design reopened from another page, else the current version."""
        return self.anchor or (self.base_spec(), self.cur)

    def spec_for(self, s):
        base, was = self.spec_base()
        return write_spec(base, s, was)

    def changes(self):
        try:
            return change_chips(self.base_spec(), self.spec_for(self.draft), self.cur, self.draft)
        except ValueError:
            return []

    def word_changed(self):
        return self.draft["V"] != self.cur["V"]

    def needs_scratch(self):
        """The word list changed, or there is no trained version to start from."""
        return self.word_changed() or not (self.base and self.base.best_checkpoint)

    def start_note(self):
        if not self.word_changed() and self.needs_scratch():
            return NO_PARENT.format(n=self.n)
        return NOTES[warm_kind(self.cur, self.draft)]

    def expectations(self):
        issues = new_issues(self.base_spec(), self.draft_spec) if self.draft_spec else ()
        versions = self.services.versions()
        facts, compares = self.attribution_inputs(versions) if self.draft["L"] != self.cur["L"] else (None, None)
        return expectations(self.cur, self.draft, versions, issues, facts, compares)

    def attribution_inputs(self, versions):
        """(facts, compares) for "last time blocks changed", from what a worker already read, else (None, None). The run
        files (the first read hashes the whole data file) and the Compare runs are read on the slow lane, once per pair of
        versions until something changes, and the line is painted again when they arrive."""
        pair = blocks_change_pair(versions)
        if pair is None:
            return None, None
        key = (pair[0].number, pair[1].number)
        if key not in self.attr_fresh and key not in self.attr_busy:
            self.attr_busy.add(key)
            services = self.services

            def work():  # no widgets here
                try:
                    return key, services.run_facts(key[0]), services.run_facts(key[1]), services.compare_runs()
                except Exception:  # whatever went wrong, the line says it could not check and the pair is not stuck busy
                    return key, {}, {}, []

            def apply(value):
                self.attr_busy.discard(key)
                self.attr_fresh.add(key)
                self.attr_cache[key] = value[1:]
                label = getattr(self, "guide_label", None)
                if not self.painting_expect and label is not None and label.winfo_exists():
                    self._paint_expect()
            self.quietly(work, apply, lane="slow")
        got = self.attr_cache.get(key)
        return (None, None) if got is None else ((lambda n, f=dict(zip(key, got[:2])): f[n]), lambda c=got[2]: c)

    def preview_preset(self, key):
        """Show a ready-made design's cost; nothing changes until it is loaded."""
        self.preset_key = key
        self._paint_preset()

    def use_preset(self):
        """Load the design being looked at into the draft: an ordinary unsaved change against the current version."""
        spec = dict(self.presets).get(self.preset_key)
        if self.phase == "edit" and spec:
            self.draft, self.anchor = preset_draft(spec)
            self.stage()
            self._paint_design()

    def import_gpt2(self):
        """Queue the one-time GPT-2 download and conversion; the new version gets GPT-2's weights when the queue finishes it."""
        if self.phase != "edit" or not messagebox.askyesno(T["import_gpt2"], T["import_gpt2_ask"]):
            return
        try:
            self.services.import_pretrained("gpt2")
            self.note = T["import_gpt2_started"]
        except ValueError as error:
            self.note = str(error)
        self.render()

    def pick(self, slot, short, block=None):
        """Module `short` for `slot`: in every block, or in block `block` only."""
        self.edit(**choose(self.draft, slot, short, block))

    def knob(self, key, value):
        """A seg click. Heads and thinking room mean nothing for a look-back or thinking room without them."""
        if key not in ("H", "W") or knob_applies(self.draft, key):
            self.edit(**{key: value})
        else:
            self._paint_design()  # put the seg back

    def stage(self):
        """Publish the unsaved draft so Quick test and Compare runs can use it."""
        spec = None
        if self.phase == "edit" and self.changes():
            try:
                spec = self.spec_for(self.draft)
            except ValueError:
                spec = None
        if spec != self._staged:
            self._staged = spec
            self.services.stage_draft(spec)

    # actions ----------------------------------------------------------------
    def discard(self):
        self.draft, self.anchor = dict(self.cur), None
        self.stage()
        self._paint_design()

    def ask_save(self):
        """"Save as version N": first a quick guess at what the change will do (skippable); version 1 has nothing to compare."""
        if self.base is None or self.draft_spec is None:
            return self.save()
        self.asking, self.picks = ("one", self.draft_spec), {}
        self._paint_tray()

    def ask_split(self):
        if self.split_plan():
            self.asking, self.picks = ("split", self.draft_spec), {}
            self._paint_tray()

    def cancel_guesses(self):
        self.asking = None
        self._paint_tray()

    def save_guessed(self, skip=False):
        """Save what the guesses were asked about, with the guesses picked (none when skipped)."""
        kind = self.asking[0] if self.asking else "one"
        picks = [{} if skip else guesses_from({key: chip.value for key, chip in chips.items()}) for chips in self.guesses]
        self.asking = None
        if kind == "split":
            self._paint_tray()  # the guess buttons go before the slow read of the data list, so a second click cannot save again
            return self.split(picks)
        self.save(picks[0] if picks else None)

    def split_plan(self):
        """split_plan for the draft, or [] (a design reopened from another page, or nothing saved yet, is saved whole)."""
        if self.anchor is not None or self.base is None or self.phase != "edit":
            return []
        key = copy.deepcopy((self.cur, self.draft))
        if key != self._split_key:
            self._split_key, self._split_plan = key, split_plan(self.base_spec(), self.cur, self.cur, self.draft)
        return self._split_plan

    def guess_steps(self):
        """[(title, old spec, new spec)] for each version the guesses are about."""
        if not self.asking:
            return []
        if self.asking[0] == "one":
            return [("", self.base_spec(), self.draft_spec)]
        out, old = [], self.base_spec()
        for i, (_s, spec, chips) in enumerate(self.split_plan()):
            out.append((T["predict_step_title"].format(n=self.n + 1 + i, change=", ".join(chips)), old, spec))
            old = spec
        return out

    def save(self, prediction=None):
        try:
            spec = self.spec_for(self.draft)
        except ValueError as error:
            self.note = str(error)
            return self.render()
        if problems := errors(spec):
            self.note = problems[0]
            return self.render()
        self.saved = self.services.save_version(spec, ", ".join(self.changes()) or ("first design" if self.base is None else "no change"),
                                                prediction or None)
        self.phase, self._staged, self.anchor = "saved", None, None
        self.services.stage_draft(None)
        self.load_chart()
        self.render()

    def queue_it(self, from_scratch):
        self.slow(lambda: text_sets(self.services.data("text")), lambda sets: self._queue(sets, from_scratch))

    def split(self, predictions=()):
        """Save the draft as one version per change and queue them one after another, each starting from the one before."""
        plan = self.split_plan()
        if not plan:
            self.note = T["one_change_cannot"]
            return self.render()
        self.slow(lambda: text_sets(self.services.data("text")), lambda sets: self._split(sets, plan, list(predictions)))

    def _split(self, sets, plan, predictions):
        if not sets:
            self.note = NO_TEXT
            return self.render()
        parent = self.base.best_checkpoint if self.base else None
        mode = "from_scratch" if plan[0][0]["V"] != self.cur["V"] or not parent else "catch_up"
        saved = [self.services.save_version(spec, ", ".join(chips), predictions[i] if i < len(predictions) else None)
                 for i, (_s, spec, chips) in enumerate(plan)]
        run_dir = getattr(getattr(self.services, "workspace", None), "run_dir", None)
        jobs = split_jobs(saved[0].number, plan, mode, parent, sets, run_dir)
        for job in jobs:
            self.services.enqueue(job)
        self.chain |= {job.id for job in jobs}
        self.base, self.saved = saved[-2], saved[-1]  # the tray follows the last one: "version N, from version N - 1"
        self.cur, self.draft, self.anchor = read_spec(saved[-2].spec), read_spec(saved[-1].spec), None
        self.job_id, self.phase, self._staged = jobs[-1].id, "queued", None
        self.note = T["split_note"].format(first=saved[0].number, last=saved[-1].number)
        self.services.stage_draft(None)
        self.load_chart()
        self.render()

    def _queue(self, sets, from_scratch):
        if not sets:
            self.note = NO_TEXT
            return self.render()
        parent = self.base.best_checkpoint if self.base else None
        mode = "from_scratch" if from_scratch or self.needs_scratch() else "catch_up"
        spec = self.saved.spec if self.saved and self.saved.spec else self.spec_for(self.draft)
        self.job_id = uuid.uuid4().hex[:8]
        self.services.enqueue(catchup_job(self.job_id, self.n + 1, spec, mode, parent, sets))
        self.phase = "queued"
        self.render()

    def to_front(self):
        """Move the job to the head of the waiting list (the running job is not interrupted)."""
        for _ in range(len(self.services.queue().up_next)):
            if self.position() == 0:
                break
            self.services.move_up(self.job_id)
        self.render()

    def outlook(self):
        return queued_outlook(self.services, self.job_id)

    def position(self):
        return next((i for i, j in enumerate(self.services.queue().up_next) if j.id == self.job_id), None)

    def new_changes(self):
        self.base = self.services.current_version() or self.saved
        self.dismissed = self.base.number if self.base else None
        self.cur = read_spec(self.base.spec) if self.base and self.base.spec else dict(self.draft)
        self.draft, self.anchor, self.phase, self.job_id = dict(self.cur), None, "edit", None
        self.load_chart()
        self.render()

    def ask_both(self):
        """Same question to both versions (those with trained weights); replies stream in."""
        self.answer_stop.set()
        self.answer_stop = stop = threading.Event()
        self.answers, self.workers = {}, []
        for number in (self.n, self.n + 1):
            v = self.version(number)
            if v and v.best_checkpoint:
                self.answers[number] = ""
                self.workers.append(threads.start(self._reply, v, stop))
        self.render()
        self._pump()

    def _reply(self, v, stop):
        put = lambda text: self.answer_q.put((v.number, text))  # noqa: E731
        try:
            final = self.services.chat(v, [{"role": "user", "content": PROMPT}], put, stop)
            if final:
                put(final)
        except Exception as error:  # noqa: BLE001 - shown in place of the reply
            put(plain_error(error))

    def _pump(self):
        try:
            while True:
                number, text = self.answer_q.get_nowait()
                if number in self.answers:
                    self.answers[number] = text
                    if number in self.answer_labels:
                        self.answer_labels[number].configure(text=text)
        except queue.Empty:
            pass
        if any(t.is_alive() for t in self.workers) or not self.answer_q.empty():
            self.after(200, self._pump)

    # view -------------------------------------------------------------------
    def render(self):
        for child in self.body.winfo_children():
            child.destroy()
        self.stage()
        self.answer_labels, self.segs = {}, {}
        self._pill()
        top = tk.Frame(self.body, bg=C["bg"])
        top.pack(fill="x", pady=(0, GUTTER))
        left = tk.Frame(top, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, GUTTER))
        right = tk.Frame(top, bg=C["bg"], width=340)
        right.pack(side="left", fill="y")
        self._tower(left)
        self._start_from(right)
        self._controls(right)
        self._costs()
        self._expect()
        self._tray()
        self._charts()
        self._notebook()
        self._look_inside()
        self._paint_design()

    def _paint_design(self):
        """Everything the draft decides, repainted in place: rebuilding the page on each setting flashed it."""
        try:
            self.draft_spec = self.spec_for(self.draft)
        except ValueError:
            self.draft_spec = None
        self._paint_tower()
        for key in self.segs:
            self._paint_seg(key)
        for key, note in self.knob_notes.items():
            note.configure(text="" if knob_applies(self.draft, key) else T["no_heads" if key == "H" else "no_room"])
        self._paint_costs()
        self._paint_expect()
        self._paint_tray()

    def _pill(self):
        q = self.services.queue()
        row = tk.Frame(self.body, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=14, pady=8)
        row.pack(anchor="w", pady=(0, GUTTER))
        self.pill_label = tk_label(row, "", 14, "text", "panel")
        self.pill_label.pack(side="left")
        self.pill_bar = Progress(row)
        self.pill_bar.configure(width=80)
        self.pill_bar.pack(side="left", padx=(10, 0))
        self._paint_pill(q)

    def _paint_pill(self, q):
        self.pill_label.configure(text=pill_text(q.job, self.base), fg=C["violet"] if q.job else C["text2"])
        self.pill_bar.set(q.step / q.total if q.job and q.total else None)

    def _tower(self, parent):
        """The model bottom to top, drawn top-down: next word, word scores, even out, the blocks, position, word table, your text."""
        box = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, highlightbackground=C["border"])
        box.pack(fill="x")
        tip(tk_label(box, T["next_word"], 15, "text", "panel2", font=font(15, bold=True)), T["next_word_desc"]).pack(anchor="w", padx=12, pady=8)
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        self.pickers = []  # (menu button, was label, slot, block index or None) of the model-wide parts
        self._part_row(parent, "head")
        row = self._row(parent)
        tip(tk_label(row, T["final_norm"], 14, "text", "panel2", font=font(14, bold=True), width=14),
            f"{T['final_norm_tip']}\n\n{part_tip('norm')}").pack(side="left", padx=12, pady=6)
        self.final_norm = tk_label(row, "", 13, "text2", "panel2")
        self.final_norm.pack(side="left")
        tip(self.final_norm, lambda: module_tip("norm", self.draft["M"]["norm"]))
        tk_label(row, f"  {T['same_as_blocks']}", 12, "muted", "panel2").pack(side="left")
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        self.block_box = tk.Frame(parent, bg=C["bg"])
        self.block_box.pack(fill="x")
        self.block_rows = []  # (row, name, heads, width, tag, header, toggle, parts, detail) per block, block 1 first; drawn top-down
        step = tk.Frame(parent, bg=C["bg"])
        step.pack(anchor="w", pady=8)
        tip(tk_label(step, f"{T['blocks']}  ", 14, "text", "bg"), f"{T['blocks_desc']}\n\n{glossary_text('block')}").pack(side="left")
        self.minus = button(step, "−", lambda: self.edit(**resize(self.draft, max(1, self.draft["L"] - 1))))
        self.plus = button(step, "+", lambda: self.edit(**resize(self.draft, min(12, self.draft["L"] + 1))))
        self.minus.pack(side="left")
        self.blocks_label = tk_label(step, "", 18, "text", "bg", width=3, anchor="center")
        self.blocks_label.pack(side="left")
        self.plus.pack(side="left")
        self.blocks_was = tk_label(step, "", 13, "muted", "bg")
        self.blocks_was.pack(side="left")
        for b in (self.minus, self.plus):
            b.state(["!disabled"] if self.phase == "edit" else ["disabled"])
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        self._part_row(parent, "position")
        self._part_row(parent, "embedding")
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        text = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, highlightbackground=C["border"])
        text.pack(fill="x")
        head = tk.Frame(text, bg=C["panel2"])
        head.pack(fill="x", padx=12, pady=(8, 0))
        tip(tk_label(head, T["your_text_title"], 14, "text", "panel2", font=font(14, bold=True)), glossary_text("tokenizer")).pack(side="left")
        self.pieces_count = tk_label(head, "", 12, "muted", "panel2")
        self.pieces_count.pack(side="left")
        self._seg(text, list(WORDS), "V", (0, 1, 2, 3), "panel2")
        for option, term in zip(WORDS, WORD_TERMS):
            tip(self.segs["V"][0].buttons[option], glossary_text(term))
        self.pieces_box = tk.Frame(text, bg=C["panel2"])
        self.pieces_box.pack(anchor="w", fill="x", padx=12, pady=(0, 8))
        self.pieces_shown = None

    def _row(self, parent):
        row = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, highlightbackground=C["border"])
        row.pack(fill="x", pady=2)
        return row

    def _part_row(self, parent, slot):
        """One model-wide part: its name and its module menu."""
        row = self._row(parent)
        tip(tk_label(row, T["part_" + slot], 14, "text", "panel2", font=font(14, bold=True), width=14), part_tip(slot)).pack(side="left", padx=12, pady=6)
        self.pickers.append((*self._picker(row, slot, lambda: None, "panel2"), slot, None))

    def _picker(self, parent, slot, block, bg):
        """(menu button, was label) choosing `slot`'s module; block() says where a pick goes: None = every block, or a block index.
        Modules that cannot work in this design are listed greyed out with the reason."""
        mb = tk.Menubutton(parent, font=font(13), relief="flat", bg=C[bg], fg=C["text"], activebackground=C["panel"], activeforeground=C["text"],
                           disabledforeground=C["muted"], takefocus=1, cursor="hand2", padx=8, pady=3, indicatoron=True,
                           highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["violet"])
        menu = tk.Menu(mb, tearoff=0, font=font(13), bg=C["panel"], fg=C["text"], activebackground=C["violet"], activeforeground=C["bg"],
                       disabledforeground=C["dim"])

        def fill():
            menu.delete(0, "end")
            base, was = self.spec_base()
            for short, name, _blurb, reason in part_options(base, self.draft, was, slot, block()):
                menu.add_command(label=f"{name} · {T['wont_fit']}: {reason}" if reason else name, state="disabled" if reason else "normal",
                                 command=lambda short=short: self.pick(slot, short, block()))
        menu.configure(postcommand=fill)
        mb.configure(menu=menu)
        for seq in ("<space>", "<Return>"):
            mb.bind(seq, lambda _e: menu.tk_popup(mb.winfo_rootx(), mb.winfo_rooty() + mb.winfo_height()) if self.phase == "edit" else None)
        tip(mb, lambda: module_tip(slot, mb.module))
        mb.pack(side="left")
        was = tk_label(parent, "", 12, "muted", bg)
        was.pack(side="left", padx=(6, 0))
        return mb, was

    def _paint_picker(self, mb, was, slot, now, before, bg):
        mb.module = now
        changed = before is not None and now != before
        mb.configure(text=module_name(slot, now), fg=C["amber"] if changed else C["text"], bg=C[bg],
                     state="normal" if self.phase == "edit" else "disabled")
        was.configure(text=f"{T['was'].format(name=module_name(slot, before))}  ·  changed" if changed else "", bg=C[bg])

    def _paint_tower(self):
        cur, dr, rows = self.cur, self.draft, self.block_rows
        for mb, was, slot, _ in self.pickers:
            self._paint_picker(mb, was, slot, dr["M"][slot], cur["M"][slot], "panel2")
        self.final_norm.configure(text=module_name("norm", dr["M"]["norm"]), fg=C["amber"] if dr["M"]["norm"] != cur["M"]["norm"] else C["text2"])
        while len(rows) < max(cur["L"], dr["L"]):  # a new block goes on top
            i = len(rows)
            row = tk.Frame(self.block_box, highlightthickness=1)
            row.pack(fill="x", pady=2, **({"before": rows[-1][0]} if rows else {}))
            header = tk.Frame(row)
            header.pack(fill="x")
            toggle = tk_label(header, "▸", 13, "muted", cursor="hand2", takefocus=1)
            toggle.pack(side="left", padx=(8, 0))
            _activate(toggle, lambda i=i: self.toggle(i))
            tip(toggle, T["block_parts_tip"])
            name = tk_label(header, f"Block {i + 1}", 13, width=8)
            name.pack(side="left", padx=4, pady=8)
            heads = tip(tk_label(header, "", 13, "pink"), glossary_text("heads"))
            heads.pack(side="left")
            width = tip(tk.Frame(header, bg=C["blue"], height=8), glossary_text("expansion"))
            width.pack(side="left", padx=10)
            parts = tk_label(header, "", 12, "muted")
            parts.pack(side="left")
            tip(parts, lambda i=i: "\n\n".join(module_tip(slot, block_parts(self.draft, i)[slot]) for slot in ("mixer", "channel")))
            tag = tk_label(header, "", 12, "amber")
            tag.pack(side="right", padx=8)
            detail = tk.Frame(row)
            rows.append((row, name, heads, width, tag, header, toggle, parts, detail))
        while len(rows) > max(cur["L"], dr["L"]):
            rows.pop()[0].destroy()
        knobs = dr["H"] != cur["H"] or dr["W"] != cur["W"]
        for i, (row, name, heads, width, tag, header, toggle, parts, detail) in enumerate(rows):
            added, removed = i >= cur["L"], i >= dr["L"]
            shown = cur if removed else dr
            mine, own = block_parts(shown, i), own_slots(shown, i)
            changed = not added and not removed and (knobs or block_parts(cur, i) != mine)
            bg = "amber_bg" if added else "violet_bg" if own else "panel2"
            row.configure(bg=C[bg], highlightbackground=C["amber"] if added else C["violet"] if own else C["border"])
            for w in (header, toggle, name, heads, parts, tag, detail):
                w.configure(bg=C[bg])
            toggle.configure(text="▾" if i in self.open and not removed else "▸", highlightbackground=C[bg])
            name.configure(fg=C["dim"] if removed else C["text"])
            heads.configure(text="●" * shown["H"] if "heads" in fields_of("mixer", mine["mixer"]) else "")
            if "d_ff" in fields_of("channel", mine["channel"]):
                width.configure(width=[36, 72, 144][shown["W"]])
                width.pack(side="left", padx=10, after=heads)
            else:
                width.pack_forget()
            parts.configure(text=f"{module_name('mixer', mine['mixer'])} · {module_name('channel', mine['channel'])}")
            text = "new" if added else "removed" if removed else "changed" if changed else T["own_parts"] if own else ""
            tag.configure(text=text, fg=C["violet"] if text == T["own_parts"] else C["amber"])
            self._paint_detail(i, detail, bg, removed)
        self.blocks_label.configure(text=str(dr["L"]), fg=C["amber"] if dr["L"] != cur["L"] else C["text"])
        self.blocks_was.configure(text=f"  was {cur['L']}" if dr["L"] != cur["L"] else "")
        self._paint_pieces()

    def _paint_pieces(self):
        v = self.draft["V"]
        chips, count = pieces_for(v, self.lists)
        self.pieces_count.configure(text=f"  {count}")
        if (v, id(self.lists)) != self.pieces_shown:
            self.pieces_shown = v, id(self.lists)
            for child in self.pieces_box.winfo_children():
                child.destroy()
            for k in range(0, len(chips), 14):  # wrap long piece lists
                line = tk.Frame(self.pieces_box, bg=C["panel2"])
                line.pack(anchor="w")
                for label, state in chips[k:k + 14]:
                    Chip(line, label, state).pack(side="left", padx=1, pady=1)

    def toggle(self, i):
        self.open ^= {i}
        self._paint_tower()

    def _paint_detail(self, i, detail, bg, removed):
        """Block i's sub-layers top-down (what a word passes through bottom-up), each with its module menu."""
        for child in detail.winfo_children():
            child.destroy()
        if i not in self.open or removed:
            detail.pack_forget()
            return
        detail.pack(fill="x", padx=(32, 12), pady=(0, 8))
        one = self.scope.get(i, False)
        options = [T["scope_all"], T["scope_one"].format(n=i + 1)]
        scope = Segmented(detail, options, options[one], lambda opt, i=i: self.set_scope(i, opt != options[0]), bg=bg)
        scope.pack(anchor="w", pady=(0, 4))
        cur, dr = self.cur, self.draft
        mine, before = block_parts(dr, i), block_parts(cur, i) if i < cur["L"] else None
        for slot in (None, "channel", "norm", None, "mixer", "norm"):  # None: the add-back
            line = tk.Frame(detail, bg=C[bg])
            line.pack(fill="x", pady=1)
            if slot is None:
                tip(tk_label(line, f"⊕ {T['add_back']}", 12, "muted", bg), glossary_text("residual")).pack(side="left")
                continue
            skipped = registry.get(slot, mine[slot]).skip
            tip(tk_label(line, T["part_" + slot], 13, "dim" if skipped else "text", bg, width=14), part_tip(slot)).pack(side="left")
            mb, was = self._picker(line, slot, lambda i=i: i if self.scope.get(i) else None, bg)
            self._paint_picker(mb, was, slot, mine[slot], before and before[slot], bg)
            if (i, slot) in dr["O"]:
                tk_label(line, T["only_here"], 12, "violet", bg).pack(side="left", padx=(8, 0))

    def set_scope(self, i, one):
        self.scope[i] = one
        self._paint_tower()

    def _seg(self, parent, options, key, values, bg="panel"):
        label, value = dict(zip(values, options)), dict(zip(options, values))
        seg = Segmented(parent, options, None, lambda opt: self.knob(key, value[opt]), bg=bg)
        seg.pack(anchor="w", padx=12 if bg == "panel2" else 0, pady=4)
        self.segs[key] = seg, label

    def _paint_seg(self, key):
        seg, label = self.segs[key]
        seg.set(label.get(self.draft[key]))
        # ponytail: Tk borders cannot be dashed, so the saved choice gets a violet outline and a "saved" in its label.
        was = label.get(self.cur[key]) if self.cur[key] != self.draft[key] else None
        for opt, b in seg.buttons.items():
            b.configure(highlightbackground=C["violet"] if opt == was else C["border"], text=f"{opt} · saved" if opt == was else opt)

    def _start_from(self, parent):
        """Pick a ready-made design, read what it costs on this computer and its recipe, then load it into the draft."""
        card = Card(parent)
        card.pack(fill="x", pady=(0, 8))
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w")
        Kicker(head, T["start_from"], bg="panel").pack(side="left")
        info_icon(head, T["start_from_tip"]).pack(side="left", padx=(4, 0))
        mb = tk.Menubutton(card, font=font(13), relief="flat", bg=C["panel"], fg=C["text"], activebackground=C["panel2"], activeforeground=C["text"],
                           disabledforeground=C["muted"], takefocus=1, cursor="hand2", padx=8, pady=3, indicatoron=True,
                           highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["violet"])
        menu = tk.Menu(mb, tearoff=0, font=font(13), bg=C["panel"], fg=C["text"], activebackground=C["violet"], activeforeground=C["bg"])
        for key, spec in self.presets:
            menu.add_command(label=spec["name"], command=lambda key=key: self.preview_preset(key))
        mb.configure(menu=menu)
        for seq in ("<space>", "<Return>"):
            mb.bind(seq, lambda _e: menu.tk_popup(mb.winfo_rootx(), mb.winfo_rooty() + mb.winfo_height()) if self.phase == "edit" else None)
        tip(mb, lambda: glossary_text(preset_term(self.preset_key)) if self.preset_key else T["start_from_tip"])
        mb.pack(anchor="w", pady=(4, 0))
        self.preset_mb = mb
        self.preset_about = tk_label(card, "", 12, "text2", "panel", wraplength=300)
        tip(self.preset_about, lambda: glossary_text(preset_term(self.preset_key)) if self.preset_key else "")
        self.preset_lines = [tk_label(card, "", 12, "muted", "panel", wraplength=300) for _ in range(4)]  # size, fit, speed, recipe
        self.preset_btn = button(card, T["preset_load"], self.use_preset)
        pretrained = Card(parent)
        pretrained.pack(fill="x", pady=(0, 8))
        self.import_btn = button(pretrained, T["import_gpt2"], self.import_gpt2)
        self.import_btn.pack(anchor="w")
        tip(self.import_btn, lambda: T["import_gpt2_tip"])
        self._paint_preset()

    def _paint_preset(self):
        spec = dict(self.presets).get(self.preset_key)
        self.preset_mb.configure(text=spec["name"] if spec else T["start_from_pick"], state="normal" if self.phase == "edit" else "disabled")
        widgets = (self.preset_about, *self.preset_lines, self.preset_btn)
        for w in widgets:
            w.pack_forget()
        if not spec:
            return
        cost = self.preset_cost = preset_cost(spec, gpu=gpu_kind())
        self.preset_about.configure(text=T["preset_" + self.preset_key])
        texts = [*cost["lines"], T["preset_recipe"].format(recipe=recipe_text(spec))]
        for label, text, bad in zip(self.preset_lines, texts, (False, cost["fits"] is False, cost["slow"], False)):
            label.configure(text=text, fg=C["amber"] if bad else C["muted"])
        self.preset_btn.state(["!disabled"] if self.phase == "edit" else ["disabled"])
        self.preset_about.pack(anchor="w", pady=(6, 2))
        for label in self.preset_lines:
            label.pack(anchor="w", pady=1)
        self.preset_btn.pack(anchor="w", pady=(6, 0))

    def _controls(self, parent):
        self.knob_notes = {}
        for name, hint, color, opts, key, values in (
                ("Look-back heads", f"{T['look_back_heads_tip']}\n\n{glossary_text('heads')}", "pink", [str(h) for h in HEADS], "H", HEADS),
                ("Thinking room", f"{T['thinking_room_tip']}\n\n{glossary_text('expansion')}", "blue", list(WIDTHS), "W", (0, 1, 2))):
            card = Card(parent)
            card.pack(fill="x", pady=(0, 8))
            head = tk.Frame(card, bg=C["panel"])
            head.pack(anchor="w")
            tk_label(head, name, 15, color, "panel", font=font(15, bold=True)).pack(side="left")
            info_icon(head, hint).pack(side="left", padx=(4, 0))
            self._seg(card, opts, key, values)
            self.knob_notes[key] = tk_label(card, "", 12, "muted", "panel", wraplength=300)
            self.knob_notes[key].pack(anchor="w")

    def _costs(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["what_costs"], bg="panel").pack(anchor="w")
        self.cost_rows = []
        for name, f, more, less in ((T["size_label"], "params", T["cost_bigger"], T["cost_smaller"]),
                                    (T["time_per_step"], "time", T["cost_slower"], T["cost_faster"]),
                                    (T["memory_while_training"], "memory", T["cost_more"], T["cost_less"])):
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", pady=4)
            label = tk_label(row, name, 14, "text", "panel", width=24)
            label.pack(side="left")
            if term := {"params": "parameters", "time": "step"}.get(f):
                tip(label, glossary_text(term))
            bars = tk.Frame(row, bg=C["panel"])
            bars.pack(side="left", fill="x", expand=True)
            drawn = []
            for label in ("now", "new"):
                line = tk.Frame(bars, bg=C["panel"])
                line.pack(fill="x")
                tk_label(line, label, 11, "dim", "panel", width=4).pack(side="left")
                drawn.append(tk.Frame(line, bg=C["dim"], height=7))
                drawn[-1].pack(side="left", pady=1)
            said = tk_label(row, "", 13, "muted", "panel", width=22)
            said.pack(side="left")
            self.cost_rows.append((name, f, more, less, *drawn, said))
        row = tk.Frame(card, bg=C["panel"])
        row.pack(fill="x", pady=(8, 2))
        tk_label(row, "Can the old weights carry over?", 14, "text", "panel", width=24).pack(side="left")
        self.warm_chip = Chip(row)
        self.warm_none = tk_label(row, "nothing to carry over: no change yet", 13, "muted", "panel")
        self.warm_text = tk_label(card, "", 12, "muted", "panel", wraplength=780)
        self.fits_row = tk.Frame(card, bg=C["panel"])
        self.fits_row.pack(anchor="w", pady=(6, 0))
        self.fits_label = tk_label(self.fits_row, "", 12, "muted", "panel")
        self.fits_label.pack(side="left")
        info_icon(self.fits_row, lambda: self.fits_text).pack(side="left", padx=(4, 0))
        self.compute_labels = [tk_label(card, "", 12, "muted", "panel", wraplength=780) for _ in range(3)]
        for label in self.compute_labels:
            label.pack(anchor="w", pady=(6, 0))
        self.compute_tip = T["compute_tip"].format(rule=TOKENS_PER_PARAM, most=MAX_EPOCHS)
        tip(self.compute_labels[0], lambda: self.compute_tip)
        row = tk.Frame(card, bg=C["panel"])
        row.pack(anchor="w", pady=(8, 0))
        self.lr_button = button(row, T["lr_find"], self.find_lr)
        self.lr_button.pack(side="left")
        info_icon(row, T["lr_tip"]).pack(side="left", padx=(4, 0))
        self.lr_label = tk_label(card, "", 12, "muted", "panel", wraplength=780)
        self.lr_label.pack(anchor="w", pady=(4, 0))
        self._paint_lr()
        self._ask_compute()

    def _ask_compute(self):
        """The size of your text, counted once per page: known at once when it was counted before, else in the background."""
        known = self.services.cached_preview([])
        if known is not None or self.compute_preview is not None:
            self.compute_preview = known if known is not None else self.compute_preview
        elif not self._asking:
            self._asking = True
            self.quietly(lambda: self.services.pretrain_preview([]), self._got_compute, lane="slow")

    def _got_compute(self, preview):
        self._asking, self.compute_preview = False, preview or {}
        if self.winfo_exists() and self.compute_labels[0].winfo_exists():
            self._paint_compute()

    def _paint_compute(self):
        spec = self.draft_spec or self.base_spec()
        if self.word_changed():
            lines = [(T["compute_recount"], False)]
        else:
            lines = compute_lines(sizes(spec, self.draft["V"])["params"], spec["dims"]["n_ctx"], spec["dims"]["n_layer"],
                                  self.compute_preview, gpu_kind())
        for label, (text, bad) in zip(self.compute_labels, lines + [("", False)] * len(self.compute_labels)):
            label.configure(text=text, fg=C["amber"] if bad else C["muted"])

    def find_lr(self):
        """Run the learning-rate range test on the draft in the background; pressed again while it runs, it stops."""
        if self.lr_stop is not None:
            self.lr_stop.set()
            return
        spec, stop = self.draft_spec or self.base_spec(), threading.Event()
        self.lr_stop, self.lr_result, self.lr_for = stop, None, spec

        def work():
            try:
                return self.services.lr_range_test(spec, stop)
            except Exception as error:  # say it under the button; the button comes back
                return {"error": plain_error(error)}

        def done(result):
            self.lr_stop, self.lr_result = None, result
            if self.winfo_exists() and self.lr_label.winfo_exists():
                self._paint_lr()
        self._paint_lr()
        self.quietly(work, done, lane="slow")

    def _paint_lr(self):
        running = self.lr_stop is not None
        current = ((self.lr_for or {}).get("train", {}).get("optimizer", {}).get("config", {}) or {}).get("lr")
        self.lr_button.configure(text=T["lr_stop"] if running else T["lr_find"])
        stale = self.lr_for != (self.draft_spec or self.base_spec())  # the draft changed since the test ran
        self.lr_label.configure(text=T["lr_running"] if running else "" if stale else lr_text(self.lr_result, current))

    def _paint_costs(self):
        a = sizes(self.base_spec(), self.cur["V"])
        b = sizes(self.draft_spec, self.draft["V"]) if self.draft_spec else a
        self.cost_say = {}
        for name, f, more, less, now, new, said in self.cost_rows:
            x, y = a[f], b[f]
            s = self.cost_say[name] = say(x, y, more, less)
            same = s == T["about_same"]
            old_pct, new_pct = bar(x, y)
            now.configure(width=max(2, old_pct * 3))
            new.configure(width=max(2, new_pct * 3), bg=C["violet"] if same else C["amber"])
            said.configure(text=s, fg=C["muted"] if same else C["amber"])
        self.warm = warm = warm_outlook(self.cur, self.draft)
        if warm:
            self.warm_none.pack_forget()
            self.warm_chip.set(warm["chip"], warm["state"])
            self.warm_chip.pack(side="left")
            self.warm_text.configure(text=warm["text"])
            self.warm_text.pack(anchor="w", before=self.fits_row)
        else:
            self.warm_chip.pack_forget()
            self.warm_text.pack_forget()
            self.warm_text.configure(text="")
            self.warm_none.pack(side="left")
        ok = fits(b)
        verdict = T["training_memory_fits"] if ok is None else ("fits" if ok else "doesn't fit")
        self.fits_text = f"{T['training_memory']} {verdict} {T['training_memory_desc']}"
        self.fits_label.configure(text=f"{T['training_memory']} {verdict}")
        self._paint_compute()

    def _expect(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["what_expect"], bg="panel").pack(anchor="w")
        row = tk.Frame(card, bg=C["panel"])
        row.pack(anchor="w", pady=(2, 0))
        self.guide_label = tk_label(row, "", 14, "text", "panel")
        self.guide_label.pack(side="left")
        info_icon(row, lambda: self.hover_text).pack(side="left", padx=(6, 0))

    def _paint_expect(self):
        self.guide_text, self.guide_details = design_expectation(self.draft, self.draft_spec or self.base_spec())
        self.painting_expect = True
        try:
            self.expect_texts = self.expectations()
        finally:
            self.painting_expect = False
        self.hover_text = "\n\n".join([self.guide_details, *self.expect_texts])  # the card itself is one line
        self.guide_label.configure(text=self.guide_text)

    def _tray(self):
        self.tray_box = tk.Frame(self.body, bg=C["bg"])
        self.tray_box.pack(fill="x", pady=(0, GUTTER))
        self.tray = StateTray(self.tray_box)
        self.tray.pack(fill="x", pady=(0, 8))
        self.chips = tk.Frame(self.tray_box, bg=C["bg"])
        self.predict_slot = tk.Frame(self.tray_box, bg=C["bg"])  # the save-time guesses
        self.predict_slot.pack(fill="x")
        self.button_slot = tk.Frame(self.tray_box, bg=C["bg"])
        self.button_slot.pack(fill="x")
        self.check_tray = StateTray(self.tray_box)  # a saved guess against what training did
        self.buttons, self.actions, self._predict_for = None, {}, None

    def _paint_tray(self):
        ph, n, nv = self.phase, self.n, self.nv
        Nv = nv[0].upper() + nv[1:]
        scratch = self.needs_scratch()
        self.chip_texts = []
        actions = []
        if self.asking and (ph != "edit" or self.asking[1] != self.draft_spec):  # the draft moved on: ask again on the next save
            self.asking = None
        if ph == "edit":
            self.chip_texts = self.changes()
            if self.chip_texts:
                state, kicker, text = "unsaved", T["not_saved"], T["not_saved_text"].format(version=nv)
                plan = self.split_plan() if len(self.chip_texts) > 1 else []
                if len(self.chip_texts) > 1 and self.base is not None:
                    text += " " + T["one_change"].format(count=len(self.chip_texts)) + " " + \
                        (T["one_change_split"].format(count=len(plan)) if plan else T["one_change_cannot"])
                if self.asking:
                    save = T["predict_split_save"].format(count=len(plan)) if self.asking[0] == "split" else T["predict_save"]
                    actions = [(save, self.save_guessed, True), (T["predict_skip"], lambda: self.save_guessed(True), False),
                               (T["predict_back"], self.cancel_guesses, False)]
                else:
                    actions = [(T["quick_test_first"], lambda: self.navigate("quicktest"), False),
                               (f"Save as {nv}", self.ask_save, True), (T["discard"], self.discard, False)]
                    if plan:
                        actions.append((T["split"].format(count=len(plan)), self.ask_split, False))
            elif self.base is None:  # nothing saved yet: the starting design itself can become version 1
                state, kicker, text = "unsaved", T["first_save"], T["first_save_text"].format(version=nv)
                actions = [(f"Save as {nv}", self.save, True), (T["quick_test_first"], lambda: self.navigate("quicktest"), False)]
            else:
                state, kicker, text = "neutral", T["no_changes"], T["no_changes_text"].format(version=n)
        elif ph == "saved":
            state = "needs_training"
            kicker = T["saved_text_needs_training" if scratch else "saved_text_needs_catchup"].format(new_version=nv)
            text = T["saved_text"].format(startNote=self.start_note(), current_version=f"version {n}", new_version=nv) if n else self.start_note()
            text += f" {if_added(self.services.queue_outlook())}."
            actions = [(T["add_training" if scratch else "add_catchup"], lambda: self.queue_it(False), True)]
            if not scratch:
                actions.append((T["start_from_scratch"], lambda: self.queue_it(True), False))
        elif ph == "queued":
            state, kicker, text = "live", T["queued"], T["queued_text"].format(new_version=Nv)
            when = self.outlook()
            text += f" {when}." if when else ""
            if self.position() and self.job_id not in self.chain:  # a split step must not run before the one it starts from
                actions = [(T["move_to_front"], self.to_front, True)]
        else:
            state = "training"
            kicker = T["training"].format(new_version=nv, status="IS TRAINING" if scratch else "IS CATCHING UP")
            text = TRAINING_TEXT.format(new=nv, n=n)
            actions = [(T["start_new_changes"], self.new_changes, True)]
        if self.note:
            text += " " + self.note
            self.note = None
        self.tray.set(state, kicker, text)
        self.tray_kicker = kicker
        self.tray_text = text
        chips = self.chips.winfo_children()
        for chip in chips[len(self.chip_texts):]:
            chip.destroy()
        for i, c in enumerate(self.chip_texts):
            if i < len(chips):
                chips[i].set(c, "unsaved")
            else:
                Chip(self.chips, c, "unsaved").pack(side="left", padx=(0, 6))
        if self.chip_texts:
            self.chips.pack(anchor="w", pady=(0, 6), after=self.tray)
        else:
            self.chips.pack_forget()  # an emptied frame keeps its last size
        if [label for label, _, _ in actions] != list(self.actions):
            if self.buttons:
                self.buttons.destroy()
            self.buttons, self.actions = tk.Frame(self.button_slot, bg=C["bg"]), {}
            self.buttons.pack(anchor="w")
            for label, cmd, primary in actions:
                self.actions[label] = btn = button(self.buttons, label, cmd, primary)
                btn.pack(side="left", padx=(0, 8))
        if self.asking != self._predict_for:
            self._predict_for = self.asking
            self._paint_predict()
        self._paint_check()

    def _paint_predict(self):
        """The save-time guesses: a row of chips per dimension worth a guess, for each version being saved."""
        for child in self.predict_slot.winfo_children():
            child.destroy()
        self.guesses = []
        steps = self.guess_steps()
        if not steps:
            return
        card = Card(self.predict_slot)
        card.pack(fill="x", pady=(0, 8))
        Kicker(card, T["predict"], bg="panel").pack(anchor="w")
        tk_label(card, T["predict_text"], 13, "muted", "panel", wraplength=780).pack(anchor="w", pady=(2, 4))
        for i, (title, old, new) in enumerate(steps):
            if title:
                tk_label(card, title, 13, "text", "panel", wraplength=780).pack(anchor="w", pady=(6, 0))
            chips = {}
            for key in prediction_questions(old, new):
                row = tk.Frame(card, bg=C["panel"])
                row.pack(anchor="w", pady=2)
                tk_label(row, T["predict_" + key], 13, "text", "panel", width=22, anchor="w").pack(side="left")
                chips[key] = GuessChips(row, [T["predict_" + d] for d in GUESSES[key]],
                                        lambda label, at=(i, key): self.picks.__setitem__(at, label))
                chips[key].set(self.picks.get((i, key)))
                chips[key].pack(side="left")
            self.guesses.append(chips)

    def _paint_check(self):
        """The guess made when the version was saved, against what training did: the current version while editing, else
        the one being trained. Step times come from the notebook's read (self.speeds); nothing here reads a file."""
        target = self.base if self.phase == "edit" else self.version(self.n + 1)
        shown = None
        if target is not None and getattr(target, "prediction", None):
            old = self.version(target.parent or target.number - 1)
            shown = check_text(old, target, self.speeds)
        self.check_shown = shown
        if shown:
            self.check_tray.set(*shown)
            self.check_tray.pack(fill="x", pady=(8, 0))
        else:
            self.check_tray.pack_forget()

    def _charts(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["while_training"], bg="panel").pack(anchor="w")
        old, new = bool(self.plot_args.get("smooth")), bool(self.plot_args.get("steps"))
        self.plot = self.ask_btn = None
        if old or new:
            self.plot = BpcPlot(card, 640, 200)
            self.plot.pack(anchor="w", pady=6)
            self.plot.set(**self.plot_args)
            legend = tk.Frame(card, bg=C["panel"])
            legend.pack(anchor="w")
            if old:
                tk_label(legend, version_name(self.n).capitalize(), 12, "violet", "panel").pack(side="left", padx=(0, 16))
            if new:
                faint = f"faint line: {self.nv}" + (f", continuing from version {self.n}'s end" if old else "")
                tk_label(legend, faint, 12, "amber", "panel").pack(side="left")
        else:
            self.no_curve = curve_note(self.n, self.phase)
            tk_label(card, self.no_curve, 13, "muted", "panel").pack(anchor="w", pady=6)
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w", pady=(8, 0))
        Kicker(head, T["when_answering"], bg="panel").pack(side="left")
        info_icon(head, T["answering_note"]).pack(side="left", padx=(4, 0))
        trained = {k for k in (self.n, self.n + 1) if self.version(k) and self.version(k).best_checkpoint}
        rows = answer_rows(self.n, trained, self.phase)
        if not rows:  # nothing can answer, so no prompt, rows or button
            self.nothing_to_ask = T["nothing_to_ask"].format(n=self.n) if self.n else T["nothing_to_ask_first"]
            tk_label(card, self.nothing_to_ask, 13, "muted", "panel", wraplength=780).pack(anchor="w", pady=6)
            return
        tk_label(card, T["example_prompt"], 14, "text", "panel", font=font(14, serif=True)).pack(anchor="w")
        for number, why_not in rows:
            if number in self.answers:
                text = self.answers[number] or "…"
            else:
                text = why_not or T["its_answer_appears"]
            row = tk.Frame(card, bg=C["panel"])
            row.pack(anchor="w", fill="x")
            tk_label(row, version_name(number), 13, "violet" if number == self.n else "amber", "panel", width=14).pack(side="left")
            label = tk_label(row, text, 13, "text" if number in self.answers else "muted", "panel", font=font(13, serif=number in self.answers), wraplength=640)  # serif is for model text only
            label.pack(side="left")
            self.answer_labels[number] = label
        both = len(trained) == 2
        self.ask_btn = button(card, "Ask both versions" if both else f"Ask {version_name(next(iter(trained)))}", self.ask_both)
        self.ask_btn.pack(anchor="w", pady=(6, 0))

    def _notebook(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w")
        Kicker(head, T["notebook"], bg="panel").pack(side="left")
        info_icon(head, T["notebook_tip"]).pack(side="left", padx=(4, 0))
        self.notebook_box = tk.Frame(card, bg=C["panel"])
        self.notebook_box.pack(anchor="w", fill="x", pady=(4, 0))
        self._paint_notebook()

    def _paint_notebook(self):
        box = getattr(self, "notebook_box", None)
        if box is None or not box.winfo_exists():
            return
        for child in box.winfo_children():
            child.destroy()
        self.notebook_texts = notebook_lines(self.notebook_rows)
        quiet = not self.notebook_rows
        for line in self.notebook_texts:
            tk_label(box, line, 13, "muted" if quiet else "text", "panel", wraplength=780, justify="left").pack(anchor="w", pady=1)

    def _look_inside(self):
        Kicker(self.body, T["look_inside"]).pack(anchor="w")
        row = tk.Frame(self.body, bg=C["bg"])
        row.pack(anchor="w", pady=6)
        self.links = {}
        for route, key in LINKS:
            link = tk_label(row, T[key], 14, "violet" if route not in ("compare", "quicktest") else "muted", cursor="hand2", takefocus=1)
            link.pack(side="left", padx=(0, 16))
            self.links[route] = link
            _activate(link, lambda r=route: self.navigate(r))

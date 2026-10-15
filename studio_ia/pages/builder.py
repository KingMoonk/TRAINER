"""Build your model: the picture is the editor. Draft edits live on this page until saved."""
from __future__ import annotations

import copy
import os
import queue
import threading
import time
import tkinter as tk
import uuid
from pathlib import Path

from studio_spec import config_from_spec, default_spec, validate_spec
from studio_versions import _facts, spec_diff
from studio_warmstart import NOTES
from studio_ia import pieces as cut
from studio_ia import threads
from studio_ia.copy import BUILDER as T
from studio_ia.expectation import expectation
from studio_ia.pages.base import Page
from studio_ia.services import Job, if_added, plain_error
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, Chip, Kicker, Progress, Segmented, StateTray, _activate, button, info_icon, tip

FF = (2, 4, 8)  # thinking-room width as a multiple of d_model: narrow / normal / wide
TOKENIZERS = ("byte", "bpe_custom", "bpe_gpt2", "word")
VS = (256, 8192, 50257, 8192)  # word-list sizes behind letters / small / large / whole words (the cap; less on little data)
WORDS = ("Letters", "Small word list", "Large word list", "Whole words")
WIDTHS = ("Narrow", "Normal", "Wide")
HEADS = (1, 4, 8)
PROMPT = "The largest planet in our solar system is"
STEPS, ROWS = 1000, 16  # a catch-up job here; the Pretraining page is where length, rows and data are chosen
LINKS = (("trace", "trace"), ("head_zoom", "inside_head"), ("value_map", "value_map"), ("loss", "loss"),
         ("head_lab", "head_lab"), ("inside3d", "3d"), ("compare", "compare_runs"), ("quicktest", "quick_test"))
EXPECT = dict(
    more="More blocks: each word gets more rounds of processing, and every training step takes longer. With little data it may memorize rather than learn; watch the test line.",
    fewer="Fewer blocks: smaller and faster. It may not get its error as low.",
    heads="Different look-back heads: same size, same speed. Each block splits its looking into {how} views. Whether that helps at this size is something to test, not assume.",
    wider="More thinking room: more to learn in every block, so bigger and slower.",
    narrower="Less thinking room: smaller and faster, with less capacity in every block.",
    letters="Letters: the word table becomes tiny, but the model has to learn spelling, and your text turns into about 5 times as many pieces, so each step covers less text.",
    small="Small word list: built from your own data. A much smaller table; text splits into somewhat more pieces.",
    large="Large word list: GPT-2’s list. A big table, short sequences.",
    words="Whole words: one entry per frequent word in your own data. Short sequences, but a word that is not in the list is lost.",
)
WARM = dict(  # kind -> (chip, colour); the sentence itself is studio_warmstart.NOTES[kind]
    same=("weights carry over exactly", "live"), heads=("weights carry over, not exactly", "attention"),
    removed=("weights carry over, minus what was removed", "attention"), scratch=("starts from scratch", "needs_training"))
TRAINING_TEXT = "Chat switches to {new} only when it beats version {n} on held-back text, or when you pick it."
NO_PARENT = "Version {n} has no trained weights to start from, so the next version trains from scratch."
NO_CURVE = "No error curve yet: version {n} has not trained on this computer. It appears here once it has."


def cfg(s):
    """s = dict(L, H, W, V, d, T) -> sizes for the cost formulas."""
    return dict(V=VS[s["V"]], d=s["d"], T=s["T"], L=s["L"], H=s["H"], ff=FF[s["W"]] * s["d"])


def P(c):
    """Parameters."""
    return c["V"] * c["d"] + c["T"] * c["d"] + c["L"] * (4 * c["d"] ** 2 + 4 * c["d"] + 2 * c["d"] * c["ff"] + c["ff"] + c["d"] + 4 * c["d"]) + 2 * c["d"]


def C_(c):
    """Training time per step: 6·(L·(4d²+2d·ff) + V·d) + 6·L·T·d."""
    return 6 * (c["L"] * (4 * c["d"] ** 2 + 2 * c["d"] * c["ff"]) + c["V"] * c["d"]) + 6 * c["L"] * c["T"] * c["d"]


def TM(c):
    """Memory while training: 16 bytes per parameter (fp32 weights, gradients, Adam m and v)."""
    return 16 * P(c)


def activations(c):
    return 2 * c["L"] * c["d"] * 4 * c["T"]


def say(x, y, more, less):
    r = y / x
    if abs(r - 1) < 0.05:
        return T["about_same"]
    return T["about_multiplier"].format(mult=f"{r if r > 1 else 1 / r:.1f}") + " " + (more if r > 1 else less)


def bar(x, y):
    m = max(x, y)
    return round(x / m * 100), round(y / m * 100)


def read_spec(spec):
    """ModelSpec -> the four editable choices plus d and T."""
    f = _facts(spec)
    word = {"letters": 0, "small word list": 1, "large word list": 2, "whole words": 3}.get(f["words"], 1)
    return dict(L=f["blocks"], H=f["heads"] or 4, W=0 if f["room"] < 4 else 1 if f["room"] == 4 else 2, V=word,
                d=f["width"], T=f["context"])


def write_spec(base, s, was=None):
    """`base` with only the choices that differ between `was` (default: base's own) and `s` rewritten."""
    spec = copy.deepcopy(validate_spec(base))
    was = was or read_spec(spec)
    if s["L"] != was["L"]:
        spec["dims"]["n_layer"] = s["L"]
    if s["H"] != was["H"]:
        spec["block"]["mixer"]["config"]["heads"] = s["H"]
    if s["W"] != was["W"]:
        spec["block"]["channel"]["config"]["d_ff"] = FF[s["W"]] * s["d"]
    if s["V"] != was["V"]:
        spec["tokenizer"] = {"id": TOKENIZERS[s["V"]], "config": {}}
    return validate_spec(spec)


def warm_kind(cur, draft):
    """How the old weights carry into the draft: scratch | heads | removed | same (the studio_warmstart rules)."""
    if draft["V"] != cur["V"]:
        return "scratch"
    if draft["H"] != cur["H"]:
        return "heads"
    if draft["L"] < cur["L"] or draft["W"] < cur["W"]:
        return "removed"
    return "same"


def warm_outlook(cur, draft):
    """None when nothing changed, else {kind, carries, preserving, chip, state, text} for the cost panel."""
    if all(cur[k] == draft[k] for k in ("L", "H", "W", "V")):
        return None
    kind = warm_kind(cur, draft)
    chip, state = WARM[kind]
    return dict(kind=kind, carries=kind != "scratch", preserving=kind == "same", chip=chip, state=state, text=NOTES[kind])


def fits(c, ram=None):
    """True/False: weights, gradients, optimizer state and activations for 16 rows fit in physical memory; None if unknown."""
    if ram is None:
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return None
    return TM(c) + ROWS * activations(c) <= ram


def last_blocks_change(versions):
    """(from, to, v_from, v_to, bpc_moved|None) for the latest block-count change in the history, else None.
    The size of the move is measured from the two versions' held-back scores, or None when either is missing."""
    vs = [v for v in versions if v.spec]
    for i in range(len(vs) - 1, 0, -1):
        a, b = _facts(vs[i - 1].spec)["blocks"], _facts(vs[i].spec)["blocks"]
        if a != b:
            x, y = vs[i - 1].scorecard.get("bpc"), vs[i].scorecard.get("bpc")
            moved = f"{abs(y - x):.2f}" if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
            return a, b, vs[i - 1].number, vs[i].number, moved
    return None


def design_expectation(s):
    """(headline, details) for the choices in `s`: what a design this size and word list can be expected to write."""
    c = cfg(s)
    return expectation(P(c), TOKENIZERS[s["V"]], embedding_params=c["V"] * c["d"])


def expectations(cur, draft, versions=()):
    a, b, out = cur, draft, []
    if b["L"] > a["L"]:
        out.append(EXPECT["more"])
    if b["L"] < a["L"]:
        out.append(EXPECT["fewer"])
    if b["L"] != a["L"] and (last := last_blocks_change(versions)):
        out.append(f"Last time blocks changed ({last[0]} → {last[1]}, version {last[2]} → {last[3]}), "
                   + (f"error moved by {last[4]} bits per character." if last[4] else "but one of them has no held-back score yet."))
    if b["H"] != a["H"]:
        out.append(EXPECT["heads"].format(how="more, narrower" if b["H"] > a["H"] else "fewer, wider"))
    if b["W"] > a["W"]:
        out.append(EXPECT["wider"])
    if b["W"] < a["W"]:
        out.append(EXPECT["narrower"])
    if b["V"] != a["V"]:
        out.append(EXPECT[("letters", "small", "large", "words")[b["V"]]])
    return out or [T["change_something"]]


MISSING = {1: "Built from your text: add some in Your data to see these pieces.",
           2: "GPT-2’s list could not be downloaded. It is fetched the first time you train with it.",
           3: "Built from your text: add some in Your data to see these pieces."}


def pieces_for(v, lists):
    """([(label, state)], count line) for the example sentence under word list v. `lists` = services.word_lists(), or None
    while it is still being read. The count line puts every list's count side by side, in the order of the buttons."""
    if lists is None:
        lists = {"tokenizers": (cut.Tokenizer("byte"), None, None, None), "sentence": cut.EXAMPLE}
        note = "Reading your text…"
    else:
        note = MISSING.get(v)
    line = " · ".join("–" if n is None else f"{n:,}" for n in cut.counts(lists)) + " pieces"
    tok = lists["tokenizers"][v]
    if tok is None:
        return [(note, "neutral")], line
    chips = [(label, "pieces" if known else "attention") for label, known in cut.chips(tok, lists["sentence"])]
    lost = sum(state == "attention" for _, state in chips)
    if lost:
        line += f", {lost} word{'s' if lost > 1 else ''} lost to “unknown”"
    return chips, line


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


def pill_text(job, base):
    """Header pill: what is training now, else where the current version stands."""
    if job:
        n = job.params.get("version", base.number if base else 0)
        if not n:
            return "Training (no version yet)"
        scratch = job.params.get("mode") == "from_scratch"
        return f"Version {n} {'is training' if scratch or job.kind != 'pretrain' else 'is catching up'}"
    return f"Version {base.number} is {base.status}" if base else "No saved version yet"


def catchup_job(job_id, number, spec, mode, parent_checkpoint, datasets):
    """The pretrain job for a saved version, in the shape the Pretraining page queues."""
    cfg_ = {**config_from_spec(spec), "steps": STEPS, "batch_size": ROWS, "validation": .1}
    scratch = mode == "from_scratch"
    return Job(job_id, "pretrain", f"Version {number} {'from scratch' if scratch else 'catch-up'}",
               params={"config": cfg_, "datasets": list(datasets), "mode": mode, "init_from": None if scratch else parent_checkpoint,
                       "output_dir": None, "user_text": False, "version": number})


def queued_outlook(services, job_id):
    """When the queued job `job_id` starts (services.queue_outlook), or "" once it has left the queue."""
    q = services.queue()
    job = next((j for j in [q.job, *q.up_next] if j is not None and j.id == job_id), None)
    return services.queue_outlook(job) if job else ""


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
        self.plot_args, self.plot = {}, None
        self._chart_token = None
        self.answers, self.answer_q, self.answer_stop, self.answer_labels, self.workers = {}, queue.Queue(), threading.Event(), {}, []
        self.pill_bar = self.pill_label = None
        self._curve_at = 0.

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
        self.load_chart()
        self.render()
        self.load_lists()

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
        base = self.base
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

    def spec_for(self, s):
        if self.anchor:
            return write_spec(self.anchor[0], s, self.anchor[1])
        return write_spec(self.base_spec(), s, self.cur)

    def changes(self):
        try:
            return spec_diff(self.base_spec(), self.spec_for(self.draft))
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

    def last_blocks_change(self):
        return last_blocks_change(self.services.versions())

    def expectations(self):
        return expectations(self.cur, self.draft, self.services.versions())

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

    def save(self):
        try:
            spec = self.spec_for(self.draft)
        except ValueError as error:
            self.note = str(error)
            return self.render()
        self.saved = self.services.save_version(spec, ", ".join(self.changes()) or ("first design" if self.base is None else "no change"))
        self.phase, self._staged, self.anchor = "saved", None, None
        self.services.stage_draft(None)
        self.load_chart()
        self.render()

    def queue_it(self, from_scratch):
        self.slow(lambda: text_sets(self.services.data("text")), lambda sets: self._queue(sets, from_scratch))

    def _queue(self, sets, from_scratch):
        if not sets:
            self.note = "There is no plain text to train on yet. Add some in Your data, then queue it."
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
        self._controls(right)
        self._costs()
        self._expect()
        self._tray()
        self._charts()
        self._look_inside()
        self._paint_design()

    def _paint_design(self):
        """Everything the draft decides, repainted in place: rebuilding the page on each setting flashed it."""
        self._paint_tower()
        for key in self.segs:
            self._paint_seg(key)
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
        box = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, highlightbackground=C["border"])
        box.pack(fill="x")
        tip(tk_label(box, T["next_word"], 15, "text", "panel2", font=font(15, bold=True)), T["next_word_desc"]).pack(anchor="w", padx=12, pady=8)
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        self.block_box = tk.Frame(parent, bg=C["bg"])
        self.block_box.pack(fill="x")
        self.block_rows = []  # (row, name, heads, width, tag) per block, block 1 first; drawn top-down
        step = tk.Frame(parent, bg=C["bg"])
        step.pack(anchor="w", pady=8)
        tip(tk_label(step, f"{T['blocks']}  ", 14, "text", "bg"), T["blocks_desc"]).pack(side="left")
        self.minus = button(step, "−", lambda: self.edit(L=max(1, self.draft["L"] - 1)))
        self.plus = button(step, "+", lambda: self.edit(L=min(12, self.draft["L"] + 1)))
        self.minus.pack(side="left")
        self.blocks_label = tk_label(step, "", 18, "text", "bg", width=3, anchor="center")
        self.blocks_label.pack(side="left")
        self.plus.pack(side="left")
        self.blocks_was = tk_label(step, "", 13, "muted", "bg")
        self.blocks_was.pack(side="left")
        for b in (self.minus, self.plus):
            b.state(["!disabled"] if self.phase == "edit" else ["disabled"])
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        text = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, highlightbackground=C["border"])
        text.pack(fill="x")
        head = tk.Frame(text, bg=C["panel2"])
        head.pack(fill="x", padx=12, pady=(8, 0))
        tk_label(head, T["your_text_title"], 14, "text", "panel2", font=font(14, bold=True)).pack(side="left")
        self.pieces_count = tk_label(head, "", 12, "muted", "panel2")
        self.pieces_count.pack(side="left")
        self._seg(text, list(WORDS), "V", (0, 1, 2, 3), "panel2")
        self.pieces_box = tk.Frame(text, bg=C["panel2"])
        self.pieces_box.pack(anchor="w", fill="x", padx=12, pady=(0, 8))
        self.pieces_shown = None

    def _paint_tower(self):
        cur, dr, rows = self.cur, self.draft, self.block_rows
        while len(rows) < max(cur["L"], dr["L"]):  # a new block goes on top
            row = tk.Frame(self.block_box, highlightthickness=1)
            row.pack(fill="x", pady=2, **({"before": rows[-1][0]} if rows else {}))
            name = tk_label(row, f"Block {len(rows) + 1}", 13, width=9)
            name.pack(side="left", padx=8, pady=8)
            heads = tk_label(row, "", 13, "pink")
            heads.pack(side="left")
            width = tk.Frame(row, bg=C["blue"], height=8)
            width.pack(side="left", padx=10)
            tag = tk_label(row, "", 12, "amber")
            tag.pack(side="right", padx=8)
            rows.append((row, name, heads, width, tag))
        while len(rows) > max(cur["L"], dr["L"]):
            rows.pop()[0].destroy()
        changed = dr["H"] != cur["H"] or dr["W"] != cur["W"]
        for i, (row, name, heads, width, tag) in enumerate(rows):
            added, removed = i >= cur["L"], i >= dr["L"]
            shown, bg = cur if removed else dr, C["amber_bg"] if added else C["panel2"]
            row.configure(bg=bg, highlightbackground=C["amber"] if added else C["border"])
            name.configure(fg=C["dim"] if removed else C["text"], bg=bg)
            heads.configure(text="●" * shown["H"], bg=bg)
            width.configure(width=[36, 72, 144][shown["W"]])
            tag.configure(text="new" if added else "removed" if removed else "changed" if changed else "", bg=bg)
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

    def _seg(self, parent, options, key, values, bg="panel"):
        label, value = dict(zip(values, options)), dict(zip(options, values))
        seg = Segmented(parent, options, None, lambda opt: self.edit(**{key: value[opt]}), bg=bg)
        seg.pack(anchor="w", padx=12 if bg == "panel2" else 0, pady=4)
        self.segs[key] = seg, label

    def _paint_seg(self, key):
        seg, label = self.segs[key]
        seg.set(label.get(self.draft[key]))
        # ponytail: Tk borders cannot be dashed, so the saved choice gets a violet outline and a "saved" in its label.
        was = label.get(self.cur[key]) if self.cur[key] != self.draft[key] else None
        for opt, b in seg.buttons.items():
            b.configure(highlightbackground=C["violet"] if opt == was else C["border"], text=f"{opt} · saved" if opt == was else opt)

    def _controls(self, parent):
        for name, hint, color, opts, key, values in (
                ("Look-back heads", T["look_back_heads_tip"], "pink", [str(h) for h in HEADS], "H", HEADS),
                ("Thinking room", T["thinking_room_tip"], "blue", list(WIDTHS), "W", (0, 1, 2))):
            card = Card(parent)
            card.pack(fill="x", pady=(0, 8))
            head = tk.Frame(card, bg=C["panel"])
            head.pack(anchor="w")
            tk_label(head, name, 15, color, "panel", font=font(15, bold=True)).pack(side="left")
            info_icon(head, hint).pack(side="left", padx=(4, 0))
            self._seg(card, opts, key, values)

    def _costs(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["what_costs"], bg="panel").pack(anchor="w")
        self.cost_rows = []
        for name, f, more, less in ((T["size_label"], P, T["cost_bigger"], T["cost_smaller"]),
                                    (T["time_per_step"], C_, T["cost_slower"], T["cost_faster"]),
                                    (T["memory_while_training"], TM, T["cost_more"], T["cost_less"])):
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", pady=4)
            tk_label(row, name, 14, "text", "panel", width=24).pack(side="left")
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

    def _paint_costs(self):
        a, b = cfg(self.cur), cfg(self.draft)
        self.cost_say = {}
        for name, f, more, less, now, new, said in self.cost_rows:
            x, y = f(a), f(b)
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
        self.guide_text, self.guide_details = design_expectation(self.draft)
        self.expect_texts = self.expectations()
        self.hover_text = "\n\n".join([self.guide_details, *self.expect_texts])  # the card itself is one line
        self.guide_label.configure(text=self.guide_text)

    def _tray(self):
        self.tray_box = tk.Frame(self.body, bg=C["bg"])
        self.tray_box.pack(fill="x", pady=(0, GUTTER))
        self.tray = StateTray(self.tray_box)
        self.tray.pack(fill="x", pady=(0, 8))
        self.chips = tk.Frame(self.tray_box, bg=C["bg"])
        self.buttons, self.actions = None, {}

    def _paint_tray(self):
        ph, n, nv = self.phase, self.n, self.nv
        Nv = nv[0].upper() + nv[1:]
        scratch = self.needs_scratch()
        self.chip_texts = []
        actions = []
        if ph == "edit":
            self.chip_texts = self.changes()
            if self.chip_texts:
                state, kicker, text = "unsaved", T["not_saved"], T["not_saved_text"].format(version=nv)
                actions = [(T["quick_test_first"], lambda: self.navigate("quicktest"), False),
                           (f"Save as {nv}", self.save, True), (T["discard"], self.discard, False)]
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
            if self.position():
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
            self.buttons, self.actions = tk.Frame(self.tray_box, bg=C["bg"]), {}
            self.buttons.pack(anchor="w")
            for label, cmd, primary in actions:
                self.actions[label] = btn = button(self.buttons, label, cmd, primary)
                btn.pack(side="left", padx=(0, 8))

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

"""The one façade pages talk to. `Services` is the protocol; `FakeServices` is an in-memory stand-in."""
from __future__ import annotations

import errno
import io
import logging
import queue
import socket
import tempfile
import threading
import time
import urllib.error
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

import studio_ui_guard as ui_guard
from studio_timing import timed
from studio_versions import NoCheckpoint


@dataclass
class Version:
    number: int
    name: str = ""
    change: str = ""
    parent: int | None = None
    spec: dict = field(default_factory=dict)
    status: str = "draft"  # draft | saved | queued | training | trained
    best_checkpoint: str | None = None
    scorecard: dict = field(default_factory=dict)
    verdict: str = ""
    prediction: dict = field(default_factory=dict)  # the guess made on saving: {"error": "lower", "step": "slower", "memory": "more"}
    train_device: str | None = None  # cuda | mps | cpu: what trained best_checkpoint
    score_device: str | None = None  # what scored it on the held-back text


@dataclass
class Job:
    id: str
    kind: str  # pretrain | quicktest | compare | post_round | restore_answers | checkpoint_now | import_pretrained
    label: str = ""
    note: str = ""
    params: dict = field(default_factory=dict)
    status: str = "waiting"
    eta_s: float | None = None
    created_at: float = 0.0


def _open_gate():
    """A Gate that lets training run; imported late because studio_queue pulls in the heavy modules."""
    from studio_queue import Gate
    return Gate()


@dataclass
class QueueState:
    job: Job | None = None  # the running job, None when idle
    step: int = 0
    total: int = 0
    phase: str = ""  # warm-up | steady | cool-down
    lr: float | None = None
    eta_s: float | None = None
    paused: bool = False  # by the user; stays true with no job running ("Paused by you")
    blocked: str | None = None  # == gate.reason (why the plan forbids training right now)
    gate: Any = field(default_factory=_open_gate)  # studio_queue.Gate: reason, resumes_at, supported
    stopping: bool = False  # stop after cool-down in progress
    up_next: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    failed: list = field(default_factory=list)  # jobs that ended failed, newest last; see studio_queue.QueueState


_log = logging.getLogger(__name__)


def plain_error(error, name=None):
    """One sentence for what went wrong and what to do next. `name` is what was being worked on (a source, a file), for the
    sentences that mention it. The one place pages and background work turn an exception into words; the log keeps the rest."""
    import studio_catalog as cat
    what = name or "that file"
    if isinstance(error, urllib.error.HTTPError):
        return f"The server returned {error.code} for {what}. The file may have moved; try again later."
    if isinstance(error, (urllib.error.URLError, socket.timeout, ConnectionError)):
        return "Could not reach the download server. Check the connection and try again."
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "The disk is full. Free space in Resources › Storage, then try again."
    if isinstance(error, MemoryError):
        return f"Ran out of memory while reading {what}."
    if isinstance(error, (cat.DownloadCancelled, ValueError)) and str(error):
        return str(error)
    _log.error("Unexpected %s%s", type(error).__name__, f" while working on {name}" if name else "", exc_info=error)
    return f"Something went wrong: {type(error).__name__}. The log has the details."


def problem_items(rule, n, rng):
    """n rule-made problems; rule None/"" spreads n evenly over every rule."""
    import studio_rules
    names = studio_rules.RULES if not rule else (rule,)
    return [i for j, r in enumerate(names) for i in studio_rules.generate(r, rng, n // len(names) + (j < n % len(names)))]


def problem_report(items, added=0, duplicates=0):
    """{items, added, duplicates, checked, passed}: checked = items with a canonical answer, passed = those the rule accepts."""
    import studio_rules
    with_answer = [i for i in items if "answer" in i]
    return {"items": items, "added": added, "duplicates": duplicates, "checked": len(with_answer),
            "passed": sum(studio_rules.check(i["rule"], i["prompt"], i["answer"]) for i in with_answer)}


def start_phrase(opens, now):
    """"tonight at 22:00", "today at 09:00" or "tomorrow at 09:00" for a Gate.resumes_at; "" when there is none."""
    if opens is None:
        return ""
    day = "tomorrow" if opens.date() > now.date() else "tonight" if opens.hour >= 18 else "today"
    return f"{day} at {opens:%H:%M}"


def plan_phrase(plan):
    """" (plan: nights only)" / " (plan: 21:00 to 06:00)" for a plan dict whose hours are limited; "" for any time."""
    hours = plan.get("hours", "any")
    if hours == "nights":
        return " (plan: nights only)"
    if hours == "set":
        return f" (plan: {plan.get('start_hour', 22):02d}:00 to {plan.get('end_hour', 7):02d}:00)"
    return ""


def if_added(outlook):
    """A queue_outlook() for a job not yet added, as the line beside its "Add to the queue" button: "If you add it: starts now"."""
    return f"If you add it: {outlook[:1].lower()}{outlook[1:]}"


def queued_outlook(services, job_id):
    """When the queued job `job_id` starts (services.queue_outlook), or "" once it has left the queue."""
    q = services.queue()
    job = next((j for j in [q.job, *q.up_next] if j is not None and j.id == job_id), None)
    return services.queue_outlook(job) if job else ""


class Services:
    """Synchronous façade; long work runs through the queue. Subclasses override; scopes add methods, never rename."""

    workspace: Any = None

    def __init__(self):
        self._subscribers: list[Callable[[str, Any], None]] = []
        self._ui = None  # a Tk widget; set by the shell so events hop onto the Tk thread
        self._ui_thread, self._events = None, queue.SimpleQueue()  # events emitted on other threads, for pump()
        self._raw: list[Callable[[str, Any], None]] = []  # watchers that take events on the emitting thread (the API)
        self._raw_lock = threading.Lock()
        self._settings = {}
        from studio_ia.snapshots import Snapshots
        self.snapshots = Snapshots(self)  # what the Tk thread reads instead of calling slow services; inert until the shell starts it

    # events ---------------------------------------------------------------
    def bind_ui(self, widget):
        self._ui, self._ui_thread = widget, threading.get_ident()
        ui_guard.bind(self._ui_thread)  # the @slow primitives read it: one running on this thread is a freeze

    def on_ui_thread(self):
        return self._ui_thread is not None and threading.get_ident() == self._ui_thread

    def subscribe(self, callback):
        """callback(kind, payload) runs on the Tk thread. Returns an unsubscribe function."""
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback) if callback in self._subscribers else None

    def watch(self, callback):
        """callback(kind, payload) runs synchronously on whichever thread emits, never through Tk; safe to call from any
        thread. For code with no Tk loop (the API server). Keep the callback quick: it runs inside emit(), on the worker
        that emitted. Returns an unsubscribe function."""
        with self._raw_lock:
            self._raw.append(callback)

        def off():
            with self._raw_lock:
                if callback in self._raw:
                    self._raw.remove(callback)
        return off

    def emit(self, kind, payload=None):
        """Safe from any thread. On the Tk thread delivery is scheduled with `after`; another thread only queues the event
        for pump(), because a Tk call from a worker can reach an app that is already destroyed and crash the process.
        Watchers (watch()) are told first, right here on the emitting thread; the Tk path below is unchanged by them."""
        with self._raw_lock:
            raw = list(self._raw)
        for callback in raw:
            callback(kind, payload)

        def deliver():
            for callback in list(self._subscribers):
                callback(kind, payload)
        if self._ui is None:
            deliver()
        elif self.on_ui_thread():
            self._ui.after(0, deliver)
        else:
            self._events.put((kind, payload))

    def pump(self):
        """Tk thread: deliver the events other threads emitted since the last call, in order."""
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                return
            for callback in list(self._subscribers):
                callback(kind, payload)

    # slow reads: what one is doing right now, polled by the shell's loading panel ---------------------------------
    loading = ("", None)

    def note(self, text, fraction=None):
        """Safe from any thread. fraction None = no measurable progress yet."""
        self.loading = (text, fraction)

    def ready(self, work):
        """True when work() (service reads only) will not keep the Tk thread waiting; otherwise the shell hands it to its worker."""
        return True

    # staged design change: the Network designer's unsaved draft, read by Quick test and Compare runs ----------------
    _staged = None

    def stage_draft(self, spec):
        """Builder publishes its unsaved draft spec (None = no change staged); other pages may also set it to reopen a design."""
        self._staged = spec

    def draft_spec(self):
        return self._staged

    # versions (S3) --------------------------------------------------------
    def versions(self) -> list:
        raise NotImplementedError

    def current_version(self):
        raise NotImplementedError

    def best_version(self):
        raise NotImplementedError

    def checkpoint_step(self, version):
        """Training step at which the version's best checkpoint was saved, or None when unknown."""
        return None

    def run_facts(self, number):
        """What version `number`'s training run did, to tell whether two versions are a fair comparison:
        {data, steps, tokens, start, lr, extra} (a fact the files do not give is left out), or {} when the run is not on this computer."""
        return {}

    def compare_runs(self):
        """The saved Compare runs: [{base, variants, mode, results, finished}] (specs as saved, results.json of each and its time)."""
        return []

    def run_speed(self, number):
        """Measured seconds per training step of version `number`'s last run, or None when it has not logged one here."""
        return None

    def save_version(self, spec, change_text, prediction=None):
        """prediction: what the person guessed the change would do ({dimension: direction}), stored with the version."""
        raise NotImplementedError

    # sharing versions between machines (studio_publish; docs/cluster.md). All of it is disk or network: call it off the Tk thread.
    def version_store(self):
        """The version store this machine shares versions through (a studio_publish.RemoteStore). Raises RuntimeError with a plain
        sentence when none is set up, and ValueError naming what is missing when a bucket is half set."""
        raise NotImplementedError

    def publish_version(self, number):
        """Upload version `number`'s weights (half precision, no optimizer state) and metadata to the version store. Returns
        {key, number, origin}. Publishing the same weights again uploads nothing. Raises studio_publish.StoreError, a sentence that
        says what to do, when the version has no weights or the store cannot be used."""
        raise NotImplementedError

    def published_versions(self):
        """What the version store holds, newest first: [{key, number, name, change, origin, device, step, published_at, size, scorecard,
        mine (published by this machine), local (this workspace has it), local_number}]. Raises like publish_version."""
        raise NotImplementedError

    def pull_version(self, number, origin=None):
        """Download published version `number` (from the machine `origin` when more than one published a version with that number)
        and add it here as a trained version. Pulling what is already here returns that version. The node that published it is asked
        first (when the workspace lists it and it answers; given `origin`, the store is not read at all); otherwise, or when it
        cannot send it, the store is used. `note` says which, and why. Returns the Version."""
        raise NotImplementedError

    def share_version(self, number):
        """(path, metadata) of version `number`'s weights-only half-precision copy, as `publish_version` would upload it, for a peer
        that pulls it from this machine. Made on the first call, then reused until the version changes. Raises NoCheckpoint when the
        version has no weights, studio_publish.StoreError when its checkpoint cannot be read."""
        raise NotImplementedError

    def setting(self, key, default=None):
        """A choice the person made that the app remembers (the Builder's Advanced toggle). Read from memory: safe on Tk."""
        return self._settings.get(key, default)

    def set_setting(self, key, value):
        """Remember a choice. WorkspaceServices writes it to disk: call it off the Tk thread."""
        self._settings[key] = value

    # queue (S2) -----------------------------------------------------------
    def queue(self) -> QueueState:
        raise NotImplementedError

    def enqueue(self, job):
        raise NotImplementedError

    def move_up(self, job_id):
        raise NotImplementedError

    def queue_outlook(self, job=None, now=None) -> str:
        """One sentence on when `job` (a queued Job; None = a job added to the end of the queue now) will start, from queue():
        "Running now", "Held; release it to run", "Paused by you", "Waiting: on battery", "Starts tonight at 22:00 (plan: nights
        only)", "Starts after <job> (about N min)" or "Starts now". Every "add to the queue" surface shows this one text."""
        from datetime import datetime
        q, now = self.queue(), now or datetime.now()
        if job is not None and q.job is not None and job.id == q.job.id:
            return "Running now"
        if job is not None and getattr(job, "status", "waiting") == "held":
            return "Held; release it to run"
        if q.paused:
            return "Paused by you"
        reason = q.gate.reason or q.blocked
        if reason and reason != "outside hours":
            return f"Waiting: {reason}"
        names = [j.id for j in q.up_next]
        before = q.up_next[:names.index(job.id)] if job is not None and job.id in names else list(q.up_next)
        ahead = ([q.job] if q.job is not None else []) + [j for j in before if getattr(j, "status", "waiting") != "held"]
        etas = [q.eta_s if j is q.job else j.eta_s for j in ahead]
        wait = f" (about {max(1, round(sum(etas) / 60))} min)" if ahead and None not in etas else ""
        after = f"Starts after {ahead[-1].label or ahead[-1].kind}{wait}" if ahead else ""
        if reason:  # outside hours
            when = start_phrase(q.gate.resumes_at, now)
            plan = plan_phrase(q.plan)
            return (f"{after}; the queue opens {when}{plan}" if after else f"Starts {when}{plan}") if when else f"Waiting: {reason}"
        return after or "Starts now"

    @timed("services.next_step")
    def next_step(self):
        """The single "what next" answer (studio_ia.next_step.NextStep) from this workspace's data, versions and queue."""
        from studio_ia.next_step import next_step
        q = self.queue()
        first = next((j for j in q.up_next if getattr(j, "status", "waiting") != "held"), None)
        return next_step(self.has_text(), self.versions(), q, self.queue_outlook(first))

    def has_text(self):
        """Whether any text has been added. The sidebar asks at launch, so a store that has not been summarised yet must not be read."""
        return self.data("text").count > 0

    def next_step_if_ready(self):
        """next_step() when it will not keep the Tk thread waiting (ready()), else None: a page's empty state keeps its own words."""
        return self.next_step() if self.ready(self.next_step) else None

    def start_queue(self):
        """Let jobs saved as waiting by an earlier session run (no-op where there is no real queue)."""

    def resume_queue_at_start(self):
        """Carry on the queue the last run of the app left going (no-op where there is no real queue)."""

    def refresh_downloads(self):
        """Rebuild catalogue downloads converted by an older app version; returns the ids rebuilt (no-op without a workspace)."""
        return []

    def move(self, job_id, index):
        """Put a waiting job at `index` in Up next (0 = runs next)."""
        raise NotImplementedError

    def hold(self, job_id, held=True):
        """Hold a waiting job (it keeps its place and is skipped) or release it."""
        raise NotImplementedError

    def remove(self, job_id):
        raise NotImplementedError

    def pause(self):
        raise NotImplementedError

    def resume(self):
        raise NotImplementedError

    def checkpoint_now(self):
        raise NotImplementedError

    def stop_after_cooldown(self):
        raise NotImplementedError

    def set_plan(self, plan):
        raise NotImplementedError

    def storage(self, days=None, data=False):
        """(items, bytes) that clean_storage would delete. days=None: all history; data=True: your data too."""
        raise NotImplementedError

    def clean_storage(self, days=None, data=False):
        raise NotImplementedError

    def leftovers(self):
        """[{id, kind, name, belongs_to, bytes, shared}] of what training left in runs/ that no result needs: interrupted attempts,
        failed runs and dataset snapshots no run uses (studio_cleanup.leftovers). Never a running job's folder, the checkpoint a
        waiting job continues from or a version's runs."""
        raise NotImplementedError

    def remove_leftovers(self, ids):
        """Delete the listed leftovers (ids from leftovers(); one that is not listed any more is ignored). Returns (items, bytes)."""
        raise NotImplementedError

    def free_space(self) -> int:
        """Bytes free on the disk that holds the workspace (where checkpoints are written)."""
        raise NotImplementedError

    def export_version(self, version, kind, path):
        """Write a trained version's best checkpoint to `path`: kind "hf" (a folder: safetensors, config, tokenizer, README) or
        "gguf" (a .gguf file). Returns studio_export's report {path, files, note, ...}. Raises NoCheckpoint for a version that
        has not trained, and ValueError (already in plain words) for a design or tokenizer that format cannot hold."""
        raise NotImplementedError

    def import_pretrained(self, name="gpt2", lora=None):
        """Start from a pretrained open model ("gpt2": OpenAI's 124M, about 548 MB to download once). Saves its design as a new version
        and queues the download and conversion in the background; the version becomes trained (with a checkpoint) when the job ends.
        lora: settings dict ({} for the defaults) to fine-tune small adapters instead of every weight. Returns the new Version."""
        raise NotImplementedError

    def pretrain_preview(self, dataset_ids, rows=16):
        """{pieces, params, context, first_rows}: pieces = tokenised size of the chosen text, params = current
        version's parameter count, first_rows = first `rows` rows of token groups, each a list of example numbers
        with -1 where an example ends."""
        raise NotImplementedError

    def cached_preview(self, dataset_ids, rows=16):
        """pretrain_preview's answer if already known without work, else None. The default always knows it."""
        return self.pretrain_preview(dataset_ids, rows)

    def cached_step_rate(self, spec, rows):
        """The step rate measure_step_rate found for `spec` at `rows` rows a step on this computer, or None while it is unknown. Never
        computes. {"seconds_per_step", "tokens_per_second", "device", "device_name", "note", "rows", "context"}."""
        return None

    def measure_step_rate(self, spec, rows, vocab, stop=None):
        """Time a few optimizer steps of a throwaway copy of `spec` (`vocab` word pieces) on the device a run of it would use, keep the
        answer (cached_step_rate) and return it. Slow: call it off the Tk thread. RuntimeError with a sentence when the device cannot
        hold the model. None when `stop` (a threading.Event) ended it."""
        raise NotImplementedError

    def loss_evidence(self, version):
        """The validation points of `version`'s newest run when that run started from step 0 on this design, {"points": [(word
        pieces read, validation loss)], "run": folder}, else None. What the plan fits a curve to."""
        return None

    def device_facts(self):
        """studio_backend.gpu_status(): {available, kind, name, ...} of the GPU this computer offers, or None when it cannot say."""
        return None

    def continue_run(self, number, kind, steps, dataset_ids=(), width=None, layers=None, spec=None, rows=16, hold=.05):
        """Queue a pretrain job that continues version `number`'s newest run (studio_ia.continue_run): kind "same" adds `steps` steps on
        the same data, "new_data" adds them on `dataset_ids` (same weights, optimizer and step), "grow" trains a bigger design
        (`spec`, or this one at `width` and `layers`, saved as the next version) from these weights for `steps` steps. The job's label
        says what it continues and from which step. Returns {"job_id", "label", "version", "kind", "from_step"}. Reads checkpoints: call it
        off the Tk thread. ValueError with a sentence when the run cannot be continued that way."""
        raise NotImplementedError

    def word_lists(self):
        """studio_ia.pieces.word_lists for the Builder: the four word lists, the small and whole-word ones fitted on a sample
        of your plain text, and the example sentence they cut. Slow the first time: pages ask for it on the slow lane."""
        raise NotImplementedError

    def lr_range_test(self, spec, stop=None):
        """studio_compute.lr_range_test for `spec` on a slice of the chosen text: a throwaway copy trained for about a hundred
        steps while the learning rate rises. Slow: call it off the Tk thread. `stop` (a threading.Event) ends it early."""
        raise NotImplementedError

    # data (S5) / evaluation (S4) / chat ------------------------------------
    def data(self, kind, query=""):
        """studio_data_kinds.DataSummary for "text" | "conversations" | "problems"; query filters the rows."""
        raise NotImplementedError

    def has_reviewed_text(self, dataset_ids):
        """True when a run that trains on reviewed examples only would have text to train on in these sets."""
        raise NotImplementedError

    def has_conversations(self, dataset_ids, reviewed_only=False, at_least=1):
        """True when these sets hold `at_least` different prompts a run's teacher can answer (only conversations marked Good with `reviewed_only`)."""
        raise NotImplementedError

    def text_sets(self):
        """[{id, name, count}] of the sets that hold plain text, the Pretraining page's tiles. Unlike data("text") it reads no rows."""
        return list(self.data("text").sets or [])

    def conversation_sets(self):
        """[{id, name, count}] of the sets that hold conversations: the places a cluster run's teacher can take its prompts from."""
        return list(self.data("conversations").sets or [])

    def data_page(self, kind, query="", cursor=None, limit=200, review_only=False):
        """(rows, next cursor) of one page of the kind's rows whose text holds `query`, read from storage so every row can be
        reached; cursor None = the first page, a next cursor of None = nothing follows. Rows are shaped like DataSummary.rows."""
        raise NotImplementedError

    def add_data(self, kind, fields):
        """fields: text -> {text}; conversations -> {user, assistant}; problems -> {prompt, rule}. True if added."""
        raise NotImplementedError

    def import_data(self, kind, paths):
        """Import files into the kind's own set. Returns one report line per file."""
        raise NotImplementedError

    def collection(self, kind="text"):
        """The recommended collection of one kind ("text" | "conversations" | "problems"): one dict per source
        {id, name, size, records, state, completed, total, error}; state is "not downloaded" | "downloading" |
        "downloaded" | "imported"."""
        raise NotImplementedError

    def download_sources(self, ids=None, kind="text", install=False):
        """Download the kind's whole recommended collection (ids None) or the listed sources, one after another, on a
        background thread; install=True also adds each one to your sets. Progress arrives as events "download"
        {id, text, completed, total}, then "download_done" {id} or "download_failed" {id, error}. False when a download
        is already running."""
        raise NotImplementedError

    def import_source(self, source_id):
        """Import one downloaded source into your sets (background thread; events "import_done" {id} or
        "download_failed" {id, error}). False when it is not downloaded or work is already running."""
        raise NotImplementedError

    def generate_problems(self, rule, n, store=True, rng=None):
        """Rule-made problems (rule None = all rules). store=False previews. Returns problem_report(...)."""
        raise NotImplementedError

    def make_drafts(self, kind, n):
        """Local-model drafts ("conversations" rewrites, "text" prose), stored with status Review. Returns how many.
        Raises RuntimeError with a plain message when no local model is reachable."""
        raise NotImplementedError

    def draft_count(self):
        """Drafts still waiting for review."""
        raise NotImplementedError

    def review_data(self, example_id, keep):
        """keep -> mark Good; otherwise delete the example."""
        raise NotImplementedError

    def data_fields(self, example_id):
        """The add_data fields of one stored example, to pre-fill an edit."""
        raise NotImplementedError

    def edit_data(self, example_id, fields):
        """Replace the add_data fields of one example. A draft stays a draft until marked Good. Raises ValueError with a
        plain message when the edit is empty, duplicates another example, or a problem no longer passes its rule."""
        raise NotImplementedError

    def references(self, version):
        raise NotImplementedError

    def gate_result(self, version):
        """How Chat weighed a version against the best version before it: {"better", "diff", "upper", "vs"} (diff and upper in
        bits per character, version minus the earlier best; vs its number), or None when it was not weighed."""
        raise NotImplementedError

    def run_curve(self, run_dir):
        raise NotImplementedError

    def run_phases(self, run_dir):
        """[(from_step, to_step, name)] of the run's schedule, from its spec alone: cheap, so the Tk thread may ask it."""
        raise NotImplementedError

    def chat(self, version, messages, on_text, stop, ablate=None):
        """ablate: optional [(block, head), ...] mean-ablated for this reply only."""
        raise NotImplementedError

    # inspect (S7): thin wrappers over studio_inspect, numpy out; version is a Version or number -----------------
    def inspect_forward(self, version, text, hooks=True):
        """{'ids' [1,T], 'tokens' [[str]], 'logits' [1,T,V], 'hooks' {name: array}} (studio_inspect.forward)."""
        raise NotImplementedError

    def inspect_logit_lens(self, version, text, top_k=5):
        """studio_inspect.logit_lens: one row per stage ('embed', 'blockN.resid.post'): {'hook', 'probabilities' [1,T,V], 'top_ids' [1,T,k], 'top_probabilities' [1,T,k]}."""
        raise NotImplementedError

    def inspect_ablate(self, version, text, targets=None, mode="zero", heads=None):
        """studio_inspect.ablate: {'baseline', 'mode', 'heads': [{'block','head','loss','delta', + head_stats}]}, one head ablated at a time (all of them when `heads` is None)."""
        raise NotImplementedError

    def inspect_head_stats(self, pattern):
        """studio_inspect.head_stats of an attention pattern [B,H,T,T]: {'entropy','mean_distance','first_token_mass','previous_token_mass'}, each [H]."""
        raise NotImplementedError

    def inspect_trace(self, version, text, pos=-1):
        """studio_inspect.trace: {'tokens', 'pos', 'rows'}."""
        raise NotImplementedError

    def inspect_prepared(self, version):
        """{'ids': [int], 'tokens': [str]}: studio_inspect.prepared_text with the version's tokenizer."""
        raise NotImplementedError

    def inspect_loss(self, version, text, pos=-1, trained=True):
        """{'tokens', 'pos', 'probabilities' [V], 'target' id, 'pieces' [str per id]}; trained=False uses fresh weights."""
        raise NotImplementedError

    def inspect_landscape(self, version, snapshot=None, mode="random", grid=21, span=1.0, seed=0, live=False, new_anchor=False, stop=None):
        """The loss on a grid of weights around a version's weights (studio_landscape): {mode, unit, grid, span, seed, axis [grid], loss
        [grid][grid] (rows are the second direction), ceiling, current {x, y, loss, step}, minimum {x, y, loss, inside}, trajectory
        [{step, x, y, loss}], center {step, name}, batch {rows, tokens}, cost {seconds, evaluations}, note, live}. snapshot picks the
        centre from viewer_snapshots (default: the best checkpoint). live is a small, throttled frame with the same directions as the
        last one (new_anchor starts them afresh). ValueError with a sentence when it cannot run. Slow: call it off the Tk thread."""
        raise NotImplementedError

    def inspect_value_map(self, version, heads, block=0, head=0):
        """{'d_model', 'heads' (the real count), 'blocks' {block: heads}, 'singular' list | None}: singular values of W_V^h W_O^h when heads matches."""
        raise NotImplementedError

    def inspect_rank_heads(self, version, mode="mean"):
        """studio_inspect.rank_heads on a fixed held-back batch: [{'block','head','rise'}]. Slow; call off the UI thread."""
        raise NotImplementedError

    def inspect_heads_off(self, version, off, mode="mean"):
        """Held-back bits/char with the (block, head) pairs in `off` switched off (mode mean | zero): {'before', 'after'}."""
        raise NotImplementedError

    def inspect_reply(self, version, prompt, off=(), mode="mean"):
        """Greedy answer to `prompt` with the (block, head) pairs in `off` switched off for this reply only."""
        raise NotImplementedError

    def teach_correction(self, version, messages, correction, whole_answer=False):
        """Save a correction as a Post-training round 3 item (teacher sees it, student does not); whole_answer=True saves a round 1 conversation instead."""
        raise NotImplementedError

    def rate_pair(self, messages, chosen, rejected):
        """Save a rated pair for round 4; chosen/rejected = {"version": n, "reply": str}."""
        raise NotImplementedError

    def verdict(self, a, b):
        """Training saved between versions a and b (numbers): studio_eval.training_saved result dict."""
        raise NotImplementedError

    def few_shot(self, version):
        """Lettered problems answered with 0, 1 and a few worked examples in the prompt and no training: studio_eval.few_shot_table's
        dict plus "tasks" and "held_back" (False when no held-back choice problem exists and training ones are asked). Slow."""
        raise NotImplementedError

    def mlperf_board(self, task_id) -> dict:
        """The Training speed section: {"tasks": [{"id", "version", "title", "target_bpc", "checksum"}], "board": studio_mlperf.leaderboard,
        "history": studio_mlperf.history for the single-device division, "devices": the choices a run can name}. Reads stored results only."""
        raise NotImplementedError

    def mlperf_compare(self, ids) -> dict:
        """Side by side summaries of stored results (studio_mlperf.compare); ValueError when they are of different task versions."""
        raise NotImplementedError

    def mlperf_enqueue(self, task_id, device, attempts=None) -> dict:
        """Queue a training-speed run (job kind "mlperf") on this machine; {"id", "label"}. It waits its turn behind any training job."""
        raise NotImplementedError

    def benchmark(self) -> dict:
        """The benchmark track as stored, never computing: {"suite" id, "sizes" {docs, choice, problems}, "dropped" {kind: n filtered
        out}, "rows": [{"number", "name", "lineage", "tokens", "result" (studio_bench.run_suite's dict, None until it has run)}]},
        one row per trained version. Reads the held-back data: call it off the Tk thread."""
        raise NotImplementedError

    def run_benchmark(self, number):
        """Run (or read back) the benchmark suite on one trained version: studio_bench.run_suite's dict. Slow."""
        raise NotImplementedError

    def post_training(self, version) -> dict:
        """{"rounds": {1..4: {"status": done|running|queued|idle, "note": str}}, "counts": {"conversations", "corrections",
        "problems", "pairs"}, "gates": {round: {"ok", "checks", "verdict", "judge": (k, n), "taught", "groups"}},
        "overlap": {"text" | "problems": {"checked", "left_out", "note"}}}; missing keys are fine. overlap counts the held-back items
        left out because a 13-gram of them also appears in training (studio_decontam)."""
        raise NotImplementedError

    def post_rows(self, kind):
        """Rows for a post_round job: conversations | problems | pairs | heldout_docs | heldout_problems."""
        raise NotImplementedError

    def post_loops(self) -> list:
        """The automated RLHF loops (studio_loop.summary dicts), newest first: id, version, status, stop_reason, config, anchor, current,
        kept, rounds [the journal's records], and so on."""
        raise NotImplementedError

    def start_loop(self, version, config=None, resume=None, judge_model=None) -> dict:
        """Queue an `rlhf_loop` job for `version` (None = the current one): {"job", "loop_id", "label"}. `judge_model` is the Ollama model
        that grades (None = the workspace's teacher, or no judge: checked problems only). `resume` is the id of an earlier
        loop to carry on, from its last kept round. Raises ValueError with a sentence when it cannot start."""
        raise NotImplementedError

    def adopt_loop(self, loop_id) -> dict:
        """Check a loop's best on all the held-back data and, if the gate passes, make it the version's checkpoint: the gate result."""
        raise NotImplementedError

    # 3D viewer (S11): a separate Panda3D process; Services only starts, stops and talks to it -------------------
    def viewer_status(self) -> dict:
        """{'available': bool, 'running': bool, 'connected': bool}. available False = the viewer cannot be launched here."""
        raise NotImplementedError

    def viewer_start(self, scene=None):
        """Launch the viewer (no-op if running), send the current version's spec and `scene` ({'level','mode'}) when given."""
        raise NotImplementedError

    def viewer_stop(self):
        raise NotImplementedError

    def viewer_send(self, message):
        """Queue one dict message for a running viewer (e.g. {'type': 'scene', 'level': 'Window', 'mode': 'Train'})."""
        raise NotImplementedError

    def viewer_poll(self) -> list:
        """Messages the viewer sent back since the last call."""
        raise NotImplementedError

    def viewer_snapshots(self, number=None) -> list:
        """A version's training snapshots, oldest first; omit number for the current version."""
        raise NotImplementedError

    def viewer_probe_step(self, text, number=None):
        """Run one real training step of the chosen (default current) version's checkpoint without touching training, save it
        as a snapshot and return its path; None when the version has no checkpoint. Slow: call it off the Tk thread."""
        raise NotImplementedError

    def viewer_token_text(self, number, ids):
        """The text of each token id (a list of strings, '' for a piece with no text), for the viewer's labels; None when the version
        has no tokenizer yet. A small file read."""
        return None

    def viewer_summary(self, number=None) -> tuple:
        """(state, summary) of a version's saved whole-context summary (studio_summary.summary_state): state is ready, stale, none or
        no_checkpoint; summary is {edges, n_ctx, step, attention, stream_norms} when ready or stale, else None. A file read."""
        from studio_summary import summary_state
        return summary_state(self._summary_checkpoint(number))

    def viewer_summarize(self, number=None) -> dict:
        """Run the chosen (default current) version's checkpoint over a full window of its held-back text, save the summary beside
        the run's snapshots and return it (studio_summary.whole_context). ValueError with a sentence when there is no checkpoint or
        no held-back text to read. Slow: call it off the Tk thread."""
        from pathlib import Path

        from studio_ia.copy import INSIDE3D
        from studio_summary import save_summary, whole_context
        checkpoint = self._summary_checkpoint(number)
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError(INSIDE3D["summary_no_checkpoint"])
        summary = whole_context(checkpoint, self._heldout())
        save_summary(checkpoint, summary)
        return summary

    def _summary_checkpoint(self, number):
        version = self.current_version() if number is None else next((v for v in self.versions() if v.number == number), None)
        return version.best_checkpoint if version is not None else None

    def _heldout(self):
        """The held-back documents as text."""
        raise NotImplementedError

    # collect (passive training-data collection, issue #447) -------------------------------------------
    def collect_options(self):
        """{topics, sources, excluded, licences, user_agent}: what the Collect page offers (studio_collect.view.options)."""
        raise NotImplementedError

    def collect_config(self):
        """The collector's settings (studio_collect.config.DEFAULTS shape)."""
        raise NotImplementedError

    def save_collect_config(self, overrides):
        """Apply a partial settings dict and return the settings. ValueError, in plain words, for a bad value. Starts the background
        collector when collection is on."""
        raise NotImplementedError

    def collect_state(self):
        """What the collector is doing: status, counters, plan, per-source progress, disk and bandwidth use, journal (studio_collect.view.state)."""
        raise NotImplementedError

    def collect_pause(self):
        """Pause collection at once (the service notices within a second). Returns the settings."""
        raise NotImplementedError

    def collect_resume(self):
        """Resume (and turn on) collection; starts the background collector. Returns the settings."""
        raise NotImplementedError

    def collect_start(self):
        """Begin keeping the collector running and registering its shards (the API calls this at start-up)."""

    def collect_stop(self):
        """Stop doing that (the API calls this at shut-down). The background collector itself keeps going."""



DEMO_RUN = "demo-run"  # the id of the run `studio_api --fake` starts with (FakeServices.demo_run)


class FakeServices(Services):
    """Empty, in-memory placeholder data so pages render before real services exist. A test fixture and `studio_api --fake`
    only, never the real app. start_queue() runs the queued pretrain jobs one after another in a worker thread: `job_steps`
    steps of `step_seconds` each (`sleep` is injectable, so a test needs no real wait), then the job's version is trained or,
    for a job with no version of its own, a new one is added. The events match studio_queue.Runner's."""

    def __init__(self, job_steps=8, step_seconds=0.05, sleep=time.sleep, snapshot_every=0):
        super().__init__()
        self.job_steps, self.step_seconds, self._sleep = job_steps, step_seconds, sleep
        self.snapshot_every = snapshot_every  # > 0: also emit the training loop's viz_snapshot event every that many steps (--fake does)
        self._worker: threading.Thread | None = None
        self._started = False  # start_queue() was called: from then on a resume carries on, as the real runner's thread does
        self._worker_lock = threading.Lock()
        self._versions: list[Version] = []
        self._queue = QueueState()
        self.viewer_log: list = []  # (action, payload): the fake viewer records calls and spawns nothing
        self._viewer_running = False
        self.snapshot_paths: list = []  # what viewer_snapshots returns
        self.probe_path = None  # what viewer_probe_step returns
        self.heldout_docs: list = []  # the held-back text viewer_summarize reads
        self.reviewed_text = True  # whether the text sets hold an example marked Good
        self.conversations_in_text = 40  # how many conversations a cluster run's teacher can answer are in the data

    def viewer_status(self):
        return {"available": True, "running": self._viewer_running, "connected": self._viewer_running}

    def viewer_start(self, scene=None):
        self._viewer_running = True
        self.viewer_log.append(("start", scene))

    def viewer_stop(self):
        self._viewer_running = False
        self.viewer_log.append(("stop", None))

    def viewer_send(self, message):
        if self._viewer_running:
            self.viewer_log.append(("send", message))

    def viewer_poll(self):
        return []

    def viewer_snapshots(self, number=None):
        return list(self.snapshot_paths)

    def viewer_token_text(self, number, ids):
        return None

    def viewer_probe_step(self, text, number=None):
        self.viewer_log.append(("probe", text))
        return self.probe_path

    def inspect_landscape(self, version, snapshot=None, mode="random", grid=21, span=1.0, seed=0, live=False, new_anchor=False, stop=None):
        """A made-up bowl with two ripples and a path that walks into it one point longer per call, so a live page has something to show."""
        import math

        import studio_landscape
        if mode not in studio_landscape.MODES:
            raise ValueError(f"Directions are {' or '.join(studio_landscape.MODES)}.")
        grid = studio_landscape.grid_size(min(grid, 15) if live else grid)
        studio_landscape.check_size(5_000, grid)
        self._landscape_frames = 0 if new_anchor else getattr(self, "_landscape_frames", 0) + (1 if live else 0)
        shift = 0.15 * math.sin(self._landscape_frames / 3.0) if live else 0.0
        axis = [span * (2 * i / (grid - 1) - 1) for i in range(grid)]

        def height(x, y):
            return 1.2 + 1.6 * ((x - shift) ** 2 + (y + 0.5 * shift) ** 2) / (span * span) + 0.15 * math.sin(5 * x / span) * math.cos(4 * y / span)

        loss = [[height(x, y) for x in axis] for y in axis]
        low = min(((loss[r][c], r, c) for r in range(grid) for c in range(grid)))
        count = 6 + self._landscape_frames if live else 8
        path = [{"step": 50 * (i + 1), "x": 0.9 * span * (1 - i / count) * math.cos(0.6 * i), "y": 0.9 * span * (1 - i / count) * math.sin(0.6 * i)} for i in range(count)]
        for point in path:
            point["loss"] = height(point["x"], point["y"])
        return {"mode": mode, "unit": "filter" if mode == "random" else "trajectory", "grid": grid, "span": span, "seed": seed, "live": live,
                "axis": axis, "loss": loss, "ceiling": sorted(v for row in loss for v in row)[int(0.95 * grid * grid)],
                "current": {"x": 0.0, "y": 0.0, "loss": loss[grid // 2][grid // 2], "step": path[-1]["step"]},
                "minimum": {"x": axis[low[2]], "y": axis[low[1]], "loss": low[0], "inside": 0 < low[1] < grid - 1 and 0 < low[2] < grid - 1},
                "trajectory": path, "center": {"step": path[-1]["step"], "name": "best.npz"}, "batch": {"rows": 4, "tokens": 128},
                "cost": {"seconds": 0.0, "evaluations": grid * grid + len(path)}, "note": "Made-up data for the fake studio."}


    def _heldout(self):
        return self.heldout_docs

    def has_reviewed_text(self, dataset_ids):
        return self.reviewed_text

    def has_conversations(self, dataset_ids, reviewed_only=False, at_least=1):
        return self.conversations_in_text >= at_least

    def conversations(self, params):
        return [{"user": f"Question {n}?", "assistant": f"Answer {n}."} for n in range(int(self.conversations_in_text))]

    def versions(self):
        return list(self._versions)

    def current_version(self):
        return self._versions[-1] if self._versions else None

    def best_version(self):
        done = [v for v in self._versions if v.best_checkpoint]
        return done[-1] if done else None

    def save_version(self, spec, change_text, prediction=None):
        v = Version(len(self._versions) + 1, f"version {len(self._versions) + 1}", change_text, spec=spec, status="saved",
                    prediction=dict(prediction or {}))
        self._versions.append(v)
        self.emit("versions", v)
        return v

    def version_store(self):
        from studio_publish import NO_STORE
        raise RuntimeError(NO_STORE)  # a cluster of one

    def publish_version(self, number):
        return self.version_store()

    def published_versions(self):
        return self.version_store()

    def pull_version(self, number, origin=None):
        return self.version_store()

    def share_version(self, number):
        raise NoCheckpoint(number)  # nothing here has weights

    def queue(self):
        return self._queue

    def set_queue(self, state: QueueState):
        """Test hook: replace the queue state and announce it."""
        self._queue = state
        self.emit("queue", state)

    def enqueue(self, job):
        self._queue.up_next.append(job)
        self.emit("queue", self._queue)

    def start_queue(self):
        """Run the waiting jobs in a worker thread, unless one is already running or the queue is paused. The first job is
        already the running job when this returns."""
        with self._worker_lock:
            self._started = True
            if self._worker is not None and self._worker.is_alive():
                return
            job = self._take_job()
            if job is None:
                return
            self._worker = start_thread(self._run_jobs, job)

    def wait_idle(self, timeout=10.0):
        """Test hook: block until the worker has run every waiting job; True when it has."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout)
        return worker is None or not worker.is_alive()

    def _take_job(self):
        """The next waiting job, now the running one; None when paused or nothing waits. Hold _worker_lock."""
        job = next((j for j in self._queue.up_next if j.status == "waiting"), None)
        if job is None or self._queue.paused:
            return None
        self._queue.up_next.remove(job)
        job.status = "running"
        self._queue.job, self._queue.step, self._queue.total, self._queue.phase = job, 0, self.job_steps, "warm-up"
        return job

    def _run_jobs(self, job):
        while job is not None:
            self._run_job(job)
            with self._worker_lock:
                job = self._take_job()
                if job is None:
                    self._worker = None

    def _run_distributed(self, job):
        """A distributed pretraining run with a made-up coordinator (studio_ia.fake_distributed): a round takes about nine steps' time."""
        import studio_diloco
        import studio_distributed
        from studio_ia.fake_distributed import FakeCoordinator
        q = self._queue
        run = studio_distributed.run_of(job)
        run.folder = tempfile.mkdtemp(prefix="fake_run_")  # the stub weights a restart starts from
        coordinator = FakeCoordinator(run, job.params.get("settings") or {}, unit=self.step_seconds * 1.5)
        studio_diloco.register(job.id, coordinator)
        studio_distributed.invite(job)
        coordinator.start()
        q.total = coordinator.rounds
        self.emit("job", {"job": job.id, "status": "running"})
        self.emit("queue", q)
        while not coordinator.finished:
            self._sleep(self.step_seconds * 2)
            coordinator.advance()
            q.step = coordinator.round
            self.emit("queue", q)
        job.status = "stopped" if coordinator.stopped else "done"
        if job.status == "done":
            studio_distributed.note_saved(job.id, job.params.get("version"))  # the made-up run "saves" the version it was for
        q.job, q.step, q.total, q.phase, q.lr, q.eta_s = None, 0, 0, "", None, None
        self.emit("job", {"job": job.id, "status": job.status, "error": ""})
        self.emit("queue", q)

    def demo_run(self, clock=time.monotonic, unit=1.):
        """A distributed run already under way, with no queue job behind it: a CUDA and an MPS machine training in rounds of about
        six seconds, so `studio_api --fake` opens the Training tab on a live run. Stop it from the tab like any other."""
        import studio_diloco
        import studio_distributed
        from studio_ia.fake_distributed import FakeCoordinator
        machines = [studio_distributed.Machine(node="mac-mini", address="local", device="mps", device_name="Apple M2"),
                    studio_distributed.Machine(node="gpu-box", address="gpu-box:8787", device="cuda", device_name="NVIDIA GeForce RTX 4090")]
        settings = {"rounds": 400, "inner_steps": 100, "outer_lr": .7, "outer_momentum": .9, "delta_dtype": "float16"}
        run = studio_distributed.Run(DEMO_RUN, "Demo run on a Mac mini and a GPU box", 1, time.time(), settings, "mac-mini", machines)
        with studio_distributed._LOCK:
            studio_distributed.RUNS[run.id] = run
        coordinator = FakeCoordinator(run, settings, clock=clock, unit=unit)
        studio_diloco.register(run.id, coordinator)
        coordinator.start()
        return run

    def _run_job(self, job):
        if job.kind == "pretrain_distributed":
            return self._run_distributed(job)
        total, q = self.job_steps, self._queue
        self.emit("job", {"job": job.id, "status": "running"})
        self.emit("queue", q)
        for step in range(1, total + 1):
            self._sleep(self.step_seconds)
            phase = "warm-up" if step * 4 <= total else "cool-down" if step * 4 > total * 3 else "steady"
            q.step, q.phase, q.lr = step, phase, 3e-4
            q.eta_s = (total - step) * self.step_seconds
            self.emit("train", {"job": job.id, "type": "stats", "step": step, "total": total, "phase": phase, "lr": q.lr,
                                "loss": round(5.0 - 3.0 * step / total, 4), "tokens_per_second": 1000.0})
            if self.snapshot_every and step % self.snapshot_every == 0:
                self.emit("train", {"job": job.id, "type": "viz_snapshot", "step": step, "path": f"viz/step_{step:08d}.npz", "markers": []})
        job.status = "done"
        q.job, q.step, q.total, q.phase, q.lr, q.eta_s = None, 0, 0, "", None, None
        self.emit("job", {"job": job.id, "status": "done", "error": ""})
        number = job.params.get("version")
        known = next((v for v in self._versions if v.number == number), None)
        if known is not None:
            known.status = "trained"
        else:
            self._versions.append(Version(len(self._versions) + 1, f"version {len(self._versions) + 1}", job.label or "Trained",
                                          spec=dict(job.params.get("spec") or {}), status="trained"))
        self.emit("versions", None)
        self.emit("queue", q)

    def move_up(self, job_id):
        q = self._queue.up_next
        i = next((k for k, j in enumerate(q) if j.id == job_id), 0)
        if i:
            q[i - 1], q[i] = q[i], q[i - 1]
        self.emit("queue", self._queue)

    def move(self, job_id, index):
        q = self._queue.up_next
        job = next((j for j in q if j.id == job_id), None)
        if job is not None:
            q.remove(job)
            q.insert(max(0, min(index, len(q))), job)
        self.emit("queue", self._queue)

    def hold(self, job_id, held=True):
        for j in self._queue.up_next:
            if j.id == job_id:
                j.status = "held" if held else "waiting"
        self.emit("queue", self._queue)

    def storage(self, days=None, data=False):
        return (0, 0)

    def clean_storage(self, days=None, data=False):
        self.cleaned = (days, data)
        return (0, 0)

    def leftovers(self):
        return []

    def remove_leftovers(self, ids):
        self.removed = list(ids)
        return (0, 0)

    def free_space(self):
        return 120 * 10 ** 9

    def export_version(self, version, kind, path):
        self.exports = [*getattr(self, "exports", []), (version, kind, path)]
        return {"path": str(path), "files": [], "note": ""}

    def import_pretrained(self, name="gpt2", lora=None):
        import studio_import
        self.imports = [*getattr(self, "imports", []), name]
        return self.save_version(studio_import.gpt2_spec(lora=lora), f"Started from {studio_import.MODELS[name]['title']}")

    def remove(self, job_id):
        self._queue.up_next = [j for j in self._queue.up_next if j.id != job_id]
        self.emit("queue", self._queue)

    def pause(self):
        self._queue.paused = True
        self.emit("queue", self._queue)

    def resume(self):
        self._queue.paused = False
        self.emit("queue", self._queue)
        if self._started:
            self.start_queue()  # the worker stopped on the pause: carry on

    def checkpoint_now(self):
        self.emit("queue", self._queue)

    def stop_after_cooldown(self):
        pass

    def word_lists(self):
        from studio_ia import pieces
        return pieces.word_lists([r["b"] for r in self._rows()["text"]])  # no GPT-2 list: nothing is downloaded here

    def lr_range_test(self, spec, stop=None):
        rates = [1e-6 * 10 ** (i * 6 / 99) for i in range(100)]
        losses = [5.5 - 2.5 * min(1., max(0., (i - 30) / 50)) for i in range(100)]
        return dict(rates=rates, losses=losses, smoothed=losses, suggestion=rates[70] / 10, steepest=70, diverged=False, cancelled=False,
                    lr_min=1e-6, lr_max=1.)

    def pretrain_preview(self, dataset_ids, rows=16):
        row = [(i // 7) % 4 if i % 7 else -1 for i in range(32)]
        return {"pieces": 2_000_000, "params": 1_000_000, "context": 256, "first_rows": [row[r:] + row[:r] for r in range(rows)]}

    def cached_step_rate(self, spec, rows):
        return self.measure_step_rate(spec, rows, 256)

    def continue_run(self, number, kind, steps, dataset_ids=(), width=None, layers=None, spec=None, rows=16, hold=.05):
        if not any(v.number == number for v in self.versions()):
            raise ValueError(f"No version {number}.")
        label = {"same": f"Version {number} continued from step 100", "new_data": f"Version {number} continued from step 100 on new data",
                 "grow": f"Version {number + 1} grown from version {number} (step 100)"}[kind]
        made = self.__dict__.setdefault("continued", [])
        job = Job(id=f"c{len(made) + 1}", kind="pretrain", label=label, params={"continue_from": {"kind": kind, "step": 100, "add": steps}})
        made.append(job)
        self.enqueue(job)
        return {"job_id": job.id, "label": label, "version": number, "kind": kind, "from_step": 100}

    def measure_step_rate(self, spec, rows, vocab, stop=None):
        return {"seconds_per_step": .25, "tokens_per_second": rows * 256 / .25, "device": "cpu", "device_name": "PyTorch / CPU", "note": "",
                "rows": rows, "context": 256}

    def set_plan(self, plan):
        self._queue.plan = dict(plan)
        self.emit("queue", self._queue)

    # data: a few canned rows per kind so the pages render without a workspace -------------------------------
    draft_source = None  # callable(kind, n) -> rows, standing in for the local model

    def _rows(self):
        if not hasattr(self, "_data_rows"):
            self._data_rows = {
                "text": [dict(id="t1", a="Short stories", b="The lamp was lit before the tide came in.", status="In use", dataset="Your files"),
                         dict(id="t2", a="Short stories", b="Nobody asked the heron.", status="Held back", dataset="Your files")],
                "conversations": [dict(id="c1", a="Name a prime.", b="7 is a prime number.", status="Good", dataset="Your conversations"),
                                  dict(id="c2", a="Say hello.", b="Hello there.", status="Review", dataset="Your conversations"),
                                  dict(id="c3", a="Shorter, please.", b="Done.", status="From Chat", dataset="Your corrections")],
                "problems": [dict(id="p1", a="Reverse: groundwork", b="reverse", status="Ready", dataset="Your problems"),
                             dict(id="p2", a="17 + 25 =", b="small addition", status="Ready", dataset="Your problems")],
            }
        return self._data_rows

    def data(self, kind, query=""):
        import studio_data_kinds as dk
        rows = self._rows()[kind]
        out = dk.DataSummary(kind, count=len(rows), used_by=dk.USED_BY[kind])
        out.characters = sum(len(r["a"]) + len(r["b"]) for r in rows)
        for r in rows:
            sets = {x["name"]: x for x in out.sets}
            sets.setdefault(r["dataset"], {"id": r["dataset"], "name": r["dataset"], "count": 0})["count"] += 1
            out.sets = list(sets.values())
            if kind == "problems":
                out.rules[r["b"]] = out.rules.get(r["b"], 0) + 1
        out.near_duplicates = sum(r["status"] == "Near-duplicate" for r in rows)
        needle = query.casefold().strip()
        out.rows = [dict(r) for r in rows if needle in f"{r['a']} {r['b']}".casefold()]
        return out

    def data_page(self, kind, query="", cursor=None, limit=200, review_only=False):
        rows, start = self.data(kind, query).rows, cursor or 0
        if review_only:
            rows = [row for row in rows if row.get("draft") and row["status"] == "Review"]
        return rows[start:start + limit], start + limit if start + limit < len(rows) else None

    def _add_row(self, kind, a, b, status="Good", dataset="Your data", draft=False):
        rows = self._rows()[kind]
        if not (a.strip() and b.strip()) or any((r["a"], r["b"]) == (a, b) for r in rows):
            return False
        rows.append(dict(id=f"{kind[0]}{len(rows) + 1}-{a[:8]}", a=a, b=b, status=status, dataset=dataset, draft=draft))
        return True

    def add_data(self, kind, fields):
        a, b = {"text": ("Your files", fields.get("text", "")), "conversations": (fields.get("user", ""), fields.get("assistant", "")),
                "problems": (fields.get("prompt", ""), fields.get("rule", ""))}[kind]
        added = self._add_row(kind, a, b, "In use" if kind == "text" else "Ready" if kind == "problems" else "Good")
        self.emit("data", {"kind": kind})
        return added

    def import_data(self, kind, paths):
        from studio_core import normalize, parse_file
        from studio_data_kinds import classify, rule_of
        report = []
        for source in paths:
            path = Path(source)
            try:
                added = duplicate = invalid = 0
                for item in parse_file(path):
                    row = normalize(item)
                    if row is None or classify(row) != kind:
                        invalid += 1
                        continue
                    a, b = (path.stem, row["text"]) if kind == "text" else \
                        (row["user"], rule_of(row)) if kind == "problems" else (row["user"], row["assistant"])
                    status = "In use" if kind == "text" else "Ready" if kind == "problems" else "Good"
                    if self._add_row(kind, a, b, status, dataset=path.stem):
                        added += 1
                    else:
                        duplicate += 1
                report.append(f"{path.name}: {added} added, {duplicate} duplicates, {invalid} invalid")
            except (OSError, ValueError) as exc:
                report.append(f"{path.name}: {plain_error(exc)}")
        self.emit("data", {"kind": kind})
        return report

    def collection(self, kind="text"):
        return getattr(self, "_collection", []) if kind == "text" else getattr(self, "_collections", {}).get(kind, [])

    def download_sources(self, ids=None, kind="text", install=False):
        call = ("download_sources", ids) if (kind, install) == ("text", False) else ("download_sources", ids, kind, install)
        self.calls = getattr(self, "calls", []) + [call]
        return True

    def import_source(self, source_id):
        self.calls = getattr(self, "calls", []) + [("import_source", source_id)]
        return True

    def generate_problems(self, rule, n, store=True, rng=None):
        import random
        items = problem_items(rule, n, rng or random.Random())
        added = sum(self._add_row("problems", i["prompt"], i["rule"], "Ready", "Generated from rules") for i in items) if store else 0
        return problem_report(items, added, len(items) - added if store else 0)

    def make_drafts(self, kind, n):
        if self.draft_source is None:
            raise RuntimeError("No local model connected.")
        pairs = [("Generated prose", r["text"]) if kind == "text" else (r["user"], r["assistant"]) for r in self.draft_source(kind, n)]
        return sum(self._add_row(kind, a, b, "Review", "Drafts", True) for a, b in pairs)

    def draft_count(self):
        return sum(1 for rows in self._rows().values() for r in rows if r.get("draft") and r["status"] == "Review")

    def review_data(self, example_id, keep):
        for rows in self._rows().values():
            for r in list(rows):
                if r["id"] == example_id:
                    if keep:
                        r["status"] = "Good"
                    else:
                        rows.remove(r)

    def _find(self, example_id):
        return next((k, r) for k, rows in self._rows().items() for r in rows if r["id"] == example_id)

    def data_fields(self, example_id):
        kind, r = self._find(example_id)
        return {"text": {"text": r["b"]}, "conversations": {"user": r["a"], "assistant": r["b"]}, "problems": {"prompt": r["a"], "rule": r["b"]}}[kind]

    def edit_data(self, example_id, fields):
        self.edits = [*getattr(self, "edits", []), (example_id, fields)]
        kind, r = self._find(example_id)
        a, b = {"text": (r["a"], fields.get("text", "")), "conversations": (fields.get("user", ""), fields.get("assistant", "")),
                "problems": (fields.get("prompt", ""), fields.get("rule", ""))}[kind]
        if not (a.strip() and b.strip()):
            raise ValueError("Nothing to save: fill in every part.")
        if any(o is not r and (o["a"], o["b"]) == (a, b) for o in self._rows()[kind]):
            raise ValueError("That is already in this set.")
        r["a"], r["b"] = a, b

    def references(self, version):
        return {"uniform_bpc": None, "unigram_bpc": None, "best_so_far_bpc": None}

    def gate_result(self, version):
        return None

    def run_curve(self, run_dir):
        return {"steps": [], "smooth": [], "heldout": [], "lr": [], "phases": [], "total": 0}

    def run_phases(self, run_dir):
        return []

    def chat(self, version, messages, on_text, stop, ablate=None):
        return f"[reply with {len(ablate)} heads off]" if ablate else "[reply]"

    def teach_correction(self, version, messages, correction, whole_answer=False):
        pass

    def rate_pair(self, messages, chosen, rejected):
        pass

    def verdict(self, a, b):
        from studio_eval import training_saved
        return training_saved([], [])

    def few_shot(self, version):
        return {"items": 40, "tasks": 2, "held_back": True, "chance": .3,
                "rows": [{"shots": 0, "used": 0, "correct": 11, "accuracy": .275}, {"shots": 1, "used": 1, "correct": 13, "accuracy": .325},
                         {"shots": 4, "used": 3.5, "correct": 14, "accuracy": .35}]}

    def mlperf_board(self, task_id):
        import studio_mlperf
        results = getattr(self, "_mlperf", [])
        return {"tasks": studio_mlperf.task_list(), "board": studio_mlperf.leaderboard(results, task_id),
                "history": studio_mlperf.history(results, task_id), "devices": ["cpu"]}

    def mlperf_compare(self, ids):
        import studio_mlperf
        return studio_mlperf.compare(getattr(self, "_mlperf", []), ids)

    def mlperf_enqueue(self, task_id, device, attempts=None):
        import studio_mlperf
        studio_mlperf.get_task(task_id)
        return {"id": "mlperf01", "label": f"Training speed: {task_id} on {device}"}

    def benchmark(self):
        import studio_bench

        def result(n, bpc, accuracy, rate):
            return {"suite": "fake", "version": n, "bpc": bpc, "bpc_band": [bpc - .05, bpc + .05], "dropped": {"docs": 0, "choice": 2, "problems": 0},
                    "decontam": "checked", "problems": {"passed": round(30 * rate), "total": 30, "rate": rate},
                    "few_shot": {"items": 40, "chance": .3, "accuracy": accuracy, "band": [accuracy - .14, accuracy + .14], "rows": []}}
        rows = [{"number": 1, "name": "Version 1", "lineage": "a", "continues": None, "tokens": 200_000, "stale": False, "result": result(1, 2.6, .30, .1)},
                {"number": 2, "name": "Version 2", "lineage": "a", "continues": 1, "tokens": 800_000, "stale": False, "result": result(2, 2.2, .35, .2)},
                {"number": 3, "name": "Version 3", "lineage": "b", "continues": None, "tokens": 500_000, "stale": False, "result": None}]
        return {"suite": "fake", "sizes": {"docs": 24, "choice": 40, "problems": 30}, "dropped": {}, "sources": [], "rows": rows,
                "lineages": studio_bench.trend(rows)}

    def run_benchmark(self, number):
        return next(r["result"] for r in self.benchmark()["rows"] if r["number"] == number) or {}

    def post_training(self, version):
        return {"rounds": {}, "counts": {}, "gates": {}}

    def post_rows(self, kind):
        return []

    def post_loops(self):
        return list(getattr(self, "_loops", []))

    def start_loop(self, version, config=None, resume=None, judge_model=None):
        loops = self.__dict__.setdefault("_loops", [])
        v = next((x for x in self._versions if x.number == version), None) if version else (self._versions[-1] if self._versions else None)
        if v is None or not v.best_checkpoint:
            raise NoCheckpoint(getattr(v, "number", None))
        loop_id = resume or uuid.uuid4().hex[:8]
        if any(j.kind == "rlhf_loop" and j.params.get("loop_id") == loop_id for j in [self._queue.job, *self._queue.up_next] if j):
            raise ValueError("This loop is already queued or running.")
        if not resume:
            loops.insert(0, {"id": loop_id, "version": v.number, "status": "running", "stop_reason": "", "config": dict(config or {}),
                             "anchor": v.best_checkpoint, "current": v.best_checkpoint, "kept": 0, "failed_streak": 0, "audit_flagged": 0,
                             "elapsed_s": 0., "created": "", "updated": "", "phase": "", "note": "", "anchor_eval": None,
                             "current_eval": None, "rounds": [], "round": 1})
        job = Job(uuid.uuid4().hex[:8], "rlhf_loop", f"RLHF loop · version {v.number}", "automated preference rounds",
                  params={"version": v.number, "checkpoint": v.best_checkpoint, "config": dict(config or {}), "loop_id": loop_id,
                          "judge_model": judge_model})
        self.enqueue(job)
        return {"job": job.id, "loop_id": loop_id, "label": job.label}

    def adopt_loop(self, loop_id):
        return {"round": 5, "ok": False, "checks": [], "verdict": "Nothing to adopt: the fake services run no loops.", "checkpoint": ""}

    # inspect stubs: small synthetic arrays, 2 blocks x 4 heads, 9 pieces, vocab 12 ---------------------------------
    def inspect_forward(self, version, text, hooks=True):
        rng, T, H, h, V = np.random.default_rng(0), 9, 4, 4, 12
        hooks = {"embed": rng.normal(size=(1, T, 16)), "final": rng.normal(size=(1, T, 16))}
        for b in range(2):
            q, k, v = (rng.normal(size=(1, H, T, h)) for _ in range(3))
            scores = q @ k.transpose(0, 1, 3, 2)
            logits = scores / np.sqrt(h)
            masked = np.where(np.tril(np.ones((T, T), bool)), logits, -1e9)
            pattern = np.exp(masked - masked.max(-1, keepdims=True))
            hooks.update({f"block{b}.attn.q": q, f"block{b}.attn.k": k, f"block{b}.attn.v": v,
                          f"block{b}.attn.scores": scores, f"block{b}.attn.logits": logits,
                          f"block{b}.attn.pattern": pattern / pattern.sum(-1, keepdims=True)})
        ids = np.arange(T)[None] % V
        return {"ids": ids, "tokens": [[f"t{i}" for i in ids[0]]], "logits": rng.normal(size=(1, T, V)), "hooks": hooks}

    def inspect_logit_lens(self, version, text, top_k=5):
        rng, T, V = np.random.default_rng(1), 9, 12
        k = max(1, min(int(top_k), V))
        rows = []
        for name in ("embed", "block0.resid.post", "block1.resid.post"):
            raw = np.exp(rng.normal(size=(1, T, V)))
            probabilities = raw / raw.sum(-1, keepdims=True)
            top = np.argsort(probabilities, axis=-1)[..., -k:][..., ::-1]
            rows.append({"hook": name, "probabilities": probabilities, "top_ids": top,
                         "top_probabilities": np.take_along_axis(probabilities, top, axis=-1)})
        return rows

    def inspect_ablate(self, version, text, targets=None, mode="zero", heads=None):
        if mode not in ("zero", "mean"):
            raise ValueError("Ablation must be zero or mean.")
        selected = [(b, h) for b in range(2) for h in range(4)] if heads is None else [tuple(h) for h in heads]
        if any(not (0 <= b < 2 and 0 <= h < 4) for b, h in selected):
            raise ValueError("Selected block/head has no attention output hook.")
        stats = self.inspect_head_stats(self.inspect_forward(version, text)["hooks"]["block0.attn.pattern"])
        rows = [{"block": b, "head": h, "loss": 2.0 + .1 * (4 * b + h), "delta": .1 * (4 * b + h),
                 **{key: float(value[h]) for key, value in stats.items()}} for b, h in selected]
        return {"baseline": 2.0, "mode": mode, "heads": rows}

    def inspect_head_stats(self, pattern):
        import studio_inspect
        return studio_inspect.head_stats(pattern)

    def inspect_trace(self, version, text, pos=-1):
        names = ["embedding", "block 1 mixer", "block 1 channel", "output"]
        rows = [{"stage": n, "kind": n.split()[-1], "norm": 1.0 + i, "rel_change": None if i == 0 else .2, "cosine": None if i == 0 else .9,
                 "top_id": 1, "top_token": "t1", "top_p": .1, "approximate": i < 3} for i, n in enumerate(names)]
        return {"tokens": [f"t{i}" for i in range(9)], "pos": 8, "rows": rows}

    def inspect_prepared(self, version):
        ids = [1, 5, 6, 7, 8, 5, 6, 7, 8]
        return {"ids": ids, "tokens": [f"t{i}" for i in ids]}

    def inspect_loss(self, version, text, pos=-1, trained=True):
        V = 12
        p = np.full(V, 1 / V)
        if trained:
            p = np.full(V, .02)
            p[3] = 1 - .02 * (V - 1)
        return {"tokens": ["a", "b"], "pos": 1, "probabilities": p, "target": 3, "pieces": [f"t{i}" for i in range(V)]}

    def inspect_value_map(self, version, heads, block=0, head=0):
        return {"d_model": 256, "heads": 4, "blocks": {0: 4, 1: 4}, "block": block, "head": head,
                "singular": [3.0, 1.5, .7, .1] if heads == 4 else None}

    def inspect_rank_heads(self, version, mode="mean"):
        return [{"block": b, "head": h, "rise": .05 * (4 * b + h)} for b in range(2) for h in range(4)]

    def inspect_heads_off(self, version, off, mode="mean"):
        return {"before": 2.0, "after": 2.0 + .1 * len(off) * (2 if mode == "zero" else 1)}

    def inspect_reply(self, version, prompt, off=(), mode="mean"):
        return f"[reply with {len(off)} heads off]" if off else "[reply]"

    # collect: the real settings, state and view over a scratch folder, with no collector process behind it -------------
    def _collect_root(self):
        if getattr(self, "_collect_dir", None) is None:
            import shutil
            import tempfile
            import weakref
            self._collect_dir = tempfile.mkdtemp(prefix="collect-fake-")
            weakref.finalize(self, shutil.rmtree, self._collect_dir, True)
        return self._collect_dir

    def collect_options(self):
        from studio_collect import view
        return view.options()

    def collect_config(self):
        from studio_collect import view
        return view.get_config(self._collect_root())

    def save_collect_config(self, overrides):
        from studio_collect import view
        config = view.set_config(self._collect_root(), overrides)
        self.emit("collect", {})
        return config

    def collect_state(self):
        from studio_collect import view
        return view.state(self._collect_root(), running=False)

    def collect_pause(self):
        from studio_collect import view
        config = view.set_paused(self._collect_root(), True)
        self.emit("collect", {})
        return config

    def collect_resume(self):
        from studio_collect import view
        config = view.set_paused(self._collect_root(), False)
        self.emit("collect", {})
        return config


# ---- the real façade -------------------------------------------------------------------------------------------

import json  # noqa: E402
import math  # noqa: E402
import uuid  # noqa: E402
from studio_ia.snapshots import start_thread  # noqa: E402
from dataclasses import asdict, fields  # noqa: E402
from pathlib import Path  # noqa: E402

VERSION_JOBS = ("pretrain", "import_pretrained", "pretrain_distributed")  # queue jobs that carry a version through queued, training and trained
SUMMARY_FORMAT = 4  # data/summaries.sqlite layout; 4: entries, rows and signatures kept apart (was data/summaries.json); 3: one entry per dataset, each with its rev. A saved summary of another format is rebuilt
RUN_DOC_CHARS = 5_000_000  # characters of a run's own text read for reference numbers (unigram baseline, characters per piece)
PREVIEWS_KEPT = 16  # Pretraining previews kept in data/previews.json (one per choice of sets and design)
RANGE_CHARS = 400_000  # characters of the chosen text the learning-rate test trains on: a hundred small steps never read more
PREVIEW_CHARS = 5_000_000  # characters of the chosen text the Pretraining preview reads, however large the text is
CHAT_MODELS = 3  # most chat models kept loaded; their bytes are also in the ledger, which the queue takes off a job's share
FEW_SHOT_ITEMS = 100  # problems one few-shot check asks, each with every choice at 0, 1 and k worked examples

LN2 = math.log(2)


def layout_rows(lengths, context, rows, cells=32):
    """First `rows` rows of the batch picture: documents laid end to end (one end-of-text piece each), cut into rows of
    `context` pieces and drawn as `cells` cells per row. Each cell holds its document number, -1 where a document ends."""
    out, doc, left, n = [], 0, list(lengths), 0
    per = max(1, context // cells)
    while len(out) < rows and doc < len(left):
        row = []
        for _ in range(cells):
            cell, end = doc, False
            for _ in range(per):
                if doc >= len(left):
                    break
                n += 1
                if n >= left[doc]:
                    n, doc, end = 0, doc + 1, True
            row.append(-1 if end else cell)
        out.append(row)
    return out


def _doc(row):
    """One held-back / training document as text (a text row, a user+assistant pair or a messages list)."""
    if row.get("text"):
        return row["text"]
    if row.get("messages"):
        return "\n".join(m["content"] for m in row["messages"])
    return "\n".join(s for s in (row.get("user"), row.get("assistant")) if s)


def _run_docs(run, limit=RUN_DOC_CHARS):
    """The first documents of a run's dataset.jsonl, up to about `limit` characters. That file is the whole training
    set (gigabytes for large text), so reference numbers read a slice of it, never all of it."""
    path, docs, chars = Path(run) / "dataset.jsonl", [], 0
    if not path.exists():
        return docs
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                docs.append(_doc(json.loads(line)))
                chars += len(docs[-1])
                if chars >= limit:
                    break
    return docs


LANDSCAPE_LIVE_GRID = 11  # a live frame is at most this many cells a side: 121 evaluations of the held-back batch
LANDSCAPE_LIVE_PATH = 6  # and draws at most this many of the newest snapshots as the path
LANDSCAPE_LIVE_THROTTLE = 3.0  # it sleeps 3 s per second of work, so a live frame takes about a quarter of one core


class _Cold(BaseException):
    """Raised inside ready(): the read asked for is not cached. Not an Exception, so no page's own handler swallows it."""


class _Slow(_Cold):
    """A _Cold that is long however small the store is (scoring a checkpoint, parsing the held-back text): ready() says no."""


class _Pending(Exception):
    """_held_scores(later=True) on the Tk thread: the checkpoint is being scored on a worker, which emits "versions" when done."""


class WorkspaceServices(Services):
    """Every Services method over the real modules. Queue events fire on the runner thread and hop to Tk via emit."""

    auto_bench = True  # a version that gets weights is scored on the frozen suite by itself (_bench_later); a test may turn it off

    def __init__(self, workspace):
        import studio_jobs
        import studio_queue

        super().__init__()
        self._reads(workspace)
        import studio_distributed
        studio_distributed.load(self.root)  # the Training tab's runs from before a restart (the queue's restart handling reads them)
        self.runner = studio_queue.Runner(self.root / "data" / "queue.json", studio_jobs.HANDLERS,
                                          prepare=lambda job: studio_jobs.prepare(job, self),
                                          ledger=self.ledger, make_room=self.evict_oldest_chat,
                                          on_start=self.chat_job_started, on_end=self.chat_job_ended)
        self.runner.subscribe(self.emit)
        self.runner.subscribe(self._record_job)
        for job in self.runner.ended_on_load:  # a distributed run the restart could not resume: its version is not training any more
            self._record_job("job", {"job": job.id, "status": job.status})
        workspace.progress = self.note
        # an older records.sqlite is brought up to date now, with progress shown, not under the first page that reads it
        self._schema = start_thread(workspace.ensure_schema) if workspace.schema_pending() else None

    def _reads(self, workspace):
        import studio_versions
        self.workspace = workspace
        self.root = Path(workspace.root)
        self.store = studio_versions.VersionStore(self.root)
        self.store.versions = lambda: self._rows()  # tolerant of the simpler list schema
        import studio_memory
        self._models, self._cache = {}, {}
        self.ledger = studio_memory.Ledger()  # the bytes of the chat models in _models
        self._chat_lock = threading.RLock()  # _models, _using, _job_running and where each model is; no Tk-thread read takes it
        self._using, self._job_running = {}, False  # per model, the lock a reply holds so the model is not moved under it
        self._lock = threading.RLock()
        self._made = {}  # cache key -> the lock held while it is being made
        self._data, self._probe = {}, threading.local()  # reads of the stored data, kept until it next changes
        self._to_score, self._scorer, self._unscorable = {}, False, set()  # held-back scores the Tk thread asked for (_held_scores)
        self._score_lock = threading.Lock()
        self._bench_lock, self._bench_wait, self._bench_busy = threading.Lock(), [], False  # versions waiting for the frozen suite (_bench_later)

    @classmethod
    def reader(cls, root):
        """The data reads a queue job makes (examples, post_rows), in its worker process: no runner, no window, no schema update."""
        from studio_core import Workspace
        self = cls.__new__(cls)
        Services.__init__(self)
        self._reads(Workspace(root))
        return self

    def close(self):
        if self._ui_thread is not None:
            ui_guard.unbind(self._ui_thread)
        if getattr(self, "_viz", None) is not None:
            self._viz.stop(wait=False)
        self.runner.shutdown(timeout=2)
        import studio_distributed
        studio_distributed.unload(self.root)
        if self._schema is not None:  # its connection holds records.sqlite open, and Windows cannot delete an open file
            self._schema.join()
        bench = getattr(self, "_bench_thread", None)
        if bench is not None:  # it writes under data/bench, which a caller removing the workspace must not race
            bench.join(60)

    # 3D viewer --------------------------------------------------------------
    _viz = None

    def viewer_status(self):
        script = Path(__file__).resolve().parent.parent / "studio_brain3d.py"
        viz = self._viz
        return {"available": script.exists(), "running": bool(viz and viz.running), "connected": bool(viz and viz.connected)}

    def viewer_start(self, scene=None):
        from studio_vizbridge import VizBridge
        if self._viz is None:
            self._viz = VizBridge(Path(__file__).resolve().parent.parent / "studio_brain3d.py")
        if not self._viz.running:
            self._viz.start(log_path=self.root / "logs" / "viewer.log")
        version = self.current_version()
        if version is not None and version.spec:
            self._viz.send({"type": "spec", "spec": version.spec})
        if scene:
            self._viz.send({"type": "scene", **scene})

    def viewer_stop(self):
        if self._viz is not None:
            self._viz.stop(wait=False)

    def viewer_send(self, message):
        if self._viz is not None:
            self._viz.send(message)

    def viewer_poll(self):
        return self._viz.poll() if self._viz is not None else []

    def viewer_snapshots(self, number=None):
        version = self.current_version() if number is None else next((v for v in self.versions() if v.number == number), None)
        if version is None:
            return []
        return [str(p) for run in version.runs for p in sorted(Path(self._abs(run)).glob("viz/step_*.npz"))]

    def viewer_token_text(self, number, ids):
        import json

        from studio_model import Tokenizer
        version = self.current_version() if number is None else next((v for v in self.versions() if v.number == number), None)
        for run in (version.runs if version is not None else []):
            path = Path(self._abs(run)) / "tokenizer.json"
            if path.is_file():
                tokenizer = Tokenizer.restore(json.loads(path.read_text(encoding="utf-8")))
                return [tokenizer.decode([int(i)]) for i in ids]
        return None

    def viewer_probe_step(self, text, number=None):
        import numpy as np

        from studio_core import uid
        from studio_model import Tokenizer, load_checkpoint
        from studio_snapshot import one_step, save
        version = self.current_version() if number is None else next((v for v in self.versions() if v.number == number), None)
        source = version.best_checkpoint if version is not None else None
        if not source or not Path(source).exists():
            return None
        data = load_checkpoint(source)
        ids = np.asarray(Tokenizer.restore(data["tokenizer"]).encode(text), dtype=np.int64)[None, -data["config"]["context"]:]
        if ids.shape[1] < 2:
            raise ValueError("the probe text needs at least two tokens")
        targets = np.roll(ids, -1, axis=1)
        targets[:, -1] = -100  # the last position has no next token to predict
        return str(save(self.workspace.run_dir / uid("probe"), one_step(source, ids, targets)))

    # versions ---------------------------------------------------------------
    def _rows(self):
        path = self.store.path
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        rows = data["versions"] if isinstance(data, dict) else data  # ponytail: list schema is read-only until the next save
        return [dict(r, runs=[x["dir"] if isinstance(x, dict) else x for x in r.get("runs", [])],
                     scorecard=r.get("scorecard") or {}) for r in rows]

    def _version(self, row):
        known = {f.name for f in fields(Version)}
        v = Version(**{k: val for k, val in row.items() if k in known and val is not None})
        v.name = v.name or f"Version {v.number}"
        v.best_checkpoint = self._abs(v.best_checkpoint)
        v.runs = row.get("runs", [])
        return v

    def _abs(self, p):
        return None if not p else str(p if Path(p).is_absolute() else self.root / p)

    def _file_hash(self, path):
        """sha256 of a file, read once per change of its size or time."""
        import hashlib
        stat = Path(path).stat()
        key = ("sha", str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._cache:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            self._cache[key] = digest.hexdigest()
        return self._cache[key]

    @staticmethod
    def _file_stamp(path):
        """(size, modified time) of a file, or None when it cannot be read."""
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def run_facts(self, number):
        """What version `number`'s last run did. A fact the files do not give is left out, never guessed: the caller reads a
        missing key as unknown. Read once per change of the run's files; the first read hashes the whole data file, so call
        it off the Tk thread."""
        try:
            runs = self.store.get(number)["runs"]
            run = Path(self._abs(runs[-1]))
            files = (run / "run.json", run / "config.json", run / "dataset.jsonl", self.root / "data" / "post_training.json")
            stamp = tuple(self._file_stamp(f) for f in files)
            if None in stamp[:3]:
                return {}
            key = ("facts", str(run), number)
            kept = self._cache.get(key)
            if kept and kept[0] == stamp:
                return dict(kept[1])
            record = json.loads(files[0].read_text(encoding="utf-8"))
            config = json.loads(files[1].read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not isinstance(config, dict):
                return {}
            facts = dict(data=self._file_hash(files[2]))
        except (KeyError, IndexError, OSError, ValueError, TypeError, AttributeError):
            return {}
        for name, value in (("steps", record.get("step")), ("tokens", record.get("trained_tokens")), ("lr", config.get("learning_rate"))):
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                facts[name] = value
        if "initialized_from" in record or "resumed_from" in record:  # None here is a run from scratch; absent is unknown
            facts["start"] = record.get("initialized_from") or record.get("resumed_from")
        try:
            facts["extra"] = any(r.get("ok") for r in self._json("post_training.json", {}).get(str(number), {}).values())  # a passed round moved the weights
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        self._cache[key] = (stamp, facts)
        return dict(facts)

    def run_speed(self, number):
        """Seconds per step of version `number`'s last run: word pieces per step (the growth in trained tokens over the growth
        in steps between the first and last rows that log both) over the median measured pieces per second. Read once per change of metrics.jsonl; call it off the
        Tk thread."""
        from studio_model import read_metrics
        try:
            path = Path(self._abs(self.store.get(number)["runs"][-1])) / "metrics.jsonl"
        except (KeyError, IndexError, TypeError):
            return None
        stamp = self._file_stamp(path)
        if stamp is None:
            return None
        key = ("speed", str(path))
        kept = self._cache.get(key)
        if kept and kept[0] == stamp:
            return kept[1]
        ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0  # noqa: E731
        try:
            rows = read_metrics(path)
        except (OSError, ValueError):
            return None
        speeds = sorted(r["tokens_per_second"] for r in rows if ok(r.get("tokens_per_second")))
        counted = [r for r in rows if ok(r.get("trained_tokens")) and ok(r.get("step"))]  # trained_tokens counts from this run's first step
        a, b = (counted[0], counted[-1]) if counted else (None, None)
        per_step = (b["trained_tokens"] - a["trained_tokens"]) / (b["step"] - a["step"]) if counted and b["step"] > a["step"] else None
        seconds = per_step / speeds[len(speeds) // 2] if speeds and per_step and per_step > 0 else None
        self._cache[key] = (stamp, seconds)
        return seconds

    def compare_runs(self):
        out = []
        for folder in sorted(Path(self.workspace.run_dir).glob("compare_*")):
            files = (folder / "experiment.json", folder / "results.json")
            stamp = tuple(self._file_stamp(f) for f in files)
            if None in stamp:
                continue
            key = ("compare", str(folder))
            kept = self._cache.get(key)
            if not kept or kept[0] != stamp:
                try:
                    experiment, results = (json.loads(f.read_text(encoding="utf-8")) for f in files)
                    variants = experiment.get("variants") or []
                    row = dict(base=experiment["base"], variants=variants, mode=experiment.get("mode"), results=results,
                               finished=stamp[1][1])
                    if not (isinstance(row["base"], dict) and isinstance(variants, list) and isinstance(results, dict)):
                        row = None
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    row = None
                kept = self._cache[key] = (stamp, row)
            if kept[1] is not None:
                out.append(kept[1])
        return out

    def _as_version(self, version):
        if isinstance(version, int):
            version = self._version(self.store.get(version))
        return version or self.best_version()

    def versions(self):
        return [self._version(r) for r in self._rows()]

    def current_version(self):
        rows = self._rows()
        return self._version(rows[-1]) if rows else None

    @timed("services.best_version")
    def best_version(self):
        """The version Chat follows: a later version takes over only when a paired test on the held-back text says it is
        better than the one it replaces (studio_gate.moves_chat, studio_eval.paired_gain), so a gap that luck could produce
        never moves Chat. One look per version, on its best checkpoint, when it has finished."""
        best = self._weigh()[0]
        return self._version(best) if best else None

    def _weigh(self):
        """(row Chat follows, {version number: how it was weighed against the best before it}), versions in order."""
        import studio_gate
        rows, best, weighed = self._rows(), None, {}
        scored = [r for r in rows if r.get("best_checkpoint") and isinstance(r.get("scorecard", {}).get("bpc"), (int, float))]
        if len(scored) > 1:  # versions are compared below and elsewhere: none may keep a number from an older scoring method
            for row in scored:
                self._rescored(row)
        for row in rows:
            bpc = row.get("scorecard", {}).get("bpc")
            if not row.get("best_checkpoint") or not isinstance(bpc, (int, float)):
                continue
            if best is None:  # with one scored version there is nothing to compare, so no held-back text is read
                best = row
                continue
            try:
                gain = self._gate(self._abs(best["best_checkpoint"]), self._abs(row["best_checkpoint"]))
            except _Pending:  # not scored yet: the stored numbers decide, as without held-back text, until the worker's "versions" event
                gain = None
            if gain is not None:
                weighed[row["number"]] = {**gain, "vs": best["number"]}
                self._rescored(best)
                self._rescored(row)
            if studio_gate.moves_chat(row["scorecard"]["bpc"], best["scorecard"]["bpc"], gain):
                best = row
        if best is None:  # nothing scored yet: the newest version that has a checkpoint
            best = next((r for r in reversed(rows) if r.get("best_checkpoint")), None)
        return best, weighed

    def gate_result(self, version):
        v = self._as_version(version)
        return self._weigh()[1].get(v.number) if v else None

    def _rescored(self, row):
        """A scorecard made before the scoring method changed holds bits per character from the old method: swap in the new one."""
        import studio_eval
        if row["scorecard"].get("bpc_method") == studio_eval.SCORE_METHOD:
            return
        try:
            scores = self._held_scores(self._abs(row["best_checkpoint"]), later=True)
        except _Pending:  # swapped in when the worker's "versions" event makes this read again
            return
        if scores is not None:
            card = {**(self.store.get(row["number"]).get("scorecard") or {}), "bpc": studio_eval.corpus_bpc(scores),
                    "bpc_method": studio_eval.SCORE_METHOD}
            self.store.set_scorecard(row["number"], card)
            self._note_devices(row["number"], self._abs(row["best_checkpoint"]), True)
            row["scorecard"] = card

    def checkpoint_step(self, version):
        from studio_model import read_metrics
        v = self._as_version(version)
        if v is None or not v.best_checkpoint:
            return None
        held = [r for r in read_metrics(Path(v.best_checkpoint).parent / "metrics.jsonl") if r.get("val_loss") is not None]
        return min(held, key=lambda r: r["val_loss"])["step"] if held else None  # best.npz is saved at the lowest held-back loss

    def setting(self, key, default=None):
        return self.workspace.state["studioConfig"].get(key, default)

    def set_setting(self, key, value):
        self.workspace.state["studioConfig"][key] = value
        self.workspace.save()

    def save_version(self, spec, change_text, prediction=None):
        row = self.store.save_version(spec, change_text or None, prediction)
        v = self._version(row)
        self.emit("versions", v)
        return v

    # sharing versions between machines -------------------------------------------
    def version_store(self):
        """The store `studioConfig.store.folder` names in the workspace, else the one the TRAINER_STORE_* variables describe."""
        import studio_publish
        store = studio_publish.configured_store(self.workspace.state["studioConfig"])
        if store is None:
            raise RuntimeError(studio_publish.NO_STORE)
        return store

    def _through(self, store, work):
        """work(), with whatever the store throws as a StoreError that says what to do."""
        import studio_publish
        try:
            return work()
        except Exception as error:  # noqa: BLE001 - disk, network and the provider's own errors all end as a sentence
            raise studio_publish.store_error(error, store) from error

    def publish_version(self, number):
        store = self.version_store()
        try:
            row = self.store.get(number)
        except KeyError:
            raise ValueError(f"No version {number}.") from None
        version = {k: row.get(k) for k in ("number", "name", "change", "parent", "spec", "scorecard", "score_device")}
        key = self._through(store, lambda: store.publish(version, self._abs(row.get("best_checkpoint"))))
        self.store.add_published(number, key)
        return {"key": key, "number": number, "origin": key.split("/")[0]}

    def published_versions(self):
        import studio_publish
        store = self.version_store()
        entries = self._through(store, store.list)
        own = studio_publish.own_origin(store)
        here = {key: row["number"] for row in self._rows() for key in row.get("published") or ()}
        return [{**{k: entry.get(k) for k in ("key", "number", "name", "change", "origin", "device", "train_device", "score_device", "step", "published_at", "size")},
                 "scorecard": entry.get("scorecard") or {}, "mine": entry.get("origin") == own,
                 "local": entry["key"] in here, "local_number": here.get(entry["key"])} for entry in reversed(entries)]

    def share_version(self, number):
        import studio_publish
        try:
            row = self.store.get(number)
        except KeyError:
            raise ValueError(f"No version {number}.") from None
        version = {k: row.get(k) for k in ("number", "name", "change", "parent", "spec", "scorecard", "score_device")}
        return self._through(None, lambda: studio_publish.shared_copy(
            version, self._abs(row.get("best_checkpoint")), self.root / "data" / "shared", studio_publish.own_origin()))

    def pull_version(self, number, origin=None):
        found, said = self._from_peer(number, origin) if origin else (None, "")
        if found is not None:
            return found
        store = self.version_store()
        from studio_publish import StoreError
        entry = self._published(self._through(store, store.list), number, origin)
        key = entry["key"]
        if not entry.get("spec"):
            raise StoreError(f"The published version {key} carries no design, so it cannot be added here. Publish it again from {entry.get('origin')}.")
        row = self._held(key)
        if self._here(row):
            self.note(f"Version {number} from {entry.get('origin')} is already here, so nothing was downloaded.")
            return self._version(row)
        if not origin:  # the store has said who published it
            found, said = self._from_peer(number, entry.get("origin"))
            if found is not None:
                return found
        path = self._through(store, lambda: store.pull(key, self.root / "data" / "pulled"))
        v = self._land(entry, path, row, store)
        self.note(said or f"Downloaded version {number} from the version store.")
        return v

    def _held(self, key):
        """The row of the version here that came from, or was published as, store key `key`."""
        return next((r for r in self._rows() if key in (r.get("published") or ())), None)

    def _here(self, row):
        return row is not None and bool(row.get("best_checkpoint")) and Path(self._abs(row["best_checkpoint"])).is_file()

    def _land(self, entry, path, row, store):
        """The pulled file `path` as a version here: under `row` when this workspace had the version and lost the file, else as a new one."""
        if row is not None:
            self.store.set_best_checkpoint(row["number"], path)
            row = self.store.get(row["number"])
        else:
            name = f"{entry.get('name') or 'Version ' + str(entry.get('number'))} from {entry.get('origin')}"
            change = f"pulled from {entry.get('origin')}: {entry.get('change') or 'no description'}"
            row = self._through(store, lambda: self.store.add_pulled(entry.get("spec"), name, change, path, entry["key"], entry.get("train_device") or entry.get("device")))
        v = self._version(row)
        self.emit("versions", v)
        return v

    def _from_peer(self, number, origin):
        """(version, "") when version `number` was copied straight from the node that `origin` names, which is asked only if the
        workspace lists it. (None, sentence for the note) when the store has to do it: that node did not answer or could not send it."""
        import studio_cluster
        import studio_peer_pull as peer
        from studio_peer import PeerError
        try:
            addresses = studio_cluster.configured_nodes(self.workspace.state)
        except ValueError:
            return None, ""  # the cluster page says what is wrong with the list
        client, off = peer.find_owner(addresses, origin, self.runner.cluster.client_for)
        if client is None:
            return None, f"{origin} did not answer, so version {number} came from the version store." if off else ""
        try:
            meta, path = peer.fetch(client, number, self.root / "data" / "pulled", lambda meta: self._here(self._held(meta["key"])))
        except (PeerError, ValueError) as error:
            why = (error.detail or "It did not answer.") if isinstance(error, PeerError) else "What it sent did not check out."
            return None, f"{origin} could not send version {number}. {why.rstrip('.')}. It came from the version store instead."
        row = self._held(meta["key"])
        if path is None:
            self.note(f"Version {number} from {origin} is already here, so nothing was downloaded.")
            return self._version(row), ""
        v = self._land(meta, path, row, None)
        self.note(f"Copied version {number} straight from {origin}.")
        return v, ""

    @staticmethod
    def _published(entries, number, origin):
        """The published entry for version `number` (the newest, when its machine published several), or a sentence on why none."""
        from studio_publish import StoreError
        found = [e for e in entries if e.get("number") == number and (origin is None or e.get("origin") == origin)]
        if not found:
            by = f" by {origin}" if origin else ""
            raise StoreError(f"Version {number} is not published{by}. Publish it from the machine that trained it, then pull it again.")
        origins = sorted({e.get("origin") for e in found})
        if len(origins) > 1:
            raise StoreError(f"Version {number} is published by {', '.join(origins)}. Say which machine to pull it from.")
        return max(found, key=lambda e: e.get("published_at") or "")

    # queue ------------------------------------------------------------------
    def queue(self):
        return self.runner.state()

    def enqueue(self, job):
        import studio_queue
        if isinstance(job, Job):
            job = studio_queue.Job(**asdict(job))
        if self.runner.thread is None:
            self.runner.start()
        if job.kind in VERSION_JOBS and job.params.get("version"):
            self._to(job.params["version"], "queued")
        return self.runner.enqueue(job)

    def continue_run(self, number, kind, steps, dataset_ids=(), width=None, layers=None, spec=None, rows=16, hold=.05):
        from studio_ia import continue_run as cont
        from studio_ia import pretraining_plan as plan
        from studio_ia import sizing
        from studio_spec import validate_spec
        try:
            row = self.store.get(number)
        except KeyError:
            raise ValueError(f"No version {number}.") from None
        if not row["runs"]:
            raise cont.ContinueError(f"Version {number} has no training run to continue yet.")
        run = cont.run_of(self._abs(row["runs"][-1]))
        best = self._abs(row.get("best_checkpoint"))
        own = False
        if kind != "same":
            own = plan.user_text(plan.sets_of(self.data("text")), list(dataset_ids))
        if kind == "same":
            label, params = cont.same_job(number, run, steps, best)
        elif kind == "new_data":
            label, params = cont.new_data_job(number, run, steps, list(dataset_ids), own, best)
        elif kind == "grow":
            design = validate_spec(spec) if spec else sizing.resized_spec(validate_spec(row["spec"]), width, layers)
            if design is None:
                raise cont.ContinueError("This design is built block by block, so it cannot be resized automatically: save the bigger design in the Network designer.")
            warm = self.root / "runs" / f"warm_{uuid.uuid4().hex[:8]}" / "warm_start.npz"
            cont.write_warm_start(run, design, warm)  # first: a design the weights cannot go into saves no version
            grown = self.save_version(design, f"grown from version {number} at step {run['step']:,}")
            label, params = cont.grow_job(number, grown, run, warm, steps, list(dataset_ids), rows, hold, own, best)
        else:
            raise cont.ContinueError(f"Unknown way to continue: {kind}.")
        job = Job(id=uuid.uuid4().hex[:8], kind="pretrain", label=label, params=params, created_at=time.time())
        self.enqueue(job)
        return {"job_id": job.id, "label": label, "version": params["version"], "kind": kind, "from_step": run["step"]}

    def start_queue(self):
        """Start the runner when jobs are waiting from an earlier session; otherwise the first enqueue starts it."""
        if self.runner.thread is None and any(j.status == "waiting" for j in self.runner.jobs):
            self.runner.start()

    def resume_queue_at_start(self):
        """Carry on the queue a restart cut short: jobs waiting from the last session run again with no Start pressed (a job that
        was running continues from its own checkpoint, studio_jobs.continue_here). A pause or stop the person made is saved in
        queue.json and still holds: the runner starts but runs nothing while paused."""
        self.start_queue()

    @timed("services.refresh_downloads")
    def refresh_downloads(self):
        import studio_catalog as cat
        rebuilt = cat.refresh_downloads(self.workspace)
        if rebuilt:
            self.emit("downloads_finished", {"ids": rebuilt})  # the Your data page reloads its summaries once
        return rebuilt

    def move_up(self, job_id):
        self.runner.move_up(job_id)

    def move(self, job_id, index):
        self.runner.move(job_id, index)

    def hold(self, job_id, held=True):
        self.runner.hold(job_id, held)

    def free_space(self):
        import shutil
        return shutil.disk_usage(self.root).free

    def export_version(self, version, kind, path):
        import studio_export
        v = self._as_version(version)
        if v is None or not v.best_checkpoint:
            raise NoCheckpoint(v.number if v else None)
        return studio_export.export(v.best_checkpoint, kind, path)

    def import_pretrained(self, name="gpt2", lora=None):
        import studio_import
        if name not in studio_import.MODELS:
            raise ValueError(f"Unknown pretrained model {name!r}.")
        title = studio_import.MODELS[name]["title"]
        with self.runner.lock:
            if any(j.kind == "import_pretrained" and j.params.get("model") == name and j.status in ("waiting", "held", "running")
                   for j in self.runner.jobs):
                raise ValueError(f"{title} is already being imported. Wait for it in the queue before starting another.")
        v = self.save_version(studio_import.gpt2_spec(lora=lora), f"Started from {title}")
        out = studio_import.new_output_dir(self.workspace.run_dir, name)
        self.enqueue(Job(uuid.uuid4().hex[:8], "import_pretrained", f"Start from {title}",
                         "downloads the model once and turns it into a version that continues text and that you can fine-tune",
                         {"model": name, "version": v.number, "output_dir": str(out)}))
        return self._version(self.store.get(v.number))

    def storage(self, days=None, data=False):
        import studio_cleanup
        paths = studio_cleanup.plan(self.root, days, data)
        return len(paths), studio_cleanup.size(paths)

    def clean_storage(self, days=None, data=False):
        import studio_cleanup
        with self.runner.lock:
            if self.runner.running:
                raise RuntimeError("Stop the running job first.")
            if days is None:  # every job points at a version or run that is about to go
                self.runner.jobs = []
                self.runner._save()
            out = studio_cleanup.clean(self.workspace, days, data)
        self.runner._emit("queue", {})
        return out

    def _queue_rows(self):
        """The queue's jobs as {id, label, status, params}: what studio_cleanup judges a folder by. Call with the runner's lock held."""
        return [{"id": j.id, "label": j.label, "status": j.status, "params": dict(j.params)} for j in self.runner.jobs]

    def leftovers(self):
        import studio_cleanup
        with self.runner.lock:
            return studio_cleanup.leftovers(self.root, self._queue_rows())

    def remove_leftovers(self, ids):
        import studio_cleanup
        with self.runner.lock:  # no job starts between the check and the delete
            return studio_cleanup.remove_leftovers(self.root, ids, self._queue_rows())

    def remove(self, job_id):
        with self.runner.lock:
            job = next((j for j in self.runner.jobs if j.id == job_id), None)
            waiting = job is not None and job.status in ("waiting", "held")  # a running job is stopped and ends through _record_job; a failed one is only dismissed
        self.runner.remove(job_id)
        if waiting and job.kind == "import_pretrained" and job.params.get("version"):  # it never started: no event will say it ended
            self._drop_empty_import(job.params["version"], "stopped")

    def pause(self):
        self.runner.pause()

    def resume(self):
        self.runner.resume()

    def checkpoint_now(self):
        return self.runner.checkpoint_now()

    def set_plan(self, plan):
        self.runner.set_plan(plan)

    # training pages: pretraining data, job results, post-training rounds, verdict ------------------------------------
    def stop_after_cooldown(self):
        self.runner.stop_after_cooldown()

    def _text_rows(self, dataset_ids):
        import studio_data_kinds
        ids = dataset_ids or [d["id"] for d in self.workspace.state["datasets"]]
        return [e for e in studio_data_kinds.pretraining_examples(self.workspace, ids) if e.get("text")]

    def examples(self, params):
        """Plain-text rows of params["datasets"] (default: every set) without the held-back documents. data/heldout.json is
        fixed the first time (exact duplicates removed, seeded split) and then kept, so every version is scored on the same text."""
        import studio_decontam
        import studio_eval
        rows = self._text_rows(params.get("datasets"))
        path = self.root / "data" / "heldout.json"
        if not path.exists() and len(rows) > 1:  # ponytail: later sets never join the held-back text; delete the file to re-split
            train, held = studio_eval.heldout_split([e["text"] for e in rows], params.get("heldout_fraction", .05))
            held, report = studio_decontam.exclude_overlap(held, studio_decontam.Index().add_all(train))  # a held-back text that shares a 13-gram with training is not scored
            self._note_overlap("text", report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([{"id": f"heldout_{i}", "text": t} for i, t in enumerate(held)]), encoding="utf-8")
        held = set(self._heldout())
        good_only = bool(params.get("reviewed_only"))  # after the split: the held-back text is the same for every run, reviewed or not
        return [e for e in rows if e["text"] not in held and (not good_only or e.get("quality") == "Good")]

    def conversations(self, params):
        """The conversations of params["datasets"] (default: every set) that can make a teacher's prompt: a run with a teacher takes its
        polish and head-to-head prompts from them, and never trains on them. Drafts are left out, and with params["reviewed_only"] so is
        everything not marked Good. A set may hold only conversations."""
        return list(self._conversation_rows(params.get("datasets"), bool(params.get("reviewed_only"))))

    def _conversation_rows(self, dataset_ids, good_only):
        """The conversations of these sets a teacher can answer: not drafts, ending in an assistant turn (what studio_rlhf.prompts_from
        keeps), and only the ones marked Good when `good_only`."""
        import studio_data_kinds
        from studio_core import messages_of
        ids = dataset_ids or [d["id"] for d in self.workspace.state["datasets"]]
        return (e for i in ids for e in self.workspace.iter_kind(i, "conversations")
                if not studio_data_kinds.is_draft(e) and (not good_only or e.get("quality") == "Good")
                and (messages_of(e) or [{}])[-1].get("role") == "assistant")

    def has_conversations(self, dataset_ids, reviewed_only=False, at_least=1):
        import studio_diloco
        # Prompts, not rows: two chats that ask the same question make one prompt (the coordinator's split dedupes the same way).
        return len(studio_diloco.unique_prompts(self._conversation_rows(list(dataset_ids), reviewed_only))) >= at_least

    def has_reviewed_text(self, dataset_ids):
        """True when a reviewed_only run on these sets has text to train on: the rows examples() would give it, so a Good example held
        back for scoring does not count."""
        return bool(self.examples({"datasets": list(dataset_ids), "reviewed_only": True}))

    def _once(self, key):
        """The lock held while one cache entry is made: other entries are not held up, and an entry both shell lanes ask for is made once."""
        with self._lock:
            return self._made.setdefault(key, threading.Lock())

    def _fresh(self, key, make):
        """make() once per state of the stored data (Workspace.revision). The Tk thread asks ready() first, so it only ever
        picks up a finished result; a lane that asks while the other makes the same entry waits and takes it."""
        revision = self.workspace.revision
        hit = self._data.get(key)
        if hit is None or hit[0] != revision:
            if getattr(self._probe, "on", False):
                raise _Cold
            with self._once(key):
                revision = self.workspace.revision
                hit = self._data.get(key)
                if hit is None or hit[0] != revision:
                    hit = self._data[key] = (revision, make())
        return hit[1]

    def ready(self, work):
        self._probe.on = True
        try:
            work()
            return True
        except _Slow:
            return False
        except _Cold:  # a small store is read in milliseconds, in place; a large one goes to the worker
            return self.workspace.is_small()
        finally:
            self._probe.on = False

    def _stamp(self):
        """Size and time of the files the stored data lives in: says whether data/summaries.sqlite still describes them."""
        db = self.workspace.db_path

        def stat(path):
            try:
                s = path.stat()
                return [s.st_mtime_ns, s.st_size] if s.st_size else None
            except OSError:
                return None
        return [stat(self.workspace.path), stat(db), stat(db.with_name(db.name + "-wal"))]

    @timed("services._summaries")
    def _summaries(self):
        """{kind: DataSummary with the first page of sample rows, "example": the first training conversation, "takes": what
        a search reads}. Kept per dataset in data/summaries.sqlite beside each dataset's `rev`, so a change to one dataset
        reads that dataset only and a launch reads nothing the file still describes."""
        import studio_data_kinds as dk

        def make():
            stamp = self._stamp()
            cache = self._summary_cache()
            cached, saved, view = cache.load()
            unchanged = saved == stamp
            datasets = list(self.workspace.state["datasets"])
            # The files differ from what the summary was made of, yet no dataset's rev moved: an older app wrote them without
            # counting, so nothing can be trusted.
            moved = any(cached.get(d["id"], {}).get("rev") != d.get("rev", 0) for d in datasets)
            rescan_all = bool(cached) and not unchanged and not moved
            try:
                entries, reread = dk.refresh(self.workspace, cached, rescan_all=rescan_all, progress=self.note)
            except (KeyError, TypeError, IndexError, AttributeError):  # an entry that does not have the shape this version writes
                entries, reread = dk.refresh(self.workspace, {}, progress=self.note)
            if reread or set(entries) != set(cached):
                self.note("Saving the summary")
                view = cache.save(datasets, entries, stamp)
                (self.root / "data" / "summaries.json").unlink(missing_ok=True)  # the layout before 4: one large file
            out, example = dk.assemble(datasets, entries, view)
            out["example"], out["takes"] = example, view["takes"]
            return out
        return self._fresh("summaries", make)

    def _summary_cache(self):
        from studio_data_cache import SummaryCache
        return SummaryCache(self.root / "data" / "summaries.sqlite", SUMMARY_FORMAT)

    def pretrain_preview(self, dataset_ids, rows=16):
        import studio_estimate
        import studio_spec
        from studio_model import read_metrics
        v = self.current_version()
        if v is None or not v.spec:
            return {}
        spec = studio_spec.validate_spec(v.spec)
        context = spec["dims"]["n_ctx"]

        def make():
            import studio_data_kinds as dk
            self.note("Reading a sample of the chosen text")
            ids = dataset_ids or [d["id"] for d in self.workspace.state["datasets"]]
            # Never the whole corpus: about 5M evenly spread characters build the word list and are counted; the
            # total size comes from stored character counts, so pieces are scaled up ("estimated") for larger text.
            sample, total = dk.text_sample(self.workspace, ids, PREVIEW_CHARS, exclude=set(self._heldout()))
            self.note("Building the word list")
            module, cfg = studio_spec.module_of("tokenizer", spec["tokenizer"])
            tok = module.fit(cfg, [{"text": d} for d in sample])
            self.note("Counting word pieces")
            encoded = [tok.encode(d) for d in sample]
            counted, chars = sum(map(len, encoded)) + len(sample), sum(map(len, sample))
            estimated = total > chars
            pieces = round(counted * total / max(1, chars)) if estimated else counted
            return {"pieces": pieces if sample else None, "estimated": estimated,
                    "params": studio_estimate.estimate(spec, tok.vocab_size)["total"], "vocab": tok.vocab_size,
                    "context": context, "first_rows": layout_rows([len(e) + 1 for e in encoded], context, rows)}
        key = self._preview_key(v, dataset_ids, rows)
        store = self._previews()
        if key not in store:
            preview = make()
            from studio_core import atomic_json
            with self._lock:
                store[key] = preview
                while len(store) > PREVIEWS_KEPT:
                    store.pop(next(iter(store)))
                atomic_json(self.root / "data" / "previews.json", store)
        return {**store[key], "tokens_per_second": self._speed(read_metrics)}

    def word_lists(self):
        def make():
            import studio_data_kinds as dk
            from studio_ia import pieces
            ids = [d["id"] for d in self.workspace.state["datasets"]]
            texts, _ = dk.text_sample(self.workspace, ids, pieces.SAMPLE_CHARS, exclude=set(self._heldout()))
            return pieces.word_lists(texts, self._cached("gpt2", self._gpt2))
        return self._fresh("word_lists", make)

    def lr_range_test(self, spec, stop=None):
        import studio_compute
        import studio_data_kinds as dk
        ids = [d["id"] for d in self.workspace.state["datasets"]]
        texts, _ = dk.text_sample(self.workspace, ids, RANGE_CHARS, exclude=set(self._heldout()))
        if not texts:
            raise ValueError("There is no plain text to try it on. Add some in Your data.")
        # the processor only: this runs on a worker thread while Tk runs, where a GPU probe or Metal work crashes the app on macOS
        return studio_compute.lr_range_test(spec, [{"text": t} for t in texts], stop=stop, config={"device": "cpu"})

    def _gpt2(self):
        """GPT-2's list, downloaded once into data/gpt2 as training with it would; None when it cannot be fetched."""
        from studio_modules.tokenizer import bpe_gpt2
        try:
            return bpe_gpt2.load_assets(bpe_gpt2.GPT2Config())
        except ValueError:
            return None

    def cached_preview(self, dataset_ids, rows=16):
        """pretrain_preview's answer when it is already known (this session or data/previews.json), else None. Never
        computes, so the page can open at once and ask for the rest in the background."""
        from studio_model import read_metrics
        v = self.current_version()
        if v is None or not v.spec:
            return {}
        hit = self._previews().get(self._preview_key(v, dataset_ids, rows))
        return None if hit is None else {**hit, "tokens_per_second": self._speed(read_metrics)}

    def cached_step_rate(self, spec, rows):
        return self._step_rates().get(self._rate_key(spec, rows))

    def measure_step_rate(self, spec, rows, vocab, stop=None):
        import studio_scaling
        from studio_core import atomic_json
        found = studio_scaling.measure_step_rate(spec, rows, vocab, stop=stop)
        if found is None:
            return None
        found = {**found, "measured_at": time.time()}
        store = self._step_rates()
        with self._lock:
            store[self._rate_key(spec, rows)] = found
            while len(store) > PREVIEWS_KEPT:
                store.pop(next(iter(store)))
            atomic_json(self.root / "data" / "step_rates.json", store)
        return found

    def _step_rates(self):
        """{design and rows -> measured step rate}, kept in data/step_rates.json: a rate belongs to this computer and its data folder."""
        if getattr(self, "_step_rate_store", None) is None:
            path = self.root / "data" / "step_rates.json"
            try:
                loaded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, ValueError):
                loaded = {}
            self._step_rate_store = loaded if isinstance(loaded, dict) else {}
        return self._step_rate_store

    @staticmethod
    def _rate_key(spec, rows):
        return json.dumps([spec, rows], sort_keys=True, default=str)

    def loss_evidence(self, version):
        from studio_core import read_json
        from studio_model import read_metrics
        try:
            run = Path(self._abs(self.store.get(version.number)["runs"][-1]))
        except (KeyError, IndexError, TypeError):
            return None
        try:
            record = read_json(run / "run.json") if (run / "run.json").exists() else {}
            config = read_json(run / "config.json")
            context = read_json(run / "spec.json")["dims"]["n_ctx"]
            rows = read_metrics(run / "metrics.jsonl")
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if record.get("initialized_from") or record.get("resumed_from"):
            return None  # a run that began from other weights is not this design's curve from step 0
        per_step = int(config.get("batch_size", 1)) * int(config.get("accumulation", 1)) * int(context)
        points = [(r["step"] * per_step, r["val_loss"]) for r in rows if r.get("val_loss") is not None and r.get("step")]
        return {"points": points, "run": str(run)} if points else None

    def device_facts(self):
        from studio_backend import gpu_status
        try:
            return gpu_status()
        except Exception:  # noqa: BLE001 - a broken driver is "no GPU" here
            return None

    def _preview_key(self, version, dataset_ids, rows):
        """Same chosen sets, design and stored data (records, held-back text) -> same preview."""
        return json.dumps([sorted(dataset_ids), rows, version.spec, self._stamp(), self._heldout_stamp()], sort_keys=True, default=str)

    def _previews(self):
        if getattr(self, "_preview_store", None) is None:
            path = self.root / "data" / "previews.json"
            try:
                loaded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, ValueError):
                loaded = {}
            self._preview_store = loaded if isinstance(loaded, dict) else {}
        return self._preview_store

    def _speed(self, read_metrics):
        """Pieces per second of the newest run that logged one (for the time estimate)."""
        for ver in reversed(self.versions()):
            for run in reversed(ver.runs):
                got = next((r["tokens_per_second"] for r in reversed(read_metrics(Path(run) / "metrics.jsonl")) if r.get("tokens_per_second")), None)
                if got:
                    return got
        return None

    # job results -> versions
    def _to(self, number, status):
        try:
            self.store.set_status(number, status)
        except (ValueError, KeyError):
            pass

    def _record_job(self, kind, payload):
        """Runner event: a pretrain job moves its version to training, then records the run, its best checkpoint and its
        held-back bits per character (the scorecard Chat follows) when it ends."""
        if kind != "job" or not isinstance(payload, dict):
            return
        job = next((j for j in self.runner.jobs if j.id == payload.get("job")), None)
        if job and job.kind in ("post_round", "restore_answers") and payload["status"] != "running" and job.params.get("result"):
            result = job.params["result"]  # the round ran in a worker: its gate result is kept here, in the app
            return self.record_round(result["round"], job.params.get("version"), result)
        number = job.params.get("version") if job and job.kind in VERSION_JOBS else None
        if not number:
            return
        if payload["status"] == "running":
            return self._to(number, "training")
        out = Path(job.params.get("output_dir") or "")
        if job.kind == "pretrain_distributed":
            return self._record_distributed(job, number, payload["status"], out)
        ckpt = next((p for p in (out / "best.npz", out / "last.npz") if p.exists()), None)
        if not ckpt and job.kind == "import_pretrained" and payload["status"] in ("failed", "stopped"):
            return self._drop_empty_import(number, payload["status"])
        if ckpt:
            import studio_eval
            self.store.add_run(number, out)
            self.store.set_best_checkpoint(number, ckpt)
            scores = self._held_scores(self._abs(ckpt))  # per document, so Chat's paired test later reads it from disk
            if scores is not None:
                card = {**(self.store.get(number).get("scorecard") or {}), "bpc": studio_eval.corpus_bpc(scores),
                        "bpc_method": studio_eval.SCORE_METHOD}
                self.store.set_scorecard(number, card)
            self._note_devices(number, ckpt, scores is not None)
        self._to(number, "trained" if ckpt and payload["status"] == "done" else "saved")
        self.emit("versions", None)
        if ckpt and payload["status"] == "done":
            self._bench_later(number)

    def _record_distributed(self, job, number, status, out):
        """A distributed run ends: only a run that finished all its rounds gives its coordinator's final weights (`last.npz` in the
        run's folder) to the version it trained, scored like a local pretrain's, with a change note naming the run and its machines.
        A run that finished under a successor coordinator after a failover (#361) gives the successor's final weights, fetched from it,
        whatever way the owner's own job ended (#453: its member may have stopped or failed while the run went on), unless the person
        stopped the run. Otherwise a stopped or failed run saves nothing, even past some rounds: the version goes back to saved."""
        import studio_distributed
        import studio_eval
        from studio_core import read_json
        record, ckpt = out / "run.json", out / "last.npz"
        done = status == "done" and ckpt.exists() and record.exists() and read_json(record).get("status") == "completed"
        if done and (out / "best.npz").exists():
            ckpt = out / "best.npz"  # the best round's weights, not the last round's (#487)
        if not done and status in ("done", "failed", "stopped") and not studio_distributed.run_of(job).stopped:
            ckpt = self._successor_weights(job.id, out)
            done = ckpt is not None
        if done:
            machines = [m.node for m in studio_distributed.run_of(job).machines if m.state != "unreachable"]
            self.store.add_run(number, out)
            self.store.set_best_checkpoint(number, ckpt)
            scores = self._held_scores(self._abs(ckpt))
            if scores is not None:
                card = {**(self.store.get(number).get("scorecard") or {}), "bpc": studio_eval.corpus_bpc(scores), "bpc_method": studio_eval.SCORE_METHOD}
                self.store.set_scorecard(number, card)
            self._note_devices(number, ckpt, scores is not None)
            self.store.set_change(number, f"{self.store.get(number)['change']} (trained in distributed run {job.id} on {', '.join(machines)})")
            ran = read_json(record) if record.exists() else {}
            self.store.set_source(number, {"kind": "distributed", "run": job.id, "checkpoint": str(ckpt), "best_round": ran.get("best_round"),
                                           "val_loss": ran.get("best_val_loss"), "stop_rule": ran.get("stop_rule", ""),
                                           "stop_reason": ran.get("stop_reason", "")})
            studio_distributed.note_saved(job.id, number)
        self._to(number, "trained" if done else "saved")
        self.emit("versions", None)
        if done:
            self._bench_later(number)

    def _successor_weights(self, run, out):
        """The final weights of `run` from the coordinator the where-file names (studio_failover writes it after a failover, term 2 or
        more), saved in the run's folder as last_term<N>.npz. None when there is no successor or it cannot send them (not finished,
        stopped, unreachable): the run then saves nothing, as one that did not finish. Network: call it off the UI thread."""
        import studio_diloco
        from studio_core import read_json
        import zipfile
        from studio_peer import PeerClient, PeerError
        try:
            where = read_json(studio_diloco.where_file(self.root, run))
            node, term, url = str(where["node"]), int(where["term"]), str(where["url"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if term < 2:
            return None
        try:
            body = PeerClient(url).get_bytes(f"/collective/{run}/weights", {"term": term})
            np.load(io.BytesIO(body), allow_pickle=False).close()  # an archive, before it is kept
        except (PeerError, OSError, ValueError, zipfile.BadZipFile) as exc:
            _log.warning("The final weights of run %s could not be fetched from %s (term %s): %s", run, node, term, exc)
            return None
        path = out / f"last_term{term}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(body)
        part.replace(path)
        return path

    def _note_devices(self, number, ckpt, scored=False):
        """Record on the version the device kind that trained `ckpt` and, when the caller just scored it, the one that scored it."""
        import studio_eval
        from studio_model import checkpoint_device
        self.store.set_devices(number, train=checkpoint_device(ckpt), score=studio_eval.SCORE_DEVICE if scored else None)

    def _drop_empty_import(self, number, how):
        """An import that ended without weights leaves a version that holds only a design. The newest can simply go; an older one
        cannot (numbers follow the order), so it says in its description that it has no weights."""
        if not self.store.remove_last(number):
            reason = "was stopped" if how == "stopped" else "did not finish"
            self.store.leave_saved(number, f", but the download {reason}, so this design has no weights")
        self.emit("versions", None)

    def _json(self, name, default):
        path = self.root / "data" / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def _save_json(self, name, value):
        from studio_core import atomic_json
        atomic_json(self.root / "data" / name, value)

    def record_round(self, number, version, result):
        """A finished post-training round: keep the result; only a passed gate moves the version's checkpoint (Chat follows it)."""
        v = self._as_version(version) if version else self.current_version()
        if v is None:
            return
        done = self._json("post_training.json", {})
        done.setdefault(str(v.number), {})[str(number)] = {k: result.get(k) for k in ("ok", "checks", "verdict", "before", "after", "band", "checkpoint", "groups", "taught", "skipped")}
        self._save_json("post_training.json", done)
        if result.get("ok"):
            card = dict(v.scorecard or {})
            after = result.get("after") or {}
            import studio_eval
            card.update({"bpc": after.get("bpc", card.get("bpc")), **({"bpc_method": studio_eval.SCORE_METHOD} if after.get("bpc") is not None else {}),
                         "problems": [after.get("pass_k"), after.get("pass_n")] if after.get("pass_n") else card.get("problems")})
            if number == 1:
                card["answers"] = True
            self.store.set_best_checkpoint(v.number, result["checkpoint"])
            self.store.set_scorecard(v.number, card)
            self._note_devices(v.number, self._abs(result["checkpoint"]))
        self.emit("versions", None)
        if result.get("ok"):
            self._bench_later(v.number)  # the weights changed: score the new file

    def _human_pairs(self):
        from studio_rlhf import PreferenceStore
        return PreferenceStore(self.root).all(["human"])

    def post_training(self, version):
        v = self._as_version(version) if version else self.current_version()
        done = self._json("post_training.json", {}).get(str(v.number), {}) if v else {}
        q = self.queue()

        def round_of(job):
            return 3 if job.kind == "restore_answers" else int(job.params.get("round", 0))
        running = round_of(q.job) if q.job and q.job.kind in ("post_round", "restore_answers") else None
        waiting = {round_of(j) for j in q.up_next if j.kind in ("post_round", "restore_answers")}
        rounds = {}
        for n in range(1, 5):
            g = done.get(str(n))
            status = "running" if running == n else "queued" if n in waiting else "done" if g else "idle"
            rounds[n] = {"status": status, "note": ("live" if g["ok"] else "kept, not live") if g and status == "done" else ""}
        convs = self.data("conversations")
        return {"rounds": rounds, "gates": {int(n): g for n, g in done.items()},
                "counts": {"conversations": convs.count, "corrections": len(self._json("corrections.json", [])),
                           "problems": self.data("problems").count, "pairs": len(self._human_pairs())},
                "overlap": self._json("overlap.json", {}),
                "example": self._summaries()["example"]}

    def _note_overlap(self, name, report):
        """Keep how many held-back items were left out for sharing a 13-gram with training (data/overlap.json, read by post_training)."""
        saved = self._json("overlap.json", {})
        if saved.get(name) != report.as_dict():
            self._save_json("overlap.json", {**saved, name: report.as_dict()})

    def _train_index(self):
        """The 13-grams of everything training reads: pretraining text and the conversations and problems that are not held back."""
        def make():
            import studio_data_kinds as dk
            import studio_decontam
            held, index = set(self._heldout()), studio_decontam.Index()
            for d in self.workspace.state["datasets"]:
                for e in self.workspace.iter_examples(d["id"]):
                    if not dk.is_draft(e) and not (e["text"] in held if dk.classify(e) == "text" else dk.stamp(e)["holdout"]):
                        index.add(_doc(e))
            return index
        return self._fresh("train_index", make)

    def post_rows(self, kind):
        import studio_data_kinds as dk
        if kind == "heldout_docs":
            return self._heldout()
        if kind == "pairs":
            return self._human_pairs()
        if kind == "corrections":
            return self._json("corrections.json", [])
        want = "problems" if kind.endswith("problems") else "conversations"
        held = kind == "heldout_problems"
        datasets = list(self.workspace.state["datasets"])
        # disk datasets: one indexed query on the kind and holdout columns; in-memory ones (under 10,000 rows each) carry the same two stamps
        on_disk = self.workspace.pick(want, held, [d["id"] for d in datasets if d.get("storage") == "sqlite"])
        out = []
        for d in datasets:
            if d["id"] in on_disk:
                rows = on_disk[d["id"]]
            else:
                rows = [e for e in self.workspace.iter_examples(d["id"]) if dk.stamp(e)["kind"] == want and bool(e["holdout"]) == held]
            out.extend({"prompt": e.get("user", ""), "rule": dk.rule_of(e), "params": e.get("params") or (e.get("metadata") or {}).get("params", {})}
                       if want == "problems" else e for e in rows)
        if held:  # a held-back problem whose text also appears in training would be answered from memory: leave it out and say how many
            import studio_decontam
            out, report = studio_decontam.exclude_overlap(out, self._train_index(), lambda row: row["prompt"])
            self._note_overlap("problems", report)
        return out

    def post_loops(self):
        import studio_loop
        return studio_loop.list_loops(self.workspace.run_dir)

    def start_loop(self, version, config=None, resume=None, judge_model=None):
        import studio_loop
        from studio_core import uid
        v = self._as_version(version) if version else self.current_version()
        if v is None:
            raise ValueError("There is no version yet. Build one first.")
        if not v.best_checkpoint:
            raise NoCheckpoint(v.number)
        loop_id = resume or uid("loop")
        folder = Path(self.workspace.run_dir) / f"rlhf_loop_{loop_id}"
        if resume:
            if studio_loop.load_state(folder) is None:
                raise ValueError(f"There is no loop {resume} to carry on.")
        else:
            studio_loop.clean_config(config)  # a refused setting is told now, not when the job starts
        with self.runner.lock:
            busy = [j for j in self.runner.jobs if j.kind == "rlhf_loop" and j.status in ("waiting", "held", "running")
                    and (j.params.get("loop_id") == loop_id or (not resume and j.params.get("version") == v.number))]
        if busy:
            raise ValueError("A loop is already queued or running for this version. Stop it first.")
        params = {"version": v.number, "checkpoint": str(v.best_checkpoint), "config": dict(config or {}), "loop_id": loop_id,
                  "output_dir": str(folder), "budget_token": uid("budget") if resume else None, "judge_model": judge_model}
        job = Job(uuid.uuid4().hex[:8], "rlhf_loop", f"RLHF loop · version {v.number}", "automated preference rounds", params=params)
        self.enqueue(job)
        return {"job": job.id, "loop_id": loop_id, "label": job.label}

    def adopt_loop(self, loop_id):
        import studio_loop
        state = studio_loop.load_state(Path(self.workspace.run_dir) / f"rlhf_loop_{loop_id}")
        if state is None:
            raise ValueError(f"There is no loop {loop_id}.")
        result = studio_loop.adoption(state, {kind: self.post_rows(kind) for kind in ("heldout_docs", "heldout_problems")})
        if result["ok"]:
            self.record_round(result["round"], state["version"], result)
        return result

    def teach_correction(self, version, messages, correction, whole_answer=False):
        """A whole new answer becomes a conversation for round 1 (dataset "From Chat"); anything else is a correction the
        round-3 teacher sees in front of the conversation (the student never does)."""
        if whole_answer:
            import studio_data_kinds as dk
            prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            self.workspace.add(dk.ensure_dataset(self.workspace, "From Chat", "Answers you wrote in Chat"),
                               [{"user": prompt, "assistant": correction, "quality": "Good"}], source="correction")
            return
        saved = self._json("corrections.json", [])
        saved.append({"messages": [dict(m) for m in messages], "correction": correction, "version": getattr(self._as_version(version), "number", None)})
        self._save_json("corrections.json", saved)

    def rate_pair(self, messages, chosen, rejected):
        from studio_rlhf import PreferenceStore
        return PreferenceStore(self.root).add(messages, chosen["reply"], rejected["reply"], "human",
                                              chosen_version=chosen.get("version"), rejected_version=rejected.get("version"))

    def _curve_for(self, number):
        """(points [(trained tokens, held-back bits/char)], schedule id, finished) of a version's latest run."""
        from studio_core import read_json
        from studio_model import read_metrics
        v = self._as_version(number)
        run = Path(v.runs[-1]) if v and v.runs else None
        if run is None or not (run / "metrics.jsonl").exists():
            return [], "", False
        curve = self.run_curve(run)
        tokens = {r["step"]: r.get("trained_tokens") or r["step"] for r in read_metrics(run / "metrics.jsonl")}
        spec = read_json(run / "spec.json") if (run / "spec.json").exists() else {}
        status = read_json(run / "run.json").get("status") if (run / "run.json").exists() else None
        return [(tokens.get(s, s), y) for s, y in curve["heldout"]], json.dumps([spec.get("train", {}).get("schedule"), curve["total"]]), status == "completed"

    def verdict(self, a, b):
        """Training saved by the newer of versions a, b over the older; no verdict unless both finished the same schedule."""
        import studio_eval
        older, newer = sorted((a, b))
        (po, so, fo), (pn, sn, fn) = self._curve_for(older), self._curve_for(newer)
        for number, done in ((older, fo), (newer, fn)):
            if not done:
                return {"percent": None, "comparable": False, "reason": f"Version {number} has no finished run to read."}
        return studio_eval.training_saved(pn, po, sn, so)

    def few_shot(self, version):
        """A seeded sample of the held-back lettered problems, asked of the version's best checkpoint; worked examples come from
        every choice problem of the same task. Cached per checkpoint file and state of the stored data."""
        import random
        import studio_eval
        ckpt = self._checkpoint(version)

        def make():
            held = [i for i in map(studio_eval.choice_item, self.post_rows("heldout_problems")) if i]
            pool = [i for i in map(studio_eval.choice_item, self.post_rows("problems")) if i]
            asked = held or pool  # ponytail: no held-back choice problem yet, so training ones are asked (flagged held_back False)
            asked = random.Random(0).sample(asked, min(FEW_SHOT_ITEMS, len(asked)))
            return {**studio_eval.few_shot(ckpt, asked, pool=held + pool, norm="char"), "tasks": len({i["task"] for i in asked}), "held_back": bool(held)}
        return self._cached(("few_shot", ckpt, Path(ckpt).stat().st_mtime_ns, self.workspace.revision), make)

    def _bench_sources(self, held):
        """{held-back document: name of the dataset it came from}, so the suite spreads its passages over the data sources."""
        import studio_data_kinds as dk
        held, out = set(held), {}
        for d in self.workspace.state["datasets"]:
            for e in self.workspace.iter_examples(d["id"]):
                if e.get("text") in held and dk.classify(e) == "text":
                    out.setdefault(e["text"], d.get("name") or d["id"])
        return out

    def _bench_build(self):
        """The suite as the stored data would make it now (studio_bench.build_suite). Read only when the frozen suite is missing or
        lacks a kind of item; items that share a 13-gram with the workspace's training text are left out, and counted."""
        import studio_bench

        def make():
            import studio_decontam
            index = self._train_index()
            text = {"docs": lambda d: d, "choice": lambda i: "\n".join([i["question"], *i["choices"]]), "problems": lambda p: p["prompt"]}

            def decontaminated(kind, items):  # an item sharing a 13-gram with training text (studio_decontam) is left out
                return items if kind == "pool" else studio_decontam.exclude_overlap(items, index, text[kind])[0]  # worked examples are training problems
            docs = self._heldout()
            return studio_bench.suite_from_rows(docs, self.post_rows("heldout_problems"), self.post_rows("problems"),
                                                filter_items=decontaminated, sources=self._bench_sources(docs))
        return self._cached(("bench_suite", self._heldout_stamp(), self.workspace.revision), make)

    def _bench_suite(self):
        """The workspace's frozen eval suite (studio_bench.frozen): chosen from the held-back data the first time it is asked for,
        then kept in data/bench/suite.json, so every version is scored on the same items."""
        import studio_bench
        return studio_bench.frozen(self.root, self._bench_build)

    def bench_suite(self):
        """The frozen suite, for a caller that must keep its own prompts away from it: {id, docs, choice, pool, problems, ...}.
        studio_bench.suite_index(suite).overlaps(text) says whether a prompt is, or contains a run of 13 words of, one of its items."""
        return self._bench_suite()

    def _bench_tokens(self, v, checkpoint):
        """Tokens seen by a version's checkpoint over every run it descends from (studio_bench.tokens_seen), or None."""
        import studio_bench

        def owner(path):  # the runs of the version that made a checkpoint that does not sit in a run folder with a log
            return next(([self._abs(r) for r in w.runs] for w in self.versions() if w.best_checkpoint and Path(w.best_checkpoint) == Path(path)), None)
        return studio_bench.tokens_seen([self._abs(r) for r in v.runs], checkpoint, owner)

    def _continues(self, v, versions):
        """The number of the version whose weights `v` was started from (run.json initialized_from), or None."""
        from studio_core import read_json
        if not v.runs:
            return None
        try:
            start = read_json(Path(self._abs(v.runs[-1])) / "run.json").get("initialized_from")
        except (OSError, ValueError, AttributeError):
            return None
        if not start:
            return None
        start = Path(start)
        for w in versions:
            if w.number == v.number:
                continue
            if (w.best_checkpoint and Path(w.best_checkpoint) == start) or any(Path(self._abs(r)) == start.parent for r in w.runs):
                return w.number
        return None

    def _lineages(self, versions):
        """{version number: (lineage id, number it continues)}. Versions that continue one another share a lineage; one that
        does not is grouped with the others of its design (same structure, trained differently), as before."""
        import hashlib
        from studio_spec import structure_key
        continues = {v.number: self._continues(v, versions) for v in versions}
        children = {n for n in continues.values() if n}

        def root(n):
            seen = {n}
            while continues.get(n) and continues[n] not in seen:
                n = continues[n]
                seen.add(n)
            return n
        out = {}
        for v in versions:
            if continues[v.number] or v.number in children:
                out[v.number] = (f"chain-{root(v.number)}", continues[v.number])
                continue
            try:
                design = hashlib.sha1(structure_key(v.spec).encode()).hexdigest()[:8]  # one design, trained differently, is one line
            except (ValueError, KeyError, TypeError):
                design = f"v{v.number}"
            out[v.number] = (design, None)
        return out

    def _bench_skip(self, v, versions):
        """(suite items this version's training text contains, studio_bench.contaminated) or None when its training text is not on this
        computer (a pulled or imported version): then nothing is dropped and the result says "unknown". The text is every run
        dataset.jsonl the weights descend from."""
        import studio_decontam
        runs, seen, current = [], set(), v
        while current is not None and current.number not in seen:
            seen.add(current.number)
            runs += [Path(self._abs(r)) for r in current.runs]
            parent = self._continues(current, versions)
            current = next((w for w in versions if w.number == parent), None)
        runs = [r for r in runs if (r / "dataset.jsonl").is_file()]
        if not runs:
            return None
        return studio_decontam.index_of_runs(runs)

    def _bench_run(self, v, checkpoint, suite, number=None):
        import studio_bench
        versions = self.versions()
        index = self._bench_skip(v, versions)
        skip = studio_bench.contaminated(suite, index) if index is not None else None
        result = studio_bench.benchmark(self.root, (v.number if number is None else number, checkpoint), suite,
                                        self._bench_tokens(v, checkpoint), skip=skip)
        return result

    def mlperf_board(self, task_id):
        import studio_backend
        import studio_mlperf
        results = studio_mlperf.load_results(self.root)
        gpu = studio_backend.gpu_status()
        return {"tasks": studio_mlperf.task_list(), "board": studio_mlperf.leaderboard(results, task_id),
                "history": studio_mlperf.history(results, task_id), "devices": ["cpu"] + ([gpu["kind"]] if gpu.get("available") else [])}

    def mlperf_compare(self, ids):
        import studio_mlperf
        return studio_mlperf.compare(studio_mlperf.load_results(self.root), ids)

    def mlperf_enqueue(self, task_id, device, attempts=None):
        import studio_mlperf
        task = studio_mlperf.get_task(task_id)
        label = f"Training speed: {task.title} v{task.version} on {device}"
        job = Job(id=uuid.uuid4().hex[:8], kind="mlperf", label=label, params={"task": task_id, "device": device, "attempts": attempts})
        self.enqueue(job)
        return {"id": job.id, "label": label}

    def benchmark(self):
        import studio_bench
        suite = self._bench_suite()
        versions = [v for v in self.versions() if v.best_checkpoint and Path(v.best_checkpoint).exists() and v.status != "training"]
        lineage = self._lineages(self.versions())
        rows = []
        for v in versions:
            result = studio_bench.cached(self.root, v, suite)
            rows.append({"number": v.number, "name": v.name, "lineage": lineage[v.number][0], "continues": lineage[v.number][1],
                         "tokens": self._bench_tokens(v, v.best_checkpoint), "result": result,
                         "stale": result is None and studio_bench.stale(self.root, v, suite)})
        return {"suite": suite["id"], "sizes": {k: len(suite[k]) for k in ("docs", "choice", "problems")}, "dropped": suite["dropped"],
                "sources": sorted({s for s in suite.get("sources", []) if s}), "rows": rows, "lineages": studio_bench.trend(rows)}

    def run_benchmark(self, number):
        v, checkpoint = self._as_version(number), self._checkpoint(number)
        return self._bench_run(v, checkpoint, self._bench_suite())

    def _bench_later(self, number):
        """Score version `number` on the frozen suite on a worker, after any already waiting, then emit "versions" so the Benchmark
        page reads it. Called when a version gets weights. Cheap on CPU (a fixed small suite, a single worker, threads capped as
        in _score_later), and resumable (studio_bench.benchmark keeps each finished stage). A failure is logged and the page's
        "Check the rest" still scores it."""
        if not self.auto_bench or not hasattr(self, "_bench_lock"):
            return
        with self._bench_lock:
            if number in self._bench_wait:
                return
            self._bench_wait.append(number)
            if self._bench_busy:
                return
            self._bench_busy = True

        def run():
            from studio_backend import cap_threads
            cap_threads()
            while True:
                with self._bench_lock:
                    if not self._bench_wait:
                        self._bench_busy = False
                        break
                    todo = self._bench_wait[0]
                try:
                    suite = self._bench_suite()
                    if any(suite[k] for k in ("docs", "choice", "problems")):
                        self.run_benchmark(todo)
                except Exception:
                    logging.getLogger(__name__).exception("scoring version %s on the frozen suite failed", todo)
                with self._bench_lock:
                    self._bench_wait.remove(todo)
            self.emit("versions", None)
        self._bench_thread = start_thread(run)

    def gate_eval(self, checkpoint, baseline=None):
        """For an automated loop (the RLHF gate): `checkpoint`'s scores on the frozen suite, and when `baseline` (a checkpoint path or a
        version number) is given, studio_bench.gate_verdict of it against the baseline. Returns {scores, baseline, verdict}.

        The suite is the same one every version is scored on, so the numbers sit on the Benchmark page's trend. Any checkpoint file
        works, a round's candidate included; its result is kept by file (path, size, time) so asking again is free. The items are
        held-back text and problems the judge never sees: keep your own prompts away from them with bench_suite() and
        studio_bench.suite_index. Slow (a minute or so on CPU the first time): call it from a worker."""
        import hashlib
        import studio_bench
        suite = self._bench_suite()

        def scores(target):
            if isinstance(target, int):
                v = self._as_version(target)
                return self._bench_run(v, self._checkpoint(target), suite)
            path = self._abs(str(target))
            key = "gate-" + hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:12]
            return studio_bench.benchmark(self.root, (key, path), suite, None)  # a file that is no version: its training text is unknown
        got = scores(checkpoint)
        base = None if baseline is None else scores(baseline)
        return {"scores": got, "baseline": base, "verdict": None if base is None else studio_bench.gate_verdict(base, got)}

    # data / evaluation ------------------------------------------------------
    def text_sets(self):
        """The text sets from the counts storage keeps exact (Workspace.kind_stats), not from the summary: answering needs no scan
        of the rows, so it takes milliseconds on a workspace of a million of them."""
        return self._sets_holding("text")

    CHAT_COUNT_CAP = 1000  # the count of a set's prompts stops here (the form shows "1000+"): reading a huge set to the end buys nothing

    def conversation_sets(self):
        """[{id, name, count}] of the sets with prompts a run's teacher can answer: count is the different prompts (what has_conversations
        counts), at most CHAT_COUNT_CAP; a set with none is not listed."""
        import studio_diloco
        from studio_rlhf import prompts_from
        out = []
        for dataset in list(self.workspace.state["datasets"]):
            keys = set()
            for row in self._conversation_rows([dataset["id"]], False):
                keys.update(studio_diloco.prompt_key(p) for p in prompts_from([row]))
                if len(keys) >= self.CHAT_COUNT_CAP:
                    break
            if keys:
                out.append({"id": dataset["id"], "name": dataset["name"], "count": len(keys)})
        return out

    def _sets_holding(self, kind):
        import studio_data_kinds as dk
        out = []
        for dataset in list(self.workspace.state["datasets"]):
            if dataset.get("storage") == "sqlite":
                stats = self.workspace.kind_stats(dataset["id"])
                count = stats.get(kind, (0,))[0] if stats is not None else sum(dk.classify(e) == kind for e in self.workspace.iter_examples(dataset["id"]))
            else:
                count = sum(dk.classify(e) == kind for e in self.workspace.iter_examples(dataset["id"]))
            if count:
                out.append({"id": dataset["id"], "name": dataset["name"], "count": count})
        return out

    def data(self, kind, query=""):
        summaries = self._summaries()
        full = summaries[kind]
        if not query.strip():
            return replace(full, rows=list(full.rows))
        names = {d["id"]: d["name"] for d in self.workspace.state["datasets"]}
        return replace(full, rows=self._summary_cache().search(kind, query, summaries["takes"][kind], names))

    def data_page(self, kind, query="", cursor=None, limit=200, review_only=False):
        import studio_data_kinds as dk
        if review_only:
            sets = [(d["id"], d["name"]) for d in self.workspace.state["datasets"] if d["name"] in self._DRAFT_SETS.values()]
            return dk.page(self.workspace, kind, query, sets, cursor, limit, review_only=True)
        summary = self.data(kind)  # the sets and the near-duplicates; the rows come from storage
        dups = {r["id"] for r in summary.rows if r["status"] == "Near-duplicate"}
        return dk.page(self.workspace, kind, query, [(x["id"], x["name"]) for x in summary.sets], cursor, limit, dups)

    def has_text(self):
        """One indexed look at the first text row of each dataset, not the whole-store summary (a cold one reads every dataset)."""
        return any(next(self.workspace.iter_kind(d["id"], "text"), None) is not None for d in list(self.workspace.state["datasets"]))

    _OWN_SETS = {"text": "Your files", "conversations": "Your conversations", "problems": "Your problems"}
    _DRAFT_SETS = {"conversations": "Rewritten conversations", "text": "Generated prose"}

    def add_data(self, kind, fields):
        import studio_data_kinds as dk
        ws, target = self.workspace, dk.ensure_dataset(self.workspace, self._OWN_SETS[kind])
        added = (bool(dk.add_problems(ws, target, [{"prompt": fields["prompt"], "rule": fields["rule"], "params": {}}])[0]) if kind == "problems"
                 else bool(ws.add(target, [{**fields, "quality": "Good"}], "manual")[0]))
        self.emit("data", {"kind": kind})  # the sidebar's Next row may have been waiting for exactly this
        return added

    def import_data(self, kind, paths):
        import studio_data_kinds as dk
        report = self.workspace.import_files(list(paths), dk.ensure_dataset(self.workspace, self._OWN_SETS[kind]), describe=plain_error)
        self.emit("data", {"kind": kind})
        return report

    # recommended collections (studio_catalog does the fetching, checking and indexing) ---------------------------
    _busy: dict = {}  # source id -> {completed, total}; swapped for a fresh dict per run, never mutated while shared
    _failed: dict = {}

    @staticmethod
    def collection_ids(kind="text"):
        """Default catalogue sources of one kind: language-stage text, problems-stage problems, every other stage conversations."""
        import studio_catalog as cat
        stage_kind = {"language": "text", "problems": "problems"}
        return [e["id"] for e in cat.CATALOG if e["default"] and stage_kind.get(e["stage"], "conversations") == kind]

    def collection(self, kind="text"):
        import studio_catalog as cat
        have = {d["catalog_id"]: d for d in cat.list_downloads(self.root)}
        made = {d["id"] for d in self.workspace.state["datasets"]}
        rows = []
        for entry in (e for e in cat.CATALOG if e["id"] in self.collection_ids(kind)):
            cid, busy = entry["id"], self._busy.get(entry["id"])
            state = ("downloading" if busy is not None else "imported" if f"catalog_{cid}" in made
                     else "downloaded" if cid in have else "not downloaded")
            rows.append({"id": cid, "name": entry["name"], "size": entry["size_hint"], "records": entry["records_hint"],
                         "state": state, "completed": (busy or {}).get("completed", 0), "total": (busy or {}).get("total", 0),
                         "error": self._failed.get(cid, "")})
        return rows

    @staticmethod
    def _source_name(cid):
        import studio_catalog as cat
        return next((e["name"] for e in cat.CATALOG if e["id"] == cid), cid)

    def _background(self, ids, work, done):
        with self._lock:
            if self._busy:
                return False
            self._busy, self._failed = {i: {} for i in ids}, {}
        def run():
            for cid in ids:
                try:
                    work(cid)
                    self._busy.pop(cid, None)
                    self.emit(done, {"id": cid})
                except Exception as error:  # shown to the person in plain words; the rest of the list still runs
                    self._busy.pop(cid, None)
                    self._failed[cid] = plain_error(error, self._source_name(cid))
                    self.emit("download_failed", {"id": cid, "error": self._failed[cid]})
            self._busy = {}
            self.emit("downloads_finished", {"ids": list(ids)})  # one data reload per batch, not one per source
        self._worker = start_thread(run)
        return True

    def download_sources(self, ids=None, kind="text", install=False):
        import studio_catalog as cat

        def work(cid):
            def progress(info):
                self._busy[cid] = {"completed": info["completed"], "total": info["total"]}
                self.emit("download", {"id": cid, **info})
            result = cat.download_dataset(cid, self.root, emit=progress)
            if install:
                cat.register_download(self.workspace, result, emit=progress)
        return self._background(list(ids or self.collection_ids(kind)), work, "download_done")

    def import_source(self, source_id):
        import studio_catalog as cat
        result = next((d for d in cat.list_downloads(self.root) if d["catalog_id"] == source_id), None)
        if result is None:
            return False

        def work(cid):
            cat.register_download(self.workspace, result, emit=lambda info: self.emit("download", {"id": cid, **info}))
        return self._background([source_id], work, "import_done")

    def generate_problems(self, rule, n, store=True, rng=None):
        import random
        import studio_data_kinds as dk
        items = problem_items(rule, n, rng or random.Random())
        if not store:
            return problem_report(items)
        return problem_report(items, *dk.add_problems(self.workspace, dk.ensure_dataset(self.workspace, "Generated from rules"), items))

    def make_drafts(self, kind, n):
        import json
        import random
        import studio_data_kinds as dk
        from studio_rlhf import OllamaTeacher
        teacher = OllamaTeacher(timeout=120)
        models = teacher.models()
        if not models:
            raise RuntimeError("Ollama has no models installed.")
        teacher.model = models[0]
        rng, rows = random.Random(), []
        if kind == "conversations":
            pool = [e for d in self.workspace.state["datasets"]  # up to n good conversations from each dataset, picked by rowid, not read whole
                    for e in self.workspace.sample(d["id"], n, rng.getrandbits(32), reviewed_only=True, kind="conversations")]
            if not pool:
                raise RuntimeError("There are no good conversations to rewrite yet.")
            work = [(json.dumps({"user": e["user"], "assistant": e["assistant"]}), True) for e in rng.sample(pool, min(n, len(pool)))]
            system = ("Reword this prompt and its answer without changing what the answer says. "
                      'Reply as JSON: {"user": "...", "assistant": "..."}.')
        else:
            work = [("Write one short passage of plain English on a topic of your choice.", False)] * n
            system = "Write plain prose only, no title and no list."
        for text, as_json in work:
            try:
                got = teacher.chat(system, text, json_mode=as_json, temperature=0.8)
            except RuntimeError:
                if rows:
                    break
                raise
            row = {"user": got.get("user"), "assistant": got.get("assistant")} if as_json else {"text": got}
            if all(row.values()):
                rows.append({**row, "quality": "Review", "synthetic": True})
        return self.workspace.add(dk.ensure_dataset(self.workspace, self._DRAFT_SETS[kind]), rows, "synthetic")[0]

    @timed("services.draft_count")
    def draft_count(self):
        names = set(self._DRAFT_SETS.values())
        return sum(self.workspace.review_count(d["id"]) for d in self.workspace.state["datasets"] if d["name"] in names)

    def review_data(self, example_id, keep):
        if keep:
            self.workspace.mark_reviewed([example_id])
        else:
            self.workspace.remove_examples([example_id])

    def data_fields(self, example_id):
        import studio_data_kinds as dk
        e = self.workspace.get_example(example_id)
        kind = dk.classify(e)
        return {"text": {"text": e["text"]}, "conversations": {"user": e["user"], "assistant": e["assistant"]},
                "problems": {"prompt": e["user"], "rule": dk.rule_of(e)}}[kind]

    def edit_data(self, example_id, fields):
        import studio_data_kinds as dk
        import studio_rules
        old = self.workspace.get_example(example_id)
        kind = dk.classify(old)
        new = {**old, "messages": []}  # normalize reads user/assistant (or text) once the old messages are cleared
        if kind == "text":
            new["text"] = fields.get("text", "")
        elif kind == "conversations":
            new.update(user=fields.get("user", ""), assistant=fields.get("assistant", ""))
        else:
            rule, prompt = fields["rule"], fields.get("prompt", "")
            params = old.get("params") or (old.get("metadata") or {}).get("params") or {}
            if old["assistant"] != dk.NO_KEY and not studio_rules.check(rule, prompt, old["assistant"], params):
                raise ValueError(f"That prompt no longer matches the stored answer under the rule \"{studio_rules.LABELS[rule]}\".")
            new.update(user=prompt, rule=rule, metadata={**(old.get("metadata") or {}), "rule": rule})
        self.workspace.update_example(new)

    def _heldout_stamp(self):
        path = self.root / "data" / "heldout.json"
        try:
            s = path.stat()
            return s.st_mtime_ns, s.st_size
        except OSError:
            return None

    @timed("services._heldout")
    def _heldout(self):
        """The held-back documents, read once per version of data/heldout.json (it is tens of MB)."""
        stamp = self._heldout_stamp()
        if stamp is None:
            return []
        if getattr(self, "_heldout_hit", (None,))[0] != stamp:
            if getattr(self._probe, "on", False):
                raise _Slow
            path = self.root / "data" / "heldout.json"
            self._heldout_hit = (stamp, [_doc(r) for r in json.loads(path.read_text(encoding="utf-8"))])
        return self._heldout_hit[1]

    def _tokenizer(self, run_dir):
        from studio_model import Tokenizer, load_checkpoint
        run_dir = Path(run_dir)
        ckpt = next(p for p in (run_dir / "best.npz", run_dir / "last.npz") if p.exists())
        return Tokenizer.restore(load_checkpoint(ckpt)["tokenizer"])

    def _cached(self, key, make):
        """make() once per key. A hit never waits (the Tk thread reads these); a miss is long, so ready() answers no for it and
        only a worker gets here. Misses of different keys run at once; two for the same key run make() once."""
        if key in self._cache:
            return self._cache[key]
        if getattr(self._probe, "on", False):
            raise _Slow
        with self._once(key):
            if key not in self._cache:
                self._cache[key] = make()
            return self._cache[key]

    def _score_key(self, ckpt):
        """(name, key) of a checkpoint's held-back scores: the file, the held-back text and the scoring method they were made
        with, so any of the three changing makes them stale. None without held-back text."""
        import studio_eval
        stamp = self._heldout_stamp()
        if stamp is None:
            return None
        try:
            s = Path(ckpt).stat()
            file = [s.st_mtime_ns, s.st_size]
        except OSError:
            file = None  # the scoring says what is wrong with it
        name = str(Path(ckpt).relative_to(self.root)) if Path(ckpt).is_relative_to(self.root) else str(ckpt)
        return name, json.dumps([name, file, stamp, studio_eval.SCORE_METHOD])

    def _held_scores(self, ckpt, later=False):
        """(documents, 2) array of nats and characters per held-back document for a checkpoint, or None without held-back text.
        Kept in data/held_back_scores/ per key (_score_key), so a new launch scores nothing it already scored. Scoring takes
        seconds per version: with later=True the Tk thread does not wait for it but gets _Pending, and a worker scores the
        checkpoint (_score_later)."""
        import hashlib
        import studio_eval
        got = self._score_key(ckpt)
        if got is None:
            return None
        name, key = got
        digest = lambda text: hashlib.sha1(text.encode()).hexdigest()[:16]  # noqa: E731
        folder, prefix = self.root / "data" / "held_back_scores", digest(name)
        file = folder / f"{prefix}-{digest(key)}.npy"
        if ("scores", key) not in self._cache and file.exists():
            try:
                self._cache[("scores", key)] = np.load(file)
            except (OSError, ValueError):
                pass  # unreadable: scored again below
        if (later and ("scores", key) not in self._cache and self.on_ui_thread()
                and not getattr(self._probe, "on", False)):  # ready() still says not ready, so a page's worker reads it in place
            self._score_later(ckpt, key)
            raise _Pending

        def make():
            docs = self._heldout()
            if not docs:
                return None
            from studio_backend import cap_threads
            cap_threads()  # a snapshot worker scores here too, not only _score_later: leave cores for Tk and training
            scores = np.array(studio_eval.score_docs(ckpt, docs), dtype=np.float64)
            with self._lock:
                folder.mkdir(parents=True, exist_ok=True)
                for old in folder.glob(f"{prefix}-*.npy"):  # an older state of the same checkpoint is dead weight
                    old.unlink(missing_ok=True)
                partial = folder / f".{file.name}"
                np.save(partial, scores)
                partial.replace(file)
            return scores
        return self._cached(("scores", key), make)

    def _score_later(self, ckpt, key):
        """Score ckpt's held-back text on one worker, after any already asked for, then emit "versions" once so whatever
        weighed versions without these scores reads again. A checkpoint whose scoring failed is not tried again from here."""
        with self._score_lock:
            if key in self._unscorable or key in self._to_score:
                return
            self._to_score[key] = ckpt
            if self._scorer:
                return
            self._scorer = True

        def run():
            from studio_backend import cap_threads
            cap_threads()
            while True:
                with self._score_lock:
                    if not self._to_score:
                        self._scorer = False
                        break
                    key, ckpt = next(iter(self._to_score.items()))
                try:
                    self._held_scores(ckpt)
                except Exception:
                    logging.getLogger(__name__).exception("scoring %s on the held-back text failed", ckpt)
                    self._unscorable.add(key)
                with self._score_lock:
                    self._to_score.pop(key, None)
            self.emit("versions", None)
        start_thread(run)

    def _gate(self, parent_ckpt, candidate_ckpt):
        """studio_eval.paired_gain of one checkpoint against another on the held-back text, or None without it. Once per pair.
        _Pending on the Tk thread while either is still being scored."""
        import studio_eval
        parent, candidate = self._held_scores(parent_ckpt, later=True), self._held_scores(candidate_ckpt, later=True)
        if parent is None or candidate is None:
            return None
        return self._cached(("gain", self._score_key(parent_ckpt)[1], self._score_key(candidate_ckpt)[1]),
                            lambda: studio_eval.paired_gain(parent, candidate))

    @timed("services.references")
    def references(self, version):
        """Reference lines for the run page. Cached per checkpoint and held-back file: the run page asks every 1.5 s."""
        import studio_eval
        v, best = self._as_version(version), self.best_version()
        out = {"uniform_bpc": None, "unigram_bpc": None, "best_so_far_bpc": None}
        docs = self._heldout()
        if v is None or not v.best_checkpoint or not docs:
            return out

        def make():
            run = Path(v.best_checkpoint).parent
            tok = self._tokenizer(run)
            train = _run_docs(run)
            scores = self._held_scores(best.best_checkpoint, later=True) if best and best.best_checkpoint else None
            return {"uniform_bpc": studio_eval.uniform_bpc(tok, docs), "unigram_bpc": studio_eval.unigram_bpc(tok, train or docs, eval_docs=docs),
                    "best_so_far_bpc": None if scores is None else studio_eval.corpus_bpc(scores)}
        key = ("refs", v.best_checkpoint, best.best_checkpoint if best else None, (best.scorecard or {}).get("bpc") if best else None,
               self._heldout_stamp())
        try:
            return self._cached(key, make)
        except _Pending:  # the best version is being scored: the lines come with the worker's "versions" event
            return out

    @timed("services.run_curve")
    def run_curve(self, run_dir):
        import studio_eval
        from studio_model import read_metrics
        run = Path(run_dir)
        rows = [r for r in read_metrics(run / "metrics.jsonl") if r.get("loss") is not None]
        phases, total = self._phases(run, rows)
        cpp = 1.0  # characters per piece; nats per piece / ln2 / cpp = bits per character
        if any((run / n).exists() for n in ("best.npz", "last.npz")):
            def chars_per_piece():  # once per run: the held-back text (or a bounded slice of the run's own text) is read only here
                docs = self._heldout() or _run_docs(run)
                tok = self._cached(("tok", str(run)), lambda: self._tokenizer(run))
                return sum(map(len, docs)) / max(1, sum(len(tok.encode(d)) for d in docs)) if docs else 1.0
            cpp = self._cached(("cpp", str(run), self._heldout_stamp()), chars_per_piece)
        bits = lambda nats: nats / LN2 / cpp  # noqa: E731
        steps = [(r["step"], bits(r["loss"])) for r in rows]
        return {"steps": steps, "smooth": list(zip([s for s, _ in steps], studio_eval.smooth([b for _, b in steps]))),
                "heldout": [(r["step"], bits(r["val_loss"])) for r in rows if r.get("val_loss") is not None],
                "lr": [(r["step"], r["lr"]) for r in rows if r.get("lr") is not None],
                "phases": phases, "total": total, **self._run_samples(run)}

    @staticmethod
    def _phases(run, rows):
        """(phases, total steps) of the run's schedule, from its spec and the last metrics row."""
        from studio_core import read_json
        train = (read_json(run / "spec.json") if (run / "spec.json").exists() else {}).get("train", {})
        total = train.get("run", {}).get("steps") or (rows[-1]["step"] if rows else 0)
        sched = train.get("schedule", {})
        warm = min(total, sched.get("config", {}).get("warmup", 0))
        stable = sched.get("config", {}).get("stable_fraction", .8)  # ponytail: schedules without a flat part use WSD's default split
        decay = warm + stable * (total - warm)
        return [(0, warm, "warm-up"), (warm, decay, "steady"), (decay, total, "cool-down")], total

    def run_phases(self, run_dir):
        """The schedule's phases alone: run_curve also converts the loss to bits per character, which loads the run's
        tokenizer and encodes the held-back text. The Queue page asks this on the Tk thread, so it must stay cheap."""
        from studio_model import read_metrics
        run = Path(run_dir)
        return self._phases(run, [r for r in read_metrics(run / "metrics.jsonl") if r.get("loss") is not None])[0]

    @staticmethod
    def _run_samples(run):
        """The latest 'what it writes now' samples train() logged (samples.jsonl), with the prompt and the steps between them."""
        from studio_core import jsonl_tail, read_json
        cfg = read_json(run / "config.json") if (run / "config.json").exists() else {}
        samples = jsonl_tail(run / "samples.jsonl", 5)
        return {"samples": [{"step": x["step"], "text": x["text"]} for x in samples],
                "sample_every": (cfg.get("sample_every") or 0) * cfg.get("eval_every", 0) or None,
                "prompt": samples[-1]["prompt"] if samples else cfg.get("sample_prompt")}

    # chat / inspect ---------------------------------------------------------
    def _checkpoint(self, version):
        v = self._as_version(version)
        if v is None or not v.best_checkpoint:
            raise NoCheckpoint(getattr(v, "number", None) or (version if isinstance(version, int) else None))
        return v.best_checkpoint

    def chat(self, version, messages, on_text, stop, ablate=None):
        from studio_model import LocalModel
        import studio_memory
        ckpt = self._checkpoint(version)
        with self._chat_lock:
            model = self._models.pop(ckpt, None)
            if model is None:
                device = self._chat_device()
                model = LocalModel(ckpt, device=device)
                self.ledger.register(ckpt, studio_memory.weight_bytes(model.model), device)
                self._using[ckpt] = threading.RLock()
            self._models[ckpt] = model  # most recently used last; the dict is the LRU order
            while len(self._models) > CHAT_MODELS:
                self._evict_chat()
            using = self._using[ckpt]
        with using:  # a move to another device waits for the reply: its cache lives on the device it started on
            return model.generate(messages, on_text=on_text, stop=stop, ablate=[tuple(h) for h in ablate] if ablate else None)

    def _chat_device(self):
        """Where a chat model should live now (the chat lock is held): the CUDA GPU while no job runs, the CPU otherwise, and on
        a Mac (MPS shares the CPU's memory) or a machine without a GPU always the CPU."""
        import studio_backend
        if self._job_running:
            return "cpu"
        status = studio_backend.gpu_status()
        return "cuda" if status["available"] and status["kind"] == "cuda" else "cpu"

    def _evict_chat(self):
        """Drop the least recently used chat model (the chat lock is held). A reply still running on it keeps it alive until it ends."""
        ckpt = next(iter(self._models))
        del self._models[ckpt]
        self._using.pop(ckpt, None)
        self.ledger.release(ckpt)

    def evict_oldest_chat(self):
        """The queue's call when a job is about to start with no room to spare. Runs on the runner thread. False when none is loaded."""
        with self._chat_lock:
            if not self._models:
                return False
            self._evict_chat()
            return True

    def _move_chat(self, device):
        """Put every loaded chat model on `device` (the chat lock is held), waiting for the reply each is giving. A model that
        cannot move (the GPU is out of memory) stays where it is, and the ledger says where that is."""
        import studio_memory
        import torch
        for ckpt, model in list(self._models.items()):
            try:
                with self._using[ckpt]:
                    model.to(device)
            except RuntimeError:  # no room on the GPU: the model keeps answering from where it is
                logging.getLogger("groundwork.chat").warning("could not move a chat model to %s", device, exc_info=True)
            self.ledger.register(ckpt, studio_memory.weight_bytes(model.model), model.device)
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()  # hand the GPU's memory back to the job

    def chat_job_started(self, job):
        """The queue's call as a job is about to start, on the runner thread: chat leaves the GPU for it."""
        with self._chat_lock:
            self._job_running = True
            self._move_chat("cpu")

    def chat_job_ended(self, job):
        """The queue's call after a job ended, on the runner thread: chat may use the CUDA GPU again."""
        with self._chat_lock:
            self._job_running = False
            self._move_chat(self._chat_device())

    def inspect_forward(self, version, text, hooks=True):
        import studio_inspect
        return studio_inspect.forward(self._checkpoint(version), text, hooks=hooks)

    def inspect_logit_lens(self, version, text, top_k=5):
        import studio_inspect
        return studio_inspect.logit_lens(self._checkpoint(version), text, top_k=top_k)

    def inspect_ablate(self, version, text, targets=None, mode="zero", heads=None):
        import studio_inspect
        return studio_inspect.ablate(self._checkpoint(version), text, targets=targets, mode=mode, heads=heads)

    def inspect_head_stats(self, pattern):
        """Per-head metrics of an attention pattern [B,H,T,T] taken from inspect_forward's hooks."""
        import studio_inspect
        return studio_inspect.head_stats(pattern)

    # Look-inside pages (S7) ---------------------------------------------------
    def _batch(self, version):
        """(checkpoint, ids [N, L], characters per piece) for the fixed held-back batch of this checkpoint."""
        import studio_inspect
        ckpt = self._checkpoint(version)
        docs = self._heldout()
        if not docs:
            raise ValueError("No held-back text yet, so there is nothing to measure the error on.")

        def make():
            model, tok = studio_inspect.probe_model(ckpt)
            return studio_inspect.heldout_batch(tok, docs, min(64, model.dims.n_ctx))
        ids, cpp = self._cached(("batch", ckpt), make)
        return ckpt, ids, cpp

    def inspect_trace(self, version, text, pos=-1):
        import studio_inspect
        return studio_inspect.trace(self._checkpoint(version), text, pos)

    def inspect_prepared(self, version):
        import studio_inspect
        return studio_inspect.prepared(self._checkpoint(version))

    def inspect_loss(self, version, text, pos=-1, trained=True):
        import studio_inspect
        return studio_inspect.loss_view(self._checkpoint(version), text, pos, trained)

    def inspect_landscape(self, version, snapshot=None, mode="random", grid=21, span=1.0, seed=0, live=False, new_anchor=False, stop=None):
        import numpy as np

        import studio_landscape as landscape
        v = self._as_version(version)
        runs = [Path(self._abs(r)) for r in getattr(v, "runs", [])] if v is not None else []
        run = next((r for r in reversed(runs) if r.is_dir()), None)
        if run is None:
            raise ValueError("This version has not trained yet, so there are no weights to look around.")
        if live:
            mode, grid = "random", min(grid, LANDSCAPE_LIVE_GRID)
        mine = landscape.run_snapshots(run)  # the model is rebuilt from this run's own files, so only its snapshots share its shape
        if snapshot is not None:
            every = [Path(p) for p in self.viewer_snapshots(v.number)]
            if not 0 <= snapshot < len(every) or every[snapshot] not in mine:
                raise ValueError("Choose a snapshot of this version's latest run.")
            center = every[snapshot]
        elif live:
            center = mine[-1] if mine else None
        else:
            center = next((p for p in (run / "best.npz", run / "last.npz") if p.exists()), mine[-1] if mine else None)
        if center is None:
            raise ValueError("Training has not saved any weights yet. The landscape appears at the first snapshot.")
        others = [p for p in mine if p != center]
        if live:
            others = others[-LANDSCAPE_LIVE_PATH:]
        docs = self._heldout()
        if not docs:
            raise ValueError("No held-back text yet, so there is nothing to measure the error on.")
        stamp = lambda p: (str(p), p.stat().st_mtime_ns, p.stat().st_size)  # noqa: E731
        key = ("landscape", stamp(center), tuple(stamp(p) for p in others), self._heldout_stamp(), mode, grid, span, seed)

        def make():
            model, tokenizer = landscape.run_model(run)
            ids, targets = self._cached(("landscape-batch", str(run), self._heldout_stamp()), lambda: landscape.landscape_batch(tokenizer, docs, model.dims.n_ctx))
            step, weights = landscape.read_weights(center)
            norms = None
            if live:
                anchors = self.__dict__.setdefault("_landscape_anchor", {})
                if new_anchor or str(run) not in anchors:
                    anchors[str(run)] = landscape.filter_norms(weights, model.roles)
                norms = anchors[str(run)]
            trajectory = [landscape.read_weights(p) for p in others]
            got = landscape.compute(model, ids, targets, weights, trajectory, mode=mode, grid=grid, span=span, seed=seed, norms=norms,
                                    stop=stop, throttle=LANDSCAPE_LIVE_THROTTLE if live else 0.)
            got["current"]["step"] = step
            tokens = int((np.asarray(targets) >= 0).sum())
            cost = {"seconds": got.pop("seconds"), "evaluations": got.pop("evaluations")}
            return dict(got, mode=mode, grid=len(got["axis"]), span=span, seed=seed, live=live, cost=cost,
                        center={"step": step, "name": center.name}, batch={"rows": int(len(ids)), "tokens": tokens},
                        note="Held-back text, one fixed batch, so every frame is measured on the same words.")
        return make() if live else self._cached(key, make)

    def inspect_value_map(self, version, heads, block=0, head=0):
        import studio_inspect
        info = studio_inspect.value_map(self._checkpoint(version), block, head)
        if heads != info["heads"]:
            info["singular"] = None
        return info

    def inspect_rank_heads(self, version, mode="mean"):
        """Rise in held-back error when each head is switched off alone, in bits per character ('rise'), largest first."""
        import studio_inspect
        ckpt, ids, cpp = self._batch(version)
        rows = self._cached(("rank", ckpt, mode), lambda: studio_inspect.rank_heads(ckpt, ids, mode))
        rows = [dict(r, rise_nats=r["rise"], rise=studio_inspect.bits_per_char(r["rise"], cpp)) for r in rows]
        return sorted(rows, key=lambda r: -r["rise"])

    def inspect_heads_off(self, version, off, mode="mean"):
        import studio_inspect
        ckpt, ids, cpp = self._batch(version)
        got = studio_inspect.ablate_set(ckpt, ids, list(off), mode)
        return {"before": studio_inspect.bits_per_char(got["baseline"], cpp), "after": studio_inspect.bits_per_char(got["loss"], cpp)}

    def inspect_reply(self, version, prompt, off=(), mode="mean"):
        import studio_inspect
        return studio_inspect.greedy_reply(self._checkpoint(version), prompt, list(off), mode)

    # collect -----------------------------------------------------------------------------------------
    def collect_manager(self):
        """The app-side manager of the background collector (starts the service, registers its shards); made on first use."""
        if getattr(self, "_collect", None) is None:
            from studio_collect.manager import Manager
            self._collect = Manager(self.workspace, self.emit)
        return self._collect

    def collect_start(self):
        self.collect_manager().start()

    def collect_stop(self):
        if getattr(self, "_collect", None) is not None:
            self._collect.close()

    def collect_options(self):
        from studio_collect import view
        return view.options()

    def collect_config(self):
        from studio_collect import view
        return view.get_config(self.root)

    def _collect_changed(self, config):
        from studio_collect import service
        if config["enabled"]:
            service.ensure_running(self.root)
        self.emit("collect", {})
        return config

    def save_collect_config(self, overrides):
        from studio_collect import view
        return self._collect_changed(view.set_config(self.root, overrides))

    def collect_state(self):
        from studio_collect import service, view
        return view.state(self.root, running=service.running(self.root))

    def collect_pause(self):
        from studio_collect import view
        return self._collect_changed(view.set_paused(self.root, True))

    def collect_resume(self):
        from studio_collect import view
        return self._collect_changed(view.set_paused(self.root, False))




class FixtureServices(WorkspaceServices):
    """WorkspaceServices over the committed fixture workspace (built once into `directory`)."""

    def __init__(self, directory):
        from tests.fixtures.ia_workspace import build
        super().__init__(build(directory))


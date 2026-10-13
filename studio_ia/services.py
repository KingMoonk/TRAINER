"""The one façade pages talk to. `Services` is the protocol; `FakeServices` is an in-memory stand-in."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np


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


@dataclass
class Job:
    id: str
    kind: str  # pretrain | quicktest | compare | post_round | restore_answers | checkpoint_now
    label: str = ""
    note: str = ""
    params: dict = field(default_factory=dict)
    status: str = "waiting"
    eta_s: float | None = None
    created_at: float = 0.0


@dataclass
class QueueState:
    job: Job | None = None  # the running job, None when idle
    step: int = 0
    total: int = 0
    phase: str = ""  # warm-up | steady | cool-down
    lr: float | None = None
    eta_s: float | None = None
    paused: bool = False
    blocked: str | None = None  # why the plan forbids training right now (hours / idle / battery)
    stopping: bool = False  # stop after cool-down in progress
    up_next: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)


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


class Services:
    """Synchronous façade; long work runs through the queue. Subclasses override; scopes add methods, never rename."""

    workspace: Any = None

    def __init__(self):
        self._subscribers: list[Callable[[str, Any], None]] = []
        self._ui = None  # a Tk widget; set by the shell so events hop onto the Tk thread

    # events ---------------------------------------------------------------
    def bind_ui(self, widget):
        self._ui = widget

    def subscribe(self, callback):
        """callback(kind, payload) runs on the Tk thread. Returns an unsubscribe function."""
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback) if callback in self._subscribers else None

    def emit(self, kind, payload=None):
        """Safe from any thread: delivery is scheduled with `after` on the bound Tk widget."""
        def deliver():
            for callback in list(self._subscribers):
                callback(kind, payload)
        if self._ui is None:
            deliver()
        else:
            self._ui.after(0, deliver)

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

    def save_version(self, spec, change_text):
        raise NotImplementedError

    # queue (S2) -----------------------------------------------------------
    def queue(self) -> QueueState:
        raise NotImplementedError

    def enqueue(self, job):
        raise NotImplementedError

    def move_up(self, job_id):
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

    def pretrain_preview(self, dataset_ids, rows=16):
        """{pieces, params, context, first_rows}: pieces = tokenised size of the chosen text, params = current
        version's parameter count, first_rows = first `rows` rows of token groups, each a list of example numbers
        with -1 where an example ends."""
        raise NotImplementedError

    # data (S5) / evaluation (S4) / chat ------------------------------------
    def data(self, kind, query=""):
        """studio_data_kinds.DataSummary for "text" | "conversations" | "problems"; query filters the rows."""
        raise NotImplementedError

    def add_data(self, kind, fields):
        """fields: text -> {text}; conversations -> {user, assistant}; problems -> {prompt, rule}. True if added."""
        raise NotImplementedError

    def import_data(self, kind, paths):
        """Import files into the kind's own set. Returns one report line per file."""
        raise NotImplementedError

    def collection(self):
        """The recommended text collection: one dict per source {id, name, size, records, state, completed, total, error};
        state is "not downloaded" | "downloading" | "downloaded" | "imported"."""
        raise NotImplementedError

    def download_sources(self, ids=None):
        """Download the whole recommended collection (ids None) or the listed sources, one after another, on a background
        thread. Progress arrives as events "download" {id, text, completed, total}, then "download_done" {id} or
        "download_failed" {id, error}. False when a download is already running."""
        raise NotImplementedError

    def import_source(self, source_id):
        """Import one downloaded source into the text sets used for pretraining (background thread; events "import_done"
        {id} or "download_failed" {id, error}). False when it is not downloaded or work is already running."""
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

    def run_curve(self, run_dir):
        raise NotImplementedError

    def chat(self, version, messages, on_text, stop, ablate=None):
        """ablate: optional [(block, head), ...] mean-ablated for this reply only."""
        raise NotImplementedError

    # inspect (S7): thin wrappers over studio_inspect, numpy out; version is a Version or number -----------------
    def inspect_forward(self, version, text, hooks=True):
        """{'ids' [1,T], 'tokens' [[str]], 'logits' [1,T,V], 'hooks' {name: array}} (studio_inspect.forward)."""
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

    def post_training(self, version) -> dict:
        """{"rounds": {1..4: {"status": done|running|queued|idle, "note": str}}, "counts": {"conversations", "corrections",
        "problems", "pairs"}, "gates": {round: {"ok", "checks", "verdict", "judge": (k, n)}}}; missing keys are fine."""
        raise NotImplementedError

    def post_rows(self, kind):
        """Rows for a post_round job: conversations | problems | pairs | heldout_docs | heldout_problems."""
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


class FakeServices(Services):
    """Empty, in-memory placeholder data so pages render before real services exist."""

    def __init__(self):
        super().__init__()
        self._versions: list[Version] = []
        self._queue = QueueState()
        self.viewer_log: list = []  # (action, payload): the fake viewer records calls and spawns nothing
        self._viewer_running = False

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

    def versions(self):
        return list(self._versions)

    def current_version(self):
        return self._versions[-1] if self._versions else None

    def best_version(self):
        done = [v for v in self._versions if v.best_checkpoint]
        return done[-1] if done else None

    def save_version(self, spec, change_text):
        v = Version(len(self._versions) + 1, f"version {len(self._versions) + 1}", change_text, spec=spec, status="saved")
        self._versions.append(v)
        self.emit("versions", v)
        return v

    def queue(self):
        return self._queue

    def set_queue(self, state: QueueState):
        """Test hook: replace the queue state and announce it."""
        self._queue = state
        self.emit("queue", state)

    def enqueue(self, job):
        self._queue.up_next.append(job)
        self.emit("queue", self._queue)

    def move_up(self, job_id):
        q = self._queue.up_next
        i = next((k for k, j in enumerate(q) if j.id == job_id), 0)
        if i:
            q[i - 1], q[i] = q[i], q[i - 1]
        self.emit("queue", self._queue)

    def storage(self, days=None, data=False):
        return (0, 0)

    def clean_storage(self, days=None, data=False):
        self.cleaned = (days, data)
        return (0, 0)

    def remove(self, job_id):
        self._queue.up_next = [j for j in self._queue.up_next if j.id != job_id]
        self.emit("queue", self._queue)

    def pause(self):
        self._queue.paused = True
        self.emit("queue", self._queue)

    def resume(self):
        self._queue.paused = False
        self.emit("queue", self._queue)

    def checkpoint_now(self):
        self.emit("queue", self._queue)

    def stop_after_cooldown(self):
        pass

    def pretrain_preview(self, dataset_ids, rows=16):
        row = [(i // 7) % 4 if i % 7 else -1 for i in range(32)]
        return {"pieces": 2_000_000, "params": 1_000_000, "context": 256, "first_rows": [row[r:] + row[:r] for r in range(rows)]}

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

    def _add_row(self, kind, a, b, status="Good", dataset="Your data", draft=False):
        rows = self._rows()[kind]
        if not (a.strip() and b.strip()) or any((r["a"], r["b"]) == (a, b) for r in rows):
            return False
        rows.append(dict(id=f"{kind[0]}{len(rows) + 1}-{a[:8]}", a=a, b=b, status=status, dataset=dataset, draft=draft))
        return True

    def add_data(self, kind, fields):
        a, b = {"text": ("Your files", fields.get("text", "")), "conversations": (fields.get("user", ""), fields.get("assistant", "")),
                "problems": (fields.get("prompt", ""), fields.get("rule", ""))}[kind]
        return self._add_row(kind, a, b, "In use" if kind == "text" else "Ready" if kind == "problems" else "Good")

    def import_data(self, kind, paths):
        return [f"{Path(p).name}: 0 added (no workspace)" for p in paths]

    def collection(self):
        return getattr(self, "_collection", [])

    def download_sources(self, ids=None):
        self.calls = getattr(self, "calls", []) + [("download_sources", ids)]
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
        return {"uniform_bpc": None, "unigram_bpc": None, "best_so_far_bpc": None, "noise_band": None}

    def run_curve(self, run_dir):
        return {"steps": [], "smooth": [], "heldout": [], "lr": [], "phases": [], "total": 0}

    def chat(self, version, messages, on_text, stop, ablate=None):
        return f"[reply with {len(ablate)} heads off]" if ablate else "[reply]"

    def teach_correction(self, version, messages, correction, whole_answer=False):
        pass

    def rate_pair(self, messages, chosen, rejected):
        pass

    def verdict(self, a, b):
        from studio_eval import training_saved
        return training_saved([], [])

    def post_training(self, version):
        return {"rounds": {}, "counts": {}, "gates": {}}

    def post_rows(self, kind):
        return []

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


# ---- the real façade -------------------------------------------------------------------------------------------

import json  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
from dataclasses import asdict, fields  # noqa: E402
from pathlib import Path  # noqa: E402

CHAT_MODELS = 3  # ponytail: a count cap on loaded chat models; P1 of issue #13 replaces it with a byte budget

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


class _Cold(BaseException):
    """Raised inside ready(): the read asked for is not cached. Not an Exception, so no page's own handler swallows it."""


class WorkspaceServices(Services):
    """Every Services method over the real modules. Queue events fire on the runner thread and hop to Tk via emit."""

    def __init__(self, workspace):
        import studio_jobs
        import studio_queue
        import studio_versions

        super().__init__()
        self.workspace = workspace
        self.root = Path(workspace.root)
        self.store = studio_versions.VersionStore(self.root)
        self.store.versions = lambda: self._rows()  # tolerant of the simpler list schema
        self.runner = studio_queue.Runner(self.root / "data" / "queue.json", studio_jobs.HANDLERS, services=self)
        self.runner.subscribe(self.emit)
        self.runner.subscribe(self._record_job)
        self._models, self._cache = {}, {}
        self._lock = threading.RLock()
        self._data, self._probe = {}, threading.local()  # reads of the stored data, kept until it next changes

    def close(self):
        if getattr(self, "_viz", None) is not None:
            self._viz.stop(wait=False)
        self.runner.shutdown(timeout=2)

    # 3D viewer --------------------------------------------------------------
    _viz = None

    def viewer_status(self):
        import sys
        script = Path(__file__).resolve().parent.parent / "studio_brain3d.py"
        frozen = getattr(sys, "frozen", False)  # bundled: no script beside the executable
        viz = self._viz
        return {"available": frozen or script.exists(), "running": bool(viz and viz.running), "connected": bool(viz and viz.connected)}

    def viewer_start(self, scene=None):
        import sys

        from studio_core import module_command
        from studio_vizbridge import VizBridge
        if self._viz is None:
            frozen = getattr(sys, "frozen", False)
            self._viz = VizBridge(Path(__file__).resolve().parent.parent / "studio_brain3d.py",
                                  command=module_command("studio_brain3d") if frozen else None)
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

    def _as_version(self, version):
        if isinstance(version, int):
            version = self._version(self.store.get(version))
        return version or self.best_version()

    def versions(self):
        return [self._version(r) for r in self._rows()]

    def current_version(self):
        rows = self._rows()
        return self._version(rows[-1]) if rows else None

    def best_version(self):
        """The version Chat follows: a later version takes over only when its held-back error is below the noise band of
        the one it replaces (studio_gate.moves_chat), so a difference inside the noise never moves Chat."""
        import studio_eval
        import studio_gate
        rows, docs, best = self._rows(), self._heldout(), None
        for row in rows:
            bpc = row.get("scorecard", {}).get("bpc")
            if not row.get("best_checkpoint") or not isinstance(bpc, (int, float)):
                continue
            ckpt = best and self._abs(best["best_checkpoint"])
            # ponytail: the band is one scoring pass over the held-back text on the caller's thread, cached per checkpoint
            band = self._cached(("band", ckpt), lambda: studio_eval.noise_band(ckpt, docs)) if best and docs else None
            if studio_gate.moves_chat(bpc, best and best["scorecard"]["bpc"], band):
                best = row
        if best is None:  # nothing scored yet: the newest version that has a checkpoint
            best = next((r for r in reversed(rows) if r.get("best_checkpoint")), None)
        return self._version(best) if best else None

    def checkpoint_step(self, version):
        from studio_model import read_metrics
        v = self._as_version(version)
        if v is None or not v.best_checkpoint:
            return None
        held = [r for r in read_metrics(Path(v.best_checkpoint).parent / "metrics.jsonl") if r.get("val_loss") is not None]
        return min(held, key=lambda r: r["val_loss"])["step"] if held else None  # best.npz is saved at the lowest held-back loss

    def save_version(self, spec, change_text):
        row = self.store.save_version(spec, change_text or None)
        v = self._version(row)
        self.emit("versions", v)
        return v

    # queue ------------------------------------------------------------------
    def queue(self):
        return self.runner.state()

    def enqueue(self, job):
        import studio_queue
        if isinstance(job, Job):
            job = studio_queue.Job(**asdict(job))
        if self.runner.thread is None:
            self.runner.start()
        if job.kind == "pretrain" and job.params.get("version"):
            self._to(job.params["version"], "queued")
        return self.runner.enqueue(job)

    def move_up(self, job_id):
        self.runner.move_up(job_id)

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

    def remove(self, job_id):
        self.runner.remove(job_id)

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
        import studio_eval
        rows = self._text_rows(params.get("datasets"))
        path = self.root / "data" / "heldout.json"
        if not path.exists() and len(rows) > 1:  # ponytail: later sets never join the held-back text; delete the file to re-split
            _, held = studio_eval.heldout_split([e["text"] for e in rows], params.get("heldout_fraction", .05))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([{"id": f"heldout_{i}", "text": t} for i, t in enumerate(held)]), encoding="utf-8")
        held = set(self._heldout())
        return [e for e in rows if e["text"] not in held]

    def _fresh(self, key, make):
        """make() once per state of the stored data (Workspace.revision). No lock: the shell's one worker does the
        long reads in turn, and the Tk thread asks ready() first, so it only ever picks up a finished result."""
        revision = self.workspace.revision
        hit = self._data.get(key)
        if hit is None or hit[0] != revision:
            if getattr(self._probe, "on", False):
                raise _Cold
            hit = self._data[key] = (revision, make())
        return hit[1]

    def ready(self, work):
        self._probe.on = True
        try:
            work()
            return True
        except _Cold:  # a small store is read in milliseconds, in place; a large one goes to the worker
            state = self.workspace.state
            return len(state["examples"]) + sum(d.get("recordCount") or 0 for d in state["datasets"]) <= 5000
        finally:
            self._probe.on = False

    def _stamp(self):
        """Size and time of the files the stored data lives in: says whether data/summaries.json still describes them."""
        db = self.workspace.db_path

        def stat(path):
            try:
                s = path.stat()
                return [s.st_mtime_ns, s.st_size] if s.st_size else None
            except OSError:
                return None
        return [stat(self.workspace.path), stat(db), stat(db.with_name(db.name + "-wal"))]

    def _summaries(self):
        """{kind: DataSummary with every sample row, "example": the first training conversation}. Read from
        data/summaries.json when the stored files have not changed since it was written, so a launch reads nothing twice."""
        import studio_data_kinds as dk
        from studio_core import atomic_json

        def make():
            path, stamp = self.root / "data" / "summaries.json", self._stamp()
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["stamp"] == stamp:
                    return {**{k: dk.DataSummary(**saved[k]) for k in dk.KINDS}, "example": saved["example"]}
            except (OSError, ValueError, KeyError, TypeError):
                pass
            scanned = dk.scan(self.workspace, progress=self.note)
            out = {k: dk.summarize(self.workspace, k, limit=None, scanned=scanned, progress=self.note) for k in dk.KINDS}
            first = next((e for e, _ in scanned[1]["conversations"] if not dk.held_back(e)), None)  # the first of post_rows("conversations")
            out["example"] = {"prompt": first["user"], "answer": first["assistant"]} if first and first.get("user") else {}
            self.note("Saving the summary")
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, {**{k: asdict(out[k]) for k in dk.KINDS}, "example": out["example"], "stamp": stamp})
            return out
        return self._fresh("summaries", make)

    def pretrain_preview(self, dataset_ids, rows=16):
        import studio_estimate
        import studio_spec
        from studio_model import read_metrics
        v = self.current_version()
        if v is None or not v.spec:
            return {}
        context = v.spec["dims"]["n_ctx"]

        def make():
            self.note("Reading the chosen text")
            held = set(self._heldout())
            docs = [e["text"] for e in self._text_rows(dataset_ids) if e["text"] not in held]
            self.note("Building the word list")
            module, cfg = studio_spec.module_of("tokenizer", v.spec["tokenizer"])
            tok = module.fit(cfg, [{"text": d} for d in docs])
            small = sum(map(len, docs)) <= 5_000_000  # larger text: the size stays [N] rather than a guess
            self.note("Counting word pieces")
            encoded = [tok.encode(d) for d in docs] if small else []
            return {"pieces": sum(map(len, encoded)) + len(docs) if small else None,
                    "params": studio_estimate.estimate(v.spec, tok.vocab_size)["total"],
                    "context": context, "first_rows": layout_rows([len(e) + 1 for e in encoded], context, rows)}
        heldout = self.root / "data" / "heldout.json"
        key = ("preview", tuple(dataset_ids), rows, json.dumps(v.spec, sort_keys=True, default=str),
               heldout.stat().st_mtime_ns if heldout.exists() else None)
        preview = self._fresh(key, make)
        speed = None
        for ver in reversed(self.versions()):
            for run in reversed(ver.runs):
                got = [r["tokens_per_second"] for r in read_metrics(Path(run) / "metrics.jsonl") if r.get("tokens_per_second")]
                speed = speed or (got[-1] if got else None)
        return {**preview, "tokens_per_second": speed}

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
        number = job.params.get("version") if job and job.kind == "pretrain" else None
        if not number:
            return
        if payload["status"] == "running":
            return self._to(number, "training")
        out = Path(job.params.get("output_dir") or "")
        ckpt = next((p for p in (out / "best.npz", out / "last.npz") if p.exists()), None)
        if ckpt:
            import studio_eval
            self.store.add_run(number, out)
            self.store.set_best_checkpoint(number, ckpt)
            docs = self._heldout()
            if docs:
                card = {**(self.store.get(number).get("scorecard") or {}), "bpc": studio_eval.bits_per_char(ckpt, docs)}
                self.store.set_scorecard(number, card)
        self._to(number, "trained" if ckpt and payload["status"] == "done" else "saved")
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
        done.setdefault(str(v.number), {})[str(number)] = {k: result.get(k) for k in ("ok", "checks", "verdict", "before", "after", "band", "checkpoint")}
        self._save_json("post_training.json", done)
        if result.get("ok"):
            card = dict(v.scorecard or {})
            after = result.get("after") or {}
            card.update({"bpc": after.get("bpc", card.get("bpc")), "problems": [after.get("pass_k"), after.get("pass_n")] if after.get("pass_n") else card.get("problems")})
            if number == 1:
                card["answers"] = True
            self.store.set_best_checkpoint(v.number, result["checkpoint"])
            self.store.set_scorecard(v.number, card)
        self.emit("versions", None)

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
                "example": self._summaries()["example"]}

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
        out = []
        for d in self.workspace.state["datasets"]:
            for e in self.workspace.iter_examples(d["id"]):
                if dk.classify(e) == want and dk.held_back(e) == held:
                    out.append({"prompt": e.get("user", ""), "rule": dk.rule_of(e), "params": e.get("params") or (e.get("metadata") or {}).get("params", {})}
                               if want == "problems" else e)
        return out

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
        from studio_model import read_metrics
        v = self._as_version(number)
        run = Path(v.runs[-1]) if v and v.runs else None
        if run is None or not (run / "metrics.jsonl").exists():
            return [], "", False
        curve = self.run_curve(run)
        tokens = {r["step"]: r.get("trained_tokens") or r["step"] for r in read_metrics(run / "metrics.jsonl")}
        spec = json.loads((run / "spec.json").read_text(encoding="utf-8")) if (run / "spec.json").exists() else {}
        status = json.loads((run / "run.json").read_text(encoding="utf-8")).get("status") if (run / "run.json").exists() else None
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

    # data / evaluation ------------------------------------------------------
    def data(self, kind, query=""):
        import studio_data_kinds as dk
        full = self._summaries()[kind]
        return replace(full, rows=dk.matching(full.rows, query))

    _OWN_SETS = {"text": "Your files", "conversations": "Your conversations", "problems": "Your problems"}
    _DRAFT_SETS = {"conversations": "Rewritten conversations", "text": "Generated prose"}

    def add_data(self, kind, fields):
        import studio_data_kinds as dk
        ws, target = self.workspace, dk.ensure_dataset(self.workspace, self._OWN_SETS[kind])
        if kind == "problems":
            return bool(dk.add_problems(ws, target, [{"prompt": fields["prompt"], "rule": fields["rule"], "params": {}}])[0])
        return bool(ws.add(target, [{**fields, "quality": "Good"}], "manual")[0])

    def import_data(self, kind, paths):
        import studio_data_kinds as dk
        return self.workspace.import_files(list(paths), dk.ensure_dataset(self.workspace, self._OWN_SETS[kind]))

    # recommended text collection (studio_catalog does the fetching, checking and indexing) -----------------------
    _busy: dict = {}  # source id -> {completed, total}; swapped for a fresh dict per run, never mutated while shared
    _failed: dict = {}

    @staticmethod
    def collection_ids():
        import studio_catalog as cat
        return [e["id"] for e in cat.CATALOG if e["stage"] == "language" and e["default"]]

    def collection(self):
        import studio_catalog as cat
        have = {d["catalog_id"]: d for d in cat.list_downloads(self.root)}
        made = {d["id"] for d in self.workspace.state["datasets"]}
        rows = []
        for entry in (e for e in cat.CATALOG if e["id"] in self.collection_ids()):
            cid, busy = entry["id"], self._busy.get(entry["id"])
            state = ("downloading" if busy is not None else "imported" if f"catalog_{cid}" in made
                     else "downloaded" if cid in have else "not downloaded")
            rows.append({"id": cid, "name": entry["name"], "size": entry["size_hint"], "records": entry["records_hint"],
                         "state": state, "completed": (busy or {}).get("completed", 0), "total": (busy or {}).get("total", 0),
                         "error": self._failed.get(cid, "")})
        return rows

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
                    self._failed[cid] = str(error) or error.__class__.__name__
                    self.emit("download_failed", {"id": cid, "error": self._failed[cid]})
            self._busy = {}
        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()
        return True

    def download_sources(self, ids=None):
        import studio_catalog as cat

        def work(cid):
            def progress(info):
                self._busy[cid] = {"completed": info["completed"], "total": info["total"]}
                self.emit("download", {"id": cid, **info})
            cat.download_dataset(cid, self.root, emit=progress)
        return self._background(list(ids or self.collection_ids()), work, "download_done")

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
            pool = [e for d in self.workspace.state["datasets"] for e in self.workspace.iter_examples(d["id"])
                    if dk.classify(e) == "conversations" and e.get("quality") == "Good"]
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

    def draft_count(self):
        names = set(self._DRAFT_SETS.values())
        return sum(1 for d in self.workspace.state["datasets"] if d["name"] in names
                   for e in self.workspace.iter_examples(d["id"]) if e.get("quality") == "Review")

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
            if old["assistant"] != dk.NO_KEY and not studio_rules.check(rule, prompt, old["assistant"]):
                raise ValueError(f"That prompt no longer matches the stored answer under the rule \"{studio_rules.LABELS[rule]}\".")
            new.update(user=prompt, rule=rule, metadata={**(old.get("metadata") or {}), "rule": rule})
        self.workspace.update_example(new)

    def _heldout(self):
        path = self.root / "data" / "heldout.json"
        return [_doc(r) for r in json.loads(path.read_text(encoding="utf-8"))] if path.exists() else []

    def _tokenizer(self, run_dir):
        from studio_model import Tokenizer, load_checkpoint
        run_dir = Path(run_dir)
        ckpt = next(p for p in (run_dir / "best.npz", run_dir / "last.npz") if p.exists())
        return Tokenizer.restore(load_checkpoint(ckpt)["tokenizer"])

    def _cached(self, key, make):
        with self._lock:
            if key not in self._cache:
                self._cache[key] = make()
            return self._cache[key]

    def references(self, version):
        import studio_eval
        v, best = self._as_version(version), self.best_version()
        out = {"uniform_bpc": None, "unigram_bpc": None, "best_so_far_bpc": None, "noise_band": None}
        docs = self._heldout()
        if v is None or not v.best_checkpoint or not docs:
            return out
        run = Path(v.best_checkpoint).parent
        tok = self._tokenizer(run)
        train = [_doc(r) for r in map(json.loads, (run / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if (run / "dataset.jsonl").exists() else [])]
        out["uniform_bpc"] = studio_eval.uniform_bpc(tok, docs)
        out["unigram_bpc"] = studio_eval.unigram_bpc(tok, train or docs, eval_docs=docs)
        out["noise_band"] = self._cached(("band", v.best_checkpoint), lambda: studio_eval.noise_band(v.best_checkpoint, docs))
        score = best.scorecard.get("bpc") if best else None
        if score is None and best and best.best_checkpoint:
            score = self._cached(("bpc", best.best_checkpoint), lambda: studio_eval.bits_per_char(best.best_checkpoint, docs))
        out["best_so_far_bpc"] = score
        return out

    def run_curve(self, run_dir):
        import studio_eval
        from studio_model import read_metrics
        run = Path(run_dir)
        rows = [r for r in read_metrics(run / "metrics.jsonl") if r.get("loss") is not None]
        spec = json.loads((run / "spec.json").read_text(encoding="utf-8")) if (run / "spec.json").exists() else {}
        train = spec.get("train", {})
        total = train.get("run", {}).get("steps") or (rows[-1]["step"] if rows else 0)
        docs = self._heldout()
        if not docs and (run / "dataset.jsonl").exists():
            docs = [_doc(json.loads(line)) for line in (run / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line]
        cpp = 1.0  # characters per piece; nats per piece / ln2 / cpp = bits per character
        if docs and any((run / n).exists() for n in ("best.npz", "last.npz")):
            tok = self._cached(("tok", str(run)), lambda: self._tokenizer(run))
            cpp = self._cached(("cpp", str(run)), lambda: sum(map(len, docs)) / max(1, sum(len(tok.encode(d)) for d in docs)))
        bits = lambda nats: nats / LN2 / cpp  # noqa: E731
        steps = [(r["step"], bits(r["loss"])) for r in rows]
        sched = train.get("schedule", {})
        warm = min(total, sched.get("config", {}).get("warmup", 0))
        stable = sched.get("config", {}).get("stable_fraction", .8)  # ponytail: schedules without a flat part use WSD's default split
        decay = warm + stable * (total - warm)
        return {"steps": steps, "smooth": list(zip([s for s, _ in steps], studio_eval.smooth([b for _, b in steps]))),
                "heldout": [(r["step"], bits(r["val_loss"])) for r in rows if r.get("val_loss") is not None],
                "lr": [(r["step"], r["lr"]) for r in rows if r.get("lr") is not None],
                "phases": [(0, warm, "warm-up"), (warm, decay, "steady"), (decay, total, "cool-down")], "total": total,
                **self._run_samples(run)}

    @staticmethod
    def _run_samples(run):
        """The latest 'what it writes now' samples train() logged (samples.jsonl), with the prompt and the steps between them."""
        def rows(name):
            path = run / name
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8")) if (run / "config.json").exists() else {}
        samples = rows("samples.jsonl")[-5:]
        return {"samples": [{"step": x["step"], "text": x["text"]} for x in samples],
                "sample_every": (cfg.get("sample_every") or 0) * cfg.get("eval_every", 0) or None,
                "prompt": samples[-1]["prompt"] if samples else cfg.get("sample_prompt")}

    # chat / inspect ---------------------------------------------------------
    def _checkpoint(self, version):
        v = self._as_version(version)
        if v is None or not v.best_checkpoint:
            raise ValueError("That version has no trained checkpoint yet.")
        return v.best_checkpoint

    def chat(self, version, messages, on_text, stop):
        from studio_model import LocalModel
        ckpt = self._checkpoint(version)
        with self._lock:
            model = self._models.pop(ckpt, None) or LocalModel(ckpt, device="cpu")
            self._models[ckpt] = model  # most recently used last; the dict is the LRU order
            while len(self._models) > CHAT_MODELS:
                del self._models[next(iter(self._models))]
        return model.generate(messages, on_text=on_text, stop=stop)

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


class FixtureServices(WorkspaceServices):
    """WorkspaceServices over the committed fixture workspace (built once into `directory`)."""

    def __init__(self, directory):
        from tests.fixtures.ia_workspace import build
        super().__init__(build(directory))


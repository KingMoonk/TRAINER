"""The one façade pages talk to. `Services` is the protocol; `FakeServices` is an in-memory stand-in."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    up_next: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)


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

    # versions (S3) --------------------------------------------------------
    def versions(self) -> list:
        raise NotImplementedError

    def current_version(self):
        raise NotImplementedError

    def best_version(self):
        raise NotImplementedError

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

    def pretrain_preview(self, dataset_ids, rows=16):
        """{pieces, params, context, first_rows}: pieces = tokenised size of the chosen text, params = current
        version's parameter count, first_rows = first `rows` rows of token groups, each a list of example numbers
        with -1 where an example ends."""
        raise NotImplementedError

    # data (S5) / evaluation (S4) / chat ------------------------------------
    def data(self, kind):
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

    def inspect_value_map(self, version, heads):
        """{'d_model', 'heads' (the real count), 'singular' list | None}: singular values of W_V^h W_O^h when heads matches."""
        raise NotImplementedError

    def inspect_rank_heads(self, version, mode="mean"):
        """studio_inspect.rank_heads on a fixed held-back batch: [{'block','head','rise'}]. Slow; call off the UI thread."""
        raise NotImplementedError

    def inspect_heads_off(self, version, off):
        """Held-back bits/char with the (block, head) pairs in `off` mean-ablated: {'before', 'after'}."""
        raise NotImplementedError

    def teach_correction(self, version, messages, correction):
        """Save a correction as a Post-training round 3 item (teacher sees it, student does not)."""
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


class FakeServices(Services):
    """Empty, in-memory placeholder data so pages render before real services exist."""

    def __init__(self):
        super().__init__()
        self._versions: list[Version] = []
        self._queue = QueueState()

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

    def data(self, kind):
        return {"kind": kind, "count": 0, "sets": []}

    def references(self, version):
        return {"uniform_bpc": None, "unigram_bpc": None, "best_so_far_bpc": None, "noise_band": None}

    def run_curve(self, run_dir):
        return {"steps": [], "smooth": [], "heldout": [], "lr": [], "phases": [], "total": 0}

    def chat(self, version, messages, on_text, stop, ablate=None):
        return f"[reply with {len(ablate)} heads off]" if ablate else "[reply]"

    def teach_correction(self, version, messages, correction):
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

    def inspect_value_map(self, version, heads):
        return {"d_model": 256, "heads": 4, "singular": [3.0, 1.5, .7, .1] if heads == 4 else None}

    def inspect_rank_heads(self, version, mode="mean"):
        return [{"block": b, "head": h, "rise": .05 * (4 * b + h)} for b in range(2) for h in range(4)]

    def inspect_heads_off(self, version, off):
        return {"before": 2.0, "after": 2.0 + .1 * len(off)}


# ---- the real façade -------------------------------------------------------------------------------------------

import json  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
from dataclasses import asdict, fields  # noqa: E402
from pathlib import Path  # noqa: E402

LN2 = math.log(2)


def _doc(row):
    """One held-back / training document as text (a text row, a user+assistant pair or a messages list)."""
    if row.get("text"):
        return row["text"]
    if row.get("messages"):
        return "\n".join(m["content"] for m in row["messages"])
    return "\n".join(s for s in (row.get("user"), row.get("assistant")) if s)


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
        self._models, self._cache = {}, {}
        self._lock = threading.RLock()

    def close(self):
        self.runner.shutdown(timeout=2)

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
        row = self.store.best_version()
        if row is None:  # nothing scored yet: the newest version that has a checkpoint
            row = next((r for r in reversed(self._rows()) if r.get("best_checkpoint")), None)
        return self._version(row) if row else None

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
        return self.runner.enqueue(job)

    def move_up(self, job_id):
        self.runner.move_up(job_id)

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

    # data / evaluation ------------------------------------------------------
    def data(self, kind):
        import studio_data_kinds
        return studio_data_kinds.summarize(self.workspace, kind)

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
            cpp = sum(map(len, docs)) / max(1, sum(len(tok.encode(d)) for d in docs))
        bits = lambda nats: nats / LN2 / cpp  # noqa: E731
        steps = [(r["step"], bits(r["loss"])) for r in rows]
        sched = train.get("schedule", {})
        warm = min(total, sched.get("config", {}).get("warmup", 0))
        stable = sched.get("config", {}).get("stable_fraction", .8)  # ponytail: schedules without a flat part use WSD's default split
        decay = warm + stable * (total - warm)
        return {"steps": steps, "smooth": list(zip([s for s, _ in steps], studio_eval.smooth([b for _, b in steps]))),
                "heldout": [(r["step"], bits(r["val_loss"])) for r in rows if r.get("val_loss") is not None],
                "lr": [(r["step"], r["lr"]) for r in rows if r.get("lr") is not None],
                "phases": [(0, warm, "warm-up"), (warm, decay, "steady"), (decay, total, "cool-down")], "total": total}

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
            if ckpt not in self._models:
                self._models[ckpt] = LocalModel(ckpt, device="cpu")
            model = self._models[ckpt]
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


class FixtureServices(WorkspaceServices):
    """WorkspaceServices over the committed fixture workspace (built once into `directory`)."""

    def __init__(self, directory):
        from tests.fixtures.ia_workspace import build
        super().__init__(build(directory))

    def teach_correction(self, version, messages, correction):
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

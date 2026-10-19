"""Training-speed benchmark in the spirit of MLPerf Training: a frozen reference task, a quality target, and time to reach it.

A task fixes everything that decides how hard it is: the text (generated here, byte for byte, from a seeded grammar, so no machine
needs a download), the design, tokenizer, batch, optimizer, step cap, evaluation interval, seeds and the target held-out bits per
character. Its checksum (a hash of the text and every setting) is stored in the code; a task whose text or settings change without
a new version and a new checksum fails verify(). Results carry the task's id, version and checksum, and the leaderboard, history and
compare views only ever set results of the same version side by side.

Run rules. Each attempt trains with one of the task's fixed seeds and evaluates on the held-out slice every `eval_every` steps. The
score is the training clock at the first evaluation at or below the target (evaluations are inside the clock, as in MLPerf); the
clock starts when the first step starts, so reading the text, building the model and (on the first step) any kernel set-up are not
hidden: setup is reported on its own, and the first step's excess over a steady step is reported as warm-up. A result is
"converged" only when every attempt reached the target within the step cap, and its time is the median of the attempts, with their
spread; otherwise it is "did_not_converge" and has no time. Throughput is the median of the steady steps (evaluations excluded).

MFU is model FLOPs per second over the device's peak. FLOPs per token are the design's (studio_estimate: forward plus backward is
three forward passes, attention counted over the whole context) plus the output projection, which studio_estimate leaves out of
its compute figure: 6 x d_model x vocabulary. Peak is the datasheet number when PEAKS knows the device and precision (source stored
with the result); otherwise it is the best fp32 matrix-multiply rate measured on the device just before the run, marked as such, so
an MFU is never quoted against a number nobody can check.

The held-out rows are the trainer's own seeded split of the task's rows (studio_model.partition_examples), so they depend on the
attempt's seed: the same on every machine for the same seed, drawn from one stationary generator, but not one identical slice for
all three seeds. The result stores the seeds used.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

RESULTS_FORMAT = 1  # bump when a result's fields change meaning
DIVISIONS = ("single", "cluster")
STATUSES = ("converged", "did_not_converge", "failed", "stopped")
LN2 = math.log(2)

# ---------------------------------------------------------------------------------------------------- the text


class _Mix:
    """splitmix64: a generator written out here so the text never depends on the interpreter's `random` internals."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def pick(self, items):
        return items[self.below(len(items))]


ADJECTIVES = ("quiet", "red", "old", "bright", "narrow", "heavy", "pale", "wild", "small", "tall", "cold", "gentle")
NOUNS = ("fox", "river", "lantern", "baker", "garden", "engine", "violin", "captain", "mountain", "letter", "harbor", "painter")
VERBS = ("found", "carried", "watched", "mended", "followed", "painted", "answered", "counted", "opened", "remembered")
PLACES = ("near the mill", "under the bridge", "beside the fire", "across the field", "behind the church", "inside the barn")
TIMES = ("at dawn", "before the rain", "after supper", "late in autumn", "on a still night", "every morning")
CLAUSES = ("because the road was closed", "while the others slept", "although nobody asked", "until the bell rang",
           "so that the river stayed clear", "since the winter was long")
NAMES = ("Mara", "Tobias", "Ines", "Caspar", "Lena", "Orrin", "Petra", "Silas")


def _sentence(mix):
    shape = mix.below(5)
    subject = f"the {mix.pick(ADJECTIVES)} {mix.pick(NOUNS)}"
    obj = f"a {mix.pick(ADJECTIVES)} {mix.pick(NOUNS)}"
    if shape == 0:
        text = f"{subject} {mix.pick(VERBS)} {obj} {mix.pick(PLACES)}"
    elif shape == 1:
        text = f"{mix.pick(NAMES)} {mix.pick(VERBS)} {obj} {mix.pick(TIMES)}"
    elif shape == 2:
        text = f"{subject} {mix.pick(VERBS)} {obj} {mix.pick(CLAUSES)}"
    elif shape == 3:
        text = f"{mix.pick(TIMES)}, {mix.pick(NAMES)} {mix.pick(VERBS)} {mix.below(90) + 10} {mix.pick(NOUNS)}s {mix.pick(PLACES)}"
    else:
        text = f"{subject} and {mix.pick(NAMES)} {mix.pick(VERBS)} {obj}"
    return text[0].upper() + text[1:] + "."


def corpus(rows, seed, sentences=(3, 6)):
    """`rows` paragraphs of generated text, each a {"text": ...} example. The same arguments give the same rows on every machine."""
    mix = _Mix(seed)
    return [{"text": " ".join(_sentence(mix) for _ in range(sentences[0] + mix.below(sentences[1] - sentences[0] + 1)))}
            for _ in range(rows)]


# ---------------------------------------------------------------------------------------------------- the tasks


@dataclass(frozen=True)
class Task:
    id: str
    version: int
    title: str
    rows: int  # paragraphs in the text
    text_seed: int
    d_model: int
    n_layer: int
    n_ctx: int
    heads: int
    batch_size: int
    learning_rate: float
    warmup: int
    step_cap: int  # the schedule's length and the most steps an attempt may take
    eval_every: int
    validation: float  # share of the rows held out
    target_bpc: float  # held-out bits per character an attempt must reach
    seeds: tuple = (11, 12, 13, 14, 15)
    attempts: int = 3  # the default number of attempts
    precision: str = "fp32"
    checksum: str = field(default="", compare=False)  # the frozen value; checksum_of() recomputes it

    def examples(self):
        return corpus(self.rows, self.text_seed)

    def spec(self):
        from studio_spec import validate_spec
        return validate_spec({"schema": 1, "name": f"mlperf {self.id} v{self.version}",
                              "dims": {"d_model": self.d_model, "n_layer": self.n_layer, "n_ctx": self.n_ctx, "dropout": 0.},
                              "block": {"mixer": {"id": "attention", "config": {"heads": self.heads}}},
                              "train": {"precision": {"id": self.precision, "config": {}}}})

    def config(self, seed, device="auto"):
        return {"name": f"mlperf {self.id}", "architecture": self.spec(), "tokenizer": "byte", "width": self.d_model, "layers": self.n_layer,
                "heads": self.heads, "context": self.n_ctx, "dropout": 0., "batch_size": self.batch_size, "accumulation": 1,
                "steps": self.step_cap, "learning_rate": self.learning_rate, "warmup": self.warmup, "eval_every": self.eval_every,
                "validation": self.validation, "seed": seed, "objective": "all", "windowing": "sequential", "device": device,
                "sample_every": 0}


def checksum_of(task):
    """SHA-256 over the task's settings (all but the stored checksum) and its text."""
    settings = {k: v for k, v in asdict(task).items() if k != "checksum"}
    body = json.dumps({"settings": settings, "text": [row["text"] for row in task.examples()]}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(body.encode()).hexdigest()


TASKS = {
    "small": Task(id="small", version=1, title="Small language model", rows=400, text_seed=1, d_model=64, n_layer=2, n_ctx=128, heads=4,
                  batch_size=16, learning_rate=2e-3, warmup=40, step_cap=400, eval_every=20, validation=.1, target_bpc=2.0,
                  attempts=3,
                  checksum="e28a4a7dd623ea3fac6aa237525f363d6b33853d76642a5ebe41979bd4ff146a"),
}


def verify(task):
    """Raise unless the task still is the one whose checksum is frozen in the code."""
    found = checksum_of(task)
    if found != task.checksum:
        raise ValueError(f"Task {task.id} v{task.version} changed: its checksum is {found[:16]}..., frozen {task.checksum[:16] or 'none'}. "
                         "A changed task needs a new version, never an edited one.")
    return found


def task_list():
    return [{"id": t.id, "version": t.version, "title": t.title, "target_bpc": t.target_bpc, "checksum": t.checksum, "attempts": t.attempts,
             "step_cap": t.step_cap, "seeds": list(t.seeds)} for t in TASKS.values()]


def get_task(task_id):
    try:
        return TASKS[task_id]
    except KeyError:
        raise ValueError(f"Unknown task {task_id}. Known: {', '.join(TASKS)}.") from None


# ---------------------------------------------------------------------------------------------------- FLOPs and peaks

# Dense (no sparsity) peak TFLOP/s per device name fragment: {"fp32": non-tensor-core FP32, "bf16": tensor-core BF16 with FP32
# accumulate}. Source: the vendor datasheets and whitepapers named in "source". A laptop part is a ceiling: its power limit holds it below.
PEAKS = (
    ("H100", {"fp32": 67., "bf16": 989.}, "NVIDIA H100 SXM datasheet"),
    ("A100", {"fp32": 19.5, "bf16": 312.}, "NVIDIA A100 datasheet"),
    ("RTX 4090", {"fp32": 82.6, "bf16": 165.2}, "NVIDIA Ada GPU architecture whitepaper"),
    ("RTX 3090", {"fp32": 35.6, "bf16": 71.}, "NVIDIA GA102 whitepaper"),
    ("RTX 4060 Laptop", {"fp32": 15.11}, "NVIDIA RTX 4060 Laptop GPU at its maximum boost clock (TechPowerUp GPU database)"),
)


def flops_per_token(task):
    """Training FLOPs per token for the task's design: studio_estimate's figure plus the output projection (6 x d_model x vocabulary)."""
    import studio_estimate
    import studio_model
    cfg = task.config(0)
    vocabulary = studio_model.vocabulary_guess(studio_model.validate_config(cfg))
    spec = task.spec()
    return studio_estimate.estimate(spec, vocabulary)["train_flops_per_token"] + 6 * task.d_model * vocabulary


def peak_for(name, precision):
    """(FLOP/s, source) from the datasheet table, or (None, "") when the table has no entry for this device and precision."""
    key = "bf16" if precision in ("bf16", "fp16") else "fp32"
    for fragment, rates, source in PEAKS:
        if fragment.lower() in (name or "").lower() and key in rates:
            return rates[key] * 1e12, f"{source}; {key} dense"
    return None, ""


def measured_peak(device, seconds=1.0, size=1024, clock=time.perf_counter):
    """Best fp32 matrix-multiply FLOP/s torch reaches on the device in about `seconds`: a ceiling anyone can re-measure."""
    import torch
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    for _ in range(3):
        a @ b
    sync()
    best, end = 0., clock() + seconds
    while True:
        start = clock()
        for _ in range(8):
            a @ b
        sync()
        best = max(best, 8 * 2 * size ** 3 / max(clock() - start, 1e-9))
        if clock() >= end:
            return best


def peak(device_info, precision, measure=measured_peak):
    """{"flops", "kind": datasheet | measured_gemm, "source"} for the device this result ran on."""
    datasheet, source = peak_for(device_info.get("name"), precision)
    if datasheet:
        return {"flops": datasheet, "kind": "datasheet", "source": source}
    flops = measure(device_info.get("kind") or "cpu")
    return {"flops": flops, "kind": "measured_gemm", "source": "best 1024x1024 fp32 matrix multiply torch reached on this device in 1 s "
            "(no datasheet entry for it), so an MFU against it is a share of what torch can do, not of the silicon's rating"}


def mfu(tokens_per_second, flops_per_tok, peak_flops):
    return tokens_per_second * flops_per_tok / peak_flops if tokens_per_second and peak_flops else None


# ---------------------------------------------------------------------------------------------------- the machine


def _driver():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=5, check=False).stdout.strip().splitlines()
        return out[0].strip() if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def describe_device(kind):
    """What the result ran on: the device, the software and the threads. `kind` is "cpu" or "cuda" or "mps"."""
    import torch
    info = {"kind": kind, "name": platform.processor() or platform.machine(), "node": platform.node(), "cpu": platform.processor(),
            "threads": torch.get_num_threads(), "cores": __import__("os").cpu_count(), "torch": torch.__version__, "python": platform.python_version(),
            "os": platform.platform(), "cuda": None, "driver": None, "memory_bytes": None}
    if kind == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(name=torch.cuda.get_device_name(0), cuda=torch.version.cuda, driver=_driver(), memory_bytes=int(props.total_memory))
    elif kind == "mps":
        info.update(name="Apple GPU (Metal)")
    return info


class Energy:
    """GPU energy over an interval from NVML's total-energy counter, where the driver reports it. joules() is None otherwise."""

    def __init__(self, kind):
        self.handle = self.start = None
        if kind != "cuda":
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.start = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:  # not installed, or a part without the counter
            self.handle = None

    def joules(self):
        if self.handle is None:
            return None
        try:
            return (self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - self.start) / 1000.
        except Exception:
            return None


def _peak_memory(kind):
    import torch
    if kind == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None  # CPU and Metal report none without a platform library the app does not require


# ---------------------------------------------------------------------------------------------------- one attempt


def spread(values):
    values = sorted(values)
    return {"median": statistics.median(values), "min": values[0], "max": values[-1], "n": len(values)} if values else None


def train_runner(task, seed, device, emit, stop, directory):
    """The real training of one attempt: studio_model.train on the task's text and config."""
    import studio_model
    return studio_model.train(task.config(seed, device), task.examples(), directory, emit=emit, stop=stop)


def run_attempt(task, seed, device, runner=train_runner, clock=time.perf_counter, stop=None, emit=None, work=None):
    """One timed attempt. The runner trains with studio_model.train's events and must stop when `stop` is set; it is told nothing of
    the target. Returns the attempt's dict (see the module text for each field)."""
    outer = stop or threading.Event()
    stop = threading.Event()  # this attempt's: reaching the target ends it without ending the benchmark
    emit = emit or (lambda event: None)
    started = clock()
    state = {"setup": None, "hit": None, "rows": [], "best": None, "stopped_by_target": False}

    def watch(event):
        emit(event)
        if outer.is_set():
            stop.set()
        kind = event.get("type")
        if kind == "history" and state["setup"] is None:
            state["setup"] = clock() - started  # the text is read, the model built: the first step is next
        elif kind == "metric" and event.get("step", 0) >= 1:
            state["rows"].append(event)
            if event.get("val_loss") is not None:
                bpc = event["val_loss"] / LN2  # a byte tokenizer: one token is one character (the two delimiters per row are under 1%)
                state["best"] = bpc if state["best"] is None else min(state["best"], bpc)
                if state["hit"] is None and bpc <= task.target_bpc:
                    state["hit"] = event
                    state["stopped_by_target"] = True
                    stop.set()  # the score is read at this evaluation: nothing after it counts

    attempt = {"seed": seed, "status": "failed", "error": None}
    own = tempfile.TemporaryDirectory(prefix="mlperf_", dir=work) if work else tempfile.TemporaryDirectory(prefix="mlperf_")
    try:
        with own as directory:
            runner(task, seed, device, watch, stop, Path(directory) / "run")
    except Exception as exc:  # a failed attempt is a result with its reason, not a crash of the benchmark
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        return attempt
    rows = state["rows"]
    if not rows:
        attempt["error"] = "training reported no step"
        return attempt
    steady = [r["tokens_per_second"] for r in rows if r["step"] >= 2 and r.get("tokens_per_second")]
    last = rows[-1]
    per_step = last.get("trained_tokens", 0) / max(last["step"], 1)
    tps = statistics.median(steady) if steady else None
    first = next((r for r in rows if r["step"] == 1), None)
    steady_step = per_step / tps if tps else None
    final = state["hit"] or last
    attempt.update(
        status="converged" if state["hit"] else ("stopped" if outer.is_set() else "did_not_converge"),
        steps=final["step"], best_bpc=state["best"], tokens=final.get("trained_tokens"),
        time_to_target_s=final.get("elapsed_seconds") if state["hit"] else None,
        tokens_per_s=tps, steps_per_s=(tps / per_step) if tps and per_step else None,
        setup_s=state["setup"], first_step_s=first.get("elapsed_seconds") if first else None,
        warmup_s=max(0., first["elapsed_seconds"] - steady_step) if first and steady_step and first.get("elapsed_seconds") else None,
        wall_s=clock() - started)
    return attempt


# ---------------------------------------------------------------------------------------------------- one result


def run_benchmark(task_id, device="cpu", attempts=None, runner=train_runner, clock=time.perf_counter, stop=None, emit=None,
                  info=None, measure=measured_peak, app_version="dev", division="single", work=None, now=time.time):
    """Run the task's attempts on one device and return the result dict (not yet stored)."""
    task = get_task(task_id)
    verify(task)
    attempts = attempts or task.attempts
    if not 1 <= attempts <= len(task.seeds):
        raise ValueError(f"Attempts must be between 1 and {len(task.seeds)}.")
    stop = stop or threading.Event()
    emit = emit or (lambda event: None)
    info = info or describe_device(device)
    limit = peak(info, task.precision, measure)
    flops = flops_per_token(task)
    energy = Energy(info.get("kind"))
    if info.get("kind") == "cuda" and __import__("torch").cuda.is_available():
        import torch
        torch.cuda.reset_peak_memory_stats()
    done = []
    for number, seed in enumerate(task.seeds[:attempts], 1):
        if stop.is_set():
            break
        emit({"type": "info", "text": f"MLPerf {task.id} v{task.version}: attempt {number} of {attempts} (seed {seed})"})
        done.append(run_attempt(task, seed, device, runner, clock, stop, emit, work))
    live = [a for a in done if a["status"] in ("converged", "did_not_converge")]
    if stop.is_set() and len(done) < attempts or any(a["status"] == "stopped" for a in done):
        status = "stopped"
    elif done and all(a["status"] == "converged" for a in done):
        status = "converged"
    elif live and not any(a["status"] == "failed" for a in done):
        status = "did_not_converge"
    else:
        status = "failed"
    times = [a["time_to_target_s"] for a in done if a["status"] == "converged"]
    tps = [a["tokens_per_s"] for a in live if a["tokens_per_s"]]
    sps = [a["steps_per_s"] for a in live if a["steps_per_s"]]
    median_tps = statistics.median(tps) if tps else None
    return {"format": RESULTS_FORMAT, "id": f"{int(now() * 1000):x}", "created_at": now(), "task": task.id, "version": task.version,
            "checksum": task.checksum, "division": division, "status": status, "app_version": app_version,
            "device": info, "label": f"{info.get('node', '')} · {info.get('name', '')}".strip(" ·"),
            "attempts": done, "attempt_count": attempts, "seeds": list(task.seeds[:attempts]), "target_bpc": task.target_bpc,
            "step_cap": task.step_cap, "precision": task.precision,
            "time_to_target": spread(times) if status == "converged" else None,
            "best_bpc": min((a["best_bpc"] for a in live if a.get("best_bpc") is not None), default=None),
            "tokens_per_s": median_tps, "steps_per_s": statistics.median(sps) if sps else None,
            "flops_per_token": flops, "peak": limit, "mfu": mfu(median_tps, flops, limit["flops"]),
            "setup_s": statistics.median([a["setup_s"] for a in live if a["setup_s"] is not None]) if live and any(a["setup_s"] is not None for a in live) else None,
            "warmup_s": statistics.median([a["warmup_s"] for a in live if a["warmup_s"] is not None]) if live and any(a["warmup_s"] is not None for a in live) else None,
            "peak_memory_bytes": _peak_memory(info.get("kind")), "energy_joules": energy.joules(),
            "error": next((a["error"] for a in done if a.get("error")), None)}


# ---------------------------------------------------------------------------------------------------- storage and views


def results_path(root):
    return Path(root) / "data" / "mlperf" / "results.jsonl"


def load_results(root):
    path = results_path(root)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a torn last line from a crash: the rest still reads
    return out


def store_result(root, result):
    """Append one result. A failed or stopped run is kept too: the page shows why it has no time."""
    path = results_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True) + "\n")
    return result


def _rank(result):
    return (result["status"] != "converged", (result.get("time_to_target") or {}).get("median", math.inf))


def leaderboard(results, task_id):
    """Per division, this task's newest version's results fastest first (converged before the rest), plus the versions held apart:
    {"task", "version", "divisions": {name: [result summaries]}, "other_versions": [{"version", "results"}]}. Versions are never ranked together."""
    task = get_task(task_id)
    mine = [r for r in results if r["task"] == task_id]
    current = [r for r in mine if r["version"] == task.version]
    board = {d: [] for d in DIVISIONS}
    for r in sorted(current, key=_rank):
        board.setdefault(r["division"], []).append(summary(r))
    older = {}
    for r in mine:
        if r["version"] != task.version:
            older[r["version"]] = older.get(r["version"], 0) + 1
    return {"task": task_id, "version": task.version, "title": task.title, "target_bpc": task.target_bpc, "divisions": board,
            "other_versions": [{"version": v, "results": n} for v, n in sorted(older.items(), reverse=True)]}


def summary(r):
    keys = ("id", "created_at", "task", "version", "division", "status", "label", "app_version", "time_to_target", "best_bpc", "tokens_per_s",
            "steps_per_s", "mfu", "peak_memory_bytes", "energy_joules", "setup_s", "warmup_s", "attempt_count", "target_bpc", "error")
    out = {k: r.get(k) for k in keys}
    out["peak_kind"] = (r.get("peak") or {}).get("kind")
    out["device"] = {k: (r.get("device") or {}).get(k) for k in ("kind", "name", "node", "torch", "cuda", "driver", "threads")}
    return out


def history(results, task_id, division="single", label=None):
    """The converged times of one task version's results in time order, with the app version each ran under: {"version", "points"}.
    `label` keeps one machine. Only the newest task version is followed: an older one is a different test."""
    task = get_task(task_id)
    rows = [r for r in results if r["task"] == task_id and r["version"] == task.version and r["division"] == division
            and (label is None or r.get("label") == label)]
    points = [{"id": r["id"], "created_at": r["created_at"], "app_version": r.get("app_version"), "label": r.get("label"),
               "status": r["status"], "median_s": (r.get("time_to_target") or {}).get("median"), "tokens_per_s": r.get("tokens_per_s")}
              for r in sorted(rows, key=lambda r: r["created_at"])]
    return {"task": task_id, "version": task.version, "division": division, "points": points}


def compare(results, ids):
    """Side-by-side summaries of the chosen results. Refuses results of different tasks or task versions: those are different tests."""
    chosen = [r for r in results if r["id"] in set(ids)]
    if len(chosen) != len(set(ids)):
        raise ValueError("A result to compare is missing.")
    if len({(r["task"], r["version"], r["checksum"]) for r in chosen}) > 1:
        raise ValueError("These results are of different tasks or task versions, so their times cannot be compared.")
    fastest = min((r["time_to_target"]["median"] for r in chosen if r.get("time_to_target")), default=None)
    rows = []
    for r in chosen:
        row = summary(r)
        median = (r.get("time_to_target") or {}).get("median")
        row["relative_time"] = median / fastest if median and fastest else None
        rows.append(row)
    return {"rows": rows}


# ---------------------------------------------------------------------------------------------------- the queue job


def job(job, context, emit, stop, pause):
    """Queue handler for kind "mlperf" (params: task, device, attempts): run the task and store the result under the workspace."""
    from studio_api.version import app_version
    p = job.params
    device = p.get("device") or "cpu"
    result = run_benchmark(p["task"], device, p.get("attempts"), stop=stop, emit=emit, app_version=app_version())
    store_result(context.root, result)
    job.params["result"] = {"id": result["id"], "status": result["status"]}
    emit({"type": "info", "text": f"Training speed {result['task']}: {result['status']}"
          + (f", {result['time_to_target']['median']:.1f} s to {result['target_bpc']} bits per character" if result["time_to_target"] else "")})

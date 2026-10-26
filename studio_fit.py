"""Fit a run to the computer that trains it: what this machine measures, and how far a solo pretraining run is scaled to it.

`hardware_profile` reads what the machine has right now (GPU memory total and free, free RAM, free disk, cores) and the speed
studio_probe last measured on it. `fit_solo` takes a pretraining config and that profile and answers whether the run fits. A run that
fits comes back unchanged, however big it is. A run that does not is scaled down only as far as it needs, in this order, and every
change is a note saying what and why:

1. a smaller batch with more accumulation (the same rows per optimizer step, in pieces that fit the device's memory);
2. the optimizer's state in host memory (studio_memory.offload_wanted's rule: only when even a batch of one does not fit);
3. a shorter context;
4. a smaller token budget (fewer rows read, so less disk and RAM, and less time) when the disk, the RAM or the wall-clock window the
   person allows cannot hold the run;
5. only last, a smaller model.

When even the smallest scaled run cannot fit it refuses, naming the resource and the number that blocks it. The notes that only say a
run is slow or tight (the window, an estimate within 10% of the memory) can be overridden with `keep_mine`; a run that physically
cannot fit never can. `fit_members` sizes a distributed run per member instead: a member that cannot fit its share is flagged, never
shrinking the others. A solo run never goes through it."""
from __future__ import annotations

import copy
import math
import os
import shutil
import time
from dataclasses import dataclass, field

DEFAULT_WINDOW_HOURS = 72.  # the time a run may take unless the person says otherwise: long enough for an overnight or multi-day run
TIGHT = 1.1  # an estimate this close above the memory is "tight", not "does not fit": the estimate is approximate and the OOM retry backs it up
CONTEXT_FLOOR = 128  # a context is never cut below this (or below the one the design asks for)
MIN_STEPS = 100  # a scaled run still trains at least this many optimizer steps
MAX_ACCUMULATION = 64  # studio_model.validate_config's limit
MAX_STEPS = 20_000_000  # studio_model.MAX_STEPS
RESIDENT_BYTES_PER_TOKEN = .5  # host memory a streamed text holds per token read: a few hundred bytes a row (studio_corpus), rows of about a thousand tokens
DISK_BYTES_PER_TOKEN = 6  # per token read, beside its text: the token cache (2 bytes + 1 of mask) and the shards (2 + 1)
TEXT_COPIES = 2  # the spool of the rows read, and the snapshot the run keeps beside its checkpoint
CORPUS_CHARS = 6  # studio_corpus.CHARS_PER_TOKEN: reading for a token budget may take this many characters a token
HEADROOM = 2 ** 30  # disk left alone when a run is sized to the free space
CHECKPOINTS = 3  # last.npz, best.npz and the copy being written
ORDER = ("tight", "batch", "offload", "context", "model")  # the memory notes in the order the scaling tries them


def size_text(n):
    return f"{n / 2 ** 30:.1f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.0f} MB"


def count_text(value):
    """133k, 4.2M, 545M, 4.0B."""
    for size, mark in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if value >= size:
            return f"{value / size:.0f}{mark}" if value >= 10 * size else f"{value / size:.1f}{mark}"
    return str(int(value))


def hours_text(hours):
    if hours < 1:
        return f"{max(1, round(hours * 60))} min"
    if hours >= 48:
        return f"{hours / 24:.1f} days"
    return f"{hours:.1f} h" if hours < 10 else f"{round(hours)} h"


@dataclass(frozen=True)
class Profile:
    """What the machine has. A figure that could not be read is None, and a check that needs it is skipped, never guessed."""

    device: str  # cuda | mps | cpu: what a run on "auto" with a big model uses
    gpu_name: str = ""
    vram_total: int = 0  # the GPU's own memory (cuda), or what Metal may address of the unified memory (mps); 0 without a GPU
    vram_free: int | None = None  # free now; None where it cannot be read (mps shares the RAM, see ram_free)
    ram_total: int | None = None
    ram_free: int | None = None
    disk_free: int | None = None
    cpu_cores: int = 1
    tokens_per_second: float | None = None  # the newest device probe on this machine (studio_probe), None before any
    measured_at: float | None = None  # when that probe was made (epoch seconds)

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, found):
        return cls(**{key: found[key] for key in cls.__dataclass_fields__ if key in found})


def hardware_profile(path=None, status=None, now=None):
    """The profile of this machine as it is now. The free figures are read on every call (another app, a chat model or an earlier run may
    hold memory), so the plan and the queue each take a fresh one right before they decide. `path` is where the run's files go (default:
    the workspace's root). `status` is studio_backend.gpu_status()."""
    import studio_memory
    import studio_probe
    from studio_backend import gpu_status
    from studio_core import ROOT
    status = gpu_status() if status is None else status
    kind = status["kind"] if status.get("available") else "cpu"
    vram_free = None
    if kind == "cuda":
        try:
            import torch
            vram_free = int(torch.cuda.mem_get_info(0)[0])
        except Exception:  # noqa: BLE001 - a driver that will not say is "unknown", not zero
            vram_free = None
    try:
        disk_free = shutil.disk_usage(path or ROOT).free
    except OSError:
        disk_free = None
    speed = at = None
    try:
        probes = studio_probe.reported(status, now=now) if kind != "cpu" else {}
    except Exception:  # noqa: BLE001 - a broken cache is no speed
        probes = {}
    if probes:
        newest = next(iter(probes.values()))
        seconds = newest["seconds"].get(newest["device"])
        speed = newest["tokens"] / seconds if seconds else None
        at = (time.time() if now is None else now) - newest["age_s"]
    return Profile(device=kind, gpu_name=status.get("name") or "", vram_total=int(status.get("memory_bytes") or 0), vram_free=vram_free,
                   ram_total=studio_memory.total_memory(), ram_free=studio_memory.available_memory(), disk_free=disk_free,
                   cpu_cores=os.cpu_count() or 1, tokens_per_second=speed, measured_at=at)


@dataclass
class Note:
    """One thing the fit changed (or kept) and why. `advisory`: it only says the run is slow or tight, so `keep_mine` drops it."""

    kind: str  # batch | offload | context | tokens | model | slow | tight
    reason: str  # vram | ram | disk | time
    text: str
    advisory: bool = False

    def to_dict(self):
        return {"kind": self.kind, "reason": self.reason, "text": self.text, "advisory": self.advisory}


@dataclass
class Fit:
    config: dict  # the config to run: the declared one, scaled
    steps: int
    token_budget: int | None  # the run's cap on tokens: the declared one, or the one the scaling set
    device: str
    wall_seconds: float | None = None  # set when the time window scaled the run: training then re-plans its schedule from the measured pace
    offload: bool = False
    notes: list = field(default_factory=list)
    changed: bool = False  # the config or the token budget differs from the one asked for
    refusal: str = ""  # a plain sentence when even the smallest scaled run cannot fit
    blocker: dict | None = None  # {"resource": vram | ram | disk, "needed": bytes, "available": bytes} behind the refusal
    can_keep_mine: bool = False  # a note is advisory: "use my settings anyway" would keep the person's figure
    kept_mine: bool = False  # the person asked for that and it kept something
    hours: float | None = None  # the run's time at the measured speed, None without one

    def to_dict(self):
        return {"notes": [n.to_dict() for n in self.notes], "scaled": self.changed, "refusal": self.refusal, "blocker": self.blocker,
                "can_keep_mine": self.can_keep_mine, "kept_mine": self.kept_mine, "offload": self.offload, "device": self.device,
                "steps": self.steps, "token_budget": self.token_budget, "hours": self.hours}

    def job_params(self):
        """What a pretrain job carries from this fit, beside its text and mode."""
        return {"config": self.config, "offload_optimizer": self.offload, **({"token_budget": self.token_budget} if self.token_budget else {}),
                **({"wall_seconds": self.wall_seconds} if self.wall_seconds else {})}


def _device(cfg, profile, device):
    if device:
        return device
    wanted = cfg["device"]
    if wanted == "cpu":
        return "cpu"
    return wanted if wanted in ("cuda", "mps") else profile.device


def _pool(profile, device, share):
    """(bytes a run on `device` may use now, what to call that memory). The GPU's own memory on CUDA; the machine's RAM on the CPU and
    on Apple's MPS, which shares it. At most `share` of the total (the queue's plan, less what chat's models hold), and never more
    than is free right now."""
    if device == "cuda":
        total, free, name = profile.vram_total, profile.vram_free, "GPU memory"
    elif device == "mps":
        known = [x for x in (profile.vram_total, profile.ram_total) if x]  # Metal may address part of the unified memory
        total, free, name = (min(known) if known else 0), profile.ram_free, "shared memory"
    else:
        total, free, name = profile.ram_total, profile.ram_free, "memory"
    if not total:
        return 0, name
    usable = int(total * share)
    return (min(usable, free) if free is not None else usable), name


def _with_architecture(cfg, spec):
    from studio_model import validate_config
    return validate_config(dict(cfg, architecture=spec))


def _shorter(cfg, context):
    from studio_spec import validate_spec
    spec = copy.deepcopy(cfg["architecture"])
    spec["dims"]["n_ctx"] = context
    return _with_architecture(cfg, validate_spec(spec))


def _designs(cfg):
    """Smaller designs of cfg, the biggest first: [(spec, parameters)]. A hand-built per-block design has none."""
    import studio_estimate
    from studio_ia import sizing
    from studio_model import effective_spec, vocabulary_guess
    spec = effective_spec(cfg)
    if spec.get("blocks"):
        return []
    vocab = vocabulary_guess(cfg)
    dims = spec["dims"]
    now = studio_estimate.estimate(spec, vocab)["total"]
    widths = {w for w, _ in sizing.LADDER if w < dims["d_model"]} | {dims["d_model"]}
    depths = {n for _, n in sizing.LADDER if n < dims["n_layer"]} | {max(1, dims["n_layer"] // k) for k in (2, 4, 8)} | {dims["n_layer"]}
    found = []
    for width in widths:
        for layers in depths:
            if (width, layers) == (dims["d_model"], dims["n_layer"]):
                continue
            smaller = sizing.resized_spec(spec, width, layers)
            if smaller is None:
                continue
            params = studio_estimate.estimate(smaller, vocab)["total"]
            if params < now:
                found.append((params, smaller))
    return [(spec2, params) for params, spec2 in sorted(found, key=lambda row: -row[0])]


class _Sizer:
    """The memory arithmetic for one machine: what a config's training step holds on the device, against what the device can give."""

    def __init__(self, profile, device, share, vocab):
        self.profile, self.device, self.vocab = profile, device, vocab
        self.usable, self.pool_name = _pool(profile, device, share)
        self.offload_ok = device == "cuda"

    def bytes_for(self, cfg, offload, batch=None):
        import studio_memory
        fixed, per_sample = studio_memory.training_bytes(cfg, self.vocab, offload)
        return fixed + per_sample * (cfg["batch_size"] if batch is None else batch)

    def largest_batch(self, cfg, offload):
        """The biggest batch of cfg (at most its own) whose step fits, 0 when not even one does."""
        import studio_memory
        fixed, per_sample = studio_memory.training_bytes(cfg, self.vocab, offload)
        if per_sample <= 0:
            return cfg["batch_size"] if fixed <= self.usable else 0
        return int(max(0, min(cfg["batch_size"], (self.usable - fixed) // per_sample)))

    def host_moments(self, cfg):
        import studio_memory
        return studio_memory.host_bytes(cfg, self.vocab, True)

    def host_holds(self, cfg):
        free = self.profile.ram_free
        return free is None or self.host_moments(cfg) <= free


def _batch_text(declared, micro, accumulation, effective, name, usable):
    if effective == micro * accumulation:
        return f"Batch {declared} -> {micro} x {accumulation} accumulation to fit the {size_text(usable)} of {name} free."
    why = f"accumulation stops at {MAX_ACCUMULATION}" if accumulation >= MAX_ACCUMULATION else f"{micro} does not divide {effective}"
    return (f"Batch {declared} -> {micro} x {accumulation} accumulation ({micro * accumulation} rows a step instead of {effective}, "
            f"because {why}) to fit the {size_text(usable)} of {name} free.")


def _context_note(context, ctx, sizer):
    return Note("context", "vram", f"Context {context} -> {ctx} tokens: the activations at {context} do not fit the {size_text(sizer.usable)} of "
                                   f"{sizer.pool_name} free, even one row at a time{' with the optimizer in RAM' if sizer.offload_ok else ''}.")


def fit_solo(config, profile, *, share=1., pieces=None, chars_per_piece=4., token_budget=None, window_hours=DEFAULT_WINDOW_HOURS,
             speed=None, keep_mine=False, fixed_architecture=False, device=None, vocab=None):
    """Fit the solo pretraining run `config` (an engine config with its design in "architecture") to `profile`.

    `share`: the part of the device's memory the queue's plan gives the run. `pieces`: the word pieces of the chosen text, when known;
    `chars_per_piece` converts them to characters for the disk. `token_budget`: the run's own cap on tokens (the design's token limit).
    `window_hours`: the wall-clock time the person allows; with `speed` (tokens a second, measured for this design here) the tokens are
    scaled to it. `keep_mine`: the person pressed "use my settings anyway" (drops the advisory notes). `fixed_architecture`: the run
    continues weights (a catch-up, a resume), so its context and size cannot change. `device` overrides the one the config and the
    profile imply. Returns a Fit; a run that fits comes back with its config as given and no notes."""
    import studio_memory
    from studio_model import validate_config, vocabulary_guess
    cfg = validate_config(config)
    device = _device(cfg, profile, device)
    sizer = _Sizer(profile, device, share, vocabulary_guess(cfg) if vocab is None else vocab)
    out = Fit(config=dict(config), steps=cfg["steps"], token_budget=token_budget, device=device)
    b0, a0, ctx0 = cfg["batch_size"], cfg["accumulation"], cfg["context"]
    effective0 = b0 * a0
    planned = cfg["steps"] * effective0 * ctx0
    if token_budget:
        planned = min(planned, token_budget)
    if device in ("cuda", "mps") and profile.device != device:
        out.refusal = f"This run is set to use the {device.upper()}, but this computer has no {device.upper()} GPU."
        return out
    if not sizer.usable:
        out.blocker = {"resource": "vram" if device == "cuda" else "ram", "needed": int(sizer.bytes_for(cfg, False, 1)), "available": 0}
        out.refusal = f"This computer reports no {sizer.pool_name} to train in."
        return out

    # the device's memory: a smaller batch, the optimizer on the host, a shorter context, last a smaller model --------------------------
    chosen, offload, notes = cfg, False, []
    micro = sizer.largest_batch(cfg, False)
    tight = micro < b0 and sizer.bytes_for(cfg, False) <= sizer.usable * TIGHT  # within the estimate's error: a person may want to try it
    if micro < 1 and sizer.offload_ok and sizer.host_holds(cfg):
        offload, micro = True, sizer.largest_batch(cfg, True)
    if micro < 1 and not fixed_architecture:
        chosen, offload, micro, notes = _smaller(cfg, sizer, ctx0)
    if micro < 1:
        _refuse_memory(out, cfg, sizer, fixed_architecture)
        return out
    micro = min(micro, b0)
    if micro < b0:  # a batch that divides the declared rows keeps the rows per step exactly, if that costs under half the batch
        micro = next((d for d in range(micro, max(0, micro // 2), -1) if effective0 % d == 0), micro)
    accumulation = a0 if micro == b0 else min(MAX_ACCUMULATION, math.ceil(effective0 / micro))
    out.can_keep_mine = tight
    if tight and keep_mine:
        chosen, offload, micro, accumulation = cfg, False, b0, a0
        out.kept_mine = True
        out.notes.append(Note("tight", "vram", f"Batch {b0} needs about {size_text(int(sizer.bytes_for(cfg, False)))} against the "
                                               f"{size_text(sizer.usable)} of {sizer.pool_name} free: tight, kept as you asked. If it does run out "
                                               "of memory the batch is halved and training goes on.", advisory=True))
    else:
        if micro != b0 or accumulation != a0:
            out.notes.append(Note("batch", "vram", _batch_text(b0, micro, accumulation, effective0, sizer.pool_name, sizer.usable), advisory=tight))
        if offload:
            out.notes.append(Note("offload", "vram", f"Optimizer state moved to RAM ({size_text(sizer.host_moments(chosen))}), because even one row at a "
                                                     f"time does not fit the {size_text(sizer.usable)} of GPU memory free with it on the GPU. Steps are slower."))
        out.notes.extend(notes)
        out.notes.sort(key=lambda n: ORDER.index(n.kind))  # the order they were tried in, whatever order they were found
    out.offload = offload
    new = dict(chosen, batch_size=micro, accumulation=accumulation)
    per_step = micro * accumulation * new["context"]
    floor_tokens = MIN_STEPS * per_step

    # the text read: disk and RAM, scaled with the token budget (a budget below one pass reads less, studio_corpus) -------------------
    weights = studio_memory.training_bytes(new, sizer.vocab, False)[0] // studio_memory.ADAMW_COPIES
    host = (sizer.host_moments(new) if offload else 0) + 2 * weights  # a checkpoint's copy of the weights, and the moments when offloaded
    checkpoints = CHECKPOINTS * weights * 3  # each holds the weights and AdamW's two moments
    reads = min(pieces, planned) if pieces else planned
    cap, why, available, hard = None, "", 0, 0
    for resource, free in (("disk", profile.disk_free), ("ram", profile.ram_free)):
        if free is None:
            continue
        room = free - (checkpoints + HEADROOM if resource == "disk" else host)
        for chars in ((chars_per_piece, CORPUS_CHARS) if resource == "disk" else (0,)):  # a budgeted read may overshoot by up to CORPUS_CHARS a token
            per_token = TEXT_COPIES * chars + DISK_BYTES_PER_TOKEN if resource == "disk" else RESIDENT_BYTES_PER_TOKEN
            allowed = max(0, int(room / per_token))
            if allowed >= reads:
                break
        else:
            if cap is None or allowed < cap:
                cap, why, available = allowed, resource, free
                hard = int(floor_tokens * per_token + (checkpoints + HEADROOM if resource == "disk" else host))
    if cap is not None:
        if cap < min(floor_tokens, reads):
            out.blocker = {"resource": why, "needed": hard, "available": int(available)}
            out.refusal = (f"Not enough disk space: {size_text(available)} free, and even the smallest run ({count_text(floor_tokens)} tokens of text, "
                           f"{MIN_STEPS} steps) needs about {size_text(hard)}. Free space in Resources > Storage." if why == "disk" else
                           f"Not enough RAM: {size_text(available)} free, and even the smallest run needs about {size_text(hard)}.")
            return out
        what = ("disk" if why == "disk" else "RAM")
        out.notes.append(Note("tokens", why, f"Token budget {count_text(token_budget or reads)} -> {count_text(cap)}: more text would not fit the "
                                             f"{size_text(available)} of {what} free" + (" once its token cache and the checkpoints are counted." if why == "disk" else ".")))
        planned = token_budget = min(planned, cap)

    # the wall-clock window ----------------------------------------------------------------------------------------------------------
    hours = planned / speed / 3600 if speed else None
    if hours is not None and hours > window_hours:
        if keep_mine:
            out.kept_mine = out.can_keep_mine = True
            out.notes.append(Note("slow", "time", f"At {speed:,.0f} tokens/s this takes about {hours_text(hours)}, longer than the "
                                                  f"{hours_text(window_hours)} window. Kept as you asked.", advisory=True))
        else:
            fitted = max(floor_tokens, int(window_hours * 3600 * speed))
            out.notes.append(Note("tokens", "time", f"Token budget {count_text(planned)} -> {count_text(fitted)} to finish in about "
                                                    f"{hours_text(window_hours)} at {speed:,.0f} tokens/s.", advisory=True))
            out.can_keep_mine = True
            planned = token_budget = fitted
            out.wall_seconds = window_hours * 3600
            hours = planned / speed / 3600
    out.hours = hours
    if not out.notes:
        return out  # it fits: exactly the run that was asked for
    out.steps = max(1, min(MAX_STEPS, math.ceil(planned / per_step)))
    out.token_budget = token_budget
    out.changed = any(n.kind != "tight" and n.kind != "slow" for n in out.notes)
    if out.changed:
        out.config = {**config, "batch_size": micro, "accumulation": accumulation, "steps": out.steps}
        if chosen is not cfg:
            out.config.update({key: chosen[key] for key in ("architecture", "context", "width", "layers", "heads", "expansion")})
    else:
        out.steps = cfg["steps"]
    return out


RECORD = "fit"  # the job param that remembers what was asked for, so the queue can fit it again to the machine it starts on


def record(fit, config, token_budget, *, window_hours, keep_mine, speed, pieces, chars_per_piece):
    """`params["fit"]` for a job built from `fit`: the settings as the person declared them (`config`, `token_budget`), everything the
    fit read besides the machine, and what it said. The queue fits the declared settings again just before the job starts."""
    return {"declared": {"config": config, "token_budget": token_budget}, "window_hours": window_hours, "keep_mine": keep_mine, "speed": speed,
            "pieces": pieces, "chars_per_piece": chars_per_piece, "notes": [n.to_dict() for n in fit.notes], "scaled": fit.changed}


def refit(params, profile, share, remote=False):
    """Fit the pretrain job whose params are `params` to `profile` as it is now, from its declared settings (the plan's record, else its
    own config), or None when it must not be scaled here: it resumes a checkpoint (its batch and design are part of that file) or has
    nothing to fit. `remote`: the job was placed here by another machine, whose measured speed says nothing about this one."""
    declared = (params.get(RECORD) or {}).get("declared") or {}
    config = declared.get("config") or params.get("config")
    if not config or params.get("resume") or params.get("continue_from") or params.get("resume_rebase"):
        return None
    out = params.get("output_dir")
    if out and (os.path.exists(os.path.join(out, "last.npz")) or os.path.exists(os.path.join(out, "config.json"))):
        return None  # an earlier attempt's folder: the job restarts from it
    seen = params.get(RECORD) or {}
    budget = declared["token_budget"] if "token_budget" in declared else params.get("token_budget")
    return fit_solo(config, profile, share=share, pieces=seen.get("pieces") or params.get("pieces_per_pass"),
                    chars_per_piece=seen.get("chars_per_piece") or 4., token_budget=budget,
                    window_hours=seen.get("window_hours") or DEFAULT_WINDOW_HOURS, speed=None if remote else seen.get("speed"),
                    keep_mine=bool(seen.get("keep_mine")), fixed_architecture=bool(params.get("init_from")) and params.get("mode") == "catch_up")


def applied(params, fit):
    """`params` of a pretrain job with `fit` put in: its scaled config, token budget, wall-clock cap and optimizer placement (a fit that
    changes nothing leaves the job as it was), and the record of what was declared and what was said."""
    seen = dict(params.get(RECORD) or {})
    declared = seen.get("declared") or {"config": params["config"], "token_budget": params.get("token_budget")}
    new = {key: value for key, value in params.items() if key not in ("wall_seconds", "token_budget")}
    new.update(fit.job_params())
    new[RECORD] = {**seen, "declared": declared, "notes": [n.to_dict() for n in fit.notes], "scaled": fit.changed}
    return new


def _smaller(cfg, sizer, context):
    """The design that fits when the declared one, even at a batch of one with the optimizer offloaded, does not: first a shorter
    context, then a smaller model at the shortest context. Returns (config, offload, largest batch, notes); batch 0 when none fits."""
    floor = min(context, CONTEXT_FLOOR)
    options = (False, True) if sizer.offload_ok else (False,)
    ctx, current = context, cfg
    while ctx > floor:
        ctx = max(floor, ctx // 2)
        current = _shorter(cfg, ctx)
        for offload in options:
            if offload and not sizer.host_holds(current):
                continue
            micro = sizer.largest_batch(current, offload)
            if micro >= 1:
                return current, offload, micro, [_context_note(context, ctx, sizer)]
    shorter = [_context_note(context, ctx, sizer)] if ctx != context else []
    for spec, params in _designs(current):
        candidate = _with_architecture(current, spec)
        for offload in options:
            if offload and not sizer.host_holds(candidate):
                continue
            micro = sizer.largest_batch(candidate, offload)
            if micro >= 1:
                return candidate, offload, micro, shorter + [Note(
                    "model", "vram", f"Model {cfg['width']} wide x {cfg['layers']} layers -> {candidate['width']} wide x {candidate['layers']} layers "
                                     f"({count_text(params)} parameters): the original does not fit the {size_text(sizer.usable)} of "
                                     f"{sizer.pool_name} free at any batch size or context.")]
    return cfg, False, 0, []


def _refuse_memory(out, cfg, sizer, fixed_architecture):
    """The refusal when nothing fits: the smallest run the scaling would reach, and what it needs against what is free."""
    smallest = cfg
    if not fixed_architecture:
        smallest = _shorter(cfg, min(cfg["context"], CONTEXT_FLOOR)) if cfg["context"] > CONTEXT_FLOOR else cfg
        designs = _designs(smallest)
        if designs:
            smallest = _with_architecture(smallest, designs[-1][0])
    offload = sizer.offload_ok and sizer.host_holds(smallest)
    need = int(sizer.bytes_for(smallest, offload, 1))
    out.blocker = {"resource": "vram" if sizer.device == "cuda" else "ram", "needed": need, "available": int(sizer.usable)}
    kept = "the model it continues from" if fixed_architecture else f"the smallest model ({smallest['width']} wide, {smallest['layers']} layers)"
    out.refusal = (f"Not enough {sizer.pool_name}: {size_text(sizer.usable)} free for this run, and even {kept} at one row and context "
                   f"{smallest['context']}{' with the optimizer in RAM' if offload else ''} needs about {size_text(need)}. "
                   + ("Close other programs that use the GPU, or pick a smaller design." if sizer.device == "cuda" else "Close other programs, or pick a smaller design."))


def fit_members(config, members, *, pieces=None, token_budget=None, window_hours=DEFAULT_WINDOW_HOURS):
    """Size a distributed run per member (the solo rules never apply to it). `members`: [{"name", "profile", "share", "speed"}], each its
    own machine's profile, memory share and measured tokens/s for this design. Every member keeps the design and context (they share one
    set of weights), so its own memory decides only its batch and accumulation (the same rows per step) and whether its optimizer is
    offloaded; a member that cannot fit even one row with its optimizer offloaded is flagged `fits: False` with the reason, and the others
    are left exactly as they would be alone. The run's token budget comes from the members' combined measured speed over the window, over
    the members that fit. Returns {"members": [{"name", "fits", "batch_size", "accumulation", "offload", "notes", "refusal"}], "flagged":
    [names], "tokens_per_second": combined | None, "token_budget": int | None, "note": str}."""
    from studio_model import validate_config, vocabulary_guess
    cfg = validate_config(config)
    out, combined = [], 0.
    b0, a0 = cfg["batch_size"], cfg["accumulation"]
    for member in members:
        profile = member["profile"]
        sizer = _Sizer(profile, _device(cfg, profile, None), member.get("share", 1.), vocabulary_guess(cfg))
        row = {"name": member["name"], "fits": True, "batch_size": b0, "accumulation": a0, "offload": False, "notes": [], "refusal": ""}
        micro = sizer.largest_batch(cfg, False) if sizer.usable else 0
        if micro < 1 and sizer.usable and sizer.offload_ok and sizer.host_holds(cfg):
            row["offload"], micro = True, sizer.largest_batch(cfg, True)
            row["notes"].append(f"{member['name']}: optimizer state moved to RAM, because even one row does not fit its "
                                f"{size_text(sizer.usable)} of GPU memory.")
        if micro < 1:
            need = int(sizer.bytes_for(cfg, sizer.offload_ok, 1))
            row.update(fits=False, refusal=f"{member['name']} cannot take part: it has {size_text(sizer.usable)} of {sizer.pool_name} for this "
                                           f"run and the design needs about {size_text(need)} even at one row.")
        else:
            if micro < b0:
                row["batch_size"], row["accumulation"] = micro, min(MAX_ACCUMULATION, math.ceil(b0 * a0 / micro))
                row["notes"].append(f"{member['name']}: batch {b0} -> {micro} x {row['accumulation']} accumulation to fit "
                                    f"{size_text(sizer.usable)} of {sizer.pool_name}.")
            combined += member.get("speed") or 0.
        out.append(row)
    tokens = int(window_hours * 3600 * combined) if combined else None
    wanted = min(x for x in (token_budget, pieces) if x) if (token_budget or pieces) else None
    budget = tokens if tokens and wanted and tokens < wanted else None
    note = (f"Token budget {count_text(wanted)} -> {count_text(tokens)} to finish in about {hours_text(window_hours)} at the {combined:,.0f} "
            "tokens/s the members that fit measure together.") if budget else ""
    return {"members": out, "flagged": [row["name"] for row in out if not row["fits"]], "tokens_per_second": combined or None,
            "token_budget": budget, "note": note}

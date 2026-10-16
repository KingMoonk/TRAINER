"""A job's memory budget and the batch that fits it, from the model's own estimate (studio_estimate) and the device.

A CUDA GPU has memory of its own; Apple's MPS shares the machine's RAM (unified memory), so does the CPU. The queue's plan
says what share of that memory a job may take, and the job lowers its batch size until a training step fits. The estimate
can only be approximate (a framework keeps workspace of its own), so the out-of-memory retry in studio_model.train stays as
the backstop for a batch the estimate accepted."""
from __future__ import annotations

import threading

import torch

ADAMW_COPIES = 4  # per trainable parameter: the weight, its gradient and AdamW's two moments
STATE_COPIES = 2  # of those, the two moments: the part that can live on the host while the update streams them through the device


def pool(device):
    """The memory a model on `device` takes from: a CUDA GPU's own, or the machine's RAM (the CPU, and Apple's MPS, which shares it)."""
    return "gpu" if device == "cuda" else "ram"


class StateHome:
    """Where an optimizer's state lives between steps when it is offloaded: in host memory (pinned on CUDA, so the copies to and
    from the device can overlap), fetched to the parameter's device for its update and copied back. The one place that touches
    devices, so a test can offload on the CPU (host = device = cpu, nothing pinned)."""

    def __init__(self, host="cpu", pin=False):
        self.host, self.pin = host, pin

    @classmethod
    def for_device(cls, device):
        """The home for a run on `device`; only CUDA has memory of its own to relieve (MPS and the CPU share the RAM)."""
        return cls(pin=device == "cuda")

    def zeros_like(self, value):
        return torch.zeros(value.shape, dtype=value.dtype, device=self.host, pin_memory=self.pin)

    def fetch(self, stored, device):
        """A copy of `stored` on `device`, in time for the update that reads it."""
        return stored.to(device, non_blocking=self.pin, copy=True)

    def store(self, stored, resident):
        stored.copy_(resident, non_blocking=self.pin)

    def synchronize(self, device):
        """Wait for the copies back to the host to land (they are asynchronous when pinned) before anything reads the host copy."""
        if self.pin and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)


def weight_bytes(model):
    """What a compiled model's weights hold, in the memory of the device they are on."""
    return sum(value.numel() * value.element_size() for value in model.p.values())


class Ledger:
    """What the app process holds in memory outside the running job: the chat models, each registered when it loads and released
    when it is evicted. The queue subtracts a pool's total from the share it hands a job on that pool; the worker process never
    sees the ledger, only the reduced share. Entries stay in the order they were registered."""

    def __init__(self):
        self._lock = threading.Lock()
        self._held = {}  # key -> (bytes, device)

    def register(self, key, nbytes, device="cpu"):
        with self._lock:
            self._held[key] = (int(nbytes), device)

    def release(self, key):
        with self._lock:
            self._held.pop(key, None)

    def total(self, device="cpu"):
        """Bytes held in the memory a job on `device` draws on."""
        with self._lock:
            return sum(n for n, d in self._held.values() if pool(d) == pool(device))

    def largest(self, device="cpu"):
        """Bytes of the biggest single entry in that memory (0 when there is none)."""
        with self._lock:
            return max((n for n, d in self._held.values() if pool(d) == pool(device)), default=0)


def device_of(cfg):
    """The device a training run of the validated config `cfg` lands on: the choice the model's build makes."""
    from studio_backend import get_backend
    d = cfg["width"], cfg["context"], cfg["layers"]
    return get_backend(cfg["device"], work=cfg["batch_size"] * d[1] * d[0] * max(1, d[2])).device


def physical_memory():
    from studio_passive import total_memory  # has the Windows path; os.sysconf is POSIX-only
    return total_memory()


def budget(device, share, ram=None, gpu=None):
    """Bytes a job may use: `share` of the GPU's memory on cuda and mps (what torch reports it can address), or of the
    machine's RAM on the CPU and on a GPU whose memory is unknown (reported as 0)."""
    total = 0
    if device in ("cuda", "mps"):
        if gpu is None:
            from studio_backend import gpu_status
            gpu = gpu_status()["memory_bytes"]
        total = gpu
    if not total:
        total = physical_memory() if ram is None else ram
    return int(total * share)


def training_bytes(cfg, vocab=None, offload=False):
    """(fixed, per_sample): what a training step of the validated config holds on the device whatever the batch (weights,
    gradients and AdamW's moments, in the weights' dtype) and per sample of the batch (one context of activations kept for
    the backward pass, plus the logits and their gradient, in the compute dtype).

    With `offload` the moments live on the host (see host_bytes): the device holds only the two moments of the parameter being
    updated, bounded by the largest module's parameters."""
    return _training_bytes(cfg, vocab, offload)[:2]


def host_bytes(cfg, vocab=None, offload=True):
    """What the optimizer state takes in the host's memory: AdamW's two moments when `offload` is on, otherwise nothing."""
    return _training_bytes(cfg, vocab, offload)[2]


def offload_wanted(cfg, share, device=None, ram=None, gpu=None, vocab=None):
    """Whether a run of the validated config should keep its optimizer state on the host: only on CUDA, and only when even a
    batch of one does not fit `share` of the GPU's memory with the state on the device. (A smaller batch is cheaper than
    streaming the state every step, so the batch is lowered first.)"""
    if device is None:
        device = device_of(cfg)
    if device != "cuda":
        return False
    fixed, per_sample = training_bytes(cfg, vocab)
    return fixed + per_sample > budget(device, share, ram, gpu)


def _training_bytes(cfg, vocab, offload):
    from studio_estimate import estimate
    from studio_model import effective_spec, vocabulary_guess
    from studio_spec import module_of
    spec = effective_spec(cfg)
    vocab = vocabulary_guess(cfg) if vocab is None else vocab
    result = estimate(spec, vocab)
    precision, precision_cfg = module_of("precision", spec["train"]["precision"])
    weight_bytes = torch.empty(0, dtype=precision.build(precision_cfg, None, None)).element_size()
    compute_bytes = torch.empty(0, dtype=getattr(torch, result["compute_dtype"])).element_size()
    fixed = result["total"] * weight_bytes * ADAMW_COPIES
    per_sample = (result["activation_bytes_per_token"] + 2 * vocab * compute_bytes) * spec["dims"]["n_ctx"]
    if not offload:
        return fixed, per_sample, 0
    largest = max((row["params"] for row in result["rows"]), default=0)  # one parameter is never bigger than its module's
    moments = result["total"] * weight_bytes * STATE_COPIES
    return fixed - moments + min(moments, largest * weight_bytes * STATE_COPIES), per_sample, moments


def batch_cap(config, share, device=None, ram=None, gpu=None, offload=False):
    """The largest batch of `config` whose training step fits `share` of the memory of the device it will run on: never
    above the declared batch size, never below 1 (the out-of-memory retry handles the rest). An estimate with nothing per
    sample (every module skipped) keeps the declared batch size. `offload`: the optimizer state is on the host, not the device."""
    from studio_model import validate_config
    cfg = validate_config(config)
    declared = cfg["batch_size"]
    fixed, per_sample = training_bytes(cfg, offload=offload)
    if per_sample <= 0:
        return declared
    if device is None:
        device = device_of(cfg)
    room = budget(device, share, ram, gpu) - fixed
    return max(1, min(declared, room // per_sample))

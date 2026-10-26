"""A job's memory budget and the batch that fits it, from the model's own estimate (studio_estimate) and the device.

A CUDA GPU has memory of its own; Apple's MPS shares the machine's RAM (unified memory), so does the CPU. The queue's plan
says what share of that memory a job may take, and the job lowers its batch size until a training step fits. The estimate
can only be approximate (a framework keeps workspace of its own), so the out-of-memory retry in studio_model.train stays as
the backstop for a batch the estimate accepted."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

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


def validated(config):
    from studio_model import validate_config
    return validate_config(config)


def probe_args(cfg):
    """(spec, vocabulary, batch) the device probe of a run of the validated config `cfg` is keyed by (studio_probe.shape_key): what the
    model's own build passes it. The vocabulary is the config's guess, which is the tokenizer's for every fixed tokenizer; the run's batch is
    the declared one unless the memory plan lowered it, in which case the probe it made is not found here and the size rule decides."""
    from studio_model import effective_spec, vocabulary_guess
    from studio_spec import validate_spec
    return validate_spec(effective_spec(cfg)), vocabulary_guess(cfg), cfg["batch_size"]


def shape_of(cfg):
    """The design shape key of the validated config (what a node reports its measured speeds under)."""
    import studio_probe
    return studio_probe.shape_key(*probe_args(cfg))


def device_of(cfg):
    """The device a training run of the validated config `cfg` lands on: the choice the model's build makes. On "auto" with a GPU, that
    is the measured pick when this machine has a cached probe of the design (studio_probe.cached; nothing is timed here), else the size rule."""
    from studio_backend import get_backend
    d = cfg["width"], cfg["context"], cfg["layers"]
    measure = None
    import studio_probe
    if cfg["device"] == "auto" and studio_probe.any_stored():  # nothing stored: do not bring the GPU up just to look
        def measure():
            from studio_backend import gpu_status
            return studio_probe.cached(*probe_args(cfg), gpu_status())
    return get_backend(cfg["device"], work=cfg["batch_size"] * d[1] * d[0] * max(1, d[2]), measure=measure).device


def process_memory():
    """Bytes this process uses now. Falls back to the peak only where the current figure cannot be read."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize)
    if sys.platform == "darwin":
        try:
            import ctypes
            info = (ctypes.c_uint64 * 12)()  # rusage_info_v0: 16-byte uuid, user and system time, two wakeup counts, pageins, wired size, resident size, ...
            if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 0, info) == 0:
                return int(info[8])
        except (OSError, AttributeError):
            pass
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024  # bytes on macOS, kilobytes on Linux


def _windows_memory():
    """(total, available) physical bytes on Windows, from GlobalMemoryStatusEx."""
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    status = Status()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def total_memory():
    if sys.platform == "win32":
        return _windows_memory()[0]
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def available_memory():
    """Bytes of RAM free for a program to take now (what the system could hand out without swapping), or None where it cannot be read."""
    if sys.platform == "win32":
        return _windows_memory()[1]
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "darwin":
        try:
            import re
            import subprocess
            text = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=True).stdout
            page = int(re.search(r"page size of (\d+) bytes", text).group(1))
            pages = sum(int(re.search(rf"Pages {name}:\s+(\d+)", text).group(1)) for name in ("free", "inactive", "speculative"))
            return pages * page
        except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
            return None
    return None


def physical_memory():
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

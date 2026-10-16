"""Compute devices: PyTorch on the CPU, a CUDA GPU or Apple's MPS, with explicit device reporting."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

DEVICES = ("auto", "cpu", "cuda", "mps")


def background_threads(cores=None):
    """How many threads torch's intra-op pool gets while the app is open: every core but one, so the Tk thread always has a
    core to draw on and does not wait on the GIL behind torch's workers."""
    return max(1, (cores or os.cpu_count() or 1) - 1)


def cap_threads():
    """Cap torch's intra-op pool for in-app background work (training jobs, scoring), where it starts. The pool is one per
    process, not per thread, so this also caps any other torch work in the app: none of it runs on the Tk thread. A standalone
    run (train.py) does not call it and keeps every core."""
    torch.set_num_threads(background_threads())


_probe = threading.Lock()


@lru_cache(maxsize=1)
def gpu_status():
    """Probe the GPU torch can see (CUDA first, then MPS) with a small calculation; a visible device alone isn't enough.
    One probe at a time: the cache only serves calls that arrive after the first one has finished, and two threads
    bringing Metal up at once (the API runs its routes in a thread pool) crash the process."""
    with _probe:
        return _probe_gpu()


def _probe_gpu():
    try:
        if torch.cuda.is_available():
            kind, name = "cuda", torch.cuda.get_device_name(0)
            memory = int(torch.cuda.get_device_properties(0).total_memory)
        elif torch.backends.mps.is_available():
            kind, name, memory = "mps", "Apple GPU (Metal)", 0
        else:
            raise RuntimeError("No CUDA or Metal GPU was found.")
        sample = torch.ones((4, 4), dtype=torch.float32, device=kind)
        result = float(torch.sqrt(sample @ sample).sum().item())
        if result != 32.:
            raise RuntimeError(f"{kind} calculation check failed.")
        return {"available": True, "kind": kind, "name": name, "reason": "", "memory_bytes": memory}
    except Exception as exc:
        return {"available": False, "kind": "", "name": "", "reason": f"{type(exc).__name__}: {exc}", "memory_bytes": 0}


@dataclass(frozen=True)
class Backend:
    xp: object
    device: str
    name: str
    reason: str = ""

    def to_cpu(self, value):
        """A NumPy copy of `value` (a tensor on any device, or anything array-like); never a view of a live tensor."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return (value.float() if value.dtype == torch.bfloat16 else value).numpy().copy()  # NumPy has no bfloat16
        return np.asarray(value)

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()


# Below this many values per step (batch x context x width x layers) the GPU's per-kernel launch cost outweighs
# its arithmetic, so "auto" stays on the CPU. Measured: CPU wins at ~1-1.5M, the GPU only pulls ahead near 4M.
AUTO_GPU_MIN_WORK = 3_000_000


def get_backend(device="auto", work=None):
    if device not in DEVICES:
        raise ValueError("Device must be auto, cpu, cuda, or mps.")
    if device == "cpu":
        return Backend(torch, "cpu", "PyTorch / CPU")
    if device == "auto" and work is not None and work < AUTO_GPU_MIN_WORK:
        return Backend(torch, "cpu", "PyTorch / CPU", f"auto chose CPU: model is small ({work:,} values per step)")
    status = gpu_status()
    kind = status.get("kind") or "cuda"
    if status["available"] and (device == "auto" or device == kind):
        return Backend(torch, kind, f"PyTorch / GPU · {status['name']}")
    reason = "GPU unavailable: " + (status["reason"] or f"no {device} device is present")
    if device != "auto":
        raise RuntimeError(reason + "\nInstall a PyTorch build for this device, or select CPU.")
    return Backend(torch, "cpu", "PyTorch / CPU", reason)

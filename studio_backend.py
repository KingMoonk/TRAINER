"""Compute devices: PyTorch on the CPU, a CUDA GPU or Apple's MPS, with explicit device reporting."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

DEVICES = ("auto", "cpu", "cuda", "mps")


@lru_cache(maxsize=1)
def gpu_status():
    """Probe the GPU torch can see (CUDA first, then MPS) with a small calculation; a visible device alone isn't enough."""
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
            return value.detach().cpu().numpy().copy()
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

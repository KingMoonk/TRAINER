"""Optional CUDA arrays, with a NumPy-only CPU path and explicit device reporting."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def gpu_status():
    """Probe kernels and cuBLAS as well as the driver; a visible GPU alone isn't enough."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA GPU was found.")
        with cp.cuda.Device(0):
            properties = cp.cuda.runtime.getDeviceProperties(0)
            name = properties["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            sample = cp.ones((4, 4), dtype=cp.float32)
            result = cp.sqrt(sample @ sample).sum()
            cp.cuda.get_current_stream().synchronize()
            if float(result) != 32.:
                raise RuntimeError("CUDA calculation check failed.")
            return {"available": True, "name": str(name), "reason": "", "memory_bytes": int(properties["totalGlobalMem"])}
    except Exception as exc:
        return {"available": False, "name": "", "reason": f"{type(exc).__name__}: {exc}", "memory_bytes": 0}


@dataclass(frozen=True)
class Backend:
    xp: object
    device: str
    name: str
    reason: str = ""

    def to_cpu(self, value):
        return self.xp.asnumpy(value) if self.device == "cuda" else np.asarray(value)

    def synchronize(self):
        if self.device == "cuda":
            self.xp.cuda.get_current_stream().synchronize()


# Below this many values per step (batch x context x width x layers) the GPU's per-kernel launch cost outweighs
# its arithmetic, so "auto" stays on the CPU. Measured: CPU wins at ~1-1.5M, the GPU only pulls ahead near 4M.
AUTO_GPU_MIN_WORK = 3_000_000


def get_backend(device="auto", work=None):
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    if device == "cpu":
        return Backend(np, "cpu", "NumPy / CPU")
    if device == "auto" and work is not None and work < AUTO_GPU_MIN_WORK:
        return Backend(np, "cpu", "NumPy / CPU", f"auto chose CPU: model is small ({work:,} values per step)")
    status = gpu_status()
    if status["available"]:
        import cupy as cp
        # CUDA current-device state is per thread, including GUI worker threads.
        cp.cuda.Device(0).use()
        return Backend(cp, "cuda", f"CuPy / GPU · {status['name']}")
    reason = "GPU unavailable: " + status["reason"]
    if device == "cuda":
        raise RuntimeError(reason + "\nInstall requirements-gpu.txt and a compatible CUDA toolkit/driver, or select CPU.")
    return Backend(np, "cpu", "NumPy / CPU", reason)

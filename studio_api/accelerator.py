"""Never train on the CPU by accident. A machine with a GPU whose torch reports `cpu` (a CPU-only wheel, a container without the
GPU passed in, a broken driver) is refused at startup unless `--allow-cpu` says the CPU is what you want. `--fake` serves no
training and is never checked."""

import platform
import shutil
import subprocess

STATUS_TIMEOUT = 5.  # seconds for `nvidia-smi -L`


def machine_accelerator(system=platform.system, machine=platform.machine, which=shutil.which, run=subprocess.run):
    """What this machine has, whatever torch can see: "mps" on an Apple Silicon Mac, "cuda" when `nvidia-smi -L` lists a GPU
    (on the host, or in a container that was given one), "" otherwise."""
    if system() == "Darwin" and machine() == "arm64":
        return "mps"
    smi = which("nvidia-smi")
    if smi:
        try:
            done = run([smi, "-L"], capture_output=True, timeout=STATUS_TIMEOUT, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        if done.returncode == 0 and b"GPU" in done.stdout:
            return "cuda"
    return ""


def torch_device(status):
    """The device torch will use: the `kind` of studio_backend.gpu_status(), "cpu" when it found no usable GPU."""
    return status.get("kind") or "cpu" if status.get("available") else "cpu"


FIXES = {
    "mps": "run the API natively on the Mac from the .venv with the default torch wheel (Docker on a Mac has no Metal)",
    "cuda": "install the CUDA build of torch (requirements-gpu.txt) or, in Docker, give the container the GPU (docs/dev-compose.md)",
}


def refusal(has, status):
    reason = status.get("reason") or "no usable GPU"
    return (f"This machine has an accelerator ({has}) but torch reports cpu ({reason}): {FIXES[has]}, "
            "or pass --allow-cpu to run on the CPU anyway.")


def check(allow_cpu=False, probe=machine_accelerator, status=None):
    """The device the API will use, as a line for the log; SystemExit with one sentence when it would be the CPU by accident.
    `status` is studio_backend.gpu_status (called only when needed)."""
    from studio_backend import gpu_status
    status = (status or gpu_status)()
    device = torch_device(status)
    has = "" if allow_cpu or device != "cpu" else probe()
    if has:
        raise SystemExit(refusal(has, status))
    return f"device: {device}" + (f" ({status['name']})" if status.get("name") else "") + (" (--allow-cpu)" if allow_cpu and device == "cpu" else "")

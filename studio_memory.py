"""A job's memory budget and the batch that fits it, from the model's own estimate (studio_estimate) and the device.

A CUDA GPU has memory of its own; Apple's MPS shares the machine's RAM (unified memory), so does the CPU. The queue's plan
says what share of that memory a job may take, and the job lowers its batch size until a training step fits. The estimate
can only be approximate (a framework keeps workspace of its own), so the out-of-memory retry in studio_model.train stays as
the backstop for a batch the estimate accepted."""
from __future__ import annotations

import torch

ADAMW_COPIES = 4  # per trainable parameter: the weight, its gradient and AdamW's two moments


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


def training_bytes(cfg, vocab=None):
    """(fixed, per_sample): what a training step of the validated config holds whatever the batch (weights, gradients and
    AdamW's moments, in the weights' dtype) and per sample of the batch (one context of activations kept for the backward
    pass, plus the logits and their gradient, in the compute dtype)."""
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
    return fixed, per_sample


def batch_cap(config, share, device=None, ram=None, gpu=None):
    """The largest batch of `config` whose training step fits `share` of the memory of the device it will run on: never
    above the declared batch size, never below 1 (the out-of-memory retry handles the rest). An estimate with nothing per
    sample (every module skipped) keeps the declared batch size."""
    from studio_backend import get_backend
    from studio_model import validate_config
    cfg = validate_config(config)
    declared = cfg["batch_size"]
    fixed, per_sample = training_bytes(cfg)
    if per_sample <= 0:
        return declared
    if device is None:
        d = cfg["width"], cfg["context"], cfg["layers"]
        device = get_backend(cfg["device"], work=declared * d[1] * d[0] * max(1, d[2])).device  # the choice the model's build makes
    room = budget(device, share, ram, gpu) - fixed
    return max(1, min(declared, room // per_sample))

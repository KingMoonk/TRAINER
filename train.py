"""Command-line entry point for the same from-scratch engine used by the studio."""
from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from studio_model import DEFAULT_CONFIG, load_checkpoint, train
from studio_backend import gpu_status
from studio_core import ROOT, normalize, parse_file, uid


def main():
    parser = argparse.ArgumentParser(description="Train a Groundwork transformer from random weights, or resume your checkpoint.")
    parser.add_argument("--data", type=Path, help="JSONL, JSON, CSV, TSV, or TXT training data")
    parser.add_argument("--config", type=Path, help="Optional architecture/training JSON configuration")
    parser.add_argument("--output", type=Path, help="Run folder (default: runs/run_<id>)")
    parser.add_argument("--resume", type=Path, help="Groundwork last.npz or best.npz checkpoint")
    parser.add_argument("--steps", type=int, help="Total optimizer steps, including any resumed steps")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Compute device; auto prefers a supported CUDA GPU and falls back to CPU")
    parser.add_argument("--list-devices", action="store_true", help="Show CPU and CUDA availability, then exit")
    args = parser.parse_args()
    if args.list_devices:
        print("cpu: NumPy CPU (available)")
        gpu = gpu_status()
        if gpu["available"]:
            print(f"cuda: {gpu['name']} (available)")
        else:
            print(f"cuda: unavailable — {gpu.get('reason', 'No supported GPU detected')}")
        print("auto: prefers CUDA when available; otherwise CPU")
        return
    cfg = dict(DEFAULT_CONFIG)
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        cfg.update(checkpoint["config"])
        cfg["steps"] = checkpoint["step"] + 300
    if args.config:
        cfg.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.device is not None:
        cfg["device"] = args.device
    if args.data:
        examples = [example for row in parse_file(args.data) if (example := normalize(row))]
    elif args.resume:
        examples = [json.loads(line) for line in (args.resume.parent / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        parser.error("--data is required for a new model")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def emit(event):
        if event["type"] == "metric":
            val = f" val={event['val_loss']:.4f}" if event.get("val_loss") is not None else ""
            print(f"step={event['step']} loss={event['loss']:.4f}{val}", flush=True)
        elif event["type"] == "info":
            print(event["text"], flush=True)
        elif event["type"] == "done":
            print(f"{event['run']['status']}: {event['directory']}", flush=True)

    train(cfg, examples, args.output or ROOT / "runs" / uid("run"), emit, stop, resume=args.resume)


if __name__ == "__main__":
    main()

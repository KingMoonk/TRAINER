"""Start from a pretrained open model: download OpenAI's GPT-2 (124M) and turn it into an ordinary studio checkpoint.

Window-free and dependency-free (urllib, hashlib, numpy). The result is a normal `.npz` checkpoint with the GPT-2 design, so it is
a version like any other: continue text with it, evaluate it, fine-tune it (a post-training round, a curriculum stage, or LoRA
through `train.run.lora`) and export it again. It is a base model: it continues text but does not answer as a chat until it is
fine-tuned, because it has never seen the USER and ASSISTANT tokens.

This reverses studio_export's mapping (Hugging Face `GPT2LMHeadModel` names to the engine's, see `gpt2_names`). Both sides keep a
linear layer as (in, out) applied as `x @ W + b` (GPT-2's Conv1D layout), the fused qkv is the same [q | k | v], and the engine's GELU
is the tanh approximation (`gelu_new`), so no weight changes value on the way in except the token table:

The engine reserves token ids 0-6 for its special tokens (PAD, BOS, EOS, USER, ASSISTANT, SYSTEM, UNK) and puts every tokenizer's own
vocabulary at +7, so GPT-2's 50,257 pieces become ids 7..50263 and the table has 50,264 rows. The 7 new rows start as follows:
BOS and EOS copy GPT-2's own `<|endoftext|>` row (GPT-2 was trained with that one token both ending a document and starting the next,
which is what the engine's BOS and EOS do), and PAD, USER, ASSISTANT, SYSTEM and UNK start as the mean of all 50,257 real rows, the usual
neutral initialization for a token the model never saw. The head is tied to the table, so these rows also set those tokens' output
scores. Chat and sampling never choose PAD/BOS/USER/SYSTEM/ASSISTANT/UNK, nor GPT-2's own `<|endoftext|>` piece (it has the same row as
EOS, the one token that ends a reply), and held-back scoring leaves the 7 rows out, so only fine-tuning ever moves them.

A file written by studio_export (which already has the 7 rows in front) is read as it is.
"""
from __future__ import annotations

import errno
import hashlib
import http.client
import json
import re
import shutil
import struct
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

import studio_spec
from studio_core import atomic_json, replace_file
from studio_export import gpt2_names, gpt2_problems
from studio_slots import BOS, EOS, OFFSET

# Pinned to one revision of https://huggingface.co/openai-community/gpt2 so the hash below stays true. The size and sha256 are from
# the Hugging Face API (https://huggingface.co/api/models/openai-community/gpt2?blobs=true) for that revision's model.safetensors.
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_URL = f"https://huggingface.co/openai-community/gpt2/resolve/{GPT2_REVISION}/model.safetensors"
GPT2_SIZE = 548_105_171
GPT2_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
HEADS = {768: 12, 1024: 16, 1280: 20, 1600: 25}   # GPT-2 small, medium, large, xl: the head count is not stored in the weights
CHUNK = 1 << 20
DTYPES = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}
IGNORED = (".attn.bias", ".attn.masked_bias")      # older files carry the causal mask as a buffer


# --- download ------------------------------------------------------------------------------------------------

def _sha256(path, stop=None):
    """The file's sha256, or None when `stop` was set while it was being read (hashing 548 MB takes seconds)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            if stop is not None and stop.is_set():
                return None
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path, size, sha256, stop):
    """True when the file is `size` bytes with the right checksum, False when not, None when stopped while checking."""
    if Path(path).stat().st_size != size:
        return False
    digest = _sha256(path, stop)
    return None if digest is None else digest == sha256


def _range_start(response):
    """Where a 206 response says its bytes begin (from Content-Range: bytes START-END/TOTAL), or None when it does not say."""
    headers = getattr(response, "headers", None)
    match = re.match(r"bytes (\d+)-", (headers.get("Content-Range") if headers is not None else None) or "")
    return int(match.group(1)) if match else None


def fetch_gpt2(dest, progress=None, stop=None, url=GPT2_URL, size=GPT2_SIZE, sha256=GPT2_SHA256):
    """Download GPT-2's model.safetensors to `dest` (about 548 MB) and return the path, or None when `stop` was set first.

    The bytes go to `<dest>.part` and are renamed to `dest` only once the size and the sha256 match, so `dest` is always whole.
    A `.part` left by an interrupted download is continued with an HTTP Range request; when the server cannot continue it (416, or
    bytes that start somewhere else) the download starts again from zero. progress(done_bytes, total_bytes).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        good = _verified(dest, size, sha256, stop)
        if good is None:
            return None
        if good:
            return dest
        dest.unlink()
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have >= size:   # a leftover that is already whole is finished as it is; a longer or wrong one is thrown away
        good = have == size and _verified(part, size, sha256, stop)
        if good is None:
            return None
        if good:
            replace_file(part, dest)
            return dest
        part.unlink()
        have = 0
    try:
        while True:
            request = urllib.request.Request(url, headers={"User-Agent": "groundwork-studio"})
            if have:
                request.add_header("Range", f"bytes={have}-")
            try:
                response = urllib.request.urlopen(request, timeout=30)
            except urllib.error.HTTPError as error:
                if error.code == 416 and have:   # the server has nothing at that offset: the leftover is stale
                    part.unlink(missing_ok=True)
                    have = 0
                    continue
                raise
            with response:
                status = getattr(response, "status", 200)
                if have and status == 206 and _range_start(response) != have:   # bytes from somewhere else would corrupt the file
                    part.unlink(missing_ok=True)
                    have = 0
                    continue
                resumed = have > 0 and status == 206
                with part.open("ab" if resumed else "wb") as stream:
                    done = have if resumed else 0
                    while True:
                        if stop is not None and stop.is_set():
                            return None
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        stream.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, size)
            break
    except OSError as error:
        if error.errno == errno.ENOSPC:
            raise ValueError(f"There is not enough free disk space for GPT-2 (it needs about {size / 1e9:.1f} GB). Free some space and "
                             "try again; what arrived is kept and the download continues from there.") from error
        raise ValueError("Could not download GPT-2. Check the internet connection and try again; what arrived is kept and the download "
                         f"continues from there. ({error})") from error
    except http.client.HTTPException as error:   # a connection cut mid-reply (IncompleteRead) is not an OSError
        raise ValueError("Could not download GPT-2. The connection was cut off; try again and the download continues from where it "
                         f"stopped. ({error})") from error
    got = part.stat().st_size
    if got != size:
        raise ValueError(f"The GPT-2 download stopped early ({got:,} of {size:,} bytes). Try again; it continues from where it stopped.")
    digest = _sha256(part, stop)
    if digest is None:
        return None
    if digest != sha256:
        part.unlink()
        raise ValueError("The downloaded GPT-2 file did not match its published checksum, so it was thrown away. Try again.")
    replace_file(part, dest)
    return dest


# --- reading and converting ----------------------------------------------------------------------------------

def read_tensors(path, stop=None):
    """{name: float32 array} of a .safetensors file, read one tensor at a time (the file is never held whole in memory).
    Returns None when `stop` was set part way."""
    path = Path(path)
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length).decode("utf-8"))
        base = 8 + length
        out = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if stop is not None and stop.is_set():
                return None
            dtype = DTYPES.get(entry["dtype"])
            if dtype is None:
                raise ValueError(f"Tensor {name} is stored as {entry['dtype']}, which this import cannot read (F32, F16 and F64 only).")
            begin, end = entry["data_offsets"]
            stream.seek(base + begin)
            raw = stream.read(end - begin)
            out[name] = np.frombuffer(raw, dtype=dtype).reshape(entry["shape"]).astype(np.float32)
    return out


FINE_TUNE_LR = 3e-5        # peak learning rate for fine-tuning every weight of a pretrained model (the scratch default, 1e-3, would wreck it)
FINE_TUNE_LORA_LR = 1e-4   # and for LoRA, whose small new adapters can take a larger step because the pretrained weights are frozen


def gpt2_spec(d_model=768, n_layer=12, n_ctx=1024, heads=12, d_ff=3072, name=None, lora=None):
    """The ModelSpec of OpenAI's GPT-2 (default: the 124M model) with its own 50,257-piece word list.

    It is meant to be fine-tuned, so its learning rate is a fine-tuning one: FINE_TUNE_LR, or FINE_TUNE_LORA_LR when `lora` (a
    settings dict, {} for the defaults) says the spec trains LoRA adapters instead.
    """
    spec = studio_spec.default_spec()
    spec["name"] = name or "GPT-2 (124M)"
    spec["train"]["optimizer"]["config"]["lr"] = FINE_TUNE_LORA_LR if lora is not None else FINE_TUNE_LR
    if lora is not None:
        spec["train"]["run"] = {"lora": lora}
    spec["dims"].update(d_model=d_model, n_layer=n_layer, n_ctx=n_ctx, dropout=.1)
    spec["tokenizer"] = {"id": "bpe_gpt2"}
    spec["block"]["mixer"] = {"id": "attention", "config": {"heads": heads}}
    spec["block"]["channel"] = {"id": "mlp_gelu", "config": {"d_ff": d_ff}}
    spec = studio_spec.validate_spec(spec)
    if problems := gpt2_problems(spec):
        raise ValueError("The GPT-2 design is not GPT-2's architecture: " + "; ".join(problems))
    return spec


def engine_weights(tensors, tokenizer):
    """(weights by engine parameter name, n_layer, ignored names) from Hugging Face GPT2LMHeadModel tensors. Token ids shift by the 7 special tokens."""
    tensors = {key.removeprefix("transformer."): value for key, value in tensors.items()}
    head = tensors.pop("lm_head.weight", None)
    ignored = [key for key in tensors if key.endswith(IGNORED)]
    for key in ignored:
        del tensors[key]
    if "wte.weight" not in tensors:
        raise ValueError("This is not a GPT-2 file: it has no wte.weight token table.")
    if head is not None and not np.array_equal(head, tensors["wte.weight"]):
        raise ValueError("This GPT-2 file has an output head that is not tied to its token table, which this app's GPT-2 design cannot hold.")
    layers = {int(key.split(".")[1]) for key in tensors if key.startswith("h.")}
    n_layer = len(layers)
    if layers != set(range(n_layer)) or not n_layer:
        raise ValueError("This is not a GPT-2 file: its blocks are not numbered from 0.")
    inverse = {theirs.removeprefix("transformer."): ours for ours, theirs in gpt2_names(n_layer).items()}
    unknown, missing = sorted(set(tensors) - set(inverse)), sorted(set(inverse) - set(tensors))
    if unknown or missing:
        raise ValueError("This is not a GPT-2 file: " + ("it has unexpected tensors " + ", ".join(unknown[:3]) if unknown
                                                         else "it lacks " + ", ".join(missing[:3])) + ".")
    weights = {inverse[key]: value for key, value in tensors.items()}
    weights["token"] = token_table(weights["token"], tokenizer)
    return weights, n_layer, ignored


def token_table(table, tokenizer):
    """GPT-2's (50257, d) table as the engine's (50264, d): the 7 special rows in front (see the module docstring)."""
    wanted = tokenizer.vocab_size
    if table.shape[0] == wanted:          # already carries the special rows (written by studio_export)
        return table
    if table.shape[0] != wanted - OFFSET:
        raise ValueError(f"The token table has {table.shape[0]:,} rows but the word list has {wanted - OFFSET:,} pieces.")
    mean = table.mean(axis=0)
    end = tokenizer.encoder.get("<|endoftext|>")
    out = np.empty((wanted, table.shape[1]), dtype=np.float32)
    out[:OFFSET] = mean
    out[OFFSET:] = table
    out[[BOS, EOS]] = table[end] if end is not None else mean
    return out


def import_gpt2(safetensors_path, out_checkpoint, tokenizer=None, heads=None, name=None, source=None, stop=None):
    """Read a Hugging Face GPT-2 `model.safetensors` and write a studio checkpoint (.npz) with the matching design. Returns a report dict.

    tokenizer: GPT-2's word list (default: the one the app downloads on first use). heads: the attention head count, which the file
    does not record (default: from the width, 12 for GPT-2 small). Returns None, writing nothing, when `stop` is set first.
    """
    import studio_model
    from studio_compile import CompiledModel
    if tokenizer is None:
        from studio_modules.tokenizer.bpe_gpt2 import GPT2Config, load_assets
        tokenizer = load_assets(GPT2Config())
    tensors = read_tensors(safetensors_path, stop)
    if tensors is None:
        return None
    weights, n_layer, _ = engine_weights(tensors, tokenizer)
    del tensors
    d_model, n_ctx, d_ff = weights["token"].shape[1], weights["position"].shape[0], weights["block0.f1.w"].shape[1]
    heads = heads or HEADS.get(d_model)
    if not heads:
        raise ValueError(f"The attention head count for a model {d_model} wide is not known; pass heads.")
    count = sum(value.size for value in weights.values())
    spec = gpt2_spec(d_model, n_layer, n_ctx, heads, d_ff, name or f"GPT-2 ({count / 1e6:.0f}M)")
    model = CompiledModel(spec, tokenizer.vocab_size, device="cpu")
    model.load_weights(weights)   # every name and shape must match the design, and every value must be finite
    del model
    if stop is not None and stop.is_set():
        return None
    # The spec alone: values passed beside it would win over the spec's own learning rate, warmup and LoRA settings.
    cfg = studio_model.validate_config({"architecture": spec, "name": spec["name"]})
    metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "step": 0, "best": None, "optimizer_steps": 0,
                "history": [], "config": cfg, "tokenizer": tokenizer.serialize(), "device": "cpu",
                "imported_from": source or str(Path(safetensors_path).name)}
    arrays = {"metadata": np.array(json.dumps(metadata))}
    arrays.update({"weight::" + key: value for key, value in weights.items()})
    out = Path(out_checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    replace_file(temporary, out)
    return {"path": str(out), "parameters": count, "spec": spec, "vocabulary": tokenizer.vocab_size}


# --- the job -------------------------------------------------------------------------------------------------

MODELS = {"gpt2": {"title": "GPT-2 (124M)", "source": "openai-community/gpt2", "size": GPT2_SIZE}}


def run_import(model, cache_dir, out_dir, emit=None, stop=None):
    """Download (once; kept in `cache_dir`) and import `model` into `out_dir`/last.npz, with the run.json a finished run has.
    Returns the run record, or None when stopped. emit gets queue progress events (the download in MB, then the conversion)."""
    if model not in MODELS:
        raise ValueError(f"Unknown pretrained model {model!r}.")
    emit = emit or (lambda event: None)
    info = MODELS[model]
    total, last = max(1, info["size"] // 2 ** 20), [0.]

    def progress(done, size):
        if time.monotonic() - last[0] > .25 or done >= size:
            last[0] = time.monotonic()
            emit({"type": "progress", "phase": f"Downloading {info['title']}", "step": done // 2 ** 20, "total": total})

    emit({"type": "info", "text": f"Downloading {info['title']} from Hugging Face ({info['size'] / 1e6:,.0f} MB, checked against its published checksum)."})
    path = fetch_gpt2(Path(cache_dir) / model / "model.safetensors", progress=progress, stop=stop)
    if path is None:
        return None
    emit({"type": "progress", "phase": "Converting to a studio checkpoint", "step": total, "total": total})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = import_gpt2(path, out / "last.npz", source=info["source"], stop=stop)
    if report is None:   # stopped during the conversion: nothing was written
        shutil.rmtree(out, ignore_errors=True)
        return None
    record = {"name": info["title"], "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"), "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "status": "completed", "step": 0, "steps": 0, "parameters": report["parameters"], "vocabulary": report["vocabulary"],
              "imported_from": info["source"], "revision": GPT2_REVISION, "backend": "torch", "device": "cpu"}
    atomic_json(out / "spec.json", report["spec"])
    atomic_json(out / "run.json", record)
    emit({"type": "done", "run": record, "directory": str(out)})
    return record


def new_output_dir(run_dir, model):
    return Path(run_dir) / f"import_{model}_{uuid.uuid4().hex[:6]}"

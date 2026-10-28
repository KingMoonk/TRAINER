"""Export a trained checkpoint: Hugging Face (safetensors, config, tokenizer files, model card) and GGUF.

Window-free and dependency-free: the safetensors and GGUF files are written by hand (both are a small header
plus raw tensor bytes), so nothing is added to requirements.

GPT-2 designs (learned positions, LayerNorm before each part, softmax attention, a GELU MLP, a tied head;
with or without projection biases, absent ones being written as zeros, which is exact) are written with Hugging Face's `GPT2LMHeadModel` key names and a GPT-2 config.json, so `transformers` loads
them as they are. Both sides keep a linear layer as an (in, out) matrix applied as `x @ W + b`, which is
GPT-2's Conv1D layout, so no transpose is needed for Hugging Face; the fused qkv is the same [q | k | v]
split in both. GGUF (llama.cpp) wants the torch `Linear` layout (out, in), so those weights are transposed
there, as llama.cpp's own converter does. The engine's GELU is the tanh approximation, which is Hugging Face's
`gelu_new` and llama.cpp's GELU. Any other design is written as raw tensors with its spec embedded, and the
README says plainly that Hugging Face cannot load it directly.

Token ids 0-6 are the engine's shared special tokens, so every exported vocabulary starts with those seven
and the vocabulary proper follows at +7. A model exported from here shares weights with no pretrained GPT-2.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

import studio_registry as registry
from studio_model import effective_spec, load_checkpoint, validate_config
from studio_modules.tokenizer.bpe_gpt2 import bytes_to_unicode
from studio_modules.tokenizer.byte import Tokenizer
from studio_slots import OFFSET
from studio_spec import block_at

SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|system|>", "<|unk|>")  # ids 0..6, studio_slots order
PAD, BOS, EOS, USER, ASSISTANT, SYSTEM, UNK = range(OFFSET)
HF_TOKENIZERS = ("byte", "bpe_gpt2", "bpe_fast")   # byte-level vocabularies Hugging Face's GPT2Tokenizer loads exactly
GGUF_TOKENIZERS = HF_TOKENIZERS        # the same vocabularies, as llama.cpp's "gpt2" (byte-level BPE) tokenizer
KINDS = ("hf", "gguf")


class ExportError(ValueError):
    """Why a checkpoint cannot be exported that way, in words a person can act on."""


# --- safetensors ---------------------------------------------------------------------------------------------

_SAFETENSORS_DTYPES = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("float64"): "F64",
                       np.dtype("int64"): "I64", np.dtype("int32"): "I32", np.dtype("uint8"): "U8"}


def write_safetensors(path, tensors, metadata=None):
    """{name: array} to a .safetensors file: u64 header size, a JSON header (dtype, shape, byte range), the raw bytes."""
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name])
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        dtype = _SAFETENSORS_DTYPES.get(np.dtype(array.dtype).newbyteorder("="))
        if dtype is None:
            raise ExportError(f"Tensor {name} has a type safetensors cannot store ({array.dtype}).")
        data = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)   # the data starts on an 8-byte boundary
    with Path(path).open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        for data in chunks:
            stream.write(data)


def read_safetensors(path):
    """(tensors, metadata) of a .safetensors file."""
    blob = Path(path).read_bytes()
    (size,) = struct.unpack_from("<Q", blob, 0)
    header = json.loads(blob[8:8 + size].decode("utf-8"))
    metadata = header.pop("__metadata__", {})
    by_name = {value: key for key, value in _SAFETENSORS_DTYPES.items()}
    tensors = {}
    for name, entry in header.items():
        begin, end = entry["data_offsets"]
        tensors[name] = np.frombuffer(blob[8 + size + begin:8 + size + end], dtype=by_name[entry["dtype"]]).reshape(entry["shape"]).copy()
    return tensors, metadata


# --- reading a checkpoint ------------------------------------------------------------------------------------

def read_checkpoint(path):
    """(spec, tokenizer, weights as float32 arrays by parameter name, step) of a Groundwork .npz checkpoint."""
    data = load_checkpoint(path, weights_only=True)
    spec = effective_spec(validate_config(data["config"]))
    tokenizer = Tokenizer.restore(data["tokenizer"])
    weights = {key: np.asarray(value, dtype=np.float32) for key, value in data["model"].items()}
    return spec, tokenizer, weights, data.get("step")


# --- GPT-2 ---------------------------------------------------------------------------------------------------

def gpt2_problems(spec):
    """Why this design is not GPT-2's architecture: [] when it is, otherwise a list of plain-words differences."""
    problems = []
    for slot, wanted, label in (("embedding", "learned", "a learned token table"), ("position", "learned", "learned positions"),
                                ("head", "tied", "an output head tied to the embedding")):
        if spec[slot]["id"] != wanted:
            problems.append(f"it does not use {label} (it uses {registry.get(slot, spec[slot]['id']).name})")
    for index in range(spec["dims"]["n_layer"]):
        block = block_at(spec, index)
        where = f"block {index + 1}" if spec["blocks"] else "its blocks"
        norm, mixer, channel = block["norm"], block["mixer"], block["channel"]
        if norm["id"] != "layernorm" or norm["config"].get("placement") != "pre":
            problems.append(f"{where} do not use LayerNorm before each part")
        if mixer["id"] != "attention":
            problems.append(f"{where} do not use softmax attention")
        cfg = channel["config"]
        if channel["id"] != "mlp_gelu" or cfg.get("layers", 1) != 1 or cfg.get("activation", "gelu") != "gelu":
            problems.append(f"{where} do not use a one-hidden-layer GELU MLP")
        if problems:
            break
    init = spec["train"]["init"]
    if init["id"] == "mup" and spec["dims"]["d_model"] != init["config"].get("base_width"):
        problems.append("μP initialization rescales attention scores and output logits at this width")
    return problems


def gpt2_names(n_layer):
    """{engine parameter name: Hugging Face GPT2LMHeadModel key} for a GPT-2 design with `n_layer` blocks."""
    names = {"token": "transformer.wte.weight", "position": "transformer.wpe.weight",
             "final.g": "transformer.ln_f.weight", "final.b": "transformer.ln_f.bias"}
    for i in range(n_layer):
        base = f"transformer.h.{i}."
        for ours, theirs in (("n1.g", "ln_1.weight"), ("n1.b", "ln_1.bias"), ("qkv.w", "attn.c_attn.weight"), ("qkv.b", "attn.c_attn.bias"),
                             ("proj.w", "attn.c_proj.weight"), ("proj.b", "attn.c_proj.bias"), ("n2.g", "ln_2.weight"), ("n2.b", "ln_2.bias"),
                             ("f1.w", "mlp.c_fc.weight"), ("f1.b", "mlp.c_fc.bias"), ("f2.w", "mlp.c_proj.weight"), ("f2.b", "mlp.c_proj.bias")):
            names[f"block{i}.{ours}"] = base + theirs
    return names


BIAS_OF = {"qkv.b": "qkv.w", "proj.b": "proj.w", "f1.b": "f1.w", "f2.b": "f2.w"}   # a GPT-2 bias and the weight whose output width it has


def to_gpt2(weights, spec):
    """The checkpoint's weights under Hugging Face GPT2LMHeadModel names. Every parameter is used exactly once.

    GPT-2 has a bias on the attention and MLP projections. A design without them is written with zero biases,
    which computes exactly the same function.
    """
    names = gpt2_names(spec["dims"]["n_layer"])
    absent = set(names) - set(weights)
    zeros = {key for key in absent if key.partition(".")[2] in BIAS_OF}
    if set(weights) - set(names) or absent - zeros:
        raise ExportError("The checkpoint's parameters do not match a GPT-2 design: "
                          + ", ".join(sorted(set(weights) ^ set(names))[:4]) + ".")
    out = {names[key]: np.ascontiguousarray(value) for key, value in weights.items()}
    for key in zeros:
        block, _, bias = key.partition(".")
        out[names[key]] = np.zeros(weights[f"{block}.{BIAS_OF[bias]}"].shape[1], dtype=np.float32)
    return out


def gpt2_config(spec, vocab_size):
    d, block = spec["dims"], block_at(spec, 0)
    return {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2", "vocab_size": vocab_size,
            "n_positions": d["n_ctx"], "n_ctx": d["n_ctx"], "n_embd": d["d_model"], "n_layer": d["n_layer"],
            "n_head": block["mixer"]["config"]["heads"], "n_inner": block["channel"]["config"].get("d_ff") or None,
            "activation_function": "gelu_new", "layer_norm_epsilon": block["norm"]["config"].get("eps", 1e-5),
            "resid_pdrop": d["dropout"], "embd_pdrop": d["dropout"], "attn_pdrop": d["dropout"], "initializer_range": .02,
            "scale_attn_weights": True, "tie_word_embeddings": True, "use_cache": True,
            "pad_token_id": PAD, "bos_token_id": BOS, "eos_token_id": EOS, "torch_dtype": "float32"}


# --- tokenizers ----------------------------------------------------------------------------------------------

def byte_level_vocab(tokenizer):
    """(tokens in id order, merges as "a b" strings) of a byte-level vocabulary, ids 0-6 the special tokens.

    Tokens use GPT-2's printable spelling of bytes. A byte tokenizer is a byte-level BPE with no merges; a
    tokenizer that is not byte-level (or whose merge rules differ from GPT-2's) raises ExportError.
    """
    if tokenizer.kind == "byte":
        table = bytes_to_unicode()
        return list(SPECIAL_TOKENS) + [table[byte] for byte in range(256)], []
    if tokenizer.kind == "bpe_fast":
        return (list(SPECIAL_TOKENS) + tokenizer.symbols,
                [f"{tokenizer.symbols[left]} {tokenizer.symbols[right]}" for left, right in tokenizer.merges])
    if tokenizer.kind == "bpe_gpt2":
        tokens = list(SPECIAL_TOKENS) + [f"<|unused{i}|>" for i in range(tokenizer.vocab_size - OFFSET)]
        for symbol, index in tokenizer.encoder.items():
            tokens[index + OFFSET] = symbol
        if len(set(tokens)) != len(tokens):
            raise ExportError("This GPT-2 vocabulary has a token that clashes with a special token name.")
        return tokens, [f"{left} {right}" for left, right in tokenizer.merges]
    raise ExportError(NO_TOKENIZER[tokenizer.kind] if tokenizer.kind in NO_TOKENIZER else f"The {tokenizer.kind} tokenizer cannot be exported.")


NO_TOKENIZER = {
    "character": "The character tokenizer has no counterpart in Hugging Face or llama.cpp (one token per character you trained on).",
}


def vocabulary_json(tokenizer):
    """A tokenizer Hugging Face cannot load, as plain JSON: the vocabulary by id and what is needed to apply it."""
    if tokenizer.kind == "character":
        pieces, how = list(tokenizer.characters), "One token per character. A character not listed becomes <|unk|>."
    else:
        raise ExportError(f"The {tokenizer.kind} tokenizer needs no vocabulary file.")
    return {"format": "groundwork-tokenizer/1", "kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size,
            "special_tokens": {name: i for i, name in enumerate(SPECIAL_TOKENS)},
            "vocabulary": list(SPECIAL_TOKENS) + pieces, "how_it_works": how}


def hf_tokenizer_files(tokenizer, n_ctx):
    """{filename: text} for the tokenizer, and whether Hugging Face's GPT2Tokenizer loads it (otherwise a vocabulary JSON)."""
    if tokenizer.kind not in HF_TOKENIZERS:
        return {"groundwork_tokenizer.json": json.dumps(vocabulary_json(tokenizer), ensure_ascii=False, indent=1)}, False
    tokens, merges = byte_level_vocab(tokenizer)
    config = {"tokenizer_class": "GPT2Tokenizer", "model_max_length": n_ctx, "add_prefix_space": False, "errors": "replace",
              "pad_token": SPECIAL_TOKENS[PAD], "bos_token": SPECIAL_TOKENS[BOS], "eos_token": SPECIAL_TOKENS[EOS],
              "unk_token": SPECIAL_TOKENS[UNK],
              "additional_special_tokens": [SPECIAL_TOKENS[USER], SPECIAL_TOKENS[ASSISTANT], SPECIAL_TOKENS[SYSTEM]]}
    special = {key: config[key] for key in ("pad_token", "bos_token", "eos_token", "unk_token", "additional_special_tokens")}
    return {"vocab.json": json.dumps({token: i for i, token in enumerate(tokens)}, ensure_ascii=False),
            "merges.txt": "\n".join(["#version: 0.2", *merges]) + "\n",
            "tokenizer_config.json": json.dumps(config, indent=1), "special_tokens_map.json": json.dumps(special, indent=1)}, True


# --- Hugging Face --------------------------------------------------------------------------------------------

def _summary(spec, tokenizer, count):
    d, block = spec["dims"], block_at(spec, 0)
    return (f"{count:,} parameters · {d['n_layer']} blocks · width {d['d_model']} · context {d['n_ctx']} · "
            f"{registry.get('mixer', block['mixer']['id']).name} + {registry.get('channel', block['channel']['id']).name} · "
            f"{registry.get('tokenizer', spec['tokenizer']['id']).name} tokenizer ({tokenizer.vocab_size:,} tokens)")


def model_card(spec, tokenizer, count, step, problems, tokenizer_loads, zero_biases=False):
    lines = ["---", "library_name: transformers" if not problems else "library_name: groundwork", "tags:", "- groundwork", "---", "",
             f"# {spec['name']}", "", "A small language model trained in Groundwork (TRAINER).", "", _summary(spec, tokenizer, count)
             + (f" · trained for {step:,} steps." if step else "."), ""]
    if problems:
        lines += ["## Loading", "",
                  "**Hugging Face cannot load this design directly.** It is not GPT-2's architecture: " + "; ".join(problems) + ". "
                  "`model.safetensors` holds the raw tensors under the names the Groundwork engine uses, and the full model "
                  "spec is embedded in the file's metadata (`groundwork_spec`) and in `config.json`. Rebuild the model "
                  "from that spec in Groundwork to run it.", ""]
    else:
        lines += ["## Loading", "", "This is GPT-2's architecture (learned positions, LayerNorm, softmax attention, a GELU MLP, "
                  "a tied output head), written under `GPT2LMHeadModel` key names."
                  + (" The design has no bias on its attention or MLP projections, so those biases are written as zeros, "
                     "which computes exactly the same function." if zero_biases else ""), "", "```python",
                  "from transformers import GPT2LMHeadModel", "", 'model = GPT2LMHeadModel.from_pretrained("path/to/this/folder")', "```", ""]
    lines += ["## Tokenizer", ""]
    if tokenizer_loads:
        lines += ["`vocab.json` and `merges.txt` are a byte-level BPE vocabulary `GPT2Tokenizer` loads "
                  f"(`GPT2Tokenizer.from_pretrained(...)`). Ids 0-{OFFSET - 1} are Groundwork's special tokens "
                  f"({', '.join(SPECIAL_TOKENS)}), so token ids are shifted by {OFFSET} from OpenAI's GPT-2: "
                  "this model shares no weights or ids with a pretrained GPT-2.", ""]
    else:
        lines += [NO_TOKENIZER.get(tokenizer.kind, "") + " The vocabulary is exported as `groundwork_tokenizer.json` "
                  "with a description of how text is split, but Hugging Face's tokenizer classes cannot load it.", ""]
    lines += ["## Chat format", "", "Groundwork frames a conversation as `<|bos|>`, then for each message its role token "
              "(`<|user|>`, `<|assistant|>`, `<|system|>`), the text and `<|eos|>`, ending on `<|assistant|>`.", ""]
    return "\n".join(lines)


def export_hf(checkpoint, out_dir):
    """Write model.safetensors, config.json, the tokenizer files and README.md into `out_dir`. Returns a report:
    {"path", "files", "architecture": "gpt2" | "raw", "huggingface_loadable", "tokenizer_loadable", "note"}."""
    spec, tokenizer, weights, step = read_checkpoint(checkpoint)
    out = Path(out_dir)
    problems, zero_biases = gpt2_problems(spec), False   # zero_biases: GPT-2 biases this design lacks and the file adds as zeros
    count = sum(value.size for value in weights.values())
    files, tokenizer_loads = hf_tokenizer_files(tokenizer, spec["dims"]["n_ctx"])
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"format": "pt", "groundwork_step": step or 0}
    if problems:
        config = {"model_type": "groundwork", "architectures": [], "vocab_size": tokenizer.vocab_size, "torch_dtype": "float32",
                  "groundwork_spec": spec, "groundwork_note": "Not loadable by transformers: " + "; ".join(problems) + "."}
        tensors, metadata["groundwork_spec"] = weights, json.dumps(spec)
    else:
        if weights["token"].shape[0] != tokenizer.vocab_size:
            raise ExportError("The checkpoint's vocabulary does not match its tokenizer.")
        config, tensors = gpt2_config(spec, tokenizer.vocab_size), to_gpt2(weights, spec)
        zero_biases = len(tensors) > len(weights)
    write_safetensors(out / "model.safetensors", tensors, metadata)
    (out / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    (out / "README.md").write_text(model_card(spec, tokenizer, count, step, problems, tokenizer_loads, not problems and zero_biases), encoding="utf-8")
    note = ("Loads in transformers as GPT2LMHeadModel." if not problems else
            "Hugging Face cannot load this design directly: " + "; ".join(problems) + ". The raw tensors and the model spec are included.")
    if not tokenizer_loads:
        note += " " + NO_TOKENIZER[tokenizer.kind] + " Its vocabulary is in groundwork_tokenizer.json."
    return {"path": str(out), "files": sorted(["model.safetensors", "config.json", "README.md", *files]),
            "architecture": "raw" if problems else "gpt2", "huggingface_loadable": not problems,
            "tokenizer_loadable": tokenizer_loads, "note": note}


# --- GGUF ----------------------------------------------------------------------------------------------------

GGUF_MAGIC, GGUF_VERSION, GGUF_ALIGNMENT = b"GGUF", 3, 32
GGUF_VALUE = {"u8": (0, "<B"), "i8": (1, "<b"), "u16": (2, "<H"), "i16": (3, "<h"), "u32": (4, "<I"), "i32": (5, "<i"),
              "f32": (6, "<f"), "bool": (7, "<?"), "u64": (10, "<Q"), "i64": (11, "<q"), "f64": (12, "<d")}
GGUF_STRING, GGUF_ARRAY = 8, 9
GGML_F32, GGML_F16 = 0, 1
TOKEN_NORMAL, TOKEN_UNKNOWN, TOKEN_CONTROL, TOKEN_UNUSED = 1, 2, 3, 5   # llama.cpp token types


def _gguf_string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_value(kind, value):
    """Bytes of one value (without its type tag). kind: a GGUF_VALUE key, "str", or ("arr", element kind)."""
    if kind == "str":
        return _gguf_string(value)
    if isinstance(kind, tuple):
        element = kind[1]
        return struct.pack("<IQ", GGUF_STRING if element == "str" else GGUF_VALUE[element][0], len(value)) + b"".join(_gguf_value(element, item) for item in value)
    return struct.pack(GGUF_VALUE[kind][1], value)


def _gguf_tag(kind):
    return GGUF_STRING if kind == "str" else GGUF_ARRAY if isinstance(kind, tuple) else GGUF_VALUE[kind][0]


def write_gguf(path, metadata, tensors, alignment=GGUF_ALIGNMENT):
    """A GGUF v3 file. metadata: [(key, kind, value)]. tensors: [(name, array)], float32 or float16, written as given:
    GGUF lists a tensor's dimensions fastest-first, so a numpy array of shape (a, b) is the ggml tensor [b, a]."""
    def pad(n):
        return -n % alignment

    infos, offset = [], 0
    for name, array in tensors:
        ggml = {np.dtype("float32"): GGML_F32, np.dtype("float16"): GGML_F16}.get(array.dtype)
        if ggml is None:
            raise ExportError(f"Tensor {name} is {array.dtype}; GGUF export writes float32 or float16.")
        infos.append(_gguf_string(name) + struct.pack("<I", array.ndim) + struct.pack(f"<{array.ndim}Q", *reversed(array.shape))
                     + struct.pack("<IQ", ggml, offset))
        offset += array.nbytes + pad(array.nbytes)
    head = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata))
    for key, kind, value in metadata:
        head += _gguf_string(key) + struct.pack("<I", _gguf_tag(kind)) + _gguf_value(kind, value)
    head += b"".join(infos)
    with Path(path).open("wb") as stream:
        stream.write(head + b"\0" * pad(len(head)))
        for _name, array in tensors:
            data = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
            stream.write(data + b"\0" * pad(len(data)))


def gguf_problem(spec):
    """"" when this design can be written as GGUF, otherwise why not, in plain words."""
    problems = gpt2_problems(spec)
    if problems:
        return ("GGUF export covers GPT-2's architecture only (llama.cpp has no loader for other designs here): "
                + "; ".join(problems) + ".")
    kind = spec["tokenizer"]["id"]
    if kind not in GGUF_TOKENIZERS:
        return NO_TOKENIZER.get(kind, f"The {kind} tokenizer cannot be exported.")
    return ""


def gguf_tensors(weights, spec, precision="f32"):
    """[(llama.cpp gpt2 tensor name, array)] in the layout llama.cpp's converter writes: torch Linear (out, in), 1-D as float32."""
    dtype = {"f32": np.float32, "f16": np.float16}.get(precision)
    if dtype is None:
        raise ExportError("Precision must be f32 or f16.")
    hf = to_gpt2(weights, spec)
    names = {"transformer.wte.weight": "token_embd.weight", "transformer.wpe.weight": "position_embd.weight",
             "transformer.ln_f.weight": "output_norm.weight", "transformer.ln_f.bias": "output_norm.bias"}
    for i in range(spec["dims"]["n_layer"]):
        for theirs, ours in (("ln_1.weight", "attn_norm.weight"), ("ln_1.bias", "attn_norm.bias"), ("attn.c_attn.weight", "attn_qkv.weight"),
                             ("attn.c_attn.bias", "attn_qkv.bias"), ("attn.c_proj.weight", "attn_output.weight"), ("attn.c_proj.bias", "attn_output.bias"),
                             ("ln_2.weight", "ffn_norm.weight"), ("ln_2.bias", "ffn_norm.bias"), ("mlp.c_fc.weight", "ffn_up.weight"),
                             ("mlp.c_fc.bias", "ffn_up.bias"), ("mlp.c_proj.weight", "ffn_down.weight"), ("mlp.c_proj.bias", "ffn_down.bias")):
            names[f"transformer.h.{i}.{theirs}"] = f"blk.{i}.{ours}"
    out = []
    for key, name in names.items():
        array = hf[key]
        if array.ndim == 2 and name.startswith("blk."):
            array = array.T   # Conv1D (in, out) -> Linear (out, in), as convert_hf_to_gguf.py does for c_attn, c_proj and c_fc
        # 1-D tensors and the position table stay float32, as in the converter
        array = array.astype(dtype if array.ndim == 2 and name != "position_embd.weight" else np.float32)
        if precision == "f16" and array.dtype == np.float16 and not np.isfinite(array).all():
            raise ExportError(f"{name} does not fit in 16 bits; export as f32.")
        out.append((name, np.ascontiguousarray(array)))
        if name == "token_embd.weight":
            out.append(("output.weight", out[-1][1]))   # tied, but the converter writes it out too
    return out


def gguf_metadata(spec, tokenizer, precision, name=None):
    d, block = spec["dims"], block_at(spec, 0)
    tokens, merges = byte_level_vocab(tokenizer)
    types = [TOKEN_CONTROL] * UNK + [TOKEN_UNKNOWN] + [TOKEN_NORMAL] * (len(tokens) - OFFSET)
    for i, token in enumerate(tokens):
        if token.startswith("<|unused"):
            types[i] = TOKEN_UNUSED
    return [("general.architecture", "str", "gpt2"), ("general.type", "str", "model"), ("general.name", "str", name or spec["name"]),
            ("general.quantization_version", "u32", 2), ("general.file_type", "u32", 1 if precision == "f16" else 0),
            ("gpt2.context_length", "u32", d["n_ctx"]), ("gpt2.embedding_length", "u32", d["d_model"]),
            ("gpt2.block_count", "u32", d["n_layer"]),
            ("gpt2.feed_forward_length", "u32", block["channel"]["config"].get("d_ff") or 4 * d["d_model"]),
            ("gpt2.attention.head_count", "u32", block["mixer"]["config"]["heads"]),
            ("gpt2.attention.layer_norm_epsilon", "f32", float(block["norm"]["config"].get("eps", 1e-5))),
            ("tokenizer.ggml.model", "str", "gpt2"), ("tokenizer.ggml.pre", "str", "gpt-2"),
            ("tokenizer.ggml.tokens", ("arr", "str"), tokens), ("tokenizer.ggml.token_type", ("arr", "i32"), types),
            ("tokenizer.ggml.merges", ("arr", "str"), merges),
            ("tokenizer.ggml.bos_token_id", "u32", BOS), ("tokenizer.ggml.eos_token_id", "u32", EOS),
            ("tokenizer.ggml.unknown_token_id", "u32", UNK), ("tokenizer.ggml.padding_token_id", "u32", PAD),
            ("tokenizer.ggml.add_bos_token", "bool", False)]


def export_gguf(checkpoint, path, precision="f32"):
    """Write a GGUF file (llama.cpp's gpt2 architecture) for a GPT-2-architecture checkpoint. precision: "f32" or "f16".
    Returns {"path", "files", "precision", "note"}; raises ExportError saying why when the design or tokenizer cannot be written."""
    spec, tokenizer, weights, _step = read_checkpoint(checkpoint)
    reason = gguf_problem(spec)
    if reason:
        raise ExportError(reason)
    if weights["token"].shape[0] != tokenizer.vocab_size:
        raise ExportError("The checkpoint's vocabulary does not match its tokenizer.")
    tensors = gguf_tensors(weights, spec, precision)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(target, gguf_metadata(spec, tokenizer, precision), tensors)
    return {"path": str(target), "files": [target.name], "precision": precision,
            "note": f"GGUF {precision}, llama.cpp's gpt2 architecture. Ids 0-{OFFSET - 1} are Groundwork's special tokens, so the vocabulary is shifted from OpenAI's GPT-2."}


def export(checkpoint, kind, path):
    """Export a checkpoint as "hf" (path = a folder) or "gguf" (path = the .gguf file)."""
    if kind == "hf":
        return export_hf(checkpoint, path)
    if kind == "gguf":
        return export_gguf(checkpoint, path)
    raise ExportError(f"Export kind must be one of {', '.join(KINDS)}.")

"""Export (studio_export): safetensors, the GPT-2 name mapping, Hugging Face folders and GGUF files."""
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import studio_export as export
import studio_spec
from studio_compile import CompiledModel
from studio_model import DEFAULT_CONFIG, LocalModel, validate_config
from studio_modules.tokenizer.bpe_fast import FastBPETokenizer
from studio_modules.tokenizer.bpe_gpt2 import GPT2Tokenizer, bytes_to_unicode
from studio_modules.tokenizer.byte import Tokenizer
from studio_slots import OFFSET

try:
    import transformers
    import torch
except ImportError:  # the Hugging Face comparison is skipped where transformers is not installed
    transformers = None


def gpt2_spec(layers=2, d_model=32, heads=4, context=32, tokenizer="byte", d_ff=0):
    spec = studio_spec.default_spec(preset="classic")
    spec["dims"].update(d_model=d_model, n_layer=layers, n_ctx=context)
    spec["tokenizer"] = {"id": tokenizer}
    spec["block"]["mixer"] = {"id": "attention", "config": {"heads": heads}}
    spec["block"]["channel"] = {"id": "mlp_gelu", "config": {"d_ff": d_ff} if d_ff else {}}
    return studio_spec.validate_spec(spec)


def bias_free_spec(**kw):
    """A GPT-2-shaped design whose attention and MLP projections have no bias (the app's older default designs)."""
    spec = gpt2_spec(**kw)
    spec["block"]["mixer"]["config"]["bias"] = False
    spec["block"]["channel"]["config"]["bias"] = False
    return studio_spec.validate_spec(spec)


def small_bpe():
    """A GPT-2 style tokenizer with the 256 byte symbols and a few merges, so no download is needed."""
    table = bytes_to_unicode()
    encoder = {table[byte]: byte for byte in range(256)}
    merges = [("h", "e"), ("l", "l"), ("he", "ll"), (table[32], "t")]
    for left, right in merges:
        encoder[left + right] = len(encoder)
    return GPT2Tokenizer(encoder, merges)


def write_checkpoint(path, spec, tokenizer, seed=0):
    """A real checkpoint file with random, non-trivial weights in every parameter."""
    model = CompiledModel(spec, tokenizer.vocab_size, device="cpu")
    rng = np.random.default_rng(seed)
    weights = {key: (rng.standard_normal(model.backend.to_cpu(value).shape) * .2).astype(np.float32) for key, value in model.p.items()}
    metadata = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "step": 7,
                "config": validate_config(dict(DEFAULT_CONFIG, architecture=spec)), "tokenizer": tokenizer.serialize()}
    arrays = {"metadata": np.array(json.dumps(metadata))}
    arrays.update({"weight::" + key: value for key, value in weights.items()})
    np.savez(path, **arrays)
    return weights


def engine_logits(path, ids):
    model = LocalModel(path, device="cpu").model
    return model.backend.to_cpu(model.forward(np.array([ids]))).astype(np.float64)[0]


def gpt2_reference(t, ids, n_layer, heads, eps=1e-5):
    """GPT-2's forward pass in plain NumPy from Hugging Face-named tensors: Conv1D is x @ W + b, c_attn is [q | k | v], gelu_new."""
    t = {key: value.astype(np.float64) for key, value in t.items()}

    def norm(x, prefix):
        centered = x - x.mean(-1, keepdims=True)
        return centered / np.sqrt((centered ** 2).mean(-1, keepdims=True) + eps) * t[prefix + ".weight"] + t[prefix + ".bias"]

    def conv1d(x, prefix):
        return x @ t[prefix + ".weight"] + t[prefix + ".bias"]

    length = len(ids)
    x = t["transformer.wte.weight"][ids] + t["transformer.wpe.weight"][:length]
    mask = np.triu(np.ones((length, length), dtype=bool), 1)
    for i in range(n_layer):
        base = f"transformer.h.{i}."
        q, k, v = (part.reshape(length, heads, -1).transpose(1, 0, 2) for part in np.split(conv1d(norm(x, base + "ln_1"), base + "attn.c_attn"), 3, axis=-1))
        scores = np.where(mask, -np.inf, q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1]))
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        x = x + conv1d((weights @ v).transpose(1, 0, 2).reshape(length, -1), base + "attn.c_proj")
        hidden = conv1d(norm(x, base + "ln_2"), base + "mlp.c_fc")
        hidden = .5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + .044715 * hidden ** 3)))
        x = x + conv1d(hidden, base + "mlp.c_proj")
    return norm(x, "transformer.ln_f") @ t["transformer.wte.weight"].T


def read_gguf(path):
    """A small independent GGUF v3 reader: (version, {key: value}, {name: (dims, ggml type, array)}, data start offset)."""
    blob = Path(path).read_bytes()
    pos = 0

    def take(fmt):
        nonlocal pos
        values = struct.unpack_from(fmt, blob, pos)
        pos += struct.calcsize(fmt)
        return values[0] if len(values) == 1 else values

    def string():
        nonlocal pos
        size = take("<Q")
        pos += size
        return blob[pos - size:pos].decode("utf-8")

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            element, count = take("<I"), take("<Q")
            return [value(element) for _ in range(count)]
        return take(scalar[kind])

    magic = blob[:4]
    pos = 4
    version, tensor_count, kv_count = take("<I"), take("<Q"), take("<Q")
    assert magic == b"GGUF"
    metadata = {}
    for _ in range(kv_count):
        key = string()
        metadata[key] = value(take("<I"))
    infos = []
    for _ in range(tensor_count):
        name, n_dims = string(), take("<I")
        dims = [take("<Q") for _ in range(n_dims)]
        infos.append((name, dims, take("<I"), take("<Q")))
    alignment = metadata.get("general.alignment", 32)
    start = pos + (-pos % alignment)
    tensors = {}
    for name, dims, ggml, offset in infos:
        dtype = {0: np.float32, 1: np.float16}[ggml]
        count = int(np.prod(dims))
        data = np.frombuffer(blob, dtype=dtype, count=count, offset=start + offset)
        tensors[name] = (dims, ggml, offset, data.reshape(list(reversed(dims))))
    return version, metadata, tensors, start


class Safetensors(unittest.TestCase):
    def test_round_trip_keeps_every_tensor_dtype_shape_and_metadata(self):
        rng = np.random.default_rng(0)
        tensors = {"a.w": rng.standard_normal((3, 5)).astype(np.float32), "b": rng.standard_normal(7).astype(np.float16),
                   "c": rng.standard_normal((2, 2, 2)), "d": np.arange(6, dtype=np.int64).reshape(2, 3), "scalar": np.array(2.5, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.safetensors"
            export.write_safetensors(path, tensors, {"format": "pt", "step": 3})
            back, metadata = export.read_safetensors(path)
            blob = path.read_bytes()
        self.assertEqual(metadata, {"format": "pt", "step": "3"})
        self.assertEqual(set(back), set(tensors))
        for key, value in tensors.items():
            self.assertEqual(back[key].dtype, value.dtype, key)
            np.testing.assert_array_equal(back[key], value, err_msg=key)
        (size,) = struct.unpack_from("<Q", blob, 0)
        self.assertEqual(size % 8, 0)   # the data starts on an 8-byte boundary
        header = json.loads(blob[8:8 + size])
        ranges = sorted(entry["data_offsets"] for key, entry in header.items() if key != "__metadata__")
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual([a[1] for a in ranges[:-1]], [b[0] for b in ranges[1:]])   # contiguous, no gaps
        self.assertEqual(8 + size + ranges[-1][1], len(blob))

    def test_unsupported_dtype_is_refused(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(export.ExportError):
            export.write_safetensors(Path(directory) / "t.safetensors", {"x": np.array([1 + 2j])})


class Gpt2Mapping(unittest.TestCase):
    def test_every_parameter_is_mapped_exactly_once(self):
        for layers in (1, 3):
            spec = gpt2_spec(layers=layers)
            model = CompiledModel(spec, 263, device="cpu")
            names = export.gpt2_names(layers)
            self.assertEqual(set(model.p), set(names))                  # no engine parameter left out, none invented
            self.assertEqual(len(set(names.values())), len(names))      # no Hugging Face key written twice
            weights = {key: model.backend.to_cpu(value) for key, value in model.p.items()}
            mapped = export.to_gpt2(weights, spec)
            self.assertEqual(len(mapped), len(model.p))
            for key, value in weights.items():
                np.testing.assert_array_equal(mapped[names[key]], value)
            self.assertFalse(any("lm_head" in key for key in mapped))   # the head is tied to wte

    def test_a_design_without_projection_biases_gets_zero_biases_and_nothing_else(self):
        spec = bias_free_spec()
        model = CompiledModel(spec, 263, device="cpu")
        names = export.gpt2_names(2)
        absent = set(names) - set(model.p)
        self.assertEqual(absent, {f"block{i}.{b}" for i in range(2) for b in ("qkv.b", "proj.b", "f1.b", "f2.b")})
        weights = {key: model.backend.to_cpu(value) for key, value in model.p.items()}
        mapped = export.to_gpt2(weights, spec)
        self.assertEqual(set(mapped), set(names.values()))
        for key in absent:
            self.assertEqual(mapped[names[key]].dtype, np.float32)
            self.assertFalse(mapped[names[key]].any())
        self.assertEqual(mapped["transformer.h.0.attn.c_attn.bias"].shape, (96,))
        self.assertEqual(mapped["transformer.h.0.mlp.c_fc.bias"].shape, (128,))
        self.assertEqual(mapped["transformer.h.1.mlp.c_proj.bias"].shape, (32,))

    def test_unmatched_parameters_are_refused(self):
        spec = gpt2_spec()
        weights = {key: np.zeros(1, np.float32) for key in export.gpt2_names(2)}
        with self.assertRaises(export.ExportError):
            export.to_gpt2({**weights, "extra": np.zeros(1, np.float32)}, spec)

    def test_default_design_is_gpt2_and_each_change_is_named(self):
        self.assertEqual(export.gpt2_problems(gpt2_spec(layers=3)), [])
        self.assertEqual(export.gpt2_problems(gpt2_spec(d_ff=96)), [])

        def changed(edit):
            spec = copy.deepcopy(gpt2_spec())
            edit(spec)
            return export.gpt2_problems(studio_spec.validate_spec(spec))

        cases = {
            "rope": lambda s: s.update(position={"id": "rope"}),
            "untied": lambda s: s.update(head={"id": "untied"}),
            "rmsnorm": lambda s: s["block"].update(norm={"id": "rmsnorm"}),
            "post-norm": lambda s: s["block"].update(norm={"id": "layernorm", "config": {"placement": "post"}}),
            "swiglu": lambda s: s["block"].update(channel={"id": "swiglu"}),
            "relu": lambda s: s["block"].update(channel={"id": "mlp_gelu", "config": {"activation": "relu"}}),
            "gru in block 2": lambda s: s.update(blocks=[{"index": 1, "mixer": {"id": "gru"}}]),
            "mup at another width": lambda s: s["train"].update(init={"id": "mup", "config": {"base_width": 256}}),
        }
        for label, edit in cases.items():
            with self.subTest(label):
                self.assertTrue(changed(edit), label)

    def test_mup_at_its_base_width_is_plain_gpt2(self):
        spec = gpt2_spec()
        spec["train"]["init"] = {"id": "mup", "config": {"base_width": spec["dims"]["d_model"]}}
        self.assertEqual(export.gpt2_problems(studio_spec.validate_spec(spec)), [])


class HuggingFaceExport(unittest.TestCase):
    def test_fast_subword_export_preserves_ids_and_merge_order(self):
        tokenizer = FastBPETokenizer([(104, 101), (108, 108), (256, 257), (32, 116)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_checkpoint(root / "m.npz", gpt2_spec(tokenizer="bpe_fast"), tokenizer)
            report = export.export_hf(root / "m.npz", root / "hf")
            self.assertTrue(report["tokenizer_loadable"])
            vocab = json.loads((root / "hf" / "vocab.json").read_text(encoding="utf-8"))
            merges = [tuple(line.split(" ")) for line in
                      (root / "hf" / "merges.txt").read_text(encoding="utf-8").splitlines()[1:]]
            rebuilt = GPT2Tokenizer({symbol: index - OFFSET for symbol, index in vocab.items() if index >= OFFSET}, merges)
            self.assertEqual(len(vocab), tokenizer.vocab_size)
            self.assertEqual(merges, [(tokenizer.symbols[a], tokenizer.symbols[b]) for a, b in tokenizer.merges])
            for text in ("hello hello", " the hell hat", "héllo ☃", "tab\tand\nline"):
                self.assertEqual(rebuilt.encode(text), tokenizer.encode(text))
            export.export_gguf(root / "m.npz", root / "m.gguf")
            _, meta, _, _ = read_gguf(root / "m.gguf")
            self.assertEqual(meta["tokenizer.ggml.tokens"], list(export.SPECIAL_TOKENS) + tokenizer.symbols)
            self.assertEqual(meta["tokenizer.ggml.merges"], [f"{a} {b}" for a, b in merges])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_gpt2_design_gets_gpt2_keys_config_and_matching_logits(self):
        spec = gpt2_spec(layers=2, d_model=32, heads=4, context=32, d_ff=96)
        weights = write_checkpoint(self.root / "m.npz", spec, Tokenizer("byte"))
        report = export.export_hf(self.root / "m.npz", self.root / "hf")
        self.assertEqual((report["architecture"], report["huggingface_loadable"], report["tokenizer_loadable"]), ("gpt2", True, True))
        out = self.root / "hf"
        self.assertEqual(sorted(path.name for path in out.iterdir()), report["files"])
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        expected = dict(model_type="gpt2", architectures=["GPT2LMHeadModel"], vocab_size=263, n_positions=32, n_embd=32, n_layer=2, n_head=4,
                        n_inner=96, activation_function="gelu_new", tie_word_embeddings=True)
        self.assertEqual({key: config[key] for key in expected}, expected)
        self.assertEqual(config["layer_norm_epsilon"], 1e-5)
        tensors, metadata = export.read_safetensors(out / "model.safetensors")
        self.assertEqual(metadata["format"], "pt")
        self.assertEqual(set(tensors), set(export.gpt2_names(2).values()))
        self.assertEqual(tensors["transformer.h.0.attn.c_attn.weight"].shape, (32, 96))   # Conv1D: (in, out), q | k | v fused
        self.assertEqual(tensors["transformer.h.1.mlp.c_fc.weight"].shape, (32, 96))
        np.testing.assert_array_equal(tensors["transformer.h.0.attn.c_attn.weight"], weights["block0.qkv.w"])
        ids = [1, 3, 10, 40, 77, 120, 200, 9, 262, 4]
        np.testing.assert_allclose(gpt2_reference(tensors, ids, 2, 4), engine_logits(self.root / "m.npz", ids), atol=2e-4, rtol=1e-4)
        readme = (out / "README.md").read_text(encoding="utf-8")
        self.assertIn("GPT2LMHeadModel", readme)
        self.assertNotIn("cannot load this design", readme)

    def test_a_design_without_projection_biases_exports_as_gpt2_with_matching_logits(self):
        spec = bias_free_spec(layers=2, d_model=32, heads=4, context=32)
        write_checkpoint(self.root / "m.npz", spec, Tokenizer("byte"))
        report = export.export_hf(self.root / "m.npz", self.root / "hf")
        self.assertEqual((report["architecture"], report["huggingface_loadable"]), ("gpt2", True))
        tensors, _meta = export.read_safetensors(self.root / "hf" / "model.safetensors")
        self.assertEqual(set(tensors), set(export.gpt2_names(2).values()))
        self.assertFalse(tensors["transformer.h.1.attn.c_proj.bias"].any())
        self.assertTrue(tensors["transformer.h.1.ln_2.bias"].any())   # LayerNorm's own shift is a real parameter
        ids = [1, 3, 10, 40, 77, 120, 200, 9, 262, 4]
        np.testing.assert_allclose(gpt2_reference(tensors, ids, 2, 4), engine_logits(self.root / "m.npz", ids), atol=2e-4, rtol=1e-4)
        self.assertIn("written as zeros", (self.root / "hf" / "README.md").read_text(encoding="utf-8"))

    def test_byte_tokenizer_files_spell_every_byte_as_the_engine_does(self):
        write_checkpoint(self.root / "m.npz", gpt2_spec(), Tokenizer("byte"))
        export.export_hf(self.root / "m.npz", self.root / "hf")
        vocab = json.loads((self.root / "hf" / "vocab.json").read_text(encoding="utf-8"))
        self.assertEqual(len(vocab), 263)
        self.assertEqual([vocab[name] for name in export.SPECIAL_TOKENS], list(range(OFFSET)))
        table, tokenizer = bytes_to_unicode(), Tokenizer("byte")
        text = "Hi, héllo ☃\n"
        self.assertEqual([vocab[table[byte]] for byte in text.encode("utf-8")], tokenizer.encode(text))
        self.assertEqual((self.root / "hf" / "merges.txt").read_text(encoding="utf-8"), "#version: 0.2\n")
        config = json.loads((self.root / "hf" / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["tokenizer_class"], config["bos_token"], config["eos_token"]), ("GPT2Tokenizer", "<|bos|>", "<|eos|>"))

    def test_bpe_files_rebuild_the_same_tokenizer(self):
        tokenizer = small_bpe()
        write_checkpoint(self.root / "m.npz", gpt2_spec(tokenizer="bpe_gpt2"), tokenizer)
        export.export_hf(self.root / "m.npz", self.root / "hf")
        vocab = json.loads((self.root / "hf" / "vocab.json").read_text(encoding="utf-8"))
        merges = [tuple(line.split(" ")) for line in (self.root / "hf" / "merges.txt").read_text(encoding="utf-8").splitlines()[1:]]
        self.assertEqual(len(vocab), tokenizer.vocab_size)
        rebuilt = GPT2Tokenizer({symbol: index - OFFSET for symbol, index in vocab.items() if index >= OFFSET}, merges)
        for text in ("hello hello", "the hell hat", "héllo ☃"):
            self.assertEqual(rebuilt.encode(text), tokenizer.encode(text), text)
        self.assertEqual(merges, tokenizer.merges)

    def test_other_designs_keep_their_own_names_with_the_spec_embedded_and_say_so(self):
        spec = gpt2_spec(layers=2)
        spec["position"] = {"id": "rope"}
        spec["block"]["norm"] = {"id": "rmsnorm"}
        spec["block"]["channel"] = {"id": "swiglu"}
        spec = studio_spec.validate_spec(spec)
        weights = write_checkpoint(self.root / "m.npz", spec, Tokenizer("byte"))
        report = export.export_hf(self.root / "m.npz", self.root / "hf")
        self.assertEqual((report["architecture"], report["huggingface_loadable"]), ("raw", False))
        self.assertIn("cannot load this design directly", report["note"])
        out = self.root / "hf"
        tensors, metadata = export.read_safetensors(out / "model.safetensors")
        self.assertEqual(set(tensors), set(weights))
        for key, value in weights.items():
            np.testing.assert_array_equal(tensors[key], value)
        self.assertEqual(json.loads(metadata["groundwork_spec"]), spec)
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["model_type"], config["architectures"], config["groundwork_spec"]), ("groundwork", [], spec))
        readme = (out / "README.md").read_text(encoding="utf-8")
        self.assertIn("Hugging Face cannot load this design directly", readme)
        self.assertIn("does not use learned positions", readme)   # names what differs from GPT-2

    def test_tokenizers_hugging_face_cannot_load_export_their_vocabulary_as_json(self):
        examples = [{"text": "the cat sat on the mat. the cat ran, the cat sat."}]
        cases = {"character": Tokenizer.fit("character", examples)}
        for kind, tokenizer in cases.items():
            with self.subTest(kind):
                out = self.root / kind
                write_checkpoint(self.root / f"{kind}.npz", gpt2_spec(tokenizer=kind), tokenizer)
                report = export.export_hf(self.root / f"{kind}.npz", out)
                self.assertFalse(report["tokenizer_loadable"])
                self.assertIn("groundwork_tokenizer.json", report["files"])
                self.assertFalse((out / "vocab.json").exists())
                data = json.loads((out / "groundwork_tokenizer.json").read_text(encoding="utf-8"))
                self.assertEqual((data["kind"], data["vocab_size"], len(data["vocabulary"])), (kind, tokenizer.vocab_size, tokenizer.vocab_size))
                self.assertEqual(data["vocabulary"][:OFFSET], list(export.SPECIAL_TOKENS))
                self.assertEqual(len(set(data["vocabulary"])), len(data["vocabulary"]))
                self.assertIn("cannot load it", (out / "README.md").read_text(encoding="utf-8"))
                # the model itself is still a GPT-2 design
                self.assertTrue(report["huggingface_loadable"])

    @unittest.skipIf(transformers is None, "transformers is not installed")
    def test_transformers_loads_the_export_and_agrees_with_the_engine(self):
        spec = gpt2_spec(layers=2, d_model=32, heads=4, context=32)
        write_checkpoint(self.root / "m.npz", spec, Tokenizer("byte"))
        export.export_hf(self.root / "m.npz", self.root / "hf")
        model = transformers.GPT2LMHeadModel.from_pretrained(self.root / "hf").eval()
        ids = [1, 3, 10, 40, 77, 120, 200, 9, 262, 4]
        with torch.no_grad():
            theirs = model(torch.tensor([ids])).logits[0].numpy()
        np.testing.assert_allclose(theirs, engine_logits(self.root / "m.npz", ids), atol=1e-4, rtol=1e-4)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(self.root / "hf")
        self.assertEqual(tokenizer.convert_tokens_to_ids("<|bos|>"), 1)


class GGUFExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_header_metadata_and_tensor_table_parse_back(self):
        spec = gpt2_spec(layers=2, d_model=32, heads=4, context=32, d_ff=96)
        tokenizer = small_bpe()
        spec = studio_spec.validate_spec(dict(spec, tokenizer={"id": "bpe_gpt2"}))
        weights = write_checkpoint(self.root / "m.npz", spec, tokenizer)
        report = export.export_gguf(self.root / "m.npz", self.root / "m.gguf")
        self.assertEqual(report["files"], ["m.gguf"])
        version, meta, tensors, start = read_gguf(self.root / "m.gguf")
        self.assertEqual(version, 3)
        self.assertEqual(start % 32, 0)
        self.assertEqual({key: meta[key] for key in ("general.architecture", "gpt2.context_length", "gpt2.embedding_length", "gpt2.block_count",
                                                      "gpt2.feed_forward_length", "gpt2.attention.head_count", "general.file_type")},
                         {"general.architecture": "gpt2", "gpt2.context_length": 32, "gpt2.embedding_length": 32, "gpt2.block_count": 2,
                          "gpt2.feed_forward_length": 96, "gpt2.attention.head_count": 4, "general.file_type": 0})
        self.assertAlmostEqual(meta["gpt2.attention.layer_norm_epsilon"], 1e-5, places=9)
        self.assertEqual((meta["tokenizer.ggml.model"], meta["tokenizer.ggml.pre"]), ("gpt2", "gpt-2"))
        self.assertEqual(len(meta["tokenizer.ggml.tokens"]), tokenizer.vocab_size)
        self.assertEqual(len(meta["tokenizer.ggml.token_type"]), tokenizer.vocab_size)
        self.assertEqual(meta["tokenizer.ggml.merges"], [f"{a} {b}" for a, b in tokenizer.merges])
        self.assertEqual(meta["tokenizer.ggml.tokens"][:OFFSET], list(export.SPECIAL_TOKENS))
        self.assertEqual([meta[f"tokenizer.ggml.{key}"] for key in ("bos_token_id", "eos_token_id", "unknown_token_id", "padding_token_id")], [1, 2, 6, 0])
        self.assertEqual(meta["tokenizer.ggml.token_type"][:OFFSET], [3, 3, 3, 3, 3, 3, 2])
        # tensor table: llama.cpp's gpt2 names, dimensions fastest-first, offsets aligned
        expected = {"token_embd.weight", "output.weight", "position_embd.weight", "output_norm.weight", "output_norm.bias"}
        for i in range(2):
            expected |= {f"blk.{i}.{name}" for name in ("attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias", "attn_output.weight",
                                                         "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias", "ffn_up.weight", "ffn_up.bias",
                                                         "ffn_down.weight", "ffn_down.bias")}
        self.assertEqual(set(tensors), expected)
        self.assertEqual(tensors["token_embd.weight"][0], [32, tokenizer.vocab_size])
        self.assertEqual(tensors["blk.0.attn_qkv.weight"][0], [32, 96])      # ggml [in, out]
        self.assertEqual(tensors["blk.1.ffn_down.weight"][0], [96, 32])
        self.assertEqual(tensors["blk.0.attn_qkv.bias"][0], [96])
        self.assertTrue(all(offset % 32 == 0 for _d, _t, offset, _a in tensors.values()))
        self.assertTrue(all(ggml == 0 for _d, ggml, _o, _a in tensors.values()))
        # the numbers are the engine's, in the layout llama.cpp's converter writes (Linear: out x in)
        np.testing.assert_array_equal(tensors["token_embd.weight"][3], weights["token"])
        np.testing.assert_array_equal(tensors["output.weight"][3], weights["token"])
        np.testing.assert_array_equal(tensors["position_embd.weight"][3], weights["position"])
        np.testing.assert_array_equal(tensors["blk.0.attn_qkv.weight"][3], weights["block0.qkv.w"].T)
        np.testing.assert_array_equal(tensors["blk.1.attn_output.weight"][3], weights["block1.proj.w"].T)
        np.testing.assert_array_equal(tensors["blk.0.ffn_up.weight"][3], weights["block0.f1.w"].T)
        np.testing.assert_array_equal(tensors["blk.1.ffn_down.bias"][3], weights["block1.f2.b"])
        np.testing.assert_array_equal(tensors["output_norm.weight"][3], weights["final.g"])
        np.testing.assert_array_equal(tensors["blk.0.ffn_norm.bias"][3], weights["block0.n2.b"])

    def test_gguf_tensors_reproduce_the_engine_logits(self):
        """Undo the layout changes the way llama.cpp's graph reads them (mul_mat of the out x in weight) and compare logits."""
        spec = gpt2_spec(layers=2, d_model=32, heads=4, context=32)
        write_checkpoint(self.root / "m.npz", spec, Tokenizer("byte"))
        export.export_gguf(self.root / "m.npz", self.root / "m.gguf")
        _v, _meta, tensors, _s = read_gguf(self.root / "m.gguf")
        ours = {"token_embd.weight": "transformer.wte.weight", "position_embd.weight": "transformer.wpe.weight",
                "output_norm.weight": "transformer.ln_f.weight", "output_norm.bias": "transformer.ln_f.bias"}
        for i in range(2):
            for theirs, name in (("ln_1.weight", "attn_norm.weight"), ("ln_1.bias", "attn_norm.bias"), ("attn.c_attn.weight", "attn_qkv.weight"),
                                 ("attn.c_attn.bias", "attn_qkv.bias"), ("attn.c_proj.weight", "attn_output.weight"), ("attn.c_proj.bias", "attn_output.bias"),
                                 ("ln_2.weight", "ffn_norm.weight"), ("ln_2.bias", "ffn_norm.bias"), ("mlp.c_fc.weight", "ffn_up.weight"),
                                 ("mlp.c_fc.bias", "ffn_up.bias"), ("mlp.c_proj.weight", "ffn_down.weight"), ("mlp.c_proj.bias", "ffn_down.bias")):
                ours[f"blk.{i}.{name}"] = f"transformer.h.{i}.{theirs}"
        self.assertEqual(set(ours.values()), set(export.gpt2_names(2).values()))
        back = {ours[name]: (array.T if name.startswith("blk.") and array.ndim == 2 else array) for name, (_d, _t, _o, array) in tensors.items() if name in ours}
        ids = [1, 3, 10, 40, 77, 120, 200, 9, 262, 4]
        np.testing.assert_allclose(gpt2_reference(back, ids, 2, 4), engine_logits(self.root / "m.npz", ids), atol=2e-4, rtol=1e-4)

    def test_a_design_without_projection_biases_writes_gguf_with_zero_biases(self):
        spec = bias_free_spec(layers=2, d_model=32, heads=4, context=32)
        self.assertEqual(export.gguf_problem(spec), "")
        write_checkpoint(self.root / "m.npz", spec, Tokenizer("byte"))
        export.export_gguf(self.root / "m.npz", self.root / "m.gguf")
        _v, _meta, tensors, _s = read_gguf(self.root / "m.gguf")
        self.assertEqual(tensors["blk.1.attn_qkv.bias"][0], [96])
        for name in ("blk.0.attn_qkv.bias", "blk.0.attn_output.bias", "blk.1.ffn_up.bias", "blk.1.ffn_down.bias"):
            self.assertFalse(tensors[name][3].any(), name)

    def test_f16_halves_the_matrices_and_keeps_vectors_and_positions_in_f32(self):
        weights = write_checkpoint(self.root / "m.npz", gpt2_spec(), Tokenizer("byte"))
        export.export_gguf(self.root / "m.npz", self.root / "full.gguf")
        export.export_gguf(self.root / "m.npz", self.root / "half.gguf", precision="f16")
        _v, meta, tensors, _s = read_gguf(self.root / "half.gguf")
        self.assertEqual(meta["general.file_type"], 1)
        kinds = {name: ggml for name, (_d, ggml, _o, _a) in tensors.items()}
        self.assertEqual((kinds["blk.0.attn_qkv.weight"], kinds["token_embd.weight"], kinds["blk.0.attn_qkv.bias"], kinds["output_norm.weight"],
                          kinds["position_embd.weight"]), (1, 1, 0, 0, 0))
        np.testing.assert_allclose(tensors["blk.0.ffn_up.weight"][3], weights["block0.f1.w"].T, atol=1e-3)
        self.assertLess((self.root / "half.gguf").stat().st_size, (self.root / "full.gguf").stat().st_size)

    def test_byte_tokenizer_is_a_gpt2_vocabulary_with_no_merges(self):
        write_checkpoint(self.root / "m.npz", gpt2_spec(), Tokenizer("byte"))
        export.export_gguf(self.root / "m.npz", self.root / "m.gguf")
        _v, meta, _t, _s = read_gguf(self.root / "m.gguf")
        self.assertEqual(len(meta["tokenizer.ggml.tokens"]), 263)
        self.assertEqual(meta["tokenizer.ggml.merges"], [])
        self.assertEqual(meta["tokenizer.ggml.token_type"][OFFSET:], [1] * 256)

    def test_designs_and_tokenizers_llama_cpp_cannot_take_say_why(self):
        spec = gpt2_spec()
        spec["position"] = {"id": "rope"}
        write_checkpoint(self.root / "rope.npz", studio_spec.validate_spec(spec), Tokenizer("byte"))
        with self.assertRaisesRegex(export.ExportError, "GPT-2's architecture only.*learned positions"):
            export.export_gguf(self.root / "rope.npz", self.root / "rope.gguf")
        self.assertFalse((self.root / "rope.gguf").exists())
        examples = [{"text": "the cat sat on the mat."}]
        for kind, tokenizer, words in (("character", Tokenizer.fit("character", examples), "character tokenizer"),):
            with self.subTest(kind):
                write_checkpoint(self.root / f"{kind}.npz", gpt2_spec(tokenizer=kind), tokenizer)
                with self.assertRaisesRegex(export.ExportError, words):
                    export.export_gguf(self.root / f"{kind}.npz", self.root / f"{kind}.gguf")
                self.assertFalse((self.root / f"{kind}.gguf").exists())
                self.assertEqual(export.gguf_problem(gpt2_spec(tokenizer=kind)).startswith("The"), True)

    def test_gguf_problem_is_empty_for_supported_designs(self):
        self.assertEqual(export.gguf_problem(gpt2_spec()), "")
        self.assertEqual(export.gguf_problem(gpt2_spec(tokenizer="bpe_gpt2")), "")

    def test_bad_precision_and_kind_are_refused(self):
        write_checkpoint(self.root / "m.npz", gpt2_spec(), Tokenizer("byte"))
        with self.assertRaises(export.ExportError):
            export.export_gguf(self.root / "m.npz", self.root / "m.gguf", precision="q4")
        with self.assertRaises(export.ExportError):
            export.export(self.root / "m.npz", "onnx", self.root / "x")


if __name__ == "__main__":
    unittest.main()

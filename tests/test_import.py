"""Import (studio_import): a Hugging Face GPT-2 file becomes a studio checkpoint that computes the same function; the download is safe."""
import errno
import hashlib
import http.client
import json
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import studio_export as export
import studio_import as imp
from studio_modules.tokenizer.bpe_gpt2 import GPT2Tokenizer, bytes_to_unicode
from studio_model import Tokenizer
from studio_slots import BOS, EOS, OFFSET
from studio_ia.services import FakeServices, FixtureServices
from tests.test_export import engine_logits, gpt2_spec, write_checkpoint

HEADS = 4
IDS = [20, 21, 22, 23, 24, 25, 26, 27]


def small_bpe():
    """A GPT-2 style tokenizer with the byte symbols, a few merges and <|endoftext|>, so no download is needed."""
    table = bytes_to_unicode()
    encoder = {table[byte]: byte for byte in range(256)}
    merges = [("h", "e"), ("l", "l"), ("he", "ll"), (table[32], "t")]
    for left, right in merges:
        encoder[left + right] = len(encoder)
    encoder["<|endoftext|>"] = len(encoder)
    return GPT2Tokenizer(encoder, merges)


def hf_file(directory, seed=0):
    """A real checkpoint, plus the same weights as a Hugging Face GPT-2 file without the engine's 7 special rows (as OpenAI ships it)."""
    tokenizer = small_bpe()
    spec = gpt2_spec(layers=2, d_model=32, heads=HEADS, context=32, tokenizer="bpe_gpt2")
    source = Path(directory) / "source.npz"
    weights = write_checkpoint(source, spec, tokenizer, seed)
    tensors = export.to_gpt2(weights, spec)
    key = "transformer.wte.weight"
    tensors[key] = np.ascontiguousarray(tensors[key][OFFSET:])
    tensors.pop("lm_head.weight", None)
    path = Path(directory) / "model.safetensors"
    export.write_safetensors(path, tensors)
    return tokenizer, spec, source, path, weights


class Conversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.tokenizer, cls.spec, cls.source, cls.hf, cls.weights = hf_file(cls.dir)
        cls.out = cls.dir / "imported.npz"
        cls.report = imp.import_gpt2(cls.hf, cls.out, tokenizer=cls.tokenizer, heads=HEADS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_imported_model_computes_what_the_original_computed(self):
        # The engine's own special rows differ from the import's initialisation, so compare on ids that never touch them.
        # Logits for a position depend only on the earlier ids' rows, and the output scores for ids 0-6 are not compared.
        original, imported = engine_logits(self.source, IDS), engine_logits(self.out, IDS)
        np.testing.assert_allclose(imported[:, OFFSET:], original[:, OFFSET:], atol=1e-4)

    def test_it_is_an_ordinary_checkpoint_with_the_gpt2_design(self):
        from studio_model import load_checkpoint
        data = load_checkpoint(self.out)
        self.assertEqual(data["step"], 0)
        self.assertEqual(data["config"]["architecture"]["tokenizer"]["id"], "bpe_gpt2")
        self.assertEqual(data["model"]["token"].shape, (self.tokenizer.vocab_size, 32))
        self.assertEqual(self.report["vocabulary"], self.tokenizer.vocab_size)
        self.assertEqual(self.report["parameters"], sum(value.size for value in data["model"].values()))
        self.assertFalse(Path(str(self.out) + ".tmp").exists())

    def test_every_real_token_row_is_kept_exactly(self):
        from studio_model import load_checkpoint
        table = load_checkpoint(self.out)["model"]["token"]
        np.testing.assert_array_equal(table[OFFSET:], self.weights["token"][OFFSET:])

    def test_the_seven_special_rows_start_sensibly(self):
        from studio_model import load_checkpoint
        table = load_checkpoint(self.out)["model"]["token"]
        real = self.weights["token"][OFFSET:]
        end = real[self.tokenizer.encoder["<|endoftext|>"]]
        np.testing.assert_array_equal(table[BOS], end)   # GPT-2 starts and ends a document with the same token
        np.testing.assert_array_equal(table[EOS], end)
        others = [i for i in range(OFFSET) if i not in (BOS, EOS)]
        for i in others:
            np.testing.assert_allclose(table[i], real.mean(axis=0), atol=1e-6)

    def test_a_file_that_already_has_the_special_rows_is_read_as_it_is(self):
        full = self.dir / "full.safetensors"
        tensors = export.to_gpt2(self.weights, self.spec)
        tensors.pop("lm_head.weight", None)
        export.write_safetensors(full, tensors)
        out = self.dir / "full.npz"
        imp.import_gpt2(full, out, tokenizer=self.tokenizer, heads=HEADS)
        np.testing.assert_allclose(engine_logits(out, IDS), engine_logits(self.source, IDS), atol=1e-4)

    def test_other_files_are_refused_in_plain_words(self):
        tensors = export.to_gpt2(self.weights, self.spec)
        broken = {key: value for key, value in tensors.items() if key != "transformer.h.0.ln_1.weight"}
        path = self.dir / "broken.safetensors"
        export.write_safetensors(path, broken)
        with self.assertRaises(ValueError) as caught:
            imp.import_gpt2(path, self.dir / "broken.npz", tokenizer=self.tokenizer, heads=HEADS)
        self.assertIn("not a GPT-2 file", str(caught.exception))
        self.assertFalse((self.dir / "broken.npz").exists())
        untied = dict(tensors, **{"lm_head.weight": tensors["transformer.wte.weight"] + 1})
        export.write_safetensors(self.dir / "untied.safetensors", untied)
        with self.assertRaises(ValueError):
            imp.import_gpt2(self.dir / "untied.safetensors", self.dir / "untied.npz", tokenizer=self.tokenizer, heads=HEADS)
        with self.assertRaises(ValueError):  # the head count is not in the file, and 32 wide is not a GPT-2 size
            imp.import_gpt2(self.hf, self.dir / "noheads.npz", tokenizer=self.tokenizer)

    def test_mask_buffers_and_a_transformer_prefix_do_not_matter(self):
        tensors = export.to_gpt2(self.weights, self.spec)
        tensors["transformer.h.0.attn.bias"] = np.ones((1, 1, 32, 32), dtype=np.float32)
        tensors["transformer.h.1.attn.masked_bias"] = np.array(-1e4, dtype=np.float32)
        tensors = {key.removeprefix("transformer."): value for key, value in tensors.items()}  # as AutoModel saves it
        tensors.pop("lm_head.weight", None)
        path = self.dir / "bare.safetensors"
        export.write_safetensors(path, tensors)
        out = self.dir / "bare.npz"
        imp.import_gpt2(path, out, tokenizer=self.tokenizer, heads=HEADS)
        np.testing.assert_allclose(engine_logits(out, IDS), engine_logits(self.source, IDS), atol=1e-4)

    def test_the_checkpoint_carries_a_fine_tuning_learning_rate_not_the_scratch_default(self):
        from studio_model import load_checkpoint
        config = load_checkpoint(self.out)["config"]
        self.assertEqual(config["learning_rate"], imp.FINE_TUNE_LR)
        self.assertEqual(config["architecture"]["train"]["optimizer"]["config"]["lr"], imp.FINE_TUNE_LR)
        self.assertLessEqual(config["learning_rate"], 1e-4)
        self.assertEqual(config["warmup"], 20)
        self.assertIsNone(config["lora"])

    def test_a_lora_design_takes_a_larger_step_because_only_the_adapters_move(self):
        from studio_model import validate_config
        plain, tuned = imp.gpt2_spec(), imp.gpt2_spec(lora={"rank": 4})
        self.assertEqual(plain["train"]["optimizer"]["config"]["lr"], imp.FINE_TUNE_LR)
        self.assertEqual(tuned["train"]["optimizer"]["config"]["lr"], imp.FINE_TUNE_LORA_LR)
        self.assertLess(imp.FINE_TUNE_LR, imp.FINE_TUNE_LORA_LR)
        self.assertLessEqual(imp.FINE_TUNE_LORA_LR, 1e-4)
        config = validate_config({"architecture": tuned})
        self.assertEqual((config["learning_rate"], config["lora"]["rank"]), (imp.FINE_TUNE_LORA_LR, 4))
        self.assertIsNone(plain["train"].get("run", {}).get("lora"))

    def test_gpt2_small_is_a_valid_design_of_about_124_million_parameters(self):
        spec = imp.gpt2_spec()
        self.assertEqual(export.gpt2_problems(spec), [])
        self.assertEqual((spec["dims"]["d_model"], spec["dims"]["n_layer"], spec["dims"]["n_ctx"]), (768, 12, 1024))
        self.assertEqual(spec["tokenizer"]["id"], "bpe_gpt2")


LAYERS, WIDTH, CONTEXT = 2, 32, 32


def random_hf_state(tokenizer, seed=3):
    """A tiny GPT-2 in Hugging Face's own tensor names and layouts (Conv1D weights are [in, out]), written by hand, not by the exporter."""
    rng = np.random.default_rng(seed)
    d, pieces = WIDTH, len(tokenizer.encoder)

    def w(*shape, scale=.15):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    state = {"transformer.wte.weight": w(pieces, d, scale=.5), "transformer.wpe.weight": w(CONTEXT, d, scale=.2),
             "transformer.ln_f.weight": 1 + w(d), "transformer.ln_f.bias": w(d)}
    for i in range(LAYERS):
        p = f"transformer.h.{i}."
        state.update({p + "ln_1.weight": 1 + w(d), p + "ln_1.bias": w(d), p + "ln_2.weight": 1 + w(d), p + "ln_2.bias": w(d),
                      p + "attn.c_attn.weight": w(d, 3 * d), p + "attn.c_attn.bias": w(3 * d),
                      p + "attn.c_proj.weight": w(d, d), p + "attn.c_proj.bias": w(d),
                      p + "mlp.c_fc.weight": w(d, 4 * d), p + "mlp.c_fc.bias": w(4 * d),
                      p + "mlp.c_proj.weight": w(4 * d, d), p + "mlp.c_proj.bias": w(d)})
    return state


def reference_logits(state, ids, tokenizer, heads=HEADS):
    """GPT-2's forward pass in plain torch, float64, straight from the Hugging Face tensors: tanh GELU, LayerNorm eps 1e-5, tied head.
    The engine's 7 special ids come first in its vocabulary; they start as GPT-2's end-of-text row (BOS, EOS) and the mean row (the rest)."""
    t = {key: torch.as_tensor(value, dtype=torch.float64) for key, value in state.items()}
    wte = t["transformer.wte.weight"]
    end, mean = wte[tokenizer.encoder["<|endoftext|>"]], wte.mean(0)
    table = torch.cat([torch.stack([mean, end, end, mean, mean, mean, mean]), wte])   # PAD BOS EOS USER SYSTEM ASSISTANT UNK
    d, count = WIDTH, len(ids)

    def norm(x, prefix):
        centred = x - x.mean(-1, keepdim=True)
        return centred / torch.sqrt((centred ** 2).mean(-1, keepdim=True) + 1e-5) * t[prefix + ".weight"] + t[prefix + ".bias"]

    def gelu(x):
        return .5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x ** 3)))

    x = table[torch.as_tensor(ids)] + t["transformer.wpe.weight"][:count]
    causal = torch.triu(torch.ones(count, count, dtype=torch.bool), diagonal=1)
    for i in range(LAYERS):
        p = f"transformer.h.{i}."
        qkv = norm(x, p + "ln_1") @ t[p + "attn.c_attn.weight"] + t[p + "attn.c_attn.bias"]
        q, k, v = (part.reshape(count, heads, d // heads).transpose(0, 1) for part in qkv.split(d, dim=-1))
        scores = (q @ k.transpose(-1, -2) / np.sqrt(d // heads)).masked_fill(causal, -torch.inf).softmax(-1)
        x = x + (scores @ v).transpose(0, 1).reshape(count, d) @ t[p + "attn.c_proj.weight"] + t[p + "attn.c_proj.bias"]
        x = x + gelu(norm(x, p + "ln_2") @ t[p + "mlp.c_fc.weight"] + t[p + "mlp.c_fc.bias"]) @ t[p + "mlp.c_proj.weight"] + t[p + "mlp.c_proj.bias"]
    return (norm(x, "transformer.ln_f") @ table.T).numpy()


class Imported(unittest.TestCase):
    """A synthetic GPT-2 imported through the real path, checked against a forward pass written out by hand."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.tokenizer = small_bpe()
        cls.state = random_hf_state(cls.tokenizer)
        cls.hf, cls.out = cls.dir / "model.safetensors", cls.dir / "imported.npz"
        export.write_safetensors(cls.hf, cls.state)
        imp.import_gpt2(cls.hf, cls.out, tokenizer=cls.tokenizer, heads=HEADS)
        cls.end = OFFSET + cls.tokenizer.encoder["<|endoftext|>"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def local(self, seed=0):
        from studio_model import LocalModel
        return LocalModel(self.out, seed=seed, device="cpu")

    def test_it_computes_the_hand_written_reference_on_every_id_including_the_special_ones(self):
        self.assertEqual(OFFSET, 7)
        ids = [1, 20, 21, 3, 22, 4, 23, 2, self.end, 5, 6, 0]   # BOS, text, USER, text, ASSISTANT, text, EOS, the real end piece, SYSTEM, UNK, PAD
        np.testing.assert_allclose(engine_logits(self.out, ids), reference_logits(self.state, ids, self.tokenizer), atol=2e-4)

    def test_bos_eos_and_the_real_end_piece_share_a_row_so_only_eos_may_stop_a_reply(self):
        from studio_model import complete, unsayable
        blocked = unsayable(self.tokenizer)
        self.assertIn(self.end, blocked)
        self.assertNotIn(EOS, blocked)
        self.assertEqual(unsayable(Tokenizer("byte")), unsayable(Tokenizer("byte")))
        self.assertNotIn(self.end, unsayable(Tokenizer("byte")))
        logits = np.full(self.tokenizer.vocab_size, -50.)
        logits[[self.end, EOS]] = 10, 9   # the real piece is the likeliest, EOS next: the reply must stop, not print the piece
        local = self.local()
        with mock.patch("studio_model.last_logits", lambda *a, **k: torch.as_tensor(logits)):
            self.assertEqual(complete(local.model, local.tokenizer, "hi"), "")
            self.assertEqual(local.generate([{"role": "user", "content": "hi"}], temperature=0), "")

    def test_sampling_does_not_count_the_end_of_text_twice(self):
        logits = np.full(self.tokenizer.vocab_size, -50.)
        logits[[self.end, EOS, OFFSET + 97]] = 2.   # three equally likely pieces; the real end piece is not one of the choices
        local, stopped = self.local(seed=7), 0
        with mock.patch("studio_model.last_logits", lambda *a, **k: torch.as_tensor(logits)):
            for _ in range(400):
                reply = local.generate([{"role": "user", "content": "hi"}], max_tokens=1, temperature=1., top_k=0)
                self.assertNotIn("<|endoftext|>", reply)
                stopped += reply == ""
        self.assertAlmostEqual(stopped / 400, .5, delta=.08)   # EOS against "a"; counted twice it would be two in three

    def test_held_back_scoring_shares_the_probability_among_the_real_pieces_only(self):
        from studio_eval import score_docs
        text = "hello that tell"
        pieces = self.tokenizer.encode(text)
        ids = [1] + pieces
        logits = reference_logits(self.state, ids[:-1], self.tokenizer)
        targets = np.array(pieces)

        def nats(columns):
            kept = logits[:, columns]
            kept = kept - np.log(np.exp(kept).sum(-1, keepdims=True))
            return float(-kept[np.arange(len(targets)), targets - columns.start].sum())

        (got, chars), = score_docs(self.local(), [text])
        self.assertEqual(chars, len(text))
        self.assertAlmostEqual(got, nats(slice(OFFSET, None)), delta=2e-3)
        self.assertGreater(abs(nats(slice(0, None)) - got), 1e-3)   # counting the special rows would have scored it differently


class FakeResponse:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status, self.headers, self.at = data, status, headers or {}, 0

    def read(self, n=-1):
        chunk = self.data[self.at:self.at + (n if n and n > 0 else len(self.data))]
        self.at += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


PAYLOAD = bytes(range(256)) * 40


class Download(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "gpt2" / "model.safetensors"
        self.part = Path(str(self.dest) + ".part")
        self.args = dict(size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest(), url="https://example.invalid/m")
        self.requests = []

    def tearDown(self):
        self.tmp.cleanup()

    def serve(self, data=PAYLOAD, status=200, headers=None):
        def urlopen(request, timeout=None):
            self.requests.append(request)
            return FakeResponse(data, status, headers)
        return mock.patch("urllib.request.urlopen", urlopen)

    def serve_each(self, *replies):
        """A server whose n-th reply is replies[n]: a FakeResponse, or an exception to raise."""
        queue = list(replies)

        def urlopen(request, timeout=None):
            self.requests.append(request)
            reply = queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return mock.patch("urllib.request.urlopen", urlopen)

    def refused(self, code):
        return urllib.error.HTTPError(self.args["url"], code, "refused", {}, None)

    def partial(self, start):
        end = len(PAYLOAD) - 1
        return FakeResponse(PAYLOAD[start:], 206, {"Content-Range": f"bytes {start}-{end}/{len(PAYLOAD)}"})

    def test_a_good_download_is_written_whole_and_atomically(self):
        seen = []
        with self.serve():
            path = imp.fetch_gpt2(self.dest, progress=lambda done, total: seen.append((done, total)), **self.args)
        self.assertEqual(path, self.dest)
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)
        self.assertFalse(self.part.exists())
        self.assertEqual(seen[-1], (len(PAYLOAD), len(PAYLOAD)))

    def test_a_file_already_there_and_whole_is_not_downloaded_again(self):
        self.dest.parent.mkdir(parents=True)
        self.dest.write_bytes(PAYLOAD)
        with self.serve():
            imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual(self.requests, [])

    def test_a_short_download_is_an_error_and_is_kept_to_continue(self):
        with self.serve(PAYLOAD[:1000]):
            with self.assertRaises(ValueError) as caught:
                imp.fetch_gpt2(self.dest, **self.args)
        self.assertIn("stopped early", str(caught.exception))
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.part.stat().st_size, 1000)

    def test_an_interrupted_download_resumes_with_a_range_request(self):
        self.dest.parent.mkdir(parents=True)
        self.part.write_bytes(PAYLOAD[:3000])
        with self.serve_each(self.partial(3000)):
            imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual(self.requests[0].get_header("Range"), "bytes=3000-")
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)
        self.assertFalse(self.part.exists())

    def test_a_leftover_the_server_cannot_continue_starts_over_cleanly(self):
        self.dest.parent.mkdir(parents=True)
        self.part.write_bytes(PAYLOAD[:3000])
        with self.serve_each(self.refused(416), FakeResponse(PAYLOAD)):   # 416: no bytes at that offset
            imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual([r.get_header("Range") for r in self.requests], ["bytes=3000-", None])
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_a_416_with_nothing_to_continue_is_an_error_not_a_loop(self):
        with self.serve_each(self.refused(416)):
            with self.assertRaises(ValueError):
                imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual(len(self.requests), 1)

    def test_bytes_that_start_at_the_wrong_place_are_not_glued_on(self):
        self.dest.parent.mkdir(parents=True)
        self.part.write_bytes(PAYLOAD[:3000])
        with self.serve_each(self.partial(0), FakeResponse(PAYLOAD)):   # the 206 starts at 0, not 3000
            imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual([r.get_header("Range") for r in self.requests], ["bytes=3000-", None])
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)
        self.dest.unlink()
        self.part.write_bytes(PAYLOAD[:3000])
        with self.serve_each(FakeResponse(PAYLOAD[3000:], 206), FakeResponse(PAYLOAD)):   # no Content-Range at all
            imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_a_whole_leftover_is_kept_and_a_wrong_or_longer_one_is_thrown_away(self):
        self.dest.parent.mkdir(parents=True)
        self.part.write_bytes(PAYLOAD)
        with self.serve():
            imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual(self.requests, [])   # already whole and right: finished without asking
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)
        self.dest.unlink()
        for stale in (b"x" * len(PAYLOAD), PAYLOAD + b"more"):
            self.part.write_bytes(stale)
            with self.serve():
                imp.fetch_gpt2(self.dest, **self.args)
            self.assertEqual(self.requests[-1].get_header("Range"), None)
            self.assertEqual(self.dest.read_bytes(), PAYLOAD)
            self.dest.unlink()

    def test_a_full_disk_says_so(self):
        class Full(FakeResponse):
            def read(self, n=-1):
                raise OSError(errno.ENOSPC, "No space left on device")
        with self.serve_each(Full(PAYLOAD)):
            with self.assertRaises(ValueError) as caught:
                imp.fetch_gpt2(self.dest, **self.args)
        self.assertIn("free disk space", str(caught.exception))

    def test_a_connection_cut_mid_reply_is_plain_words(self):
        class Cut(FakeResponse):
            def read(self, n=-1):
                raise http.client.IncompleteRead(b"abc", 100)
        with self.serve_each(Cut(PAYLOAD)):
            with self.assertRaises(ValueError) as caught:
                imp.fetch_gpt2(self.dest, **self.args)
        self.assertIn("connection", str(caught.exception))

    def test_stopping_while_the_file_is_checked_returns_nothing_and_keeps_it(self):
        stop = threading.Event()
        stop.set()
        self.dest.parent.mkdir(parents=True)
        self.dest.write_bytes(PAYLOAD)
        with self.serve():
            self.assertIsNone(imp.fetch_gpt2(self.dest, stop=stop, **self.args))
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)   # it was fine; stopping must not delete it
        self.dest.unlink()
        self.part.write_bytes(PAYLOAD)
        with self.serve():
            self.assertIsNone(imp.fetch_gpt2(self.dest, stop=stop, **self.args))
        self.assertTrue(self.part.exists())
        self.assertEqual(self.requests, [])

    def test_a_server_that_ignores_the_range_starts_over(self):
        self.dest.parent.mkdir(parents=True)
        self.part.write_bytes(b"junk" * 100)
        with self.serve(PAYLOAD, status=200):
            imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_a_wrong_checksum_throws_the_file_away(self):
        corrupt = bytes([PAYLOAD[0] ^ 1]) + PAYLOAD[1:]
        with self.serve(corrupt):
            with self.assertRaises(ValueError) as caught:
                imp.fetch_gpt2(self.dest, **self.args)
        self.assertIn("checksum", str(caught.exception))
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.part.exists())

    def test_a_network_failure_is_plain_words(self):
        def broken(request, timeout=None):
            raise OSError("no route")
        with mock.patch("urllib.request.urlopen", broken):
            with self.assertRaises(ValueError) as caught:
                imp.fetch_gpt2(self.dest, **self.args)
        self.assertIn("internet connection", str(caught.exception))
        self.assertFalse(self.dest.exists())

    def test_stopping_returns_nothing_and_keeps_what_arrived(self):
        stop = threading.Event()
        stop.set()
        with self.serve():
            self.assertIsNone(imp.fetch_gpt2(self.dest, stop=stop, **self.args))
        self.assertFalse(self.dest.exists())

    def test_the_pinned_download_is_a_revision_of_the_official_repository(self):
        self.assertIn(imp.GPT2_REVISION, imp.GPT2_URL)
        self.assertTrue(imp.GPT2_URL.startswith("https://huggingface.co/openai-community/gpt2/resolve/"))
        self.assertEqual(len(imp.GPT2_SHA256), 64)
        self.assertEqual(imp.MODELS["gpt2"]["size"], imp.GPT2_SIZE)


class Job(unittest.TestCase):
    def test_run_import_makes_a_finished_run_directory_and_reports_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tokenizer, spec, source, hf, weights = hf_file(tmp)
            events, out = [], tmp / "run"
            with mock.patch("studio_import.fetch_gpt2", return_value=hf), \
                    mock.patch("studio_modules.tokenizer.bpe_gpt2.load_assets", return_value=tokenizer), \
                    mock.patch.dict(imp.HEADS, {32: HEADS}):
                record = imp.run_import("gpt2", tmp / "cache", out, emit=events.append)
            self.assertEqual(record["status"], "completed")
            self.assertTrue((out / "last.npz").exists())
            self.assertEqual(json.loads((out / "run.json").read_text())["imported_from"], "openai-community/gpt2")
            self.assertEqual(json.loads((out / "spec.json").read_text())["tokenizer"]["id"], "bpe_gpt2")
            self.assertEqual(events[-1]["type"], "done")
            self.assertTrue(any(event.get("phase") == "Converting to a studio checkpoint" for event in events))

    def test_stopping_during_the_download_leaves_no_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("studio_import.fetch_gpt2", return_value=None):
                self.assertIsNone(imp.run_import("gpt2", Path(tmp) / "cache", Path(tmp) / "run"))
            self.assertFalse((Path(tmp) / "run" / "last.npz").exists())

    def test_stopping_during_the_conversion_writes_nothing(self):
        stop = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tokenizer, _, _, hf, _ = hf_file(tmp)

            def fetched(*args, **kwargs):
                stop.set()   # the stop arrives after the download, while the file is being converted
                return hf
            with mock.patch("studio_import.fetch_gpt2", fetched), \
                    mock.patch("studio_modules.tokenizer.bpe_gpt2.load_assets", return_value=tokenizer), \
                    mock.patch.dict(imp.HEADS, {32: HEADS}):
                self.assertIsNone(imp.run_import("gpt2", tmp / "cache", tmp / "run", stop=stop))
            self.assertFalse((tmp / "run").exists())

    def test_an_unknown_model_is_refused(self):
        with self.assertRaises(ValueError):
            imp.run_import("nope", "cache", "run")


class Services(unittest.TestCase):
    def test_the_fake_services_save_a_gpt2_version(self):
        svc = FakeServices()
        version = svc.import_pretrained("gpt2")
        self.assertEqual(svc.imports, ["gpt2"])
        self.assertEqual(version.spec["tokenizer"]["id"], "bpe_gpt2")

    def test_the_queue_downloads_converts_and_the_version_ends_up_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer, _, _, hf, _ = hf_file(tmp)
            svc = FixtureServices(str(Path(tmp) / "workspace"))
            self.addCleanup(svc.close)
            before = len(svc.versions())
            with mock.patch("studio_import.fetch_gpt2", return_value=hf), \
                    mock.patch("studio_modules.tokenizer.bpe_gpt2.load_assets", return_value=tokenizer), \
                    mock.patch.dict(imp.HEADS, {32: HEADS}):
                version = svc.import_pretrained("gpt2")
                self.assertEqual(len(svc.versions()), before + 1)
                deadline = time.time() + 60
                while time.time() < deadline and svc.store.get(version.number)["status"] not in ("trained", "saved"):
                    time.sleep(.1)
            row = svc.store.get(version.number)
            self.assertEqual(row["status"], "trained")
            self.assertTrue(Path(row["best_checkpoint"]).exists())
            self.assertEqual(Path(row["best_checkpoint"]).name, "last.npz")

    def import_with(self, tmp, fetch):
        """Workspace services whose GPT-2 download is `fetch`."""
        svc = FixtureServices(str(Path(tmp) / "workspace"))
        self.addCleanup(svc.close)
        patch = mock.patch("studio_import.fetch_gpt2", fetch)
        patch.start()
        self.addCleanup(patch.stop)
        return svc

    def wait_for(self, condition):
        deadline = time.time() + 30
        while time.time() < deadline and not condition():
            time.sleep(.05)
        self.assertTrue(condition())

    def test_a_failed_download_removes_the_empty_version(self):
        def failing(*args, **kwargs):
            raise ValueError("Could not download GPT-2.")
        with tempfile.TemporaryDirectory() as tmp:
            svc = self.import_with(tmp, failing)
            before = len(svc.versions())
            svc.import_pretrained("gpt2")
            self.wait_for(lambda: svc.runner.jobs and svc.runner.jobs[-1].status == "failed")
            self.wait_for(lambda: len(svc.versions()) == before)
            self.assertEqual(len(svc.versions()), before)

    def test_a_stopped_import_removes_the_empty_version(self):
        started = threading.Event()

        def waits(dest, progress=None, stop=None, **kwargs):
            started.set()
            stop.wait(30)
            return None
        with tempfile.TemporaryDirectory() as tmp:
            svc = self.import_with(tmp, waits)
            before = len(svc.versions())
            svc.import_pretrained("gpt2")
            self.assertTrue(started.wait(30))
            self.assertEqual(len(svc.versions()), before + 1)   # listed at once, as the copy says
            svc.remove(svc.runner.jobs[-1].id)                  # stops the running job
            self.wait_for(lambda: len(svc.versions()) == before)

    def test_an_older_empty_version_is_marked_because_numbers_cannot_be_reused(self):
        release = threading.Event()

        def failing(*args, **kwargs):
            release.wait(30)
            raise ValueError("Could not download GPT-2.")
        with tempfile.TemporaryDirectory() as tmp:
            svc = self.import_with(tmp, failing)
            version = svc.import_pretrained("gpt2")
            later = svc.save_version(imp.gpt2_spec(name="Another"), "A later design")
            release.set()
            self.wait_for(lambda: svc.store.get(version.number)["status"] == "saved")
            kept = svc.store.get(version.number)
            self.assertIn("no weights", kept["change"])
            self.assertIn("did not finish", kept["change"])
            self.assertEqual(svc.store.get(later.number)["change"], "A later design")
            self.assertEqual(len(svc.versions()), later.number)

    def test_a_second_import_is_refused_while_one_is_waiting_or_running(self):
        release = threading.Event()

        def slow(dest, progress=None, stop=None, **kwargs):
            release.wait(30)
            return None
        with tempfile.TemporaryDirectory() as tmp:
            svc = self.import_with(tmp, slow)
            self.addCleanup(release.set)
            svc.import_pretrained("gpt2")
            count = len(svc.versions())
            with self.assertRaises(ValueError) as caught:
                svc.import_pretrained("gpt2")
            self.assertIn("already being imported", str(caught.exception))
            self.assertEqual(len(svc.versions()), count)   # and it saved no second version
            self.assertEqual(len([j for j in svc.runner.jobs if j.kind == "import_pretrained"]), 1)

    def test_the_saved_design_fine_tunes_gently(self):
        version = FakeServices().import_pretrained("gpt2")
        self.assertEqual(version.spec["train"]["optimizer"]["config"]["lr"], imp.FINE_TUNE_LR)


if __name__ == "__main__":
    unittest.main()

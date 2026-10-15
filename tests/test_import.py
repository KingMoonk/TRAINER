"""Import (studio_import): a Hugging Face GPT-2 file becomes a studio checkpoint that computes the same function; the download is safe."""
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import studio_export as export
import studio_import as imp
from studio_modules.tokenizer.bpe_gpt2 import GPT2Tokenizer, bytes_to_unicode
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

    def test_gpt2_small_is_a_valid_design_of_about_124_million_parameters(self):
        spec = imp.gpt2_spec()
        self.assertEqual(export.gpt2_problems(spec), [])
        self.assertEqual((spec["dims"]["d_model"], spec["dims"]["n_layer"], spec["dims"]["n_ctx"]), (768, 12, 1024))
        self.assertEqual(spec["tokenizer"]["id"], "bpe_gpt2")


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status, self.at = data, status, 0

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

    def serve(self, data=PAYLOAD, status=200):
        def urlopen(request, timeout=None):
            self.requests.append(request)
            return FakeResponse(data, status)
        return mock.patch("urllib.request.urlopen", urlopen)

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
        with self.serve(PAYLOAD[3000:], status=206):
            imp.fetch_gpt2(self.dest, **self.args)
        self.assertEqual(self.requests[0].get_header("Range"), "bytes=3000-")
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)
        self.assertFalse(self.part.exists())

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


if __name__ == "__main__":
    unittest.main()

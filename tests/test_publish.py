"""studio_publish: a version leaves one machine as small half-precision weights and arrives on another as a plain checkpoint."""
import json
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import studio_publish
from studio_model import load_checkpoint, train
from studio_publish import FolderStore, RemoteStore, S3Store, build_artifact, sha256_of
from studio_versions import NoCheckpoint

WORDS = ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")
EXAMPLES = [{"user": f"Copy: {w}", "assistant": w} for w in WORDS] * 3
TUNING = [{"user": f"Copy: {w}", "assistant": w[::-1]} for w in WORDS] * 3
TINY = dict(device="cpu", width=32, layers=2, heads=2, context=32, expansion=2, dropout=0, batch_size=4,
            steps=30, learning_rate=.004, warmup=0, validation=0, eval_every=10, objective="assistant")
VERSION = {"number": 3, "name": "Version 3", "change": "more blocks (2 → 3)", "parent": 2,
           "spec": {"dims": {"n_layer": 3}}, "scorecard": {"bpc": 1.91, "error": .4}}


def close(actual, expected, **kwargs):
    """float16 keeps 11 bits: a relative error under 2e-3 for ordinary weights, and under 1e-7 absolute for the tiny ones."""
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-7, **kwargs)


def merged_close(actual, expected, **kwargs):
    """A merged LoRA weight is half-rounded base + adapter product: the base's rounding (up to 2^-11 of its size, about 1e-4 here)
    shows as an absolute error even where the sum is small."""
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4, **kwargs)


class Checkpoints(unittest.TestCase):
    """A base model and a LoRA tune of it, trained once for the whole module."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        train(TINY, EXAMPLES, cls.dir / "base")
        train(dict(TINY, steps=20, learning_rate=.01, lora={"rank": 4, "alpha": 8}), TUNING, cls.dir / "tuned",
              init_from=cls.dir / "base" / "last.npz")
        cls.base, cls.tuned = cls.dir / "base" / "last.npz", cls.dir / "tuned" / "last.npz"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        (self.scratch / "shared").mkdir()
        (self.scratch / "dest").mkdir()
        self.shared, self.dest = self.scratch / "shared", self.scratch / "dest"


class Artifact(Checkpoints):
    def build(self, checkpoint=None, origin="mac"):
        return build_artifact(VERSION, checkpoint or self.base, self.scratch, origin)

    def test_the_artifact_drops_the_optimizer_and_halves_the_weights(self):
        source = np.load(self.base, allow_pickle=False)
        self.assertTrue(any(name.startswith("adam_") for name in source.files))
        artifact, meta = self.build()
        with np.load(artifact, allow_pickle=False) as published:
            self.assertFalse(any(name.startswith("adam_") for name in published.files))
            weights = [name for name in published.files if name.startswith("weight::")]
            self.assertEqual(sorted(weights), sorted(name for name in source.files if name.startswith("weight::")))
            for name in weights:
                self.assertEqual(published[name].dtype, np.float16, name)
                close(published[name].astype(np.float32), source[name], err_msg=name)
            inner = json.loads(str(published["metadata"].item()))
        self.assertEqual(inner["published"], {"weights_only": True, "stored_dtype": "float16"})
        self.assertEqual({key: inner[key] for key in ("config", "tokenizer", "step", "device")},
                         {key: json.loads(str(source["metadata"].item()))[key] for key in ("config", "tokenizer", "step", "device")})

    def test_the_metadata_says_what_a_node_needs_to_list_it(self):
        artifact, meta = self.build(origin="Mac Studio.local")
        self.assertEqual((meta["number"], meta["name"], meta["parent"], meta["spec"], meta["scorecard"]),
                         (3, "Version 3", 2, VERSION["spec"], VERSION["scorecard"]))
        self.assertEqual(meta["origin"], "mac-studio.local")
        self.assertEqual(meta["device"], "cpu")  # the checkpoint records the device that trained it
        self.assertEqual(meta["weights_dtype"], "float16")
        self.assertEqual((meta["sha256"], meta["size"]), (sha256_of(artifact), artifact.stat().st_size))
        self.assertEqual(meta["key"], f"mac-studio.local/v3-{meta['sha256'][:12]}")
        self.assertRegex(meta["published_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        json.dumps(meta)

    def test_a_checkpoint_with_no_device_recorded_has_none(self):
        arrays = dict(np.load(self.base, allow_pickle=False))
        inner = json.loads(str(arrays["metadata"].item()))
        del inner["device"]
        arrays["metadata"] = np.array(json.dumps(inner))
        np.savez(self.scratch / "old.npz", **arrays)
        self.assertIsNone(self.build(self.scratch / "old.npz")[1]["device"])

    def test_the_published_file_is_well_under_the_checkpoint(self):
        artifact, meta = self.build()
        full = self.base.stat().st_size
        self.assertLess(meta["size"], .4 * full, f"{meta['size']} of {full} bytes")

    def test_the_same_checkpoint_always_gives_the_same_bytes_and_key(self):
        first = self.build()
        (self.scratch / "again").mkdir()
        second = build_artifact(VERSION, self.base, self.scratch / "again", "mac")
        self.assertEqual((first[1]["sha256"], first[1]["key"]), (second[1]["sha256"], second[1]["key"]))
        self.assertNotEqual(self.build(self.tuned)[1]["key"], first[1]["key"])

    def test_lora_adapters_are_kept_whole(self):
        artifact, _ = self.build(self.tuned)
        with np.load(self.tuned, allow_pickle=False) as source, np.load(artifact, allow_pickle=False) as published:
            adapters = [name for name in source.files if name.startswith("lora::")]
            self.assertEqual(len(adapters), 2 * 2 * 2 + 1)
            for name in adapters:
                np.testing.assert_array_equal(published[name], source[name], err_msg=name)
                self.assertEqual(published[name].dtype, source[name].dtype)

    def test_trained_weights_sit_far_inside_what_float16_holds(self):
        """Why float16 and not bfloat16 bits: nothing a trained model has needs the wider exponent range."""
        weights = load_checkpoint(self.tuned, weights_only=True, merged=False)["model"]
        peak = max(float(np.abs(value).max()) for value in weights.values())
        self.assertLess(peak, studio_publish.HALF_MAX / 1000)
        for name, value in weights.items():
            half = value.astype(np.float16)
            self.assertTrue(np.isfinite(half).all(), name)
            self.assertFalse(((half == 0) & (value != 0) & (np.abs(value) > 1e-7)).any(), name)  # nothing real flushes to zero
            close(half.astype(np.float32), value, err_msg=name)

    def test_a_weight_beyond_half_precision_is_refused_in_plain_words(self):
        arrays = dict(np.load(self.base, allow_pickle=False))
        arrays["weight::block0.qkv.w"] = arrays["weight::block0.qkv.w"] * 0 + np.float32(7e4)
        np.savez(self.scratch / "diverged.npz", **arrays)
        with self.assertRaises(ValueError) as caught:
            self.build(self.scratch / "diverged.npz")
        self.assertIn("block0.qkv.w", str(caught.exception))
        self.assertIn("cannot be published", str(caught.exception))

    def test_no_checkpoint_a_missing_file_and_a_foreign_archive_are_plain_errors(self):
        with self.assertRaises(NoCheckpoint):
            build_artifact(VERSION, None, self.scratch)
        with self.assertRaises(FileNotFoundError):
            build_artifact(VERSION, self.scratch / "gone.npz", self.scratch)
        np.savez(self.scratch / "other.npz", metadata=np.array(json.dumps({"architecture": "something_else"})))
        with self.assertRaises(ValueError):
            build_artifact(VERSION, self.scratch / "other.npz", self.scratch)


class FolderRoundTrip(Checkpoints):
    def test_a_published_version_is_listed_and_pulled_as_a_checkpoint_load_checkpoint_accepts(self):
        store = FolderStore(self.shared, origin="mac")
        self.assertIsInstance(store, RemoteStore)
        self.assertEqual(store.list(), [])
        key = store.publish(VERSION, self.base)
        self.assertTrue(store.exists(key))
        self.assertFalse(store.exists("mac/v3-000000000000"))
        (entry,) = store.list()
        self.assertEqual((entry["key"], entry["number"], entry["origin"], entry["scorecard"]), (key, 3, "mac", VERSION["scorecard"]))
        pulled = store.pull(key, self.dest / "incoming")
        self.assertEqual(pulled.parent, self.dest / "incoming")
        self.assertEqual(json.loads(pulled.with_suffix(".json").read_text())["key"], key)
        loaded = load_checkpoint(pulled, weights_only=True)
        original = load_checkpoint(self.base)
        self.assertEqual(loaded["step"], original["step"])
        self.assertEqual(loaded["config"], original["config"])
        self.assertEqual(loaded["optimizer_state"], {})  # weights only: a resume restarts the optimizer
        self.assertEqual(set(loaded["model"]), set(original["model"]))
        for name, value in loaded["model"].items():
            self.assertEqual(value.dtype, np.float32, name)  # widened again on pull
            close(value, original["model"][name], err_msg=name)

    def test_a_lora_version_pulls_with_its_adapters_and_merges_the_same(self):
        store = FolderStore(self.shared, origin="mac")
        pulled = store.pull(store.publish(VERSION, self.tuned), self.dest)
        raw = load_checkpoint(pulled, weights_only=True, merged=False)
        original = load_checkpoint(self.tuned, merged=False)
        self.assertEqual({name for name in raw["model"] if name.startswith("lora.")},
                         {name for name in original["model"] if name.startswith("lora.")})
        merged, expected = load_checkpoint(pulled, weights_only=True)["model"], load_checkpoint(self.tuned)["model"]
        self.assertEqual(set(merged), set(expected))
        for name, value in merged.items():
            merged_close(value, expected[name], err_msg=name)

    def test_publishing_the_same_checkpoint_again_changes_nothing(self):
        store = FolderStore(self.shared, origin="mac")
        key = store.publish(VERSION, self.base)
        stamp = (self.shared / f"{key}.json").read_text()
        self.assertEqual(store.publish(VERSION, self.base), key)
        self.assertEqual((self.shared / f"{key}.json").read_text(), stamp)
        self.assertEqual(len(store.list()), 1)
        self.assertEqual(sorted(path.name for path in (self.shared / "mac").iterdir()), [f"{key.split('/')[1]}.json", f"{key.split('/')[1]}.npz"])

    def test_two_machines_can_publish_the_same_version_number(self):
        mac, windows = FolderStore(self.shared, origin="mac"), FolderStore(self.shared, origin="windows")
        a, b = mac.publish(VERSION, self.base), windows.publish(VERSION, self.tuned)
        self.assertNotEqual(a, b)
        self.assertEqual({entry["origin"] for entry in FolderStore(self.shared).list()}, {"mac", "windows"})
        self.assertTrue(FolderStore(self.shared).exists(b))

    def test_an_upload_that_never_finished_is_not_listed_and_is_redone(self):
        store = FolderStore(self.shared, origin="mac")
        artifact, meta = build_artifact(VERSION, self.base, self.scratch, "mac")
        (self.shared / "mac").mkdir()
        (self.shared / f"{meta['key']}.npz").write_bytes(artifact.read_bytes()[:100])  # the weights came part way, no .json
        self.assertEqual(store.list(), [])
        self.assertFalse(store.exists(meta["key"]))
        self.assertEqual(store.publish(VERSION, self.base), meta["key"])
        self.assertEqual(sha256_of(self.shared / f"{meta['key']}.npz"), meta["sha256"])

    def test_a_download_that_does_not_match_its_hash_is_refused(self):
        store = FolderStore(self.shared, origin="mac")
        key = store.publish(VERSION, self.base)
        path = self.shared / f"{key}.npz"
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaises(ValueError):
            store.pull(key, self.dest)
        self.assertEqual([p for p in self.dest.iterdir()], [])  # nothing half-written is left in the destination

    def test_only_keys_the_store_made_are_accepted(self):
        store = FolderStore(self.shared, origin="mac")
        for bad in ("../outside/v1-abc", "/etc/passwd", "mac/../v1-abc", "", "mac\\v1", ".hidden/v1", None):
            with self.assertRaises(ValueError, msg=bad):
                store.exists(bad)
            with self.assertRaises(ValueError, msg=bad):
                store.pull(bad, self.dest)
        with self.assertRaises(KeyError):
            store.pull("mac/v9-000000000000", self.dest)

    def test_an_unconnected_shared_folder_says_so(self):
        store = FolderStore(self.scratch / "not-mounted", origin="mac")
        self.assertEqual(store.list(), [])
        with self.assertRaisesRegex(FileNotFoundError, "not there"):
            store.publish(VERSION, self.base)
        self.assertFalse((self.scratch / "not-mounted").exists())

    def test_files_in_the_folder_that_are_not_versions_are_ignored(self):
        store = FolderStore(self.shared, origin="mac")
        key = store.publish(VERSION, self.base)
        (self.shared / "notes.json").write_text("{}")
        (self.shared / "mac" / "junk.json").write_text("not json")
        (self.shared / "mac" / "x.json.tmp").write_text("{}")
        self.assertEqual([entry["key"] for entry in store.list()], [key])


class StubError(Exception):
    """What botocore's ClientError carries: the response dict with the error code."""

    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class StubS3:
    """The six boto3 client methods the store calls, over a dict. list_objects_v2 pages one key at a time to exercise paging."""

    def __init__(self):
        self.objects, self.log, self.fail_head = {}, [], None
        self.sizes = {}  # key -> the size list_objects_v2 reports instead of the real one: a bucket that is nearly full, without the bytes
        self.pending = {}  # (key, upload id) -> the sizes of the parts of a multipart upload that was started and not finished

    def upload_file(self, filename, bucket, key):
        self.log.append(("upload_file", bucket, key))
        self.objects[bucket, key] = Path(filename).read_bytes()

    def put_object(self, Bucket, Key, Body, **extra):
        self.log.append(("put_object", Bucket, Key))
        self.objects[Bucket, Key] = bytes(Body)

    def download_file(self, bucket, key, filename):
        self.log.append(("download_file", bucket, key))
        if (bucket, key) not in self.objects:
            raise StubError("404")
        Path(filename).write_bytes(self.objects[bucket, key])

    def get_object(self, Bucket, Key):
        self.log.append(("get_object", Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise StubError("NoSuchKey")

        class Body:
            def read(_):
                return self.objects[Bucket, Key]
        return {"Body": Body()}

    def head_object(self, Bucket, Key):
        self.log.append(("head_object", Bucket, Key))
        if self.fail_head:
            raise StubError(self.fail_head)
        if (Bucket, Key) not in self.objects:
            raise StubError("404")
        return {}

    def delete_object(self, Bucket, Key):
        self.log.append(("delete_object", Bucket, Key))
        self.objects.pop((Bucket, Key), None)

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        self.log.append(("list_objects_v2", Bucket, Prefix))
        keys = sorted(key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + 1]
        more = start + 1 < len(keys)
        return {"Contents": [{"Key": key, "Size": self.sizes.get(key, len(self.objects[Bucket, key]))} for key in page], "IsTruncated": more,
                **({"NextContinuationToken": str(start + 1)} if more else {})}

    def list_multipart_uploads(self, Bucket, KeyMarker=None, UploadIdMarker=None):
        self.log.append(("list_multipart_uploads", Bucket))
        ordered = sorted(self.pending)
        start = ordered.index((KeyMarker, UploadIdMarker)) + 1 if KeyMarker else 0
        page, more = ordered[start:start + 1], start + 1 < len(ordered)
        return {"Uploads": [{"Key": key, "UploadId": upload} for key, upload in page], "IsTruncated": more,
                **({"NextKeyMarker": page[-1][0], "NextUploadIdMarker": page[-1][1]} if more else {})}

    def list_parts(self, Bucket, Key, UploadId, PartNumberMarker=0):
        self.log.append(("list_parts", Bucket, Key))
        if (Key, UploadId) not in self.pending:
            raise StubError("NoSuchUpload")
        sizes, start = self.pending[Key, UploadId], int(PartNumberMarker or 0)
        page, more = sizes[start:start + 1], start + 1 < len(sizes)
        return {"Parts": [{"PartNumber": start + 1, "Size": size} for size in page], "IsTruncated": more,
                "NextPartNumberMarker": start + 1}  # S3 sends this on the last page too


CONFIG = {"endpoint": "https://abc.r2.cloudflarestorage.com", "bucket": "versions", "prefix": "/trainer/", "key_id": "AKID",
          "secret": "s3cr3t-value"}


class Bucket(Checkpoints):
    def store(self, **config):
        self.client = StubS3()
        return S3Store({**CONFIG, **config}, client=self.client, origin="windows")

    def test_a_published_version_round_trips_through_the_bucket(self):
        store = self.store()
        self.assertIsInstance(store, RemoteStore)
        key = store.publish(VERSION, self.tuned)
        self.assertEqual(sorted(self.client.objects), [("versions", f"trainer/{key}.json"), ("versions", f"trainer/{key}.npz")])
        puts = [call[2] for call in self.client.log if call[0] in ("upload_file", "put_object")]
        self.assertEqual(puts, [f"trainer/{key}.npz", f"trainer/{key}.json"])  # the weights first: the .json is what lists the version
        (entry,) = store.list()
        self.assertEqual((entry["key"], entry["origin"], entry["number"]), (key, "windows", 3))
        self.assertTrue(store.exists(key))
        pulled = store.pull(key, self.dest)
        merged, expected = load_checkpoint(pulled, weights_only=True)["model"], load_checkpoint(self.tuned)["model"]
        for name, value in merged.items():
            self.assertEqual(value.dtype, np.float32, name)
            merged_close(value, expected[name], err_msg=name)

    def test_what_goes_up_is_the_half_precision_weights_and_nothing_else(self):
        store = self.store()
        key = store.publish(VERSION, self.base)
        uploaded = self.client.objects["versions", f"trainer/{key}.npz"]
        self.assertLess(len(uploaded), .4 * self.base.stat().st_size)
        (self.scratch / "up.npz").write_bytes(uploaded)
        with np.load(self.scratch / "up.npz", allow_pickle=False) as published:
            self.assertFalse(any(name.startswith("adam_") for name in published.files))
            self.assertTrue(all(published[name].dtype == np.float16 for name in published.files if name.startswith("weight::")))
        self.assertEqual(json.loads(self.client.objects["versions", f"trainer/{key}.json"])["sha256"], sha256_of(self.scratch / "up.npz"))

    def test_publishing_twice_uploads_once(self):
        store = self.store()
        key = store.publish(VERSION, self.base)
        before = len([call for call in self.client.log if call[0] in ("upload_file", "put_object")])
        self.assertEqual(store.publish(VERSION, self.base), key)
        self.assertEqual(len([call for call in self.client.log if call[0] in ("upload_file", "put_object")]), before)

    def test_listing_follows_every_page_and_ignores_other_objects(self):
        store = self.store()
        keys = {store.publish(VERSION, self.base), store.publish(dict(VERSION, number=4), self.base), store.publish(VERSION, self.tuned)}
        self.client.objects["versions", "trainer/readme.txt"] = b"hi"
        self.client.objects["versions", "elsewhere/mac/v1-abcdabcdabcd.json"] = b"{}"
        self.assertEqual({entry["key"] for entry in store.list()}, keys)
        self.assertGreater(len([call for call in self.client.log if call[0] == "list_objects_v2"]), 3)

    def test_without_a_prefix_objects_sit_at_the_top_of_the_bucket(self):
        store = self.store(prefix="")
        key = store.publish(VERSION, self.base)
        self.assertIn(("versions", f"{key}.json"), self.client.objects)
        self.assertEqual([entry["key"] for entry in store.list()], [key])

    def test_a_missing_key_is_not_found_and_other_errors_are_not_hidden(self):
        store = self.store()
        with self.assertRaises(KeyError):
            store.pull("windows/v9-000000000000", self.dest)
        self.assertFalse(store.exists("windows/v9-000000000000"))
        self.client.fail_head = "403"
        with self.assertRaises(StubError):
            store.exists("windows/v9-000000000000")

    def test_a_pulled_file_that_is_not_what_was_published_is_refused(self):
        store = self.store()
        key = store.publish(VERSION, self.base)
        self.client.objects["versions", f"trainer/{key}.npz"] += b"x"
        with self.assertRaises(ValueError):
            store.pull(key, self.dest)
        self.assertEqual([p for p in self.dest.iterdir()], [])


class CheckpointMetadata(unittest.TestCase):
    def test_a_checkpoint_is_told_from_a_version_by_its_name_and_change(self):
        row = studio_publish.checkpoint_version("ab12", 40)
        self.assertEqual(studio_publish.checkpoint_job(row), "ab12")
        self.assertIsNone(studio_publish.checkpoint_job({**row, "change": "result at step 40"}))
        self.assertIsNone(studio_publish.checkpoint_job({"name": "Version 3", "change": "checkpoint at step 4"}))
        self.assertIsNone(studio_publish.checkpoint_job({"name": "job:", "change": "checkpoint at step 4"}))
        self.assertIsNone(studio_publish.checkpoint_job(None))

    def test_metadata_is_none_when_missing_and_empty_when_unreadable(self):
        client = StubS3()
        store = S3Store(CONFIG, client=client)
        self.assertIsNone(store.metadata("gpu-box/v0-abc"))
        client.objects["versions", "trainer/gpu-box/v0-abc.json"] = b"not json"
        self.assertEqual(store.metadata("gpu-box/v0-abc"), {})
        client.objects["versions", "trainer/gpu-box/v0-abc.json"] = b'{"name": "job:x"}'
        self.assertEqual(store.metadata("gpu-box/v0-abc"), {"name": "job:x"})


class Configuration(unittest.TestCase):
    ENVIRONMENT = {"TRAINER_STORE_ENDPOINT": "https://s3.us-west-004.backblazeb2.com", "TRAINER_STORE_BUCKET": "versions",
                   "TRAINER_STORE_PREFIX": "trainer", "TRAINER_STORE_KEY_ID": "AKID", "TRAINER_STORE_SECRET": "s3cr3t-value",
                   "TRAINER_STORE_REGION": "us-west-004"}

    def test_nothing_set_means_no_bucket(self):
        self.assertIsNone(S3Store.from_env({}))
        self.assertIsNone(S3Store.from_env({"TRAINER_STORE_BUCKET": "  ", "PATH": "/bin"}))

    def test_the_environment_names_the_bucket(self):
        with mock.patch.object(studio_publish, "_boto_client", return_value=StubS3()) as make:
            store = S3Store.from_env(self.ENVIRONMENT)
        make.assert_called_once_with({"endpoint": "https://s3.us-west-004.backblazeb2.com", "bucket": "versions", "prefix": "trainer",
                                      "key_id": "AKID", "secret": "s3cr3t-value", "region": "us-west-004"})
        self.assertEqual((store.bucket, store.prefix), ("versions", "trainer/"))

    def test_the_process_environment_is_read_by_default(self):
        with mock.patch.dict("os.environ", self.ENVIRONMENT), mock.patch.object(studio_publish, "_boto_client", return_value=StubS3()):
            self.assertEqual(S3Store.from_env().bucket, "versions")

    def test_a_half_set_environment_names_what_is_missing(self):
        partial = {name: value for name, value in self.ENVIRONMENT.items() if name != "TRAINER_STORE_SECRET"}
        with self.assertRaises(ValueError) as caught:
            S3Store.from_env(partial)
        self.assertIn("TRAINER_STORE_SECRET", str(caught.exception))
        self.assertNotIn("TRAINER_STORE_BUCKET", str(caught.exception))

    def test_an_explicit_config_needs_the_same_four_things(self):
        for name in ("endpoint", "bucket", "key_id", "secret"):
            with self.assertRaises(ValueError, msg=name):
                S3Store({**CONFIG, name: ""}, client=StubS3())

    def test_the_secret_stays_out_of_the_repr(self):
        self.assertNotIn(CONFIG["secret"], repr(S3Store(CONFIG, client=StubS3())))
        self.assertNotIn(CONFIG["key_id"], repr(S3Store(CONFIG, client=StubS3())))

    def test_without_boto3_the_message_says_how_to_get_it(self):
        with mock.patch.dict(sys.modules, {"boto3": None}), self.assertRaisesRegex(RuntimeError, "boto3"):
            studio_publish._boto_client({**CONFIG})


class Zip(unittest.TestCase):
    def test_entries_carry_no_clock_so_equal_arrays_give_equal_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            items = [("a", np.arange(5, dtype=np.float16)), ("b", np.array("text"))]
            first, second = Path(directory) / "1.npz", Path(directory) / "2.npz"
            studio_publish._write_npz(first, items)
            studio_publish._write_npz(second, items)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertEqual({info.date_time for info in archive.infolist()}, {(1980, 1, 1, 0, 0, 0)})
            with np.load(first, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["a"], items[0][1])
                self.assertEqual(str(loaded["b"].item()), "text")


if __name__ == "__main__":
    unittest.main()


GB = 10**9
CAP = studio_publish.DEFAULT_CAP_BYTES
UPLOADS = ("upload_file", "put_object")


class FreeTier(Checkpoints):
    """The bucket store never takes an upload that would take the whole bucket past the free tier."""

    def setUp(self):
        super().setUp()
        self.client = StubS3()

    def store(self, limit=None, **config):
        return S3Store({**CONFIG, **config}, client=self.client, origin="windows", limit=limit)

    def fill(self, size, key="elsewhere/other-project/data.bin"):
        """Make the bucket report `size` more bytes, in an object outside the store's prefix."""
        self.client.objects["versions", key] = b"x"
        self.client.sizes[key] = size

    def uploads(self):
        return [call for call in self.client.log if call[0] in UPLOADS]

    def needed(self, checkpoint=None):
        """What publishing `checkpoint` puts in the bucket, measured on a bucket of its own."""
        probe = StubS3()
        S3Store(CONFIG, client=probe, origin="windows").publish(VERSION, checkpoint or self.base)
        return sum(len(data) for data in probe.objects.values())

    def test_the_default_limit_keeps_half_a_gigabyte_of_headroom_under_the_free_tier(self):
        self.assertEqual(studio_publish.FREE_TIER_BYTES, 10 * GB)
        self.assertEqual(studio_publish.DEFAULT_CAP_BYTES, 9_500_000_000)
        self.assertEqual(studio_publish.store_limit(None), 9_500_000_000)
        self.assertEqual(studio_publish.store_limit(""), 9_500_000_000)

    def test_the_default_cap_refuses_what_the_free_tier_itself_would_take(self):
        needed = self.needed()
        self.fill(9_500_000_000 - needed + 1)  # under 10 GB, over the cap
        with self.assertRaises(studio_publish.StoreLimitError) as caught:
            self.store().publish(VERSION, self.base)
        self.assertIn("over the 9.50 GB limit (the default cap, 0.5 GB under the R2 free tier)", str(caught.exception))
        self.assertEqual(self.uploads(), [])
        self.store(limit="10").publish(VERSION, self.base)  # the last half GB is for whoever sets it on purpose

    def test_the_setting_sets_the_limit_up_to_the_free_tier_and_never_above_it(self):
        self.assertEqual(studio_publish.store_limit("5"), 5 * GB)
        self.assertEqual(studio_publish.store_limit(" 0.5 "), GB // 2)
        self.assertEqual(studio_publish.store_limit("10"), 10 * GB)
        self.assertEqual(studio_publish.store_limit("9.8"), 9_800_000_000)  # between the cap and the free tier: allowed when asked for
        for bad in ("10.5", "11", "1000", "0", "-1", "ten", "nan", "inf"):
            with self.assertRaises(studio_publish.StoreLimitError, msg=bad) as caught:
                studio_publish.store_limit(bad)
            self.assertIn("TRAINER_STORE_LIMIT_GB", str(caught.exception))
        with self.assertRaisesRegex(studio_publish.StoreLimitError, "above the 10 GB free tier"):
            studio_publish.store_limit("12")

    def test_an_upload_that_fits_goes_up(self):
        self.fill(5 * GB)
        key = self.store().publish(VERSION, self.base)
        self.assertIn(("versions", f"trainer/{key}.json"), self.client.objects)

    def test_an_upload_that_would_pass_the_limit_is_refused_naming_the_sizes_and_nothing_goes_up(self):
        needed = self.needed()
        self.fill(CAP - needed + 1)
        store = self.store()
        with self.assertRaises(studio_publish.StoreLimitError) as caught:
            store.publish(VERSION, self.base)
        said = str(caught.exception)
        self.assertIn(f"needs {studio_publish.size_text(needed)}", said)
        self.assertIn(f"already holds {studio_publish.size_text(CAP - needed + 1)}", said)
        self.assertIn("over the 9.50 GB limit (the default cap", said)
        self.assertTrue(caught.exception.plain)
        self.assertEqual(self.uploads(), [])  # neither the weights nor the .json
        self.assertEqual(studio_publish.store_error(caught.exception, store), caught.exception)  # the UI shows it as it is

    def test_the_limit_is_inclusive_to_the_byte(self):
        needed = self.needed()
        self.fill(CAP - needed)
        self.store().publish(VERSION, self.base)  # exactly the cap: allowed
        self.assertEqual(len(self.uploads()), 2)
        again = self.store()
        self.fill(1, key="elsewhere/one-more-byte")
        with self.assertRaises(studio_publish.StoreLimitError):
            again.publish(dict(VERSION, number=4), self.tuned)

    def test_the_size_is_checked_once_for_the_weights_and_the_json_together_before_the_first_put(self):
        self.store().publish(VERSION, self.base)
        first_put = next(i for i, call in enumerate(self.client.log) if call[0] in UPLOADS)
        listings = [i for i, call in enumerate(self.client.log) if call[0] == "list_objects_v2"]
        self.assertTrue(listings and max(listings) < first_put)
        self.client.log.clear()
        self.store().publish(VERSION, self.base)  # already published: nothing to upload, so no listing of the bucket either
        self.assertEqual([call[0] for call in self.client.log if call[0] == "list_objects_v2"], [])

    def test_a_json_that_alone_tips_the_bucket_over_is_refused_before_the_weights_go_up(self):
        needed = self.needed()
        self.fill(CAP - needed + 1)  # the weights alone would fit; with the .json they do not
        with self.assertRaises(studio_publish.StoreLimitError):
            self.store().publish(VERSION, self.base)
        self.assertEqual(self.uploads(), [])

    def test_the_whole_bucket_counts_not_just_the_prefix_and_every_page_of_the_listing(self):
        needed = self.needed()
        store = self.store()
        store.publish(VERSION, self.tuned)  # a few small objects in the prefix
        for index in range(4):
            self.fill(GB, key=f"zzz/other-{index}")  # the stub pages one key at a time: the big ones are on later pages
        self.client.log.clear()
        self.fill(CAP - 4 * GB - needed + 1, key="zzzz/last-page")  # the cap - needed + 1 together with the 4 GB above, past the limit
        with self.assertRaises(studio_publish.StoreLimitError):
            store.publish(dict(VERSION, number=5), self.base)
        pages = [c for c in self.client.log if c[0] == "list_objects_v2"]
        self.assertGreater(len(pages), 3)
        self.assertTrue(all(call[2] == "" for call in pages), pages)  # no Prefix: the free tier counts everything

    def test_a_bucket_whose_size_cannot_be_read_refuses_the_upload(self):
        store = self.store()

        def broken(**kwargs):
            raise ConnectionError("no route to host")

        self.client.list_objects_v2 = broken
        with self.assertRaises(studio_publish.StoreLimitError) as caught:
            store.publish(VERSION, self.base)
        self.assertIn("usage could not be read", str(caught.exception))
        self.assertNotIn("Not uploaded", str(caught.exception))  # it does not say the limit was reached
        self.assertNotIn("over the", str(caught.exception))
        self.assertIn("Could not reach the version store", str(caught.exception))
        self.assertEqual(self.uploads(), [])
        self.assertNotIn(CONFIG["secret"], str(caught.exception))

    def test_a_refused_key_while_reading_the_size_says_so(self):
        def denied(**kwargs):
            raise StubError("AccessDenied")

        self.client.list_objects_v2 = denied
        with self.assertRaisesRegex(studio_publish.StoreLimitError, "The bucket refused the key"):
            self.store().publish(VERSION, self.base)
        self.assertEqual(self.uploads(), [])

    def test_a_lower_limit_from_the_setting_is_enforced(self):
        self.fill(2 * GB)
        with self.assertRaises(studio_publish.StoreLimitError) as caught:
            self.store(limit="2").publish(VERSION, self.base)
        self.assertIn("over the 2.00 GB limit (TRAINER_STORE_LIMIT_GB)", str(caught.exception))
        self.assertEqual(self.uploads(), [])
        self.store(limit="3").publish(VERSION, self.base)

    def test_a_setting_above_the_free_tier_refuses_every_upload(self):
        for bad in ("11", "ten"):
            with self.assertRaises(studio_publish.StoreLimitError, msg=bad):
                self.store(limit=bad).publish(VERSION, self.base)
        self.assertEqual(self.uploads(), [])

    def test_the_environment_carries_the_setting_to_the_store_without_handing_it_to_boto(self):
        environment = {**Configuration.ENVIRONMENT, "TRAINER_STORE_LIMIT_GB": "1"}
        with mock.patch.object(studio_publish, "_boto_client", return_value=self.client) as make:
            store = S3Store.from_env(environment)
        self.assertEqual(store.limit, "1")
        self.assertNotIn("limit", make.call_args.args[0])
        self.fill(GB)
        with self.assertRaises(studio_publish.StoreLimitError):
            store.publish(VERSION, self.base)

    def test_the_limit_alone_does_not_make_a_bucket(self):
        self.assertIsNone(S3Store.from_env({"TRAINER_STORE_LIMIT_GB": "5"}))

    def test_two_uploads_at_once_cannot_both_take_the_last_room(self):
        needed = self.needed()
        self.fill(CAP - needed - needed // 2)  # room for one version, not two
        store, results = self.store(), []
        upload = self.client.upload_file

        def slowly(*args):
            time.sleep(.2)  # long enough for the other thread to look at the bucket while this one is still uploading
            upload(*args)

        self.client.upload_file = slowly

        def publish(number):
            try:
                results.append(store.publish(dict(VERSION, number=number), self.base if number == 3 else self.tuned))
            except studio_publish.StoreLimitError as error:
                results.append(error)

        threads = [threading.Thread(target=publish, args=(number,)) for number in (3, 4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(type(r).__name__ for r in results), ["StoreLimitError", "str"])

    def test_the_parts_of_unfinished_uploads_count_over_the_whole_bucket_and_every_page(self):
        needed = self.needed()
        self.client.pending["elsewhere/big.bin", "u1"] = [GB, GB]
        self.client.pending["zzz/other.bin", "u2"] = [GB, GB, GB]  # the stub pages one upload and one part at a time
        self.fill(CAP - 5 * GB - needed + 1)
        with self.assertRaises(studio_publish.StoreLimitError) as caught:
            self.store().publish(VERSION, self.base)
        self.assertNotIsInstance(caught.exception, studio_publish.StoreUsageError)
        self.assertIn(f"already holds {studio_publish.size_text(CAP - needed + 1)}", str(caught.exception))
        self.assertEqual(self.uploads(), [])
        self.assertEqual(sum(1 for call in self.client.log if call[0] == "list_multipart_uploads"), 2)
        self.assertEqual(sum(1 for call in self.client.log if call[0] == "list_parts"), 5)
        self.client.sizes["elsewhere/other-project/data.bin"] -= 1  # one byte less: exactly the cap
        self.store().publish(VERSION, self.base)
        self.assertEqual(len(self.uploads()), 2)

    def test_usage_adds_objects_and_unfinished_parts_and_says_which_is_which(self):
        self.fill(3 * GB)
        self.client.pending["a.bin", "u"] = [GB, 2 * GB]
        store = self.store()
        self.assertEqual(store.usage(), (1, 6 * GB))
        self.assertEqual(store.usage_detail(), (1, 3 * GB, 1, 3 * GB))

    def test_an_upload_that_finished_while_it_was_being_counted_is_not_an_error(self):
        self.client.pending["a.bin", "u"] = [GB]
        gone = self.client.list_parts

        def finished(**kwargs):
            self.client.pending.clear()
            return gone(**kwargs)

        self.client.list_parts = finished
        self.assertEqual(self.store().usage(), (0, 0))

    def test_unfinished_uploads_that_cannot_be_listed_refuse_the_upload(self):
        for name, failure in (("list_multipart_uploads", ConnectionError("no route")), ("list_parts", StubError("AccessDenied"))):
            client = self.client = StubS3()
            client.pending["a.bin", "u"] = [GB]

            def broken(failure=failure, **kwargs):
                raise failure

            setattr(client, name, broken)
            with self.assertRaises(studio_publish.StoreUsageError, msg=name) as caught:
                self.store().publish(VERSION, self.base)
            self.assertIsInstance(caught.exception, studio_publish.StoreLimitError)
            self.assertEqual(self.uploads(), [], name)

    def test_a_listing_that_does_not_add_up_refuses_the_upload_instead_of_undercounting(self):
        def listing(**fields):
            return lambda **kwargs: {"Contents": [{"Key": "k", "Size": GB}], **fields}

        broken = {
            "truncated with no token": ("list_objects_v2", listing(IsTruncated=True)),
            "a token on a page that says it is the last": ("list_objects_v2", listing(IsTruncated=False, NextContinuationToken="2")),
            "a token and no IsTruncated": ("list_objects_v2", listing(NextContinuationToken="2")),
            "the same token twice": ("list_objects_v2", listing(IsTruncated=True, NextContinuationToken="same")),
            "no size": ("list_objects_v2", lambda **kwargs: {"Contents": [{"Key": "k"}], "IsTruncated": False}),
            "an empty size": ("list_objects_v2", lambda **kwargs: {"Contents": [{"Key": "k", "Size": None}], "IsTruncated": False}),
            "a negative size": ("list_objects_v2", lambda **kwargs: {"Contents": [{"Key": "k", "Size": -1}], "IsTruncated": False}),
            "uploads truncated with no marker": ("list_multipart_uploads", lambda **kwargs: {"Uploads": [], "IsTruncated": True}),
            "uploads that go round": ("list_multipart_uploads", lambda **kwargs: {
                "Uploads": [{"Key": "a", "UploadId": "u"}], "IsTruncated": True, "NextKeyMarker": "a", "NextUploadIdMarker": "u"}),
            "an upload with no id": ("list_multipart_uploads", lambda **kwargs: {"Uploads": [{"Key": "a"}], "IsTruncated": False}),
        }
        for what, (method, replacement) in broken.items():
            self.client = StubS3()
            setattr(self.client, method, replacement)
            with self.assertRaises(studio_publish.StoreLimitError, msg=what) as caught:
                self.store().publish(VERSION, self.base)
            self.assertIsInstance(caught.exception, studio_publish.StoreUsageError, what)
            self.assertIn("usage could not be added up", str(caught.exception), what)
            self.assertEqual(self.uploads(), [], what)

    def test_parts_that_do_not_add_up_refuse_the_upload_too(self):
        for what, parts in (("truncated with no marker", {"Parts": [{"Size": GB}], "IsTruncated": True}),
                            ("no size", {"Parts": [{"PartNumber": 1}], "IsTruncated": False}),
                            ("parts that go round", {"Parts": [{"Size": 1}], "IsTruncated": True, "NextPartNumberMarker": 1})):
            self.client = StubS3()
            self.client.pending["a.bin", "u"] = [1]
            self.client.list_parts = lambda parts=parts, **kwargs: parts
            with self.assertRaises(studio_publish.StoreUsageError, msg=what):
                self.store().publish(VERSION, self.base)
            self.assertEqual(self.uploads(), [], what)

    def test_the_last_page_of_a_complete_listing_may_carry_s3s_trailing_marker(self):
        self.client.pending["a.bin", "u"] = [5]  # the stub sends NextPartNumberMarker on the last page, as S3 does
        self.assertEqual(self.store().usage(), (0, 5))

    def test_two_publishes_of_one_version_at_once_count_and_upload_once(self):
        needed = self.needed()
        self.fill(CAP - needed)  # room for exactly one copy
        store, results = self.store(), []
        upload = self.client.upload_file

        def slowly(*args):
            time.sleep(.2)
            upload(*args)

        self.client.upload_file = slowly

        def publish():
            results.append(store.publish_reporting(VERSION, self.base))

        threads = [threading.Thread(target=publish) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 2)  # neither was refused for the other's copy
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(sorted(created for _, created in results), [False, True])
        self.assertEqual(len(self.uploads()), 2)

    def test_publish_reporting_says_whether_this_call_put_the_version_up(self):
        store = self.store()
        key, created = store.publish_reporting(VERSION, self.base)
        self.assertTrue(created)
        self.assertEqual(store.publish_reporting(VERSION, self.base), (key, False))
        self.assertEqual(store.publish(VERSION, self.base), key)

    def test_delete_removes_the_listing_first_then_the_weights_and_not_a_bad_key(self):
        store = self.store()
        key = store.publish(VERSION, self.base)
        self.client.log.clear()
        store.delete(key)
        self.assertEqual([call[2] for call in self.client.log if call[0] == "delete_object"], [f"trainer/{key}.json", f"trainer/{key}.npz"])
        self.assertFalse(store.exists(key))
        self.assertEqual(self.client.objects, {})
        store.delete(key)  # already gone: nothing to say
        with self.assertRaises(ValueError):
            store.delete("../escape")

    def test_a_folder_store_deletes_too(self):
        folder = FolderStore(self.shared, origin="mac")
        key = folder.publish(VERSION, self.base)
        folder.delete(key)
        self.assertFalse(folder.exists(key))
        folder.delete(key)

    def test_a_shared_folder_has_no_limit_here(self):
        key = FolderStore(self.shared, origin="mac").publish(VERSION, self.base)
        self.assertTrue(FolderStore(self.shared).exists(key))

    def test_sizes_read_in_the_unit_a_provider_counts_in(self):
        self.assertEqual([studio_publish.size_text(n) for n in (0, 999, 1_500, 2_345_678, 9_990_000_000, 10 * GB)],
                         ["0 B", "999 B", "1.50 KB", "2.35 MB", "9.99 GB", "10.00 GB"])


class Check(Checkpoints):
    """`python -m studio_publish --check`: what the bucket holds against its limit, with a fake client in place of the real bucket."""

    ENVIRONMENT = {k: v for k, v in Configuration.ENVIRONMENT.items() if k != "TRAINER_STORE_REGION"}

    def run_check(self, environ, client=None, args=("--check",)):
        lines = []
        patch = mock.patch.object(studio_publish, "_boto_client", return_value=client or StubS3())
        with patch:
            code = studio_publish.main(list(args), environ=environ, out=lines.append)
        return code, "\n".join(lines)

    def test_with_no_keys_set_it_says_not_configured_and_does_not_touch_the_network(self):
        with mock.patch.object(studio_publish, "_boto_client", side_effect=AssertionError("no client without keys")):
            lines = []
            code = studio_publish.main(["--check", "--write"], environ={}, out=lines.append)
        self.assertEqual(code, 1)
        self.assertIn("Not configured", lines[0])
        self.assertIn("docs/store.env.example", lines[0])

    def test_a_half_set_environment_names_what_is_missing(self):
        code, said = self.run_check({"TRAINER_STORE_BUCKET": "versions"})
        self.assertEqual(code, 1)
        self.assertIn("TRAINER_STORE_ENDPOINT", said)
        self.assertNotIn("TRAINER_STORE_BUCKET,", said)

    def test_it_prints_objects_bytes_limit_and_room_left_and_never_the_keys(self):
        client = StubS3()
        client.objects["versions", "trainer/a.bin"] = b"x" * 1000
        client.objects["versions", "other/b.bin"] = b"y" * 500  # outside the prefix: still counted
        client.sizes["trainer/a.bin"] = 2 * GB
        code, said = self.run_check(self.ENVIRONMENT, client)
        self.assertEqual(code, 0, said)
        self.assertIn("2 objects, 2.00 GB", said)
        self.assertIn("Unfinished uploads: 0, holding 0 B.", said)
        self.assertIn("Limit: 9.50 GB (the default cap, 0.5 GB under the R2 free tier)", said)
        self.assertIn("Room left: 7.50 GB", said)
        self.assertIn("Write: not tried", said)
        for secret in ("s3cr3t-value", "AKID"):
            self.assertNotIn(secret, said)
        self.assertEqual([c[0] for c in client.log if c[0] in UPLOADS + ("delete_object",)], [])

    def test_write_puts_reads_back_and_deletes_a_test_object(self):
        client = StubS3()
        code, said = self.run_check(self.ENVIRONMENT, client, ("--check", "--write"))
        self.assertEqual(code, 0, said)
        self.assertIn("Write: ok", said)
        self.assertEqual(client.objects, {})
        self.assertEqual([c[0] for c in client.log if c[0] in UPLOADS + ("delete_object", "get_object")],
                         ["put_object", "get_object", "delete_object"])

    def test_a_lower_limit_is_reported_and_a_limit_above_the_free_tier_fails(self):
        code, said = self.run_check({**self.ENVIRONMENT, "TRAINER_STORE_LIMIT_GB": "4"})
        self.assertEqual(code, 0, said)
        self.assertIn("Limit: 4.00 GB (TRAINER_STORE_LIMIT_GB)", said)
        code, said = self.run_check({**self.ENVIRONMENT, "TRAINER_STORE_LIMIT_GB": "20"})
        self.assertEqual(code, 1)
        self.assertIn("above the 10 GB free tier", said)

    def test_a_bucket_over_the_limit_is_flagged_and_write_is_refused(self):
        client = StubS3()
        client.objects["versions", "x"] = b"x"
        client.sizes["x"] = 11 * GB
        code, said = self.run_check(self.ENVIRONMENT, client, ("--check", "--write"))
        self.assertEqual(code, 1)
        self.assertIn("Room left: 0 B  OVER THE LIMIT", said)
        self.assertIn("Write: failed", said)
        self.assertEqual(client.objects, {("versions", "x"): b"x"})

    def test_a_refused_key_or_an_unreachable_bucket_fails_with_what_to_do(self):
        for failure, expected in ((StubError("InvalidAccessKeyId"), "The bucket refused the key"),
                                  (ConnectionError("no route"), "Could not reach the version store")):
            client = StubS3()

            def broken(failure=failure, **kwargs):
                raise failure

            client.list_objects_v2 = broken
            code, said = self.run_check(self.ENVIRONMENT, client)
            self.assertEqual(code, 1)
            self.assertIn(expected, said)
            self.assertNotIn("s3cr3t-value", said)

    # ---- the job checkpoints left over ----
    def bucket_with_checkpoints(self, queue, jobs):
        """A bucket holding a checkpoint for each of `jobs` ({id: step}), a published version, and a queue file with `queue` ([job rows])."""
        client = StubS3()
        store = S3Store(CONFIG, client=client, origin="gpu-box")
        for job_id, step in jobs.items():
            store.publish(studio_publish.checkpoint_version(job_id, step), self.base_for_check(step))
        store.publish(VERSION, self.base_for_check(99))
        path = Path(self.dir.name) / "queue.json"
        path.write_text(json.dumps({"jobs": queue}), encoding="utf-8")
        return client, path

    def base_for_check(self, step):
        path = Path(self.dir.name) / f"c{step}.npz"
        meta = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "step": step}
        np.savez(path, metadata=np.array(json.dumps(meta)), **{"weight::w": np.arange(4, dtype=np.float32)})
        return path

    def test_it_lists_the_checkpoints_no_unfinished_job_of_the_queue_can_resume_from(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        queue = [{"id": "live", "status": "waiting"}, {"id": "ended", "status": "stopped"}, {"id": "away-e2", "status": "running", "owner": "x:1"}]
        client, path = self.bucket_with_checkpoints(queue, {"live": 1, "ended": 2, "away": 3, "gone": 4})
        before = len(client.log)
        code, said = self.run_check(self.ENVIRONMENT, client, ("--check", "--queue", str(path)))
        self.assertEqual(code, 0, said)
        self.assertIn("Checkpoints left over: 2 of 4", said)
        self.assertIn("job ended here (stopped), job ended,", said)
        self.assertIn("job not in this queue, job gone,", said)
        self.assertNotIn("job live", said)  # still waiting: it resumes from it
        self.assertNotIn("job away", said)  # a job this machine runs for another node
        self.assertNotIn("Version 3", said)  # a published version is not a checkpoint
        self.assertIn("trainer/gpu-box/v0-", said)
        self.assertEqual([c[0] for c in client.log[before:] if c[0] in UPLOADS + ("delete_object",)], [])  # it only lists

    def test_with_every_checkpoint_in_use_it_says_none_are_left_over(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        client, path = self.bucket_with_checkpoints([{"id": "live", "status": "running"}], {"live": 1})
        code, said = self.run_check(self.ENVIRONMENT, client, ("--check", "--queue", str(path)))
        self.assertIn("Checkpoints: 1 job checkpoint object, none left over.", said)

    def test_without_a_readable_queue_it_counts_the_checkpoints_and_does_not_guess(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        client, _ = self.bucket_with_checkpoints([], {"a": 1, "b": 2})
        code, said = self.run_check(self.ENVIRONMENT, client, ("--check", "--queue", str(Path(self.dir.name) / "nothing.json")))
        self.assertEqual(code, 0, said)
        self.assertIn("2 job checkpoint objects. The queue at", said)
        self.assertIn("could not be read", said)
        self.assertNotIn("left over:", said)

    def test_it_runs_as_a_module(self):
        import os
        import subprocess
        # The real environment minus the store settings: a hand-made one lacks what Python needs to start on Windows.
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRAINER_STORE_")}
        env["PYTHONPATH"] = ""
        done = subprocess.run([sys.executable, "-m", "studio_publish", "--check"], capture_output=True, text=True, timeout=60,
                              cwd=Path(studio_publish.__file__).parent, env=env)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("Not configured", done.stdout, done.stderr)

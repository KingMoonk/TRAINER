"""studio_publish: a version leaves one machine as small half-precision weights and arrives on another as a plain checkpoint."""
import json
import sys
import tempfile
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

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        self.log.append(("list_objects_v2", Bucket, Prefix))
        keys = sorted(key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + 1]
        more = start + 1 < len(keys)
        return {"Contents": [{"Key": key, "Size": len(self.objects[Bucket, key])} for key in page], "IsTruncated": more,
                **({"NextContinuationToken": str(start + 1)} if more else {})}


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

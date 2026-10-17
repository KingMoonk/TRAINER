"""Publishing, listing and pulling versions through the API, over real workspaces and a FolderStore (and a stub bucket for the errors):
what is stored, what a second workspace sees and gets, and the sentence a failed store ends a job with."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_core import Workspace
from studio_ia.services import FakeServices, WorkspaceServices
from studio_spec import default_spec
from tests.test_publish import CONFIG, StubError, StubS3

ENV_NAMES = ("TRAINER_STORE_FOLDER", "TRAINER_STORE_ENDPOINT", "TRAINER_STORE_BUCKET", "TRAINER_STORE_PREFIX", "TRAINER_STORE_KEY_ID",
             "TRAINER_STORE_SECRET", "TRAINER_STORE_REGION", "TRAINER_STORE_LIMIT_GB")
WEIGHTS = np.random.default_rng(3).integers(-64, 64, 4096).astype(np.float32) / 8  # exact in float16: a pull must give them back unchanged


def write_checkpoint(path, step=40):
    """A checkpoint the way training saves one: weights, plus the optimizer state that doubles it and a publish must leave behind."""
    meta = {"architecture": "groundwork_numpy_decoder", "format_version": 1, "step": step, "device": "cpu"}
    np.savez(path, metadata=np.array(json.dumps(meta)), **{"weight::w": WEIGHTS, "adam_m::w": WEIGHTS * 3, "adam_v::w": WEIGHTS * WEIGHTS})
    return Path(path)


class Node:
    """One machine: a workspace, its Services, and the API over them."""

    def __init__(self, test, root):
        self.test, self.root = test, Path(root)
        self.services = WorkspaceServices(Workspace(self.root))
        test.addCleanup(self.services.close)
        self.client = test.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))

    def use_folder(self, folder):
        self.services.set_setting("store", {"folder": str(folder)})

    def train(self, step=40):
        """Save a version and give it a trained checkpoint; returns the checkpoint's path."""
        version = self.services.save_version(default_spec(), "first design")
        path = write_checkpoint(self.root / f"v{version.number}.npz", step)
        self.services.store.set_best_checkpoint(version.number, path)
        return path

    def job(self, response, expect="done"):
        self.test.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        for _ in range(200):
            job = self.client.get(f"/jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                self.test.assertEqual(job["status"], expect, job)
                return job
            time.sleep(0.05)
        self.test.fail("job did not finish")

    def publish(self, number=1, **kwargs):
        return self.job(self.client.post(f"/versions/{number}/publish"), **kwargs)

    def pull(self, number=1, origin=None, **kwargs):
        return self.job(self.client.post(f"/versions/{number}/pull", params={"origin": origin} if origin else {}), **kwargs)

    def published(self, expect=200):
        response = self.client.get("/versions/published")
        self.test.assertEqual(response.status_code, expect, response.text)
        return response.json()


class PublishCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.shared = self.tmp / "shared"
        self.shared.mkdir()
        self.enterContext(mock.patch.dict(os.environ))
        for name in ENV_NAMES:
            os.environ.pop(name, None)
        self.a, self.b = Node(self, self.tmp / "a"), Node(self, self.tmp / "b")

    def stored(self):
        return sorted(p for p in self.shared.rglob("*") if p.is_file())


class RoundTrip(PublishCase):
    def setUp(self):
        super().setUp()
        self.a.use_folder(self.shared)
        self.b.use_folder(self.shared)

    def test_a_published_version_is_available_to_a_second_workspace_and_pulls_with_the_same_weights(self):
        source = self.a.train()
        with mock.patch("studio_publish.socket.gethostname", return_value="mac"):
            published = self.a.publish()["result"]
        self.assertEqual(published["number"], 1)
        self.assertEqual(published["origin"], "mac")

        (entry,) = self.b.published()
        self.assertEqual((entry["key"], entry["number"], entry["origin"]), (published["key"], 1, "mac"))
        self.assertEqual((entry["local"], entry["local_number"]), (False, None))
        self.assertEqual(self.b.client.get("/versions").json(), [])

        with mock.patch("studio_publish.socket.gethostname", return_value="pc"):
            pulled = self.b.pull()["result"]
            (entry,) = self.b.published()
        self.assertEqual((pulled["number"], pulled["status"]), (1, "trained"))
        self.assertEqual((entry["mine"], entry["local"], entry["local_number"]), (False, True, 1))
        (version,) = self.b.client.get("/versions").json()
        self.assertEqual(version["spec"], default_spec())
        with np.load(version["best_checkpoint"], allow_pickle=False) as weights, np.load(source, allow_pickle=False) as original:
            np.testing.assert_array_equal(weights["weight::w"], original["weight::w"])
            self.assertEqual(weights["weight::w"].dtype, np.float32)  # widened again: any part of the engine reads it
            self.assertFalse([name for name in weights.files if name.startswith("adam_")])
        self.assertEqual(self.b.services.current_version().number, 1)

    def test_what_is_uploaded_is_the_weights_and_the_metadata_only(self):
        source = self.a.train()
        key = self.a.publish()["result"]["key"]
        self.assertEqual([p.relative_to(self.shared).as_posix() for p in self.stored()], [f"{key}.json", f"{key}.npz"])
        weights, meta = self.shared / f"{key}.npz", json.loads((self.shared / f"{key}.json").read_text())
        self.assertEqual(meta["size"], weights.stat().st_size)
        self.assertLess(weights.stat().st_size, source.stat().st_size / 2)  # float16 weights, no optimizer: a third of the checkpoint
        self.assertLess(weights.stat().st_size, WEIGHTS.nbytes)  # and under even the float32 weights alone
        with np.load(weights, allow_pickle=False) as stored:
            self.assertEqual(sorted(stored.files), ["metadata", "weight::w"])
            self.assertEqual(stored["weight::w"].dtype, np.float16)

    def test_publishing_the_same_weights_again_uploads_nothing(self):
        self.a.train()
        first = self.a.publish()["result"]["key"]
        before = {p: p.stat().st_mtime_ns for p in self.stored()}
        self.assertEqual(self.a.publish()["result"]["key"], first)
        self.assertEqual({p: p.stat().st_mtime_ns for p in self.stored()}, before)

    def test_the_listing_says_which_versions_are_here_and_who_published_them(self):
        self.a.train()
        self.a.train(step=80)
        with mock.patch("studio_publish.socket.gethostname", return_value="mac"):
            self.a.publish(1)
        with mock.patch("studio_publish.socket.gethostname", return_value="pc"):
            self.b.train()
            self.b.publish(1)
            self.assertEqual([(e["origin"], e["mine"], e["local"], e["local_number"]) for e in self.b.published()],
                             [("pc", True, True, 1), ("mac", False, False, None)])  # newest first
            self.assertEqual([(e["origin"], e["number"]) for e in self.a.published()], [("pc", 1), ("mac", 1)])
            self.assertEqual([(e["origin"], e["local"]) for e in self.a.published()], [("pc", False), ("mac", True)])

    def test_two_machines_with_a_version_one_each_are_told_apart_by_origin(self):
        self.a.train()
        self.b.train(step=80)
        with mock.patch("studio_publish.socket.gethostname", return_value="mac"):
            self.a.publish()
        with mock.patch("studio_publish.socket.gethostname", return_value="pc"):
            self.b.publish()
        c = Node(self, self.tmp / "c")
        c.use_folder(self.shared)
        said = c.pull(expect="failed")["error"]
        self.assertIn("published by mac, pc", said)
        self.assertEqual(c.services.versions(), [])
        pulled = c.pull(origin="pc")["result"]
        self.assertIn("pc", pulled["name"])
        self.assertEqual([(v.number, v.change.startswith("pulled from pc")) for v in c.services.versions()], [(1, True)])

    def test_pulling_what_is_already_here_adds_nothing(self):
        self.a.train()
        self.a.publish()
        first = self.b.pull()["result"]
        again = self.b.pull()["result"]
        self.assertEqual((first["number"], again["number"]), (1, 1))
        self.assertEqual(len(self.b.services.versions()), 1)
        self.assertEqual(len(self.a.services.versions()), 1)
        self.assertEqual(self.a.pull()["result"]["number"], 1)  # its own version: nothing to download

    def test_a_pulled_file_that_was_cleaned_away_is_fetched_again_under_the_same_version(self):
        self.a.train()
        self.a.publish()
        path = Path(self.b.pull()["result"]["best_checkpoint"])
        path.unlink()
        self.assertEqual(self.b.pull()["result"]["number"], 1)
        self.assertTrue(path.is_file())
        self.assertEqual(len(self.b.services.versions()), 1)

    def test_a_version_that_is_not_published_says_so(self):
        said = self.b.pull(7, expect="failed")["error"]
        self.assertEqual(said, "Version 7 is not published. Publish it from the machine that trained it, then pull it again.")

    def test_a_published_version_is_found_through_the_environment_when_the_workspace_names_no_folder(self):
        self.a.train()
        self.a.publish()
        c = Node(self, self.tmp / "c")
        os.environ["TRAINER_STORE_FOLDER"] = str(self.shared)
        self.assertEqual(len(c.published()), 1)

    def test_the_workspace_folder_wins_over_the_environment(self):
        self.a.train()
        self.a.publish()
        os.environ["TRAINER_STORE_FOLDER"] = str(self.tmp / "elsewhere")
        self.assertEqual(len(self.b.published()), 1)


class Refusals(PublishCase):
    def test_with_no_store_every_route_says_what_to_set_up(self):
        self.a.train()
        for method, route in (("get", "/versions/published"), ("post", "/versions/1/publish"), ("post", "/versions/1/pull")):
            with self.subTest(route=route):
                response = getattr(self.a.client, method)(route)
                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("No version store is set up", response.json()["detail"])
                self.assertIn("TRAINER_STORE_FOLDER", response.json()["detail"])

    def test_the_fake_services_answer_the_same_sentence(self):
        client = self.enterContext(TestClient(create_app(FakeServices(), open=True, hosts=ANY)))
        client.post("/versions", json={"spec": {"dims": {"n_ctx": 8}}})
        for method, route in (("get", "/versions/published"), ("post", "/versions/1/publish"), ("post", "/versions/1/pull")):
            response = getattr(client, method)(route)
            self.assertEqual(response.status_code, 409, route)
            self.assertIn("No version store is set up", response.json()["detail"])

    def test_an_unknown_version_is_a_404_and_one_with_no_weights_a_409_before_any_job_starts(self):
        self.a.use_folder(self.shared)
        self.assertEqual(self.a.client.post("/versions/3/publish").status_code, 404)
        self.a.services.save_version(default_spec(), "never trained")
        response = self.a.client.post("/versions/1/publish")
        self.assertEqual(response.status_code, 409)
        self.assertIn("has not trained yet", response.json()["detail"])
        self.assertEqual(self.stored(), [])

    def test_a_half_set_bucket_names_what_is_missing(self):
        os.environ["TRAINER_STORE_BUCKET"] = "versions"
        response = self.a.client.get("/versions/published")
        self.assertEqual(response.status_code, 400)
        self.assertIn("TRAINER_STORE_ENDPOINT", response.json()["detail"])


class StoreErrors(PublishCase):
    def test_a_shared_folder_that_is_not_connected_ends_the_job_with_what_to_do_and_a_retry_works(self):
        self.a.train()
        gone = self.tmp / "not-mounted"
        self.a.use_folder(gone)
        failed = self.a.publish(expect="failed")
        self.assertEqual(failed["error"], f"The shared folder {gone} is not there. Check that the drive or share is connected.")
        response = self.a.client.get("/versions/published")  # listing an unmounted folder is simply empty: nothing there to show
        self.assertEqual((response.status_code, response.json()), (200, []))
        gone.mkdir()
        done = self.a.publish()
        self.assertTrue(done["result"]["key"])
        self.assertEqual(len(self.stored_in(gone)), 2)

    def stored_in(self, folder):
        return [p for p in folder.rglob("*") if p.is_file()]

    def bucket(self, client):
        for name, value in {"endpoint": CONFIG["endpoint"], "bucket": CONFIG["bucket"], "key_id": CONFIG["key_id"],
                            "secret": CONFIG["secret"]}.items():
            os.environ[f"TRAINER_STORE_{name.upper()}"] = value
        return mock.patch("studio_publish._boto_client", lambda config: client)

    def test_a_refused_key_ends_the_job_with_what_to_do_and_a_retry_with_the_right_key_works(self):
        self.a.train()
        client = StubS3()
        client.fail_head = "InvalidAccessKeyId"
        with self.bucket(client):
            said = self.a.publish(expect="failed")["error"]
            self.assertEqual(said, "The bucket refused the key. Check TRAINER_STORE_KEY_ID and TRAINER_STORE_SECRET, and that the key "
                                   "can read and write this bucket, then try again.")
            self.assertNotIn(CONFIG["secret"], said)
            self.assertEqual(self.a.client.get("/versions/published").status_code, 200)  # listing needs no head_object
            client.fail_head = None
            key = self.a.publish()["result"]["key"]
        self.assertEqual(sorted(k for _, k in client.objects), [f"{key}.json", f"{key}.npz"])

    def test_an_upload_past_the_free_tier_ends_the_job_with_the_sizes_and_uploads_nothing_and_room_made_lets_a_retry_through(self):
        self.a.train()
        client = StubS3()
        client.objects["versions", "elsewhere/big.bin"] = b"x"
        client.sizes["elsewhere/big.bin"] = 9_500_000_000 - 100  # 100 bytes of room under the cap: no version fits
        with self.bucket(client):
            job = self.a.publish(expect="failed")
            self.assertRegex(job["error"], r"^Not uploaded: this upload needs [\d.]+ KB and the bucket already holds 9\.50 GB, which would "
                                           r"make 9\.50 GB, over the 9\.50 GB limit \(the default cap, 0\.5 GB under the R2 free tier\)\. "
                                           r"Delete versions")
            self.assertNotIn("StoreLimitError", job["error"])  # the sentence as it is, without the exception's name
            self.assertEqual(self.a.client.get(f"/jobs/{job['id']}").json()["error"], job["error"])
            self.assertEqual([c for c in client.log if c[0] in ("upload_file", "put_object")], [])
            self.assertEqual(self.a.services.store.get(1).get("published") or [], [])
            client.sizes.clear()  # versions deleted from the bucket
            self.a.publish()
        self.assertEqual(len([c for c in client.log if c[0] in ("upload_file", "put_object")]), 2)

    def test_a_bucket_whose_usage_cannot_be_read_ends_the_job_saying_so_and_not_that_the_limit_was_hit(self):
        self.a.train()

        class Unreadable(StubS3):
            def list_multipart_uploads(self, **kwargs):
                raise ConnectionError("no route to host")

        client = Unreadable()
        with self.bucket(client):
            said = self.a.publish(expect="failed")["error"]
        self.assertIn("the bucket's usage could not be read", said)
        self.assertNotIn("Not uploaded", said)
        self.assertEqual([c for c in client.log if c[0] in ("upload_file", "put_object")], [])

    def test_a_limit_setting_above_the_free_tier_ends_the_job_naming_the_setting(self):
        self.a.train()
        os.environ["TRAINER_STORE_LIMIT_GB"] = "50"
        client = StubS3()
        with self.bucket(client):
            said = self.a.publish(expect="failed")["error"]
        self.assertIn("TRAINER_STORE_LIMIT_GB is 50, above the 10 GB free tier", said)
        self.assertEqual(client.objects, {})

    def test_an_unreachable_bucket_ends_the_job_with_what_to_do_and_a_retry_works(self):
        self.a.train()

        class Down(StubS3):
            def head_object(self, Bucket, Key):
                raise ConnectionError("no route to host")

        with self.bucket(Down()):
            said = self.a.publish(expect="failed")["error"]
            self.assertEqual(said, "Could not reach the version store at versions. Check the connection and TRAINER_STORE_ENDPOINT, then try again.")
        client = StubS3()
        with self.bucket(client):
            self.a.publish()
            self.assertEqual(len(self.a.published()), 1)

    def test_a_missing_bucket_and_a_failing_pull_say_what_to_check(self):
        self.a.train()
        client = StubS3()
        with self.bucket(client):
            self.a.publish()
            client.fail_head = "NoSuchBucket"
            said = self.a.publish(expect="failed")["error"]
            self.assertIn("The bucket versions was not found", said)
            client.objects.pop(next(k for k in client.objects if k[1].endswith(".npz")))  # the weights vanished from the bucket
            said = self.b.pull(expect="failed")["error"]
            self.assertIn("is missing from the version store", said)

    def test_a_listing_that_fails_is_a_409_with_the_sentence(self):
        class Refusing(StubS3):
            def list_objects_v2(self, **kwargs):
                raise StubError("AccessDenied")

        with self.bucket(Refusing()):
            response = self.a.client.get("/versions/published")
        self.assertEqual(response.status_code, 409)
        self.assertIn("The bucket refused the key", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

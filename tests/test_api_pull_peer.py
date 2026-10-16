"""Pulling a published version straight from the node that holds it: machine A's API serves its weights and machine B pulls them, over
real workspaces. B reaches A through a PeerClient whose opener is A's API in process, so what crosses is what the HTTP routes send."""
import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import studio_cluster
from studio_api.app import create_app
from studio_ia.services import FakeServices
from studio_peer import PeerClient
from studio_publish import FolderStore, sha256_of
from tests.test_api_node import probed
from tests.test_api_publish import PublishCase, write_checkpoint


class Wire:
    """urllib's opener, answered by one node's API in process. `log` has every request; `bodies` the bytes of each path's last answer."""

    def __init__(self, client):
        self.client, self.down, self.log, self.bodies, self.timeouts, self.tamper = client, False, [], {}, [], None

    def __call__(self, request, timeout=None):
        path = request.full_url.partition("://")[2].partition("/")[2]
        self.log.append(path.partition("?")[0])
        self.timeouts.append(timeout)
        if self.down:
            raise urllib.error.URLError("no route to host")
        response = self.client.request(request.get_method(), "/" + path, content=request.data)
        body = self.bodies[path] = response.content
        if self.tamper and path.endswith("/weights"):
            body = self.tamper(body)
        if response.status_code >= 400:
            raise urllib.error.HTTPError(request.full_url, response.status_code, "error", {}, io.BytesIO(body))
        return io.BytesIO(body)


class Unreadable(FolderStore):
    """A store that fails the test when anything is read from it."""

    def list(self):
        raise AssertionError("the store was listed")

    def pull(self, key, dest_dir):
        raise AssertionError("the store was pulled from")

    def exists(self, key):
        raise AssertionError("the store was asked")


class PeerCase(PublishCase):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch("socket.gethostname", return_value="mac"))
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
        self.a.use_folder(self.shared)
        self.b.use_folder(self.shared)
        self.wire = Wire(self.a.client)
        self.b.services.runner.cluster.client_for = lambda address, **options: PeerClient(address, opener=self.wire, **options)
        self.b.services.set_setting("nodes", ["mac"])
        self.source = self.a.train()
        self.key = self.a.publish()["result"]["key"]

    def pulled(self, expect="done", **kwargs):
        return self.b.pull(1, "mac", expect=expect, **kwargs)

    def calls(self):
        return self.wire.log


class FromThePeer(PeerCase):
    def test_with_the_owner_online_the_weights_come_from_it_and_the_store_is_not_read(self):
        self.b.services.version_store = lambda: Unreadable(self.shared)
        job = self.pulled()
        self.assertEqual(self.calls(), ["node", "versions/1/weights/meta", "versions/1/weights"])
        self.assertEqual(job["progress"], "Copied version 1 straight from mac.")
        self.assertEqual((job["result"]["number"], job["result"]["status"]), (1, "trained"))
        (version,) = self.b.client.get("/versions").json()
        self.assertEqual(version["name"], "Version 1 from mac")
        self.assertTrue(version["change"].startswith("pulled from mac: first design"))
        with np.load(version["best_checkpoint"], allow_pickle=False) as weights, np.load(self.source, allow_pickle=False) as original:
            np.testing.assert_array_equal(weights["weight::w"], original["weight::w"])
            self.assertEqual(weights["weight::w"].dtype, np.float32)  # widened again, as from the store
            self.assertEqual(sorted(weights.files), ["metadata", "weight::w"])
        self.assertEqual(self.b.services.current_version().number, 1)

    def test_only_the_weights_and_the_description_cross_the_wire(self):
        self.pulled()
        sent = self.wire.bodies["versions/1/weights"]
        with np.load(io.BytesIO(sent), allow_pickle=False) as arrays:
            self.assertEqual(sorted(arrays.files), ["metadata", "weight::w"])  # no adam_ array
            self.assertEqual(arrays["weight::w"].dtype, np.float16)
        meta = json.loads(self.wire.bodies["versions/1/weights/meta"])
        self.assertEqual((meta["size"], meta["key"]), (len(sent), self.key))
        self.assertLess(len(sent), self.source.stat().st_size / 2)
        self.assertEqual(sorted(p.name for p in (self.a.root / "data" / "shared").iterdir()), ["v1.json", "v1.npz"])
        self.assertEqual(sorted(p.name for p in (self.b.root / "data" / "pulled").iterdir()),
                         [self.key.replace("/", "-") + ".json", self.key.replace("/", "-") + ".npz"])

    def test_it_is_the_same_version_the_store_lists_so_the_store_pull_finds_it_already_here(self):
        self.pulled()
        (entry,) = self.b.published()
        self.assertEqual((entry["key"], entry["local"], entry["local_number"]), (self.key, True, 1))
        self.b.services.version_store = lambda: FolderStore(self.shared)
        self.wire.log.clear()
        job = self.b.pull(1)  # no origin: the store answers, and finds it here
        self.assertEqual((job["result"]["number"], self.calls()), (1, []))
        self.assertEqual(len(self.b.services.versions()), 1)

    def test_without_an_origin_the_store_names_the_owner_and_only_the_listing_is_read(self):
        class ListedOnly(FolderStore):
            def pull(self, key, dest_dir):
                raise AssertionError("the weights came from the store")

        self.b.services.version_store = lambda: ListedOnly(self.shared)
        job = self.b.pull(1)
        self.assertEqual(self.calls(), ["node", "versions/1/weights/meta", "versions/1/weights"])
        self.assertEqual(job["progress"], "Copied version 1 straight from mac.")

    def test_pulling_what_is_already_here_downloads_nothing_and_says_so(self):
        first = self.pulled()["result"]
        self.wire.log.clear()
        again = self.pulled()
        self.assertEqual(again["result"]["number"], first["number"])
        self.assertEqual(self.calls(), ["node", "versions/1/weights/meta"])  # the description, to know it is the same; no weights
        self.assertEqual(again["progress"], "Version 1 from mac is already here, so nothing was downloaded.")
        self.assertEqual(len(self.b.services.versions()), 1)

    def test_a_file_cleaned_away_is_fetched_again_under_the_same_version(self):
        path = self.pulled()["result"]["best_checkpoint"]
        os.unlink(path)
        self.assertEqual(self.pulled()["result"]["number"], 1)
        self.assertEqual(len(self.b.services.versions()), 1)
        self.assertTrue(os.path.isfile(path))

    def test_the_probe_is_short_and_the_download_waits_longer(self):
        self.pulled()
        self.assertEqual(self.wire.timeouts[0], studio_cluster.POLL_TIMEOUT)
        self.assertGreater(min(self.wire.timeouts[1:]), studio_cluster.POLL_TIMEOUT)

    def test_a_node_is_found_by_its_hostname_whatever_its_address(self):
        self.b.services.set_setting("nodes", ["http://100.64.0.7:8787"])
        self.pulled()
        self.assertEqual(self.calls(), ["node", "versions/1/weights/meta", "versions/1/weights"])


class FromTheStore(PeerCase):
    def test_with_the_owner_offline_the_store_is_used_and_the_note_says_so(self):
        self.wire.down = True
        job = self.pulled()
        self.assertEqual(job["progress"], "mac did not answer, so version 1 came from the version store.")
        self.assertEqual(self.calls(), ["node"])  # one short try, not a retry loop
        (version,) = self.b.client.get("/versions").json()
        with np.load(version["best_checkpoint"], allow_pickle=False) as weights, np.load(self.source, allow_pickle=False) as original:
            np.testing.assert_array_equal(weights["weight::w"], original["weight::w"])

    def test_an_owner_that_cannot_send_it_falls_back_with_its_reason(self):
        self.source.unlink()
        job = self.pulled()
        self.assertIn("mac could not send version 1. Version 1's checkpoint is not at", job["progress"])
        self.assertTrue(job["progress"].endswith("It came from the version store instead."))
        self.assertEqual(self.calls(), ["node", "versions/1/weights/meta"])
        self.assertEqual(len(self.b.services.versions()), 1)

    def test_a_download_that_does_not_hash_right_is_not_kept(self):
        self.wire.tamper = lambda body: body[:-16] + bytes(16)
        job = self.pulled()
        self.assertEqual(job["progress"], "mac could not send version 1. What it sent did not check out. It came from the version store instead.")
        self.assertEqual(len(self.b.services.versions()), 1)
        (version,) = self.b.client.get("/versions").json()
        with np.load(version["best_checkpoint"], allow_pickle=False) as weights, np.load(self.source, allow_pickle=False) as original:
            np.testing.assert_array_equal(weights["weight::w"], original["weight::w"])  # the store's, intact

    def test_without_an_origin_the_store_names_the_owner_and_an_owner_that_is_off_is_not_waited_for(self):
        self.wire.down = True
        job = self.b.pull(1)
        self.assertEqual(job["progress"], "mac did not answer, so version 1 came from the version store.")
        self.assertEqual(self.calls(), ["node"])

    def test_a_node_that_is_not_listed_is_not_asked(self):
        self.b.services.set_setting("nodes", [])
        job = self.pulled()
        self.assertEqual(self.calls(), [])
        self.assertEqual(job["progress"], "Downloaded version 1 from the version store.")

    def test_a_listed_node_that_is_someone_else_is_not_the_owner(self):
        self.b.services.set_setting("nodes", ["elsewhere"])
        self.wire.down = True
        job = self.pulled()
        self.assertEqual(job["progress"], "Downloaded version 1 from the version store.")  # nothing says "elsewhere" is "mac"

    def test_with_the_owner_offline_and_the_store_down_the_job_fails_with_the_stores_sentence(self):
        self.wire.down = True
        gone = self.tmp / "not-mounted"
        self.b.use_folder(gone)
        said = self.pulled(expect="failed")["error"]
        self.assertIn("Version 1 is not published", said)


class OwnerRoutes(PeerCase):
    def test_the_description_says_what_crossed_to_the_store_too(self):
        meta = self.a.client.get("/versions/1/weights/meta").json()
        listed = json.loads((self.shared / f"{self.key}.json").read_text())
        for field in ("key", "number", "name", "origin", "spec", "size", "sha256", "step", "device"):
            self.assertEqual(meta[field], listed[field], field)

    def test_the_weights_are_the_file_the_description_hashes(self):
        meta = self.a.client.get("/versions/1/weights/meta").json()
        response = self.a.client.get("/versions/1/weights")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(response.headers["content-length"]), meta["size"])
        shared = self.a.root / "data" / "shared" / "v1.npz"
        self.assertEqual(sha256_of(shared), meta["sha256"])
        self.assertEqual(response.content, shared.read_bytes())

    def test_the_copy_is_made_once_and_made_again_when_the_checkpoint_changes(self):
        first = self.a.client.get("/versions/1/weights/meta").json()
        built = (self.a.root / "data" / "shared" / "v1.npz").stat().st_mtime_ns
        self.assertEqual(self.a.client.get("/versions/1/weights/meta").json(), first)
        self.a.client.get("/versions/1/weights")
        self.assertEqual((self.a.root / "data" / "shared" / "v1.npz").stat().st_mtime_ns, built)
        write_checkpoint(self.source, step=80)
        again = self.a.client.get("/versions/1/weights/meta").json()
        self.assertNotEqual(again["key"], first["key"])
        self.assertEqual(sha256_of(self.a.root / "data" / "shared" / "v1.npz"), again["sha256"])

    def test_a_version_that_does_not_exist_is_a_404_and_one_that_has_not_trained_a_409(self):
        for route in ("weights/meta", "weights"):
            with self.subTest(route=route):
                self.assertEqual(self.a.client.get(f"/versions/9/{route}").status_code, 404)
        self.a.services.save_version(self.a.services.versions()[0].spec, "never trained")
        for route in ("weights/meta", "weights"):
            with self.subTest(route=route):
                response = self.a.client.get(f"/versions/2/{route}")
                self.assertEqual(response.status_code, 409)
                self.assertIn("has not trained yet", response.json()["detail"])

    def test_fake_services_have_no_weights_to_send(self):
        client = self.enterContext(TestClient(create_app(FakeServices(), open=True)))
        client.post("/versions", json={"spec": {"dims": {"n_ctx": 8}}})
        self.assertEqual(client.get("/versions/1/weights/meta").status_code, 409)
        self.assertEqual(client.get("/versions/1/weights").status_code, 409)


if __name__ == "__main__":
    unittest.main()

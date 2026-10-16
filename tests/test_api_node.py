"""studio_api.routers.node over FakeServices: GET /node answers with studio_cluster's payload. It emits no events."""
import json
import os
import unittest
from unittest import mock

import studio_cluster
from fastapi.testclient import TestClient
from studio_api.__main__ import ARGV, make, reloaded
from studio_api.app import create_app
from studio_api.version import app_version
from studio_ia.services import FakeServices, Job, QueueState
from tests.api_tap import EventTap

GPU = {"available": True, "kind": "cuda", "name": "Test GPU", "memory_bytes": 8 * 2 ** 30, "reason": ""}
real_own_info = studio_cluster.own_info


def probed(**kwargs):
    """own_info with the GPU and memory probes answered, so the test never imports torch."""
    return real_own_info(**kwargs, status=GPU, memory=16 * 2 ** 30)


class NodeApiTest(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.app = create_app(self.services, open=True)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))

    def test_the_payload_is_what_a_peer_polls(self):
        body = self.client.get("/node").json()
        self.assertEqual(body, studio_cluster.own_info(busy=False, version=app_version()).payload())
        self.assertEqual((body["device"], body["device_name"], body["memory_bytes"], body["dtypes"]),
                         ("cuda", "Test GPU", 8 * 2 ** 30, ["float32", "bfloat16", "float16"]))
        self.assertEqual(body["version"], app_version())

    def test_a_peer_can_read_it(self):
        info = studio_cluster.info_from_payload("box", self.client.get("/node").json())
        self.assertTrue(info.free)

    def test_busy_follows_the_running_job(self):
        self.assertFalse(self.client.get("/node").json()["busy"])
        self.services.set_queue(QueueState(job=Job("j1", "pretrain")))
        self.assertTrue(self.client.get("/node").json()["busy"])
        self.services.set_queue(QueueState(up_next=[Job("j2", "pretrain")], paused=True))
        self.assertFalse(self.client.get("/node").json()["busy"])

    def test_it_emits_nothing(self):
        self.client.get("/node")
        self.assertEqual([event for event in self.tap.events if event[0] != "status"], [])  # initial shell snapshot is independent

    def test_the_server_listens_where_peers_look_for_it(self):
        _, _, port = make(["--fake"])
        self.assertEqual(port, studio_cluster.DEFAULT_PORT)

    def test_reload_builds_each_app_from_the_saved_command_line(self):
        with mock.patch.dict(os.environ, {ARGV: json.dumps(["--fake", "--bind", "0.0.0.0"])}):
            app = reloaded()
        self.addCleanup(app.state.broadcaster.close)
        self.addCleanup(app.state.jobs.close)
        self.assertTrue(app.state.open)  # --fake is open whatever the bind
        self.assertIsInstance(app.state.services, FakeServices)


if __name__ == "__main__":
    unittest.main()

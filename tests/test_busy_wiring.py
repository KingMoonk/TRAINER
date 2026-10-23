"""Issue #578: dataset import, catalogue downloads and the cluster peer survey report through the busy registry (the loading bar)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import studio_cluster
from studio_core import Workspace
from studio_ia.services import WorkspaceServices
from tests.sse_harness import ApiCase
from tests.test_api_cluster import FakePeers, node_payload, probed
from tests.test_ia_training_pages import RecommendedCollection


def shown(events):
    """The busy events as [(label, done, total), ...] per event: what the bar would have drawn, in order."""
    return [[(i["label"], i["done"], i["total"]) for i in payload["items"]] for kind, payload in events if kind == "busy"]


def watch_busy(case, services):
    case.busy_events = []
    case.enterContext(mock.patch("studio_ia.busy.PROGRESS_EVERY", 0))  # every step is sent, not one in a tenth of a second
    services.watch(lambda kind, payload: case.busy_events.append((kind, payload)))


class ImportingFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = WorkspaceServices(Workspace(self.tmp.name))
        self.addCleanup(self.svc.close)
        watch_busy(self, self.svc)

    def write(self, name, text):
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
        return path

    def test_one_step_per_file_with_its_name_then_the_bar_ends(self):
        a, b = self.write("a.jsonl", "a first story."), self.write("b.jsonl", "a second story.")
        report = self.svc.import_data("text", [a, b])
        self.assertEqual(len(report), 2)
        self.assertIn("1 added", report[0])
        steps = shown(self.busy_events)
        self.assertEqual(steps[0], [("Importing your files", 0, 2)])  # start
        self.assertIn([("Importing a.jsonl", 0, 2)], steps)
        self.assertIn([("Importing a.jsonl", 1, 2)], steps)
        self.assertIn([("Importing b.jsonl", 2, 2)], steps)
        self.assertEqual(steps[-1], [])  # end
        self.assertEqual(self.svc.busy.items(), [])

    def test_the_bar_ends_when_a_file_cannot_be_read(self):
        report = self.svc.import_data("text", [Path(self.tmp.name) / "missing.jsonl"])
        self.assertIn("missing.jsonl", report[0])
        self.assertEqual(shown(self.busy_events)[-1], [])


class DownloadingSources(RecommendedCollection):
    @staticmethod
    def fake_fetch(url, path, stop=None, emit=None, maximum=None, range_bytes=None):
        info = RecommendedCollection.fake_fetch(url, path, stop, emit, maximum, range_bytes)
        if emit:
            emit({"text": "Downloading", "completed": 5, "total": 10})
        return info

    def labelled(self):
        return [step for step in shown(self.busy_events) if step]

    def test_a_download_and_the_import_of_it_each_show_their_source_name(self):
        watch_busy(self, self.svc)
        name = self.svc._source_name("tinystories")
        with mock.patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            self.assertTrue(self.svc.download_sources(["tinystories"]))
            self.svc._worker.join(10)
            downloaded = self.labelled()
            self.assertEqual(downloaded[0], [(f"Downloading {name}", 0, None)])  # start, size not known yet
            self.assertIn([(f"Downloading {name}", 5, 10)], downloaded)  # a step, with the total it learned
            self.assertEqual(shown(self.busy_events)[-1], [])
            self.busy_events.clear()
            self.assertTrue(self.svc.import_source("tinystories"))
            self.svc._worker.join(10)
        self.assertEqual(self.labelled()[0], [(f"Adding {name}", 0, None)])
        self.assertEqual(shown(self.busy_events)[-1], [])
        self.assertEqual(self.svc.busy.items(), [])

    def test_a_failed_download_still_ends_its_bar(self):
        watch_busy(self, self.svc)
        with mock.patch("studio_catalog._fetch", side_effect=OSError("no network")):
            self.assertTrue(self.svc.download_sources(["tinystories"]))
            self.svc._worker.join(10)
        self.assertEqual(shown(self.busy_events)[-1], [])
        self.assertEqual(self.svc.busy.items(), [])


class AskingThePeers(ApiCase):
    prefix = "/cluster/nodes"

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.object(studio_cluster, "own_info", probed))
        self.peers = FakePeers(up=node_payload("up-box"), down=OSError("refused"))
        self.enterContext(mock.patch.object(studio_cluster, "fetch_json", self.peers))
        watch_busy(self, self.services)
        self.call("PUT", "/cluster/nodes", json={"addresses": ["up", "down"]})

    def test_the_survey_is_one_task_with_a_step_for_each_peer(self):
        self.call("GET", "/cluster/nodes")
        steps = shown(self.busy_events)
        self.assertEqual(steps[0], [("Checking the other computers", 0, 2)])
        self.assertIn([("Checking the other computers", 2, 2)], steps)  # both answered, one of them with a failure
        self.assertEqual(steps[-1], [])
        self.assertEqual(self.services.busy.items(), [])

    def test_this_node_alone_asks_nobody_and_shows_nothing(self):
        self.call("PUT", "/cluster/nodes", json={"addresses": []})
        self.busy_events.clear()
        self.call("GET", "/cluster/nodes")
        self.assertEqual(shown(self.busy_events), [])


if __name__ == "__main__":
    unittest.main()

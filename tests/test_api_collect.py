"""studio_api.routers.collect over FakeServices, and the app-side manager that registers the collector's shards."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_core
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_collect import config, register, view
from studio_collect.manager import Manager
from studio_collect.pipeline import shard_dir
from studio_collect.store import State
from studio_ia.services import FakeServices, WorkspaceServices
from tests.api_tap import EventTap
from tests.collect_fakes import prose


class CollectApi(unittest.TestCase):
    def setUp(self):
        self.app = create_app(FakeServices(), open=True, hosts=ANY)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def test_options_say_what_is_never_collected_and_why(self):
        options = self.client.get("/collect/options").json()
        hosts = {row["host"]: row["reason"] for row in options["excluded"]}
        self.assertEqual(sorted(hosts), ["quora.com", "reddit.com", "twitter.com", "x.com"])
        self.assertTrue(all("forbids" in reason for reason in hosts.values()))
        self.assertEqual([s["id"] for s in options["sources"]], ["hub", "stackexchange", "github", "web"])
        self.assertTrue(all(s["terms"] for s in options["sources"]))
        self.assertIn("GroundworkTrainer-Collector", options["user_agent"])
        self.assertEqual({t["id"] for t in options["topics"]}, set(config.TOPICS))

    def test_the_defaults_are_off_and_cautious(self):
        got = self.client.get("/collect/config").json()
        self.assertEqual((got["enabled"], got["paused"]), (False, False))
        self.assertEqual(got["licences"], {"non_commercial": False, "unknown": False, "copyleft": False})
        self.assertTrue(got["idle"]["only_on_mains"] and got["idle"]["only_when_idle"])

    def test_limits_are_saved_and_kept_between_requests(self):
        put = self.client.put("/collect/config", json={"limits": {"disk_mb": 512, "bandwidth_mb_per_day": 100}, "idle": {"unmetered_confirmed": True}, "topics": ["science"]})
        self.assertEqual(put.status_code, 200, put.text)
        got = self.client.get("/collect/config").json()
        self.assertEqual((got["limits"]["disk_mb"], got["limits"]["bandwidth_mb_per_day"], got["limits"]["max_dataset_mb"]), (512, 100, 200))
        self.assertEqual((got["idle"]["unmetered_confirmed"], got["idle"]["only_on_mains"], got["topics"]), (True, True, ["science"]))
        self.tap.wait("collect", 1)

    def test_a_bad_setting_is_a_sentence_not_a_crash(self):
        for body, words in (({"limits": {"disk_mb": -5}}, "between"), ({"topics": ["astrology"]}, "Unknown topic"),
                            ({"web": {"sites": [{"host": "old.reddit.com", "licence": "cc-by-4.0"}]}}, "cannot be added"),
                            ({"web": {"sites": [{"host": "example.org", "licence": ""}]}}, "licence")):
            got = self.client.put("/collect/config", json=body)
            self.assertEqual(got.status_code, 400, body)
            self.assertIn(words, got.json()["detail"])
        self.assertEqual(self.client.put("/collect/config", json={"nonsense": 1}).status_code, 422)

    def test_a_website_is_added_with_its_licence(self):
        put = self.client.put("/collect/config", json={"sources": {"web": True}, "web": {"sites": [{"host": "docs.example.org", "licence": "cc-by-4.0", "pages": 50}]}})
        self.assertEqual(put.json()["web"]["sites"], [{"host": "docs.example.org", "seeds": ["https://docs.example.org/"], "licence": "cc-by-4.0", "pages": 50}])

    def test_pause_and_resume(self):
        paused = self.client.post("/collect/pause").json()
        self.assertTrue(paused["paused"])
        self.tap.wait("collect", 1)
        resumed = self.client.post("/collect/resume").json()
        self.assertEqual((resumed["paused"], resumed["enabled"]), (False, True))
        self.assertEqual(self.client.get("/collect/config").json()["enabled"], True)
        self.tap.wait("collect", 2)

    def test_state_before_anything_ran(self):
        state = self.client.get("/collect").json()
        self.assertEqual((state["status"], state["running"], state["plan"], state["sets"], state["waiting_shards"]), ("off", False, [], [], 0))
        self.assertEqual((state["stats"]["kept"], state["stats"]["dropped"]["duplicate"]), (0, 0))


class ViewOfState(unittest.TestCase):
    def test_the_view_joins_the_services_state_with_what_was_registered(self):
        with tempfile.TemporaryDirectory() as root:
            blank = State(root)
            stats = {"seen": 10, "kept": 6, "dropped": {"licence": 1, "quality": 1, "pii": 0, "duplicate": 2, "near_duplicate": 0, "decontam": 0, "invalid": 0}, "redactions": 3, "bytes_kept": 900}
            blank.save(status="running", heartbeat=1000.0, stats=stats, groups={"topic-science": stats}, pid=1,
                       plan=[{"id": "a/b", "source": "hub", "topic": "science", "group": "topic-science", "licence": "mit", "licence_class": "permissive", "bytes": 5, "status": "needs_approval",
                              "reason": "5 MB", "score": 0.7, "files": [{}], "url": "u"}],
                       sources={"hub:a/b": {"status": "reading", "name": "a/b", "files": 3, "file": 1, "topic": "science"}, "github:api": {"blocked_until": 5}})
            config.save(root, {"approved": ["a/b"]})
            state = view.state(root, running=True, now=lambda: 1010.0)
            self.assertEqual((state["status"], state["heartbeat_age"]), ("running", 10.0))
            self.assertEqual(state["sets"][0]["dropped"]["duplicate"], 2)
            self.assertEqual((state["sets"][0]["name"], state["plan"][0]["approved"]), ("Collected · science", True))
            self.assertEqual([s["detail"] for s in state["sources"]], ["file 2 of 3"])                  # the GitHub rate-limit record is not a source
            self.assertEqual(view.state(root, running=False)["status"], "off")                          # a state file with no process behind it


class ManagerTick(unittest.TestCase):
    def test_shards_are_registered_and_the_service_is_started_only_when_wanted(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = studio_core.Workspace(root)
            started, told = [], []
            manager = Manager(workspace, lambda kind, payload=None: told.append(kind), ensure=lambda r: started.append(r) or 1, running=lambda r: False)
            self.assertFalse(manager.tick())
            self.assertEqual(started, [])
            config.save(root, {"enabled": True})
            folder = shard_dir(root, "topic-science")
            folder.mkdir(parents=True)
            row = studio_core.normalize({"text": prose(1), "provenance": {"source_url": "https://x"}, "license": "mit", "source_url": "https://x"})
            for name in ("id", "datasetId", "updatedAt"):
                row.pop(name)
            (folder / "a.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (folder / "b.jsonl.part").write_text("half a lin", encoding="utf-8")                        # still being written: left alone
            self.assertTrue(manager.tick())
            self.assertEqual((len(started), told), (1, ["collect"]))
            workspace = studio_core.Workspace(root)
            self.assertEqual(workspace._dataset("collect_topic-science")["recordCount"], 1)
            self.assertTrue((folder / "b.jsonl.part").exists())
            self.assertEqual(register.totals(root)["collect_topic-science"]["records"], 1)


class RealServices(unittest.TestCase):
    def test_settings_state_pause_and_resume_over_a_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            services = WorkspaceServices.reader(root)
            self.assertFalse(services.collect_config()["enabled"])
            with mock.patch("studio_collect.service.ensure_running") as ensure:
                saved = services.save_collect_config({"enabled": True, "topics": ["law"]})
                self.assertEqual((saved["topics"], ensure.call_count), (["law"], 1))
                services.collect_pause()
                self.assertTrue(services.collect_config()["paused"])
                self.assertEqual(ensure.call_count, 2)             # a paused collector is kept alive too: it waits and notices Resume within seconds
                services.collect_resume()
                self.assertEqual((services.collect_config()["paused"], ensure.call_count), (False, 3))
            with self.assertRaises(ValueError):
                services.save_collect_config({"limits": {"disk_mb": 0}})
            state = services.collect_state()
            self.assertEqual((state["running"], state["status"]), (False, "off"))          # a service that is not running is off, whatever was saved
            self.assertEqual(config.load(root)["topics"], ["law"])


class Wiring(unittest.TestCase):
    def test_the_pages_files_are_found_under_data_collect(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(Path(view.state(root)["file"]), Path(root) / "data" / "collect")


if __name__ == "__main__":
    unittest.main()

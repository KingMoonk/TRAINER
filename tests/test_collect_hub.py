"""Open datasets by topic over a fake hub: ranking, capped and resumable downloads, records with provenance, drops, registration, restart."""
import json
import tempfile
import unittest
from pathlib import Path

import studio_core
from studio_collect import config, hub, register, service
from studio_collect.netio import Http
from studio_collect.pipeline import shard_dir
from studio_collect.store import Ledger, LimitReached, State
from tests.collect_fakes import FakeWeb, hub_entry, idle_probes, jsonl, prose, serve_hub


def science_rows(start, n):
    return [{"text": prose(i, 10)} for i in range(start, start + n)]


class Planning(unittest.TestCase):
    def setUp(self):
        self.web = FakeWeb()
        self.datasets = {
            "acme/science-corpus": (hub_entry("acme/science-corpus", "apache-2.0", 90000), {"data/train.jsonl": jsonl(science_rows(0, 6))}),
            "lab/science-notes": (hub_entry("lab/science-notes", "cc-by-sa-4.0", 800), {"data/train.jsonl": jsonl(science_rows(100, 6))}),
            "mystery/science-dump": (hub_entry("mystery/science-dump", "other", 500000), {"train.jsonl": jsonl(science_rows(200, 3))}),
            "corp/science-nc": (hub_entry("corp/science-nc", "cc-by-nc-4.0", 70000), {"train.jsonl": jsonl(science_rows(300, 3))}),
            "gate/science-gated": (hub_entry("gate/science-gated", "mit", 70000, gated="auto"), {"train.jsonl": jsonl(science_rows(400, 3))}),
            "bench/science-bench": (hub_entry("bench/science-bench", "mit", 70000, tags=["benchmark:official"]), {"train.jsonl": jsonl(science_rows(500, 3))}),
        }
        serve_hub(self.web, self.datasets)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = config.merge({"topics": ["science"]})
        self.http = Http(Ledger(self.tmp.name), self.config["limits"], transport=self.web, sleep=lambda s: None)

    def plan(self, **overrides):
        return {i["id"]: i for i in hub.build_plan(self.http, config.merge({"topics": ["science"], **overrides}))}

    def test_ranks_by_licence_and_fit_and_refuses_what_it_must(self):
        plan = self.plan()
        ranked = [i["id"] for i in hub.build_plan(self.http, self.config) if i["status"] != "skipped"]
        self.assertEqual(ranked, ["acme/science-corpus", "lab/science-notes"])
        self.assertEqual(plan["acme/science-corpus"]["licence_class"], "permissive")
        self.assertEqual(plan["lab/science-notes"]["licence_class"], "share-alike")
        self.assertIn("unknown", plan["mystery/science-dump"]["reason"])
        self.assertIn("non-commercial", plan["corp/science-nc"]["reason"])
        self.assertIn("gated", plan["gate/science-gated"]["reason"])
        self.assertIn("benchmark", plan["bench/science-bench"]["reason"])
        self.assertTrue(all(plan[i]["status"] == "skipped" for i in ("mystery/science-dump", "corp/science-nc", "gate/science-gated", "bench/science-bench")))

    def test_unknown_and_non_commercial_licences_come_in_only_when_opted_in(self):
        plan = self.plan(licences={"unknown": True, "non_commercial": True})
        self.assertEqual(plan["mystery/science-dump"]["status"], "planned")
        self.assertEqual(plan["corp/science-nc"]["status"], "planned")
        self.assertEqual(plan["gate/science-gated"]["status"], "skipped")     # opting in to licences never opens a gated dataset

    def test_the_plan_lists_files_and_bytes_before_anything_is_downloaded(self):
        plan = self.plan()
        item = plan["acme/science-corpus"]
        self.assertEqual([f["path"] for f in item["files"]], ["data/train.jsonl"])
        self.assertEqual(item["bytes"], len(self.datasets["acme/science-corpus"][1]["data/train.jsonl"]))
        self.assertEqual(self.web.asked("/resolve/"), [])

    def test_a_dataset_above_the_approval_threshold_waits_for_approval(self):
        plan = self.plan(limits={"auto_approve_mb": 0})
        self.assertEqual(plan["acme/science-corpus"]["status"], "needs_approval")
        self.assertEqual(self.plan(limits={"auto_approve_mb": 0}, approved=["acme/science-corpus"])["acme/science-corpus"]["status"], "planned")

    def test_evaluation_splits_and_oversize_files_are_not_planned(self):
        tree = [{"type": "file", "path": "train-0.parquet", "size": 5_000_000}, {"type": "file", "path": "validation.jsonl", "size": 100},
                {"type": "file", "path": "test-00000.parquet", "size": 100}, {"type": "file", "path": "train-1.parquet", "size": 5_000_000},
                {"type": "file", "path": "huge.jsonl", "size": 9_000_000_000}, {"type": "file", "path": "README.md", "size": 10}]
        files, planned, total = hub.pick_files(tree, {**self.config["limits"], "max_dataset_mb": 6, "max_file_mb": 1024})
        self.assertEqual([f["path"] for f in files], ["train-0.parquet"])
        self.assertEqual(planned, 5_000_000)

    def test_a_term_fits_only_at_the_start_of_a_word(self):
        self.assertEqual(hub.fit_score({"id": "ops/decode-logs"}, ("code",)), 0)
        self.assertEqual(hub.fit_score({"id": "zoo/claw-machine"}, ("law",)), 0)
        self.assertGreater(hub.fit_score({"id": "nvidia/OpenMathInstruct"}, ("math",)), 0)

    def test_media_and_non_english_datasets_are_skipped(self):
        cfg = self.config
        self.assertIn("not text only", hub.screen(hub_entry("lab/science-vqa", tags=["modality:image"]), cfg))
        self.assertIn("not English", hub.screen(hub_entry("gov/science-th", tags=["language:th"]), cfg))
        self.assertEqual(hub.screen(hub_entry("lab/science-en", tags=["language:en", "language:fr", "modality:tabular"]), cfg), "")
        self.assertIn("not text", hub.screen({**hub_entry("m/science-prices"), "tags": ["license:mit", "modality:timeseries"]}, cfg))

    def test_a_plan_made_under_older_screening_rules_is_made_again(self):
        self.assertEqual(hub.Adapter.plan_inputs(self.config)["screen"], hub.SCREEN)


class Download(unittest.TestCase):
    def test_a_dropped_connection_resumes_with_a_range_request_and_verifies_the_hash(self):
        data = jsonl(science_rows(0, 40))
        web = FakeWeb()
        web.add("https://huggingface.co/datasets/a/b/resolve/x/f.jsonl", data)
        with tempfile.TemporaryDirectory() as tmp:
            http = Http(Ledger(tmp), config.merge({})["limits"], transport=web, sleep=lambda s: None)
            web.cut_after = 5000
            dest = Path(tmp) / "f.jsonl"
            from tests.collect_fakes import sha256
            with self.assertRaises(OSError):
                hub.fetch(http, "https://huggingface.co/datasets/a/b/resolve/x/f.jsonl", dest, len(data), sha256(data))
            self.assertEqual(dest.with_name("f.jsonl.part").stat().st_size, 5000)
            hub.fetch(http, "https://huggingface.co/datasets/a/b/resolve/x/f.jsonl", dest, len(data), sha256(data))
            self.assertEqual(dest.read_bytes(), data)
            self.assertEqual(web.headers_log[-1]["Range"], "bytes=5000-")
            self.assertFalse(dest.with_name("f.jsonl.part").exists())

    def test_a_wrong_hash_is_discarded(self):
        web = FakeWeb()
        web.add("https://huggingface.co/x", b"abcdef")
        with tempfile.TemporaryDirectory() as tmp:
            http = Http(Ledger(tmp), config.merge({})["limits"], transport=web, sleep=lambda s: None)
            with self.assertRaisesRegex(ValueError, "checksum"):
                hub.fetch(http, "https://huggingface.co/x", Path(tmp) / "f", 6, "0" * 64)
            self.assertFalse((Path(tmp) / "f").exists())

    def test_the_daily_bandwidth_cap_stops_a_download_and_keeps_the_part(self):
        web = FakeWeb()
        web.add("https://huggingface.co/big", b"x" * 3_000_000)
        with tempfile.TemporaryDirectory() as tmp:
            http = Http(Ledger(tmp), {**config.merge({})["limits"], "bandwidth_mb_per_day": 1}, transport=web, sleep=lambda s: None)
            with self.assertRaises(LimitReached):
                hub.fetch(http, "https://huggingface.co/big", Path(tmp) / "big", 3_000_000)
            self.assertTrue((Path(tmp) / "big.part").exists())


class Rows(unittest.TestCase):
    item = {"id": "a/b", "group": "topic-science", "licence": "apache-2.0", "author": "a", "sha": "s", "topic": "science"}

    def test_pairs_chats_and_text(self):
        pair = hub.row_to_candidates({"instruction": "Explain the idea of entropy in simple words.", "input": "", "output": "Entropy measures disorder."}, self.item, "r")[0]
        chat = hub.row_to_candidates({"messages": [{"role": "user", "content": "Hello there, what is a prime?"}, {"role": "assistant", "content": "A number with two divisors."}]}, self.item, "r")[0]
        text = hub.row_to_candidates({"text": "x" * 300}, self.item, "r")[0]
        self.assertEqual((pair.kind, chat.kind, text.kind), ("qa", "qa", "text"))
        self.assertEqual(pair.url, "https://huggingface.co/datasets/a/b")
        self.assertIn("Hugging Face", pair.attribution)
        self.assertEqual(hub.row_to_candidates({"id": 4, "label": 1}, self.item, "r"), [])

    def test_a_json_array_larger_than_one_read_is_read_whole(self):
        rows = [{"text": prose(i, 10)} for i in range(120)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            self.assertGreater(path.stat().st_size, 64 * 1024)
            self.assertEqual(list(hub.iter_rows(path)), rows)


class EndToEnd(unittest.TestCase):
    """The service over the fake hub: shards with provenance, the drop counts, the workspace, and a restart in the middle."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        rows = science_rows(0, 30)
        rows += [rows[0], rows[1]]                                                         # exact duplicates inside the file
        rows += [{"text": rows[2]["text"] + " One more closing sentence about the model."}]  # a near-duplicate
        rows += [{"text": "Contact me at someone@example.org. " + prose(900, 9)}]          # personal data is stripped, not dropped
        rows += [{"text": "tiny"}]                                                         # quality
        self.rows = rows
        self.web = FakeWeb()
        serve_hub(self.web, {"acme/science-corpus": (hub_entry("acme/science-corpus", "apache-2.0"), {"train.jsonl": jsonl(rows)})})
        config.save(self.root, {"enabled": True, "topics": ["science"], "idle": {"unmetered_confirmed": True}})

    def drain(self, **kw):
        for _ in range(40):
            outcome = service.run(self.root, transport=self.web, probes=idle_probes(), sleep=lambda s: None, once=True, lock=False, **kw)
            if outcome in ("idle", "off", "error", "limit"):
                return outcome
        self.fail("the service never went idle")

    def shard_rows(self):
        out = []
        for files in register.pending(self.root).values():
            for path in files:
                out += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return out

    def test_records_land_with_provenance_and_the_drops_are_counted(self):
        existing = studio_core.Workspace(self.root)
        mine = existing.create_dataset("mine", storage="sqlite")
        existing.add(mine, [self.rows[5]])                                                  # already in the workspace
        Path(self.root, "data", "heldout.json").write_text(json.dumps([{"id": "h", "text": self.rows[6]["text"]}]), encoding="utf-8")
        self.assertEqual(self.drain(), "idle")
        kept = self.shard_rows()
        state = State(self.root).data
        self.assertEqual(len(kept), 30 - 1 - 1 + 1)                                           # 30, minus one in the workspace, one held out, plus the PII row
        for row in kept:
            p = row["provenance"]
            self.assertEqual((p["source"], p["source_url"], p["licence"]), ("hub:acme/science-corpus", "https://huggingface.co/datasets/acme/science-corpus", "apache-2.0"))
            self.assertRegex(p["retrieved_at"], r"^\d{4}-\d\d-\d\dT")
            self.assertIn("acme/science-corpus", p["attribution"])
        self.assertFalse(any("someone@example.org" in r["text"] for r in kept))
        self.assertTrue(any("[EMAIL]" in r["text"] for r in kept))
        dropped = state["stats"]["dropped"]
        self.assertEqual((dropped["duplicate"], dropped["near_duplicate"], dropped["decontam"], dropped["quality"]), (3, 1, 1, 1))
        self.assertEqual(state["stats"]["redactions"], 1)
        self.assertEqual(state["stats"]["kept"], len(kept))

    def test_the_records_join_the_workspace_as_a_text_set(self):
        self.drain()
        workspace = studio_core.Workspace(self.root)
        report = register.register(workspace)
        self.assertEqual(report["added"], 31)
        self.assertEqual(register.pending(self.root), {})
        reloaded = studio_core.Workspace(self.root)
        dataset = reloaded._dataset("collect_topic-science")
        self.assertEqual((dataset["name"], dataset["storage"], dataset["recordCount"]), ("Collected · science", "sqlite", 31))
        example = next(iter(reloaded.iter_examples("collect_topic-science")))
        self.assertTrue(example["text"] and example["license"] == "apache-2.0" and example["provenance"]["source_url"])
        self.assertEqual(register.register(reloaded)["added"], 0)

    def test_a_second_pass_finds_nothing_new(self):
        self.drain()
        before = len(self.shard_rows())
        self.assertEqual(self.drain(), "idle")
        self.assertEqual(len(self.shard_rows()), before)

    def test_it_waits_off_battery_and_busy_machines_and_resumes(self):
        for probes, expect in ((idle_probes(on_battery=lambda: True), "battery|mains"), (idle_probes(metered=lambda: True), "metered"),
                               (idle_probes(queue_busy=lambda root: True), "training"), (idle_probes(cpu=lambda: 99.0), "busy")):
            outcome = service.run(self.root, transport=self.web, probes=probes, sleep=lambda s: None, once=True, lock=False)
            self.assertEqual(outcome, "waiting")
            self.assertRegex(State(self.root).data["reason"], f"(?i){expect}")
        self.assertEqual(self.web.log, [])
        config.save(self.root, {"paused": True})
        self.assertEqual(service.run(self.root, transport=self.web, probes=idle_probes(), sleep=lambda s: None, once=True, lock=False), "waiting")
        self.assertEqual(State(self.root).data["status"], "paused")
        config.save(self.root, {"paused": False})
        self.assertEqual(self.drain(), "idle")
        self.assertTrue(self.shard_rows())

    def test_it_resumes_where_it_stopped_after_a_restart(self):
        clock = iter(range(0, 10 ** 6, 1))
        calls = {"n": 0}

        downloaded = Path(self.root) / "data" / "collect" / "downloads" / "acme-science-corpus" / "train.jsonl"

        def busy_after_a_while(root):
            calls["n"] += downloaded.exists()
            return calls["n"] > 12                                                         # the machine gets busy part-way through reading the file
        outcome = service.run(self.root, transport=self.web, probes=idle_probes(queue_busy=busy_after_a_while), sleep=lambda s: None,
                              clock=lambda: next(clock), once=True, lock=False)
        self.assertEqual(outcome, "stopped")
        first = State(self.root).data
        partial = first["stats"]["kept"]
        self.assertTrue(0 < partial < 31, first["stats"])
        self.assertGreater(first["sources"]["hub:acme/science-corpus"]["offset"], 0)
        self.assertEqual(self.drain(), "idle")                                                 # a new process: a new State read from disk
        self.assertEqual(State(self.root).data["stats"]["kept"], 31)
        self.assertEqual(len(self.shard_rows()), 31)
        self.assertEqual(len(self.web.asked("/resolve/")), 1)                                  # the file was not fetched again

    def test_a_disk_cap_stops_collection(self):
        config.save(self.root, {"limits": {"disk_mb": 64}})
        (Path(self.root) / "data" / "collect" / "filler.bin").write_bytes(b"0" * 65_000_000)
        self.assertEqual(self.drain(), "limit")
        self.assertEqual(State(self.root).data["status"], "waiting")
        self.assertIn("disk cap", State(self.root).data["reason"])


class Registration(unittest.TestCase):
    def test_a_line_separator_inside_a_record_does_not_split_it(self):
        with tempfile.TemporaryDirectory() as root:
            folder = shard_dir(root) / "topic-poetry"
            folder.mkdir(parents=True)
            rows = [{"text": "one line\u2028and the next, as the page had it"}, {"text": "a plain second record"}]
            (folder / "a.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(register.register(studio_core.Workspace(root))["added"], 2)
            self.assertEqual(register.pending(root), {})


if __name__ == "__main__":
    unittest.main()

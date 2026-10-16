import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from studio_api.routers import data as data_router
from studio_ia.services import FakeServices
from tests.sse_harness import ApiCase

NOTES = b'{"text": "x"}\n'


class EventingServices(FakeServices):
    """FakeServices that emit what WorkspaceServices emits for imports and downloads, and read the uploaded files."""

    def __init__(self):
        super().__init__()
        self.imported = []

    def import_data(self, kind, paths):
        self.imported.append((kind, {Path(p).name: Path(p).read_bytes() for p in paths}, [Path(p).parent for p in paths]))
        self.emit("data", {"kind": kind})
        return [f"{Path(p).name}: 1 added" for p in paths]

    def download_sources(self, ids=None, kind="text", install=False):
        self.emit("download", {"id": "s1", "text": "Fetching", "completed": 1, "total": 2})
        self.emit("download_done", {"id": "s1"})
        self.emit("downloads_finished", {"ids": ["s1"]})
        return super().download_sources(ids, kind, install)

    def import_source(self, source_id):
        self.emit("import_done", {"id": source_id})
        return super().import_source(source_id)


class DataRouterTests(ApiCase):
    prefix = "/data"

    def make_services(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.root = Path(self.home.name).resolve()
        (self.root / "corpus").mkdir()
        (self.root / "corpus" / "stories.txt").write_text("Once upon a tide.")
        (self.root / "notes.jsonl").write_bytes(NOTES)  # bytes: write_text would turn \n into \r\n on Windows
        (self.root / ".hidden").write_text("no")
        services = EventingServices()
        services.workspace = SimpleNamespace(root=self.root)
        services.draft_source = lambda kind, n: [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(n)]
        services._collection = [dict(id="s1", name="Stories", size="6 MB download", records="15,488 paragraphs", state="not downloaded",
                                     completed=0, total=0, error="")]
        return services

    def test_every_route_answers(self):
        self.assertTrue(self.call("GET", "/data/has_text")["has_text"])
        self.assertEqual(self.call("GET", "/data/drafts"), {"count": 0})

        summary = self.job("POST", "/data/{kind}/summary", path={"kind": "text"}, json={})
        self.assertEqual(summary["status"], "done")
        self.assertEqual((summary["result"]["kind"], summary["result"]["count"], len(summary["result"]["rows"])), ("text", 2, 2))
        self.assertEqual(summary["result"]["sets"], [{"id": "Your files", "name": "Your files", "count": 2}])
        found = self.job("POST", "/data/{kind}/summary", path={"kind": "text"}, json={"query": "heron"})
        self.assertEqual([r["id"] for r in found["result"]["rows"]], ["t2"])

        page = self.call("GET", "/data/{kind}/rows", path={"kind": "conversations"}, params={"limit": 2})
        self.assertEqual([r["id"] for r in page["rows"]], ["c1", "c2"])
        self.assertIsInstance(page["next_cursor"], str)
        rest = self.call("GET", "/data/{kind}/rows", path={"kind": "conversations"}, params={"limit": 2, "cursor": page["next_cursor"]})
        self.assertEqual(([r["id"] for r in rest["rows"]], rest["next_cursor"]), (["c3"], None))

        self.assertEqual(self.call("POST", "/data/{kind}/rows", path={"kind": "text"}, json={"fields": {"text": "Fresh prose."}}), {"added": True})
        self.assertEqual(self.call("POST", "/data/{kind}/rows", path={"kind": "text"}, json={"fields": {"text": "Fresh prose."}}), {"added": False})

        self.assertEqual(self.call("GET", "/data/examples/{example_id}/fields", path={"example_id": "c1"})["fields"],
                         {"user": "Name a prime.", "assistant": "7 is a prime number."})
        edited = self.call("PUT", "/data/examples/{example_id}", path={"example_id": "c1"},
                           json={"fields": {"user": "Name a prime.", "assistant": "11 is prime."}})
        self.assertEqual(edited["fields"]["assistant"], "11 is prime.")

        made = self.job("POST", "/data/{kind}/drafts", path={"kind": "conversations"}, json={"n": 2})
        self.assertEqual((made["status"], made["result"]), ("done", {"count": 2}))
        self.assertEqual(self.call("GET", "/data/drafts"), {"count": 2})
        drafts = self.call("GET", "/data/{kind}/rows", path={"kind": "conversations"}, params={"query": "q"})["rows"]
        keep, drop = [r["id"] for r in drafts if r["status"] == "Review"][:2]
        self.assertIsNone(self.call("POST", "/data/examples/{example_id}/review", 204, path={"example_id": keep}, json={"keep": True}))
        self.assertIsNone(self.call("POST", "/data/examples/{example_id}/review", 204, path={"example_id": drop}, json={"keep": False}))
        self.assertEqual(self.call("GET", "/data/drafts"), {"count": 0})

        report = self.call("POST", "/data/problems/generate", json={"rule": "reverse", "n": 3, "store": False})
        self.assertEqual((len(report["items"]), report["added"], report["checked"] == report["passed"]), (3, 0, True))
        self.assertGreater(self.call("POST", "/data/problems/generate", json={"n": 4})["added"], 0)

        sources = self.call("GET", "/data/collections/{kind}", path={"kind": "text"})
        self.assertEqual((sources[0]["id"], sources[0]["state"]), ("s1", "not downloaded"))
        self.assertEqual(self.call("GET", "/data/collections/{kind}", path={"kind": "problems"}), [])
        self.assertEqual(self.call("POST", "/data/collections/{kind}/download", path={"kind": "text"}, json={"ids": ["s1"], "install": True}),
                         {"started": True})
        self.assertEqual(self.call("POST", "/data/sources/{source_id}/import", path={"source_id": "s1"}), {"started": True})
        self.assertEqual(self.job("POST", "/data/refresh_downloads")["result"], [])

        uploaded = self.job("POST", "/data/{kind}/import", path={"kind": "text"},
                            files=[("files", ("a.txt", b"hello")), ("files", ("b.jsonl", b'{"text": "x"}\n'))])
        self.assertEqual(uploaded["result"], ["a.txt: 1 added", "b.jsonl: 1 added"])

        self.assertEqual([e["path"] for e in self.call("GET", "/data/browse")["entries"]][0], str(self.root))
        picked = self.job("POST", "/data/{kind}/import_paths", path={"kind": "text"},
                          json={"paths": [str(self.root / "corpus" / "stories.txt")]})
        self.assertEqual(picked["result"], ["stories.txt: 1 added"])
        self.assertAllRoutesHit()

    def test_an_upload_reaches_the_service_by_its_own_name_and_is_cleaned_up(self):
        self.job("POST", "/data/conversations/import",
                 files=[("files", ("../../evil.txt", b"one")), ("files", ("same.txt", b"two")), ("files", ("sub/same.txt", b"three"))])
        kind, files, folders = self.services.imported[0]
        self.assertEqual(kind, "conversations")
        self.assertEqual(sorted(files.values()), [b"one", b"three", b"two"])
        self.assertIn("evil.txt", files)
        self.assertEqual(len(set(folders)), 1)
        self.assertFalse(folders[0].exists(), "the temporary folder is removed once the import is over")
        self.assertEqual(folders[0].parent, data_router.UPLOADS)

    def test_a_failed_import_still_cleans_up(self):
        def broken(kind, paths):
            self.seen = Path(paths[0]).parent
            raise ValueError("No chat pairs or text records found.")
        self.services.import_data = broken
        failed = self.job("POST", "/data/text/import", files=[("files", ("x.txt", b"?"))])
        self.assertEqual(failed["status"], "failed")
        self.assertIn("No chat pairs or text records found.", failed["error"])
        self.assertFalse(self.seen.exists())

    def test_browse_lists_the_workspace_and_refuses_anything_else(self):
        start = self.call("GET", "/data/browse")
        self.assertIsNone(start["path"])
        self.assertIn(str(self.root), [e["path"] for e in start["entries"]])
        self.assertIn(str(Path.home().resolve()), [e["path"] for e in start["entries"]])

        folder = self.call("GET", "/data/browse", params={"path": str(self.root)})
        self.assertEqual([(e["name"], e["is_dir"]) for e in folder["entries"]], [("corpus", True), ("notes.jsonl", False)])  # folders first, no dotfiles
        self.assertEqual(folder["entries"][1]["size"], len(NOTES))
        inner = self.call("GET", "/data/browse", params={"path": str(self.root / "corpus")})
        self.assertEqual(inner["parent"], str(self.root))
        self.assertEqual([e["name"] for e in inner["entries"]], ["stories.txt"])

        outside = Path.home().resolve().parent
        for path in (str(outside), str(self.root / "corpus" / ".." / ".." / ".." / ".."), str(Path(outside.anchor))):
            if Path(path).resolve() in data_router.roots(self.services) or data_router.inside(Path(path).resolve(), data_router.roots(self.services)):
                continue  # this machine keeps its workspace under that path
            self.call("GET", "/data/browse", 403, params={"path": path})
        self.call("GET", "/data/browse", 400, params={"path": "relative/folder"})
        self.call("GET", "/data/browse", 404, params={"path": str(self.root / "notes.jsonl")})

    def test_a_link_cannot_lead_out_of_the_roots(self):
        outside = Path.home().resolve().parent
        if data_router.inside(outside, data_router.roots(self.services)):
            self.skipTest("the workspace is not under the home folder's parent here")
        try:
            os.symlink(outside, self.root / "escape", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("no symbolic links here")
        self.call("GET", "/data/browse", 403, params={"path": str(self.root / "escape")})
        self.call("POST", "/data/text/import_paths", 403, json={"paths": [str(self.root / "escape")]})

    def test_picking_paths_is_checked(self):
        self.call("POST", "/data/text/import_paths", 400, json={"paths": []})
        self.call("POST", "/data/text/import_paths", 400, json={"paths": [str(self.root / "corpus")]})  # a folder
        self.call("POST", "/data/text/import_paths", 400, json={"paths": [str(self.root / "missing.txt")]})
        self.call("POST", "/data/text/import_paths", 403, json={"paths": [str(Path.home().resolve().parent / "etc" / "passwd")]})
        self.assertEqual(self.services.imported, [])

    def test_bad_requests_say_what_is_wrong(self):
        self.call("GET", "/data/nonsense/rows", 422)
        self.call("POST", "/data/problems/rows", 400, json={"fields": {"prompt": "no rule"}})
        self.call("POST", "/data/conversations/drafts", 202, json={"n": 1})
        self.call("POST", "/data/problems/drafts", 422, json={"n": 1})
        self.call("POST", "/data/problems/generate", 422, json={"n": 0})
        self.call("GET", "/data/conversations/rows", 400, params={"cursor": "not-a-cursor!"})
        self.call("GET", "/data/examples/{example_id}/fields", 404, path={"example_id": "nope"})
        self.call("POST", "/data/examples/{example_id}/review", 404, path={"example_id": "nope"}, json={"keep": True})
        self.call("PUT", "/data/examples/{example_id}", 400, path={"example_id": "c1"}, json={"fields": {"user": "only half"}})
        empty = self.client.put("/data/examples/c1", json={"fields": {"user": "", "assistant": ""}})
        self.assertEqual((empty.status_code, empty.json()["detail"]), (400, "Nothing to save: fill in every part."))
        self.call("POST", "/data/text/import", 422)  # no files

    def test_a_busy_download_is_a_409(self):
        self.services.download_sources = lambda *args: False
        self.services.import_source = lambda source_id: False
        self.call("POST", "/data/collections/text/download", 409, json={})
        self.call("POST", "/data/sources/s1/import", 409)

    def test_no_local_model_fails_the_drafts_job_in_words(self):
        self.services.draft_source = None
        failed = self.job("POST", "/data/text/drafts", json={"n": 1})
        self.assertEqual(failed["status"], "failed")
        self.assertIn("No local model connected.", failed["error"])

    def test_every_event_a_data_route_emits_reaches_events(self):
        self.call("POST", "/data/text/rows", json={"fields": {"text": "Another."}})
        self.call("POST", "/data/collections/text/download", json={})
        self.call("POST", "/data/sources/s1/import")
        self.job("POST", "/data/conversations/import", files=[("files", ("a.txt", b"x"))])
        self.job("POST", "/data/text/import_paths", json={"paths": [str(self.root / "notes.jsonl")]})
        self.job("POST", "/data/conversations/drafts", json={"n": 1})
        self.assertDelivered()
        for kind in ("data", "download", "download_done", "downloads_finished", "import_done", "job"):
            self.assertTrue(self.events.of(kind), f"no {kind} event reached /events")
        self.assertEqual(self.events.of("download")[0], {"id": "s1", "text": "Fetching", "completed": 1, "total": 2})
        self.assertEqual(self.events.of("data")[0], {"kind": "text"})


if __name__ == "__main__":
    unittest.main()

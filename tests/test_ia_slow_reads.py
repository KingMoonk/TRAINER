"""Stored-data reads are done once per change, in one pass, and ready() tells the shell when one would be long."""
import tempfile
import unittest
from unittest import mock

import studio_data_kinds as dk
from studio_core import Workspace
from studio_ia.services import WorkspaceServices


class SlowReads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Workspace(self.tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)
        self.big = self.ws.create_dataset("Stories", storage="sqlite")
        self.ws.add(self.big, [{"text": f"story number {i} about a fox"} for i in range(30)])
        self.ws.add(self.ws.create_dataset("Talk"), [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(3)])

    def test_one_pass_serves_every_kind_until_the_data_changes(self):
        with mock.patch.object(dk, "scan", wraps=dk.scan) as scan:
            counts = {k: self.svc.data(k).count for k in dk.KINDS}
            self.svc.data("text", "number 7")
            self.svc.post_training(None)
            self.assertEqual((counts, scan.call_count), ({"text": 30, "conversations": 3, "problems": 0}, 2))  # one read per dataset
            self.ws.add(self.big, [{"text": "one more story"}])
            self.assertEqual((self.svc.data("text").count, scan.call_count), (31, 3))  # and the change reads only its dataset

    def test_cached_summary_matches_a_direct_read_and_filters_by_query(self):
        for kind in dk.KINDS:
            self.assertEqual(self.svc.data(kind), dk.summarize(self.ws, kind))
        self.assertEqual([r["b"] for r in self.svc.data("text", "number 7").rows], ["story number 7 about a fox"])
        self.assertEqual(self.svc.post_training(None)["example"], {"prompt": self.svc.post_rows("conversations")[0]["user"],
                                                                   "answer": self.svc.post_rows("conversations")[0]["assistant"]})

    def test_ready_sends_only_a_large_uncached_read_to_the_worker(self):
        read = lambda: self.svc.data("text")  # noqa: E731
        self.assertTrue(self.svc.ready(read))  # small store: read in place
        self.ws._dataset(self.big)["recordCount"] = 10 ** 6
        self.assertFalse(self.svc.ready(read))
        self.assertNotIn("summaries", self.svc._data)  # asking did not read anything
        read()
        self.assertTrue(self.svc.ready(read))
        self.ws.add(self.big, [{"text": "another"}])
        self.ws._dataset(self.big)["recordCount"] = 10 ** 6
        self.assertFalse(self.svc.ready(read))

    def test_summary_is_saved_and_a_new_launch_reads_it_until_the_files_change(self):
        first = {k: self.svc.data(k) for k in dk.KINDS}
        example = self.svc.post_training(None)["example"]
        with mock.patch.object(dk, "scan", wraps=dk.scan) as scan:
            again = WorkspaceServices(Workspace(self.tmp.name))
            self.addCleanup(again.close)
            self.assertEqual(({k: again.data(k) for k in dk.KINDS}, again.post_training(None)["example"], scan.call_count), (first, example, 0))
            later = Workspace(self.tmp.name)
            later.add(self.big, [{"text": "written after the summary was saved"}])
            after = WorkspaceServices(Workspace(self.tmp.name))
            self.addCleanup(after.close)
            self.assertEqual((after.data("text").count, scan.call_count), (31, 1))

    def test_scan_reports_what_it_reads(self):
        notes = []
        dk.scan(self.ws, progress=lambda text, fraction: notes.append((text, fraction)))
        self.assertEqual(notes, [("Reading Stories", 0.0)])


if __name__ == "__main__":
    unittest.main()

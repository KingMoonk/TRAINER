"""The label of a machine that shares its hostname with another, against tests/fixtures/node-labels.json. The web test
(web/tests/node-name.spec.ts) reads the same file for nodeName, so the two rules cannot drift."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import studio_cluster
import studio_cluster_queue as cq

CASES = json.loads((Path(__file__).parent / "fixtures" / "node-labels.json").read_text(encoding="utf-8"))["cases"]


class NodeLabels(unittest.TestCase):
    def test_every_fixture_case(self):
        for case in CASES:
            with self.subTest(case["name"]):
                infos = [studio_cluster.NodeInfo(address=n["address"], hostname=n["hostname"]) for n in case["nodes"]]
                own, peers = infos[0], [cq.Peer(info) for info in infos[1:]]
                self.assertEqual(own.address, studio_cluster.LOCAL)
                # a waiting job pinned by address reads its machine's label back from the row
                jobs = [SimpleNamespace(id=n.address, kind="pretrain", label="", status="waiting", params={"pinned": n.address}, node="", note="")
                        for n in infos]
                rows = cq.merge(own, peers, jobs)
                self.assertEqual({row.id: row.hostname for row in rows}, case["labels"])


if __name__ == "__main__":
    unittest.main()

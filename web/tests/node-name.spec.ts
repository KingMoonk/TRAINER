import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { nodeName } from "../src/lib/node-name";

// The naming rule on its own, no browser and no server: every case in tests/fixtures/node-labels.json, the same file
// tests/test_node_labels.py checks studio_cluster_queue.merge against.

type Case = { name: string; nodes: { address: string; hostname: string }[]; labels: Record<string, string> };
const { cases } = JSON.parse(readFileSync(path.resolve(__dirname, "..", "..", "tests", "fixtures", "node-labels.json"), "utf8")) as {
  cases: Case[];
};

for (const { name, nodes, labels } of cases) {
  test(`node names: ${name}`, () => {
    const listed = nodes.map((n) => ({ ...n, local: n.address === "local" }));
    expect(Object.fromEntries(listed.map((n) => [n.address, nodeName(n, listed)]))).toEqual(labels);
  });
}

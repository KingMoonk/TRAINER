import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, test } from "vitest";
import { nodeName } from "./node-name";

// The naming rule on its own: every case in tests/fixtures/node-labels.json, the same file
// tests/test_node_labels.py checks studio_cluster_queue.merge against.

type Case = { name: string; nodes: { address: string; hostname: string }[]; labels: Record<string, string> };
const { cases } = JSON.parse(readFileSync(path.resolve(__dirname, "..", "..", "..", "tests", "fixtures", "node-labels.json"), "utf8")) as {
  cases: Case[];
};

describe("the shared node-label fixture", () => {
  test("has cases to run", () => expect(cases.length).toBeGreaterThan(0));

  for (const { name, nodes, labels } of cases) {
    test(`node names: ${name}`, () => {
      const listed = nodes.map((n) => ({ ...n, local: n.address === "local" }));
      expect(Object.fromEntries(listed.map((n) => [n.address, nodeName(n, listed)]))).toEqual(labels);
    });
  }
});

describe("nodeName", () => {
  const node = (address: string, hostname: string, local = false) => ({ address, hostname, local });

  test("a machine with no name is called by its address", () => {
    const a = node("100.64.0.1", "");
    expect(nodeName(a, [a])).toBe("100.64.0.1");
  });

  test("a machine alone in the list keeps its bare name", () => {
    const a = node("100.64.0.1", "studio");
    expect(nodeName(a, [a])).toBe("studio");
  });

  test("two remote machines with one name get their addresses added, in any letter case", () => {
    const a = node("100.64.0.1", "Studio");
    const b = node("100.64.0.2", "studio");
    const list = [a, b];
    expect(nodeName(a, list)).toBe("Studio (100.64.0.1)");
    expect(nodeName(b, list)).toBe("studio (100.64.0.2)");
  });

  test("this machine keeps its bare name even when a peer shares it, while the peer gets its address", () => {
    const mine = node("local", "studio", true);
    const peer = node("100.64.0.2", "studio");
    expect(nodeName(mine, [mine, peer])).toBe("studio");
    expect(nodeName(peer, [mine, peer])).toBe("studio (100.64.0.2)");
  });

  test("a peer without a name never clashes", () => {
    const a = node("100.64.0.1", "studio");
    const b = node("100.64.0.2", "");
    expect(nodeName(a, [a, b])).toBe("studio");
    expect(nodeName(b, [a, b])).toBe("100.64.0.2");
  });

  test("a node is not compared with itself", () => {
    const a = node("100.64.0.1", "studio");
    expect(nodeName(a, [a, a])).toBe("studio");
  });
});

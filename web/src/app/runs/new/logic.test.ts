import { describe, expect, test } from "vitest";
import { ctaLabel, defaultSftVersion, localForced, peersOf, pinnedJob, runHref, shapeOf, workersOf, type Node } from "./logic";

const node = (address: string, over: Partial<Node> = {}): Node => ({
  address, busy: false, device: "cuda", device_name: "", dtypes: [], error: "", hostname: address, local: false, memory_bytes: 0, online: true, tailnet_name: "", version: "", ...over,
});
const nodes = [node("local", { local: true, hostname: "mac" }), node("beast"), node("lab")];

describe("workersOf", () => {
  test("Pretrain keeps what is ticked, and a run over several peers brings this computer in", () => {
    expect(workersOf("pretrain", ["local"], nodes)).toEqual(["local"]);
    expect(workersOf("pretrain", ["beast"], nodes)).toEqual(["beast"]);
    expect(workersOf("pretrain", ["local", "beast"], nodes)).toEqual(["local", "beast"]);
    expect(workersOf("pretrain", ["beast", "lab"], nodes)).toEqual(["local", "beast", "lab"]);
    expect(localForced("pretrain", ["beast", "lab"], nodes)).toBe(true);
    expect(localForced("pretrain", ["local", "beast"], nodes)).toBe(false);
  });

  test("SFT and Ablation run on one machine, and a machine that is not listed is dropped", () => {
    expect(workersOf("sft", ["beast", "lab"], nodes)).toEqual(["beast"]);
    expect(workersOf("ablation", ["gone"], nodes)).toEqual([]);
  });
});

test("peersOf leaves this computer out", () => {
  expect(peersOf(["local", "beast"], nodes)).toEqual(["beast"]);
});

describe("shapeOf", () => {
  test("names the topology", () => {
    expect(shapeOf([], nodes).label).toBe("Invalid");
    expect(shapeOf(["local"], nodes).label).toBe("Local");
    expect(shapeOf(["beast"], nodes)).toMatchObject({ label: "Remote", tip: expect.stringContaining("beast") });
    expect(shapeOf(["local", "beast", "lab"], nodes).label).toBe("DiLoCo · |W| = 3 · c = this computer");
  });
});

test("the button says Start only when the queue starts the job at once", () => {
  expect(ctaLabel("Starts now")).toBe("Start");
  expect(ctaLabel("Starts after Version 1 (about 4 min)")).toBe("Add to queue");
  expect(ctaLabel(undefined)).toBe("Add to queue");
});

test("pinnedJob keeps the plan's params and adds the peer", () => {
  const job = { kind: "pretrain" as const, label: "Version 2", params: { version: 2 } };
  expect(pinnedJob(job, null)).toEqual({ kind: "pretrain", label: "Version 2", note: "", params: { version: 2 } });
  expect(pinnedJob(job, "beast").params).toEqual({ version: 2, pinned: "beast" });
});

test("SFT starts from the best trained version, else the latest", () => {
  const v = (number: number, best: string | null) => ({ number, best_checkpoint: best }) as Parameters<typeof defaultSftVersion>[0][number];
  expect(defaultSftVersion([v(1, "a"), v(2, null)])?.number).toBe(1);
  expect(defaultSftVersion([v(1, null), v(2, null)])?.number).toBe(2);
  expect(defaultSftVersion([])).toBeUndefined();
});

test("a run's page is a query parameter", () => {
  expect(runHref("run 1")).toBe("/runs/view/?id=run%201");
});

import { describe, expect, test } from "vitest";
import { busyPeers, ctaLabel, defaultSftVersion, durationText, nameList, runningText, localForced, peersOf, pinnedJob, runHref, shapeOf, sourceFacts, workersOf, type Node, type Source } from "./logic";

const node = (address: string, over: Partial<Node> = {}): Node => ({
  address, busy: false, device: "cuda", device_name: "", dtypes: [], error: "", hostname: address, local: false, memory_bytes: 0, online: true, tailnet_name: "", origin: "", origin_down: false, version: "", ...over,
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

describe("busy machines", () => {
  const some = [node("local", { local: true, busy: true }), node("beast", { busy: true }), node("lab"), node("off", { busy: true, online: false })];
  const rows = [{ status: "running", node: "beast", label: "Post-training", kind: "sft", eta_s: 90 }, { status: "waiting", node: "lab", label: "Later", kind: "sft", eta_s: null }] as Parameters<typeof runningText>[1];

  test("only the picked peers that answer and are busy are left out; this computer is never counted", () => {
    expect(busyPeers(["local", "beast", "lab", "off"], some).map((n) => n.address)).toEqual(["beast"]);
    expect(busyPeers(["local", "lab"], some)).toEqual([]);
  });

  test("the names read as a sentence part", () => {
    expect([nameList([]), nameList(["A"]), nameList(["A", "B"]), nameList(["A", "B", "C"])]).toEqual(["", "A", "A and B", "A, B and C"]);
  });

  test("what a machine runs: its running job with the time left, else nothing", () => {
    expect(runningText("beast", rows)).toBe("Post-training · about 2 min left");
    expect(runningText("lab", rows)).toBe("");
    expect(runningText("beast", [{ ...rows[0], eta_s: 3 * 3600 }])).toBe("Post-training · about 3 hours left");
  });
});

test("durationText reads a wall time in plain words", () => {
  expect([45, 720, 5400, 7200, 183_600, 172_800].map(durationText)).toEqual(["45 s", "12 min", "1 h 30 min", "2 h", "2 d 3 h", "2 d"]);
});

describe("sourceFacts", () => {
  const source = (over: Partial<Source> = {}): Source => ({
    version: 4, name: "deep", change: "", status: "stopped", finished_at: "2026-10-19T08:00:00", steps: 300, steps_planned: null, seconds: 600, tokens: null,
    params: 2_500_000, best_val_loss: 1.98765, train_loss: null, device: null, rounds: null, inner_steps: null, same_design: true, parent: false, this_version: false, ...over,
  });

  test("names the run and lists only what it recorded", () => {
    expect(sourceFacts(source())).toEqual({
      title: "Version 4 · deep", when: "Stopped 19 Oct 2026", facts: ["300 steps", "10 min", "Held-out loss 1.988", "2.5M params"], design: "Same design: every weight is copied",
    });
  });

  test("the version itself says a run from it trains it further", () => {
    expect(sourceFacts(source({ this_version: true })).title).toBe("Version 4 · deep · this version: trains it further");
  });

  test("a run with no record says so", () => {
    const bare = source({ name: "", status: "", finished_at: null, steps: null, seconds: null, params: null, best_val_loss: null, same_design: false, parent: true });
    expect(sourceFacts(bare)).toMatchObject({ title: "Version 4 · saved from it", when: "", facts: ["Its run left no record of what it did."] });
  });
});

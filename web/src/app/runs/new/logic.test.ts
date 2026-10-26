import { describe, expect, test } from "vitest";
import { busyPeers, ctaLabel, defaultSftVersion, durationText, memberLine, nameList, runningText, localForced, peersOf, runHref, shapeOf, sizingBlocker, sourceFacts, workersOf, type Node, type Sizing, type Source } from "./logic";

const member = (over: Partial<Sizing["members"][number]>): Sizing["members"][number] => ({
  name: "box", address: "box", sized: true, fits: true, batch_size: 16, accumulation: 1, offload: false, notes: [], refusal: "", ...over,
});
const sizing = (over: Partial<Sizing> = {}): Sizing => ({ members: [], flagged: [], rounds: 10, rounds_cut: false, note: "", ...over });

describe("per-member sizing", () => {
  test("a member's line says its batch, accumulation and where its optimizer lives", () => {
    expect(memberLine(member({}))).toBe("box: batch 16, 1 accumulation steps");
    expect(memberLine(member({ batch_size: 1, accumulation: 16, offload: true }))).toBe("box: batch 1, 16 accumulation steps, optimizer in RAM");
  });

  test("a member that cannot fit is named with the reason, and one that reports nothing says so", () => {
    expect(memberLine(member({ fits: false, refusal: "box cannot take part: it has 200 MB of GPU memory for this run." }))).toBe("box cannot take part. it has 200 MB of GPU memory for this run.");
    expect(memberLine(member({ sized: false }))).toBe("box: did not report what it has free, so it trains as the design says.");
  });

  test("a flagged peer blocks Start until it is left out; this computer blocks it always", () => {
    expect(sizingBlocker(undefined, "mac", false)).toBe("");
    expect(sizingBlocker(sizing(), "mac", false)).toBe("");
    expect(sizingBlocker(sizing({ flagged: ["box"] }), "mac", false)).toContain("box cannot hold the design");
    expect(sizingBlocker(sizing({ flagged: ["box"] }), "mac", true)).toBe("");
    expect(sizingBlocker(sizing({ flagged: ["mac"] }), "mac", true)).toContain("This computer cannot hold the design");
  });
});

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

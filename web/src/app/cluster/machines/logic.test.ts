import { describe, expect, test } from "vitest";
import type { QueueRow } from "@/lib/cluster";
import { changed, cleanList, contactText, deviceText, factsAge, fraction, jobRows, lastSeen, machineJobs, progressText, refusedUs, runPage, waitingPlace, type ClusterNode } from "./logic";

const node = (address: string, over: Partial<ClusterNode> = {}): ClusterNode => ({
  address, local: false, hostname: "", version: "", device: "", device_name: "", memory_bytes: 0, busy: false, dtypes: [], online: false, error: "", tailnet_name: "", origin: "", origin_down: false, ...over,
});
const row = (id: string, label: string, over: Partial<QueueRow> = {}): QueueRow => ({
  id, kind: "pretrain", label, status: "waiting", node: "", hostname: "", queued_on: "mac-mini", queued_here: true, queued_address: "local",
  pinned: "", waiting_reason: "", stale: false, note: "", step: 0, total: 0, eta_s: null, ...over,
});
const job = (id: string, over: Record<string, unknown> = {}) => ({ id, kind: "pretrain", label: id.toUpperCase(), note: "", params: {}, status: "waiting", eta_s: null, created_at: 0, error: "", node: "", owner: "", epoch: 0, ...over });
const quiet = { job: null, up_next: [], paused: false, step: 0, total: 0, phase: "", lr: null, eta_s: null, blocked: null, stopping: false };

describe("machines", () => {
  test("device text, and who refused this machine", () => {
    expect(deviceText(node("m", { online: true, device: "mps", device_name: "Apple M2", memory_bytes: 16 * 2 ** 30 }))).toBe("Apple M2 (mps, 16 GB)");
    expect(deviceText(node("x", { online: true, device: "cpu" }))).toBe("cpu");
    expect(deviceText(node("beast", { online: false, device: "cpu" }))).toBe("");  // the API's placeholder for a machine that did not answer
    expect(deviceText(node("down"))).toBe("");
    expect(refusedUs(node("a", { error: "a answered 401 to GET /node: no" }))).toBe(true);
    expect(refusedUs(node("a", { online: true, error: "a answered 401" }))).toBe(false);
    expect(refusedUs(node("a", { error: "a did not answer within 2 seconds." }))).toBe(false);
  });

  test("the list is trimmed, deduplicated and compared in order", () => {
    expect(cleanList([" a ", "", "b", "a", "A"])).toEqual(["a", "b", "A"]);
    expect(changed(["a", "b"], ["a", "b"])).toBe(false);
    expect(changed(["b", "a"], ["a", "b"])).toBe(true);
  });

  test("last seen: nothing for this machine, a time, or that it never answered", () => {
    expect(lastSeen(node("local", { local: true, online: true }), 0)).toBe("");
    expect(lastSeen(node("g", { online: true }), 0)).toBe("");
    expect(lastSeen(node("d"), 0)).toBe("Has not answered yet");
    expect(lastSeen(node("g", { online: true }), 1_700_000_000)).toMatch(/^Last answered /);
  });
});

describe("jobs", () => {
  const rows = [
    row("a", "A", { status: "running", node: "gpu-box", queued_address: "local", step: 1, total: 4 }),
    row("b", "B", { status: "held", queued_address: "gpu-box" }),
    row("c", "C", { queued_address: "local" }),
  ];

  test("a machine runs jobs placed on it and holds the waiting jobs of its own queue", () => {
    expect(machineJobs("gpu-box", rows).running.map((r) => r.id)).toEqual(["a"]);
    expect(machineJobs("gpu-box", rows).waiting.map((r) => r.id)).toEqual(["b"]);
    expect(machineJobs("local", rows).waiting.map((r) => r.id)).toEqual(["c"]);
    expect(machineJobs("local", rows).running).toEqual([]);
  });

  test("progress in words, and as a fraction", () => {
    expect(progressText(rows[0])).toBe("Step 1 of 4");
    expect(progressText(row("z", "Z", { status: "running", step: 100, total: 1000, eta_s: 600 }))).toBe("Step 100 of 1,000 · about 10 min left");
    expect(progressText(row("z", "Z", { status: "running" }))).toBe("Starting");
    expect(fraction(rows[0])).toBe(0.25);
    expect(fraction(row("z", "Z"))).toBe(0);
  });

  test("where a waiting job will run", () => {
    expect(waitingPlace(rows[2], {})).toBe("Runs on any machine that fits it");
    expect(waitingPlace(row("p", "P", { pinned: "gpu-box", hostname: "gpu-box" }), {})).toBe("Pinned to gpu-box");
  });

  test("only a run queued here has a page of its own", () => {
    expect(runPage(rows[0])).toBe("/training-run");
    expect(runPage(rows[1])).toBeNull();
    expect(runPage({ ...rows[0], queued_here: false })).toBeNull();
    expect(runPage({ ...rows[0], kind: "versions" })).toBeNull();
  });

  test("a cluster of one has no merged rows, so its own queue stands in", () => {
    const queue = { ...quiet, job: job("r", { status: "running" }), step: 2, total: 8, up_next: [job("w", { status: "held" }), job("x")] };
    const own = jobRows({ nodes: [], jobs: [] }, queue as never);
    expect(own.map((r) => [r.id, r.status, r.node, r.queued_address])).toEqual([["r", "running", "local", "local"], ["w", "held", "local", "local"], ["x", "waiting", "local", "local"]]);
    expect(own[0].step).toBe(2);
    expect(jobRows({ nodes: [], jobs: [] }, undefined)).toEqual([]);
  });

  test("with other machines listed the merged rows are used as they are", () => {
    const merged = { nodes: [{ address: "local", hostname: "m", pin: "m", local: true, online: true, error: "", seen_at: 0, origin: "", origin_down: false, listed: true }], jobs: rows };
    expect(jobRows(merged, { ...quiet, job: job("r", { status: "running" }) } as never)).toBe(rows);
  });
});

describe("how old the facts about a machine are here", () => {
  test("this machine and one the store holds nothing of say nothing", () => {
    expect(factsAge(node("local", { local: true, age_s: 0 }))).toBe("");
    expect(factsAge(node("beast", { age_s: null }))).toBe("");
  });

  test("another machine says the age of this computer's copy, and when it is down says so", () => {
    expect(factsAge(node("beast", { origin: "beast", age_s: 5 }))).toBe("Its facts here are 5 s old");
    expect(factsAge(node("beast", { origin: "beast", age_s: 120, origin_down: true }))).toContain("2 min old");
  });
});

describe("when a machine was last heard from and will be asked next", () => {
  test("this machine says nothing", () => {
    expect(contactText(node("local", { local: true }))).toBe("");
  });

  test("a machine says how long ago it was heard and when it is asked next", () => {
    expect(contactText(node("beast", { last_heard_s: 40, next_check_s: 8 }))).toBe("Last heard 40 s ago · next check in 8 s");
    expect(contactText(node("beast", { last_heard_s: 0, next_check_s: 0.4 }))).toBe("Heard just now · next check due now");
    expect(contactText(node("beast", { last_heard_s: 300, next_check_s: null }))).toBe("Last heard 5 min ago");
  });

  test("a machine never heard from says so", () => {
    expect(contactText(node("beast"))).toBe("Not heard yet");
  });
});

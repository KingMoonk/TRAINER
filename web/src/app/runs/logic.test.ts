import { describe, expect, test } from "vitest";
import type { ClusterNode, Run } from "@/app/cluster/training/logic";
import type { Job, Queue } from "@/app/queue/logic";
import { buildRows, completedText, etaText, inSection, machineName, progressPercent, rolesOn, runEta, type ClusterJob } from "./logic";

const node = (address: string, hostname: string, over: Partial<ClusterNode> = {}): ClusterNode => ({
  address, hostname, local: address === "local", version: "1", device: "mps", device_name: "", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "", origin: "", origin_down: false, ...over,
});
const NODES = [node("local", "mac"), node("beast:8000", "beast", { device: "cuda" })];
const job = (id: string, over: Partial<Job> = {}): Job => ({ id, kind: "pretrain", label: id, note: "", params: {}, status: "waiting", eta_s: null, created_at: 0, epoch: 0, error: "", details: "", node: "", owner: "", ...over });
const queue = (over: Partial<Queue> = {}): Queue => ({
  job: null, step: 0, total: 0, phase: "", lr: null, eta_s: null, paused: false, blocked: null, stopping: false,
  gate: { supported: true }, up_next: [], failed: [], plan: { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true },
  ...over,
});
const round = (n: number, loss: number | null, val: number | null, seconds = 600) => ({ round: n, loss, val_loss: val, seconds, step: n * 100, members: 2, contributed: 2, dropped: [], bytes_up: 0, bytes_down: 0, best: false, pivot: "", polish: "" as const, lessons: 0 });
const run = (id: string, over: Partial<Run> = {}): Run => ({
  id, label: "from scratch", version: 5, saved_version: null, state: "running", note: "", started_at: 100, owner: "mac", round: 7, rounds: 40, inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9,
  delta_dtype: "float16", term: 1, coordinator: "", handoff: "", origin: "", age_s: 0, origin_down: false, can_stop: true, paused: false, can_pause: false, can_resume: false, can_restart: false, can_join: false, best_round: null, best_val_loss: null, stop_rule: "", stop_reason: "", lessons: 0, lessons_left_out: 0, reviewed_only: false, history: [round(6, 3.5, 3.6), round(7, 3.41, 3.47)],
  phase: "training", deadline_s: null, grace_s: 2, window_s: null, contributed: 0, expected: 2, theta_bytes: 0, now_s: 0, timeline: [], events: [],
  members: [
    { node: "mac", address: "local", local: true, state: "training" },
    { node: "beast", address: "beast:8000", local: false, state: "training" },
  ] as Run["members"],
  ...over,
});

describe("machineName", () => {
  test("local, an address and a hostname all read as the machine's name; an unknown one is kept", () => {
    expect(machineName("local", NODES)).toBe("mac");
    expect(machineName("beast:8000", NODES)).toBe("beast");
    expect(machineName("BEAST", NODES)).toBe("beast");
    expect(machineName("ghost", NODES)).toBe("ghost");
  });
});

describe("buildRows", () => {
  test("a DiLoCo run and its owner job are one row, active, with the coordinator and workers from its members", () => {
    const rows = buildRows({ queue: queue({ job: job("r1", { kind: "pretrain_distributed", status: "running" }) }), runs: [run("r1")], clusterJobs: [], nodes: NODES });
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ id: "r1", section: "active", coordinator: "mac", workers: ["mac", "beast"], round: 7, rounds: 40, trainLoss: 3.41, valLoss: 3.47, version: 5 });
    expect(rows[0].etaS).toBe(600 * 33);
  });

  test("a waiting job is scheduled in queue order; workers and coordinator are the one node it is pinned to, else any idle node", () => {
    const rows = buildRows({ queue: queue({ up_next: [job("a", { params: { pinned: "beast:8000" } }), job("b", { status: "held" })] }), runs: [], clusterJobs: [], nodes: NODES });
    expect(inSection(rows, "scheduled").map((r) => r.id)).toEqual(["a", "b"]);
    expect(rows[0]).toMatchObject({ workers: ["beast"], coordinator: "beast", queueIndex: 0 });
    expect(rows[1]).toMatchObject({ workers: ["any idle node"], coordinator: "any idle node", held: true, queueIndex: 1 });
  });

  test("the running local job runs on this computer, with the queue's step and total", () => {
    const rows = buildRows({ queue: queue({ job: job("j", { status: "running" }), step: 50, total: 200, eta_s: 90 }), runs: [], clusterJobs: [], nodes: NODES });
    expect(rows[0]).toMatchObject({ section: "active", workers: ["mac"], coordinator: "mac", step: 50, total: 200, etaS: 90 });
    expect(progressPercent(rows[0])).toBe(25);
  });

  test("a job placed on a peer takes that peer as its node; a job run for another machine is active here", () => {
    const cluster: ClusterJob[] = [
      { id: "p", kind: "pretrain", label: "placed", role: "owner", node: "beast:8000", epoch: 1, status: "running", lease: "held", waiting_reason: "", checkpoint_step: null },
      { id: "q", kind: "pretrain", label: "for beast", role: "runner", node: "beast:8000", epoch: 1, status: "running", lease: "held", waiting_reason: "", checkpoint_step: null },
    ];
    const rows = buildRows({ queue: queue({ up_next: [job("p")] }), runs: [], clusterJobs: cluster, nodes: NODES });
    expect(rows.find((r) => r.id === "p")).toMatchObject({ workers: ["beast"], coordinator: "beast" });
    expect(rows.find((r) => r.id === "q")).toMatchObject({ section: "active", workers: ["mac"], forOwner: "beast" });
  });

  test("finished, stopped and failed runs are completed, newest first; a queued run is scheduled", () => {
    const rows = buildRows({
      queue: queue(),
      runs: [run("old", { state: "finished", started_at: 1 }), run("new", { state: "failed", note: "out of memory", started_at: 9 }), run("soon", { state: "queued" })],
      clusterJobs: [],
      nodes: NODES,
    });
    expect(inSection(rows, "completed").map((r) => r.id)).toEqual(["new", "old"]);
    expect(inSection(rows, "scheduled").map((r) => r.id)).toEqual(["soon"]);
  });
});

describe("a member machine", () => {
  // Live data from the Mac while BEAST coordinates: `local` is relative to the coordinator, and the Mac is listed by its tailnet URL.
  const MAC = "Michaels-Mac.local";
  const URL = "https://michaels-mac.tailb22e91.ts.net";
  const nodes = [node("local", MAC), node("beast:8000", "BEAST", { device: "cuda" })];
  const members = [
    { node: "BEAST", address: "local", local: true, state: "training" },
    { node: MAC, address: URL, local: false, state: "training" },
  ] as Run["members"];
  const remote = run("e693047f", { owner: "BEAST", members, version: 4 });
  const memberJob = job("m1", { kind: "diloco_member", label: "Version 4 across 2 machines (member)", status: "running", params: { run: "e693047f", node: MAC } });

  test("the run shows once, named by node, and its member job on this machine is not a second row", () => {
    const rows = buildRows({ queue: queue({ job: memberJob }), runs: [remote], clusterJobs: [], nodes });
    expect(rows.map((r) => r.id)).toEqual(["e693047f"]);
    expect(rows[0]).toMatchObject({ coordinator: "BEAST", workers: ["BEAST", MAC] });
    expect(rolesOn("BEAST", rows)[0].roles).toBe("c, w");
    expect(rolesOn(MAC, rows)).toEqual([{ roles: "w", run: "v4", runName: "from scratch" }]);
  });

  test("a member job whose run is not listed is never the coordinator", () => {
    const [row] = buildRows({ queue: queue({ job: memberJob }), runs: [], clusterJobs: [], nodes });
    expect(row).toMatchObject({ workers: [MAC], coordinator: "" });
    expect(rolesOn(MAC, [row])[0].roles).toBe("w");
  });
});

describe("text", () => {
  test("rolesOn gives c, w for the coordinator that also trains", () => {
    const rows = buildRows({ queue: queue(), runs: [run("r1")], clusterJobs: [], nodes: NODES });
    expect(rolesOn("mac", rows)).toEqual([{ roles: "c, w", run: "v5", runName: "from scratch" }]);
    expect(rolesOn("beast", rows)).toEqual([{ roles: "w", run: "v5", runName: "from scratch" }]);
    expect(rolesOn("other", rows)).toEqual([]);
  });

  test("etaText and runEta", () => {
    expect(etaText(5 * 3600 + 600)).toBe("5h 10m");
    expect(etaText(42 * 60)).toBe("42 min");
    expect(etaText(10)).toBe("under a minute");
    expect(runEta(run("x", { history: [] }))).toBeNull();
  });

  test("completedText gives the failure reason, or val loss and rounds", () => {
    const [failed, done] = buildRows({ queue: queue(), runs: [run("f", { state: "failed", note: "out of memory", started_at: 2 }), run("d", { state: "finished", started_at: 1 })], clusterJobs: [], nodes: NODES });
    expect(completedText(failed)).toBe("failed: out of memory");
    expect(completedText(done)).toBe("val 3.47 · 7 of 40 rounds");
  });
});

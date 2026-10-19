import { describe, expect, it } from "vitest";
import type { ClusterQueue, QueueRow } from "@/lib/cluster";
import type { ClusterNode, StartRun } from "./logic";
import { allReady, busyText, readiness, unreachableText, waitForAll } from "./readiness";

const node = (address: string, over: Partial<ClusterNode> = {}): ClusterNode => ({
  address, local: false, hostname: address, version: "1", device: "cuda", device_name: "RTX 4090", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "", origin: "", origin_down: false, ...over,
});
const row = (over: Partial<QueueRow> = {}): QueueRow => ({
  id: "g1", kind: "pretrain", label: "Version 1", node: "BEAST", hostname: "BEAST", status: "running", eta_s: 720, note: "", pinned: "", queued_address: "", queued_here: false, queued_on: "BEAST", stale: false, ...over,
} as QueueRow);
const cluster = (...jobs: QueueRow[]): ClusterQueue => ({ jobs, nodes: [] });

describe("readiness of the chosen machines", () => {
  it("reads free, busy and unreachable machines, only for the ones chosen", () => {
    const nodes = [node("BEAST", { busy: true }), node("old-laptop", { online: false, error: "old-laptop did not answer within 2 seconds." }), node("quiet"), node("other")];
    const items = readiness(nodes, ["BEAST", "old-laptop", "quiet"], cluster(row()));
    expect(items.map((i) => [i.name, i.state])).toEqual([["BEAST", "busy"], ["old-laptop", "unreachable"], ["quiet", "ready"]]);
    expect(items[0]).toMatchObject({ label: "Version 1", eta_s: 720 });
    expect(items[1]).toMatchObject({ why: "old-laptop did not answer within 2 seconds.", paste: "https://old-laptop" });
  });

  it("calls a machine busy when its queue shows a running job even before the node says so", () => {
    expect(readiness([node("BEAST")], ["BEAST"], cluster(row()))[0].state).toBe("busy");
  });

  it("calls a machine busy without a label when the queue does not say which job", () => {
    const [item] = readiness([node("BEAST", { busy: true })], ["BEAST"], cluster());
    expect(item).toMatchObject({ state: "busy", label: "" });
    expect(busyText(item)).toBe("BEAST is busy with another job.");
  });

  it("is plain only when every chosen machine is ready", () => {
    expect(allReady(readiness([node("a")], ["a"], undefined))).toBe(true);
    expect(allReady(readiness([node("a", { busy: true })], ["a"], undefined))).toBe(false);
    expect(allReady([])).toBe(true);
  });

  it("words busy and unreachable machines with the job, the time left and the reason", () => {
    const [busy] = readiness([node("BEAST", { busy: true })], ["BEAST"], cluster(row()));
    expect(busyText(busy)).toBe("BEAST is training 'Version 1', about 12 min left.");
    const [down] = readiness([node("old", { online: false, error: "no route" })], ["old"], undefined);
    expect(unreachableText(down)).toBe("old did not answer: no route");
    expect(unreachableText({ ...down, why: "" })).toBe("old did not answer.");
  });
});

describe("waitForAll", () => {
  const body = { version: null, dataset_ids: ["s"], machines: ["a", "b", "c"], rounds: 1, inner_steps: 1, outer_lr: 0.7, outer_momentum: 0.9, overlap: false } as StartRun;
  const items = (...states: ("ready" | "busy" | "unreachable")[]) =>
    states.map((state, i) => ({ address: `m${i}`, name: `m${i}`, state, label: "", eta_s: null, why: "", paste: "" }));

  it("waits for this computer and every machine that answers, not the unreachable ones", () => {
    expect(waitForAll(body, items("ready", "busy", "unreachable")).min_members).toBe(3);
  });

  it("never lowers a minimum the person set, and respects the maximum", () => {
    expect(waitForAll({ ...body, min_members: 5 }, items("busy")).min_members).toBe(5);
    expect(waitForAll({ ...body, max_members: 2 }, items("busy", "busy", "busy")).min_members).toBe(2);
  });
});

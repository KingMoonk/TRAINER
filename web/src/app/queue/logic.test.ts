import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/api";
import type { ClusterQueue, QueueNode, QueueRow } from "@/lib/cluster";
import { T, fill } from "./copy";
import {
  ALL_MACHINES,
  allDoneLine,
  canRunOnAll,
  memberLines,
  dropIndex,
  errorText,
  minutes,
  nodeLabel,
  parseHour,
  pinChoice,
  placeText,
  progressLine,
  resumesText,
  rowFor,
  stateLine,
  toggleState,
  totalHours,
  type Job,
  type Queue,
} from "./logic";

const queue = (extra: Record<string, unknown> = {}) =>
  ({ job: null, up_next: [], paused: false, blocked: "", step: 0, total: 0, phase: "", lr: null, eta_s: null, ...extra }) as unknown as Queue;
const job = (id: string, eta_s: number | null = null, extra = {}) => ({ id, kind: "train", label: "", eta_s, ...extra });
const node = (extra: Partial<QueueNode> = {}) =>
  ({ address: "10.0.0.2", hostname: "alpha", local: false, online: true, pin: "10.0.0.2", ...extra }) as QueueNode;

describe("minutes", () => {
  it("is empty without a time", () => {
    expect(minutes(null)).toBe("");
    expect(minutes(undefined)).toBe("");
  });
  it("rounds to minutes and never shows less than one", () => {
    expect(minutes(0)).toBe("about 1 min");
    expect(minutes(29)).toBe("about 1 min");
    expect(minutes(95)).toBe("about 2 min");
    expect(minutes(3600)).toBe("about 60 min");
  });
});

describe("totalHours", () => {
  it("adds the running job's live estimate and each waiting job's estimate", () => {
    const state = queue({ job: job("a", 999), eta_s: 1800, up_next: [job("b", 1800), job("c", 3600)] });
    expect(totalHours(state)).toBe(2);
  });
  it("counts only waiting jobs when nothing is running", () => expect(totalHours(queue({ up_next: [job("b", 7200)] }))).toBe(2));
  it("is null when any job has no estimate, or there are no jobs", () => {
    expect(totalHours(queue({ job: job("a"), eta_s: null }))).toBeNull();
    expect(totalHours(queue({ job: job("a"), eta_s: 60, up_next: [job("b", null)] }))).toBeNull();
    expect(totalHours(queue())).toBeNull();
  });
  it("accepts an estimate of zero", () => expect(totalHours(queue({ job: job("a"), eta_s: 0 }))).toBe(0));
});

describe("allDoneLine", () => {
  it("states the finish time to one decimal", () => expect(allDoneLine(queue({ job: job("a"), eta_s: 5400 }))).toBe(fill(T.allDoneIn, { hours: "1.5" })));
  it("explains the missing finish time when jobs wait without an estimate", () => {
    expect(allDoneLine(queue({ job: job("a"), eta_s: 60, up_next: [job("b")] }))).toBe(T.finishNeedsEstimates);
  });
  it("is empty with nothing waiting and no estimate", () => {
    expect(allDoneLine(queue())).toBe("");
    expect(allDoneLine(queue({ job: job("a"), eta_s: null }))).toBe("");
  });
});

describe("stateLine", () => {
  it("names the job and whether it runs, waits or is paused", () => {
    const j = job("a", null, { label: "Tiny run" });
    expect(stateLine(queue({ job: j }))).toBe("Tiny run · running");
    expect(stateLine(queue({ job: j, blocked: "battery" }))).toBe("Tiny run · waiting: battery");
    expect(stateLine(queue({ job: j, blocked: "battery", paused: true }))).toBe("Tiny run · paused");
  });
  it("falls back to the kind, then to nothing", () => {
    expect(stateLine(queue({ job: job("a") }))).toBe("train · running");
    expect(stateLine(queue())).toBe(" · running");
  });
});

describe("progressLine", () => {
  it("says starting with no total, and learning rate auto", () => expect(progressLine(queue())).toBe("starting · learning rate: auto"));
  it("shows the step, phase, learning rate and time left", () => {
    const state = queue({ step: 1200, total: 5000, phase: "warmup", lr: 0.00123, eta_s: 600 });
    expect(progressLine(state)).toBe("step 1,200 of 5,000 · schedule: warmup · learning rate: auto (0.0012) · about 10 min left");
  });
  it("omits the phase and the time left when unknown", () => expect(progressLine(queue({ step: 1, total: 2, lr: 0.5 }))).toBe("step 1 of 2 · learning rate: auto (0.5)"));
});

describe("toggleState", () => {
  it("disables only the idle option on a computer that cannot detect idleness", () => {
    expect(toggleState("idle_only", "note", false)).toEqual({ enabled: false, note: T.idleUnsupported });
    expect(toggleState("idle_only", "note", true)).toEqual({ enabled: true, note: "note" });
    expect(toggleState("keep_awake", "note", false)).toEqual({ enabled: true, note: "note" });
  });
});

describe("dropIndex", () => {
  it("takes the position of the row dropped on", () => expect(dropIndex(2, 5)).toBe(2));
  it("stays within the list", () => {
    expect(dropIndex(-3, 5)).toBe(0);
    expect(dropIndex(9, 5)).toBe(4);
    expect(dropIndex(0, 1)).toBe(0);
  });
});

describe("errorText", () => {
  it("prefers the server's detail", () => expect(errorText(new ApiError(409, "/queue", JSON.stringify({ detail: "Already running." })))).toBe("Already running."));
  it("shows the status line with a non-JSON body", () => expect(errorText(new ApiError(502, "/queue", "Bad gateway"))).toBe("502 from /queue: Bad gateway"));
  it("shows the status line with the raw JSON when there is no string detail", () => {
    expect(errorText(new ApiError(422, "/queue", '{"detail":[1]}'))).toBe('422 from /queue: {"detail":[1]}');
  });
  it("shows just the status line for an empty body", () => expect(errorText(new ApiError(500, "/queue", ""))).toBe("500 from /queue"));
  it("handles other errors and values", () => {
    expect(errorText(new Error("offline"))).toBe("offline");
    expect(errorText("boom")).toBe("boom");
    expect(errorText(undefined)).toBe("undefined");
  });
});

describe("parseHour", () => {
  it.each([["0", 0], ["7", 7], ["07", 7], ["23", 23], [" 9 ", 9]])("reads %j as %d", (text, hour) => expect(parseHour(text)).toBe(hour));
  it.each([[""], [" "], ["24"], ["-1"], ["1.5"], ["abc"], ["123"], ["9am"]])("rejects %j", (text) => expect(parseHour(text)).toBeNull());
});

describe("resumesText", () => {
  it("is empty without a time, or with an unreadable one", () => {
    expect(resumesText(null)).toBe("");
    expect(resumesText(undefined)).toBe("");
    expect(resumesText("")).toBe("");
    expect(resumesText("not a date")).toBe("");
  });
  it("shows the clock time in the person's own format", () => {
    const iso = "2026-03-04T21:30:00Z";
    const when = new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    expect(resumesText(iso)).toBe(fill(T.blockedAt, { when }));
  });
});

describe("rowFor and placeText", () => {
  const rows = [{ id: "a", status: "running", node: "n1", hostname: "alpha", pinned: "" }, { id: "b", status: "queued", node: "", hostname: "", pinned: "10.0.0.3" }] as unknown as QueueRow[];
  const cluster = { jobs: rows, nodes: [] } as unknown as ClusterQueue;
  it("finds a job's row", () => {
    expect(rowFor(cluster, "b")).toBe(rows[1]);
    expect(rowFor(cluster, "zzz")).toBeUndefined();
    expect(rowFor(undefined, "a")).toBeUndefined();
  });
  it("says where a running job is, by name, else by node", () => {
    expect(placeText(rows[0])).toBe(fill(T.runningOn, { node: "alpha" }));
    expect(placeText({ ...rows[0], hostname: "" })).toBe(fill(T.runningOn, { node: "n1" }));
  });
  it("says where a waiting job is pinned, or that it can run anywhere", () => {
    expect(placeText(rows[1])).toBe(fill(T.pinnedTo, { node: "10.0.0.3" }));
    expect(placeText({ ...rows[1], hostname: "beta" })).toBe(fill(T.pinnedTo, { node: "beta" }));
    expect(placeText({ ...rows[1], pinned: "" })).toBe(T.runsAnywhere);
  });
});

describe("nodeLabel", () => {
  it("marks this machine and machines that do not answer", () => {
    const nodes = [node({ local: true, hostname: "me" }), node(), node({ hostname: "far", address: "10.0.0.9", online: false })];
    expect(nodes.map((n) => nodeLabel(n, nodes))).toEqual([`me (${T.thisMachine})`, "alpha", `far (${T.offline})`]);
  });
  it("adds the address when another machine shares the name", () => {
    const nodes = [node({ address: "10.0.0.2" }), node({ address: "10.0.0.3", hostname: "ALPHA" })];
    expect(nodeLabel(nodes[0], nodes)).toBe("alpha (10.0.0.2)");
  });
  it("marks this machine as local ahead of offline", () => {
    const me = node({ local: true, online: false, hostname: "me" });
    expect(nodeLabel(me, [me])).toBe(`me (${T.thisMachine})`);
  });
});

describe("pinChoice", () => {
  const nodes = [node({ pin: "pin-a", address: "10.0.0.2", hostname: "alpha" }), node({ pin: "pin-b", address: "10.0.0.3", hostname: "beta" })];
  it("matches a pin or address in any case, ignoring surrounding space", () => {
    expect(pinChoice(nodes, "PIN-B")).toBe("pin-b");
    expect(pinChoice(nodes, " 10.0.0.2 ")).toBe("pin-a");
  });
  it("matches a name only when exactly one machine has it", () => {
    expect(pinChoice(nodes, "Beta")).toBe("pin-b");
    const twins = [...nodes, node({ pin: "pin-c", address: "10.0.0.4", hostname: "beta" })];
    expect(pinChoice(twins, "beta")).toBe("beta");
  });
  it("keeps the raw pin when no machine answers to it, and '' for any machine", () => {
    expect(pinChoice(nodes, "gone")).toBe("gone");
    expect(pinChoice(nodes, "")).toBe("");
    expect(pinChoice([], "x")).toBe("x");
  });
});

describe("all machines (DiLoCo)", () => {
  const node = (address: string) => ({ address, hostname: address, pin: address, local: address === "local", online: true, error: "", seen_at: 0 }) as QueueNode;

  it("is offered only for a pretrain job in a cluster of two or more machines", () => {
    expect(canRunOnAll("pretrain", [node("local"), node("box")])).toBe(true);
    expect(canRunOnAll("pretrain", [node("local")])).toBe(false);
    expect(canRunOnAll("quicktest", [node("local"), node("box")])).toBe(false);
    expect(ALL_MACHINES).not.toBe("");
  });

  it("lists a run's members and one line for each machine left out", () => {
    const run = {
      id: "r1", kind: "pretrain_distributed", label: "", params: {
        machines: [{ node: "mac" }, { node: "box" }],
        left_out: [{ node: "old", reason: "old did not answer within 2 seconds." }],
      },
    } as unknown as Job;
    expect(memberLines(run)).toEqual({
      members: "Trains on mac, box together",
      leftOut: ["Left out: old did not answer within 2 seconds."],
    });
    expect(memberLines({ ...run, kind: "pretrain" })).toEqual({ members: "", leftOut: [] });
    expect(memberLines({ ...run, params: {} })).toEqual({ members: "", leftOut: [] });
  });
});

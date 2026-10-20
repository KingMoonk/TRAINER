import { describe, expect, it } from "vitest";
import { QUICKTEST } from "./copy";
import { NO_RUN, applyEvent, checkMark, secondsLeft, signalText, type Result, type RunState } from "./logic";

const result = (extra: Record<string, unknown> = {}) => ({ verdict: "better", minutes: 3, broken: null, ...extra }) as unknown as Result;

describe("secondsLeft", () => {
  it("projects the pace so far onto the steps still to do", () => expect(secondsLeft(0, 10, 100, 20_000)).toBe(180));
  it("is null before the first step or without a total", () => {
    expect(secondsLeft(0, 0, 100, 5000)).toBeNull();
    expect(secondsLeft(0, 5, null, 5000)).toBeNull();
    expect(secondsLeft(0, 5, 0, 5000)).toBeNull();
  });
  it("is zero at the end, and never negative if the clock reads earlier", () => {
    expect(secondsLeft(0, 100, 100, 9000)).toBe(0);
    expect(secondsLeft(10_000, 5, 10, 5000)).toBe(0);
  });
});

describe("checkMark", () => {
  it("ticks a pass, crosses a fail, and shows an ellipsis while unknown", () => {
    expect(checkMark(true)).toBe("✓");
    expect(checkMark(false)).toBe("✗");
    expect(checkMark(null)).toBe("…");
    expect(checkMark(undefined)).toBe("…");
  });
});

describe("signalText", () => {
  it("states the verdict", () => {
    expect(signalText(result(), "Base")).toBe("After the same 3 minutes, your idea's error is better than Base's.");
    expect(signalText(result({ verdict: "worse" }), "Base")).toBe("After the same 3 minutes, your idea's error is worse than Base's.");
  });
  it("words an equal verdict naturally", () => {
    expect(signalText(result({ verdict: "about the same" }), "Base")).toBe("After the same 3 minutes, your idea's error is about the same as Base's.");
  });
  it("points at the part to look at when a check failed, replacing the verdict", () => {
    const check = QUICKTEST.checks[1];
    expect(signalText(result({ broken: check.id }), "Base")).toBe(`A check failed: ${check.name}. ${QUICKTEST.lookAt[check.id]}`);
  });
  it("names an unknown check by its id and adds no advice", () => {
    expect(signalText(result({ broken: "mystery" }), "Base")).toBe("A check failed: mystery.");
  });
});

describe("applyEvent", () => {
  const train = (extra: Record<string, unknown>) => ({ job: "j1", type: "quicktest", ...extra });

  it("ignores events with no job being watched, another job, or no payload", () => {
    expect(applyEvent(NO_RUN, null, "train", train({ step: 1 }), 0)).toBe(NO_RUN);
    expect(applyEvent(NO_RUN, "j1", "train", { ...train({ step: 1 }), job: "j2" }, 0)).toBe(NO_RUN);
    expect(applyEvent(NO_RUN, "j1", "train", null, 0)).toBe(NO_RUN);
    expect(applyEvent(NO_RUN, "j1", "train", "x", 0)).toBe(NO_RUN);
  });
  it("records progress per run, with the total when numeric", () => {
    const run = applyEvent(NO_RUN, "j1", "train", train({ which: "current", step: 5, total: 100 }), 1000);
    expect(run.progress).toEqual({ current: { step: 5, total: 100 } });
    expect(run.firstSeen).toEqual({ current: { at: 1000, step: 5 } });
    expect(applyEvent(NO_RUN, "j1", "train", train({ which: "draft", step: "7", total: "x" }), 0).progress).toEqual({ draft: { step: 7, total: null } });
  });
  it("keeps the first sighting of a run, while advancing its progress", () => {
    const first = applyEvent(NO_RUN, "j1", "train", train({ which: "current", step: 5, total: 100 }), 1000);
    const next = applyEvent(first, "j1", "train", train({ which: "current", step: 9, total: 100 }), 4000);
    expect(next.firstSeen).toEqual({ current: { at: 1000, step: 5 } });
    expect(next.progress.current).toEqual({ step: 9, total: 100 });
    const other = applyEvent(next, "j1", "train", train({ which: "draft", step: 1, total: 100 }), 5000);
    expect(other.firstSeen.draft).toEqual({ at: 5000, step: 1 });
    expect(other.firstSeen.current).toEqual({ at: 1000, step: 5 });
  });
  it("ignores a quicktest event with no step yet", () => {
    expect(applyEvent(NO_RUN, "j1", "train", train({ which: "current", step: 0 }), 0)).toBe(NO_RUN);
  });
  it("clears an earlier end message when progress resumes", () => {
    const ended: RunState = { ...NO_RUN, ended: "The quick test failed." };
    expect(applyEvent(ended, "j1", "train", train({ which: "current", step: 1 }), 0).ended).toBe("");
  });
  it("stores the finished result", () => {
    const results = result();
    const run = applyEvent({ ...NO_RUN, ended: "old" }, "j1", "train", { job: "j1", type: "quicktest_done", results }, 0);
    expect(run.result).toBe(results);
    expect(run.ended).toBe("");
  });
  it("records a failed or stopped job, with the error when there is one", () => {
    expect(applyEvent(NO_RUN, "j1", "job", { job: "j1", status: "failed", error: "out of memory" }, 0).ended).toBe(QUICKTEST.finished("failed", "out of memory"));
    expect(applyEvent(NO_RUN, "j1", "job", { job: "j1", status: "stopped" }, 0).ended).toBe(QUICKTEST.finished("stopped"));
    expect(applyEvent(NO_RUN, "j1", "job", { job: "j1", status: "failed", error: "" }, 0).ended).toBe(QUICKTEST.finished("failed"));
    expect(applyEvent(NO_RUN, "j1", "job", { job: "j1", status: "failed", error: 5 }, 0).ended).toBe(QUICKTEST.finished("failed"));
  });
  it("ignores other job statuses, other train types and other kinds", () => {
    expect(applyEvent(NO_RUN, "j1", "job", { job: "j1", status: "running" }, 0)).toBe(NO_RUN);
    expect(applyEvent(NO_RUN, "j1", "job", { job: "j1", status: "done" }, 0)).toBe(NO_RUN);
    expect(applyEvent(NO_RUN, "j1", "train", { job: "j1", type: "step", step: 4 }, 0)).toBe(NO_RUN);
    expect(applyEvent(NO_RUN, "j1", "queue", { job: "j1", type: "quicktest_done", results: {} }, 0)).toBe(NO_RUN);
  });
  it("does not mutate the run it was given", () => {
    const before: RunState = { result: null, progress: {}, firstSeen: {}, ended: "" };
    const snapshot = JSON.stringify(before);
    applyEvent(before, "j1", "train", train({ which: "current", step: 2, total: 4 }), 10);
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});

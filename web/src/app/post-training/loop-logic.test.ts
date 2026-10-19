import { describe, expect, it } from "vitest";
import { LOOP_DEFAULTS, elapsed, headline, loopFor, pairsLine, roundEvidence, startRequest, wholeNumber, type Loop, type LoopRound } from "./loop-logic";

const round = (patch: Partial<LoopRound> = {}) => ({ round: 1, status: "kept", reason: "", pairs: { total: 0 }, checks: [], ...patch }) as unknown as LoopRound;
const loop = (patch: Partial<Loop> = {}) =>
  ({ id: "a", version: 1, status: "stopped", stop_reason: "", round: 3, kept: 1, elapsed_s: 0, rounds: [round(), round({ round: 2, status: "rolled_back" })], active: false, ...patch }) as unknown as Loop;

describe("startRequest", () => {
  it("turns the form into the request", () => {
    expect(startRequest({ ...LOOP_DEFAULTS, hours: "1.5", judge: "qwen" }, 2, null)).toEqual({
      body: { version: 2, resume: null, judge_model: "qwen", config: { max_rounds: 20, patience: 4, max_hours: 1.5 } },
    });
    expect(startRequest(LOOP_DEFAULTS, null, "loop1")).toEqual({ body: { version: null, resume: "loop1", judge_model: null, config: { max_rounds: 20, patience: 4 } } });
  });
  it("names the field that is wrong", () => {
    expect(startRequest({ ...LOOP_DEFAULTS, rounds: "0" }, 1, null)).toEqual({ error: "Rounds must be a whole number from 1 to 100,000." });
    expect(startRequest({ ...LOOP_DEFAULTS, patience: "x" }, 1, null)).toEqual({ error: "Stop-after must be a whole number from 1 to 1,000." });
    expect(startRequest({ ...LOOP_DEFAULTS, hours: "-1" }, 1, null)).toEqual({ error: "Hours must be more than 0, or left empty." });
  });
});

describe("wholeNumber", () => {
  it("accepts digits in range only", () => {
    expect(wholeNumber("5", 1, 9)).toBe(5);
    expect(wholeNumber("10", 1, 9)).toBeNull();
    expect(wholeNumber("1.5", 1, 9)).toBeNull();
    expect(wholeNumber("", 1, 9)).toBeNull();
  });
});

describe("loopFor", () => {
  it("picks the newest loop of the version", () => {
    const list = [loop({ id: "b", version: 2 }), loop({ id: "a", version: 1 }), loop({ id: "c", version: 1 })];
    expect(loopFor(list, 1)?.id).toBe("a");
    expect(loopFor(list, 3)).toBeUndefined();
    expect(loopFor(undefined, 1)).toBeUndefined();
  });
});

describe("headline", () => {
  it("says what the loop is doing or why it is not", () => {
    expect(headline(loop({ phase: "round 3: training" }), true)).toBe("round 3: training. 1 of 2 rounds kept.");
    expect(headline(loop({ status: "waiting_judge" }), true)).toContain("Waiting for the judge before round 3");
    expect(headline(loop({ status: "finished" }), false)).toBe("Finished. 1 of 2 rounds kept.");
    expect(headline(loop({ status: "running" }), false)).toContain("Interrupted before round 3");
  });
});

describe("round text", () => {
  it("shows the evidence it has", () => {
    const r = round({ after: { pass_k: 7, pass_n: 10 }, before: { pass_k: 5, pass_n: 10 }, judge: { win_rate: 0.8 }, length_ratio: 1.1 } as Partial<LoopRound>);
    expect(roundEvidence(r)).toBe("solved 7 of 10 (was 5) · judge preferred it 80% · 1.10x as long");
    expect(roundEvidence(round())).toBe("");
  });
  it("counts how the pairs were made", () => {
    const r = round({ pairs: { total: 5, verified: 3, comparison: 2, skipped: { "scores too close": 4 } } } as Partial<LoopRound>);
    expect(pairsLine(r)).toBe("5 pairs (3 checked by a rule, 2 judged); skipped: scores too close ×4");
  });
  it("formats working time", () => {
    expect([elapsed(30), elapsed(600), elapsed(36000)]).toEqual(["30 s", "10 min", "10.0 h"]);
  });
});

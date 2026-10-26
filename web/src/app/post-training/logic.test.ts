import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/api";
import { T, fill } from "./copy";
import { count, errorText, firstOpenRound, gateState, roundBlocker, rowText, spreadText, stepLine, type Status } from "./logic";

const rounds = (statuses: Record<string, string>) => ({ rounds: Object.fromEntries(Object.entries(statuses).map(([k, status]) => [k, { status }])) }) as unknown as Status;

describe("roundBlocker", () => {
  it("needs a trained checkpoint before anything", () => {
    for (const round of [1, 2, 3, 4]) expect(roundBlocker(round, { conversations: 5, problems: 5, corrections: 5, pairs: 5 }, false, true, false)).toBe("Train this version in Pretraining first.");
  });
  it("round 1 needs a conversation", () => {
    expect(roundBlocker(1, {}, false, false, true)).toBe("Add conversations in Your data first.");
    expect(roundBlocker(1, { conversations: 0 }, false, false, true)).toBe("Add conversations in Your data first.");
    expect(roundBlocker(1, { conversations: 1 }, false, false, true)).toBeNull();
  });
  it("round 2 needs a problem a rule can check", () => {
    expect(roundBlocker(2, { conversations: 9 }, false, false, true)).toBe("Make problems a rule can check first.");
    expect(roundBlocker(2, { problems: 1 }, false, false, true)).toBeNull();
  });
  it("round 4 needs at least two rated pairs", () => {
    expect(roundBlocker(4, { pairs: 1 }, false, false, true)).toBe("Rate at least 2 pairs in Chat first.");
    expect(roundBlocker(4, {}, false, false, true)).toBe("Rate at least 2 pairs in Chat first.");
    expect(roundBlocker(4, { pairs: 2 }, false, false, true)).toBeNull();
  });
  it("round 3 learns from another version, or from corrections", () => {
    expect(roundBlocker(3, {}, true, false, true)).toBe("There is no other trained version to learn from yet.");
    expect(roundBlocker(3, {}, true, true, true)).toBeNull();
    expect(roundBlocker(3, { corrections: 0 }, false, true, true)).toBe("Teach a correction in Chat first.");
    expect(roundBlocker(3, { corrections: 2 }, false, false, true)).toBeNull();
  });
});

describe("count", () => {
  it("groups thousands, and shows a dash for no number", () => {
    expect(count(1234567)).toBe("1,234,567");
    expect(count(0)).toBe("0");
    expect(count(null)).toBe("–");
    expect(count(undefined)).toBe("–");
  });
});

describe("stepLine", () => {
  it("says starting without a queue state or a total", () => {
    expect(stepLine(undefined)).toBe("starting");
    expect(stepLine({ step: 0, total: 0, eta_s: null })).toBe("starting");
  });
  it("shows the step, and the time left when known", () => {
    expect(stepLine({ step: 1200, total: 5000, eta_s: null })).toBe("step 1,200 of 5,000");
    expect(stepLine({ step: 1200, total: 5000, eta_s: 600 })).toBe("step 1,200 of 5,000 · about 10 min left");
  });
  it("rounds minutes and never shows less than one", () => {
    expect(stepLine({ step: 1, total: 2, eta_s: 5 })).toContain("about 1 min left");
    expect(stepLine({ step: 1, total: 2, eta_s: 0 })).toContain("about 1 min left");
    expect(stepLine({ step: 1, total: 2, eta_s: 95 })).toContain("about 2 min left");
  });
  it("still shows the time left when the total is unknown", () => expect(stepLine({ step: 0, total: 0, eta_s: 120 })).toBe("starting · about 2 min left"));
});

describe("spreadText", () => {
  it("is empty without groups", () => expect(spreadText(0, 0)).toBe(""));
  it("says how many groups taught the model, in the plural", () => {
    expect(spreadText(2, 5)).toBe(fill(T.spreadSome, { taught: "2", groups: "5", groupsWord: "groups" }));
  });
  it("uses the singular for one group", () => {
    expect(spreadText(1, 1)).toContain("1 of 1 group had");
    expect(spreadText(0, 1)).toContain("None of the 1 group had");
  });
  it("says nothing was learned when no group taught", () => {
    expect(spreadText(0, 1200)).toBe(fill(T.spreadNone, { groups: "1,200", groupsWord: "groups" }));
  });
});

describe("firstOpenRound", () => {
  it("is the first round that is not done", () => {
    expect(firstOpenRound(rounds({ 1: "done", 2: "done", 3: "idle", 4: "idle" }))).toBe(3);
    expect(firstOpenRound(rounds({ 1: "running", 2: "done" }))).toBe(1);
    expect(firstOpenRound(rounds({ 1: "done", 2: "failed", 3: "done", 4: "done" }))).toBe(2);
  });
  it("treats a missing round as idle", () => expect(firstOpenRound(rounds({ 1: "done" }))).toBe(2));
  it("opens round 1 with no status, and when every round is done", () => {
    expect(firstOpenRound(undefined)).toBe(1);
    expect(firstOpenRound({} as Status)).toBe(1);
    expect(firstOpenRound(rounds({ 1: "done", 2: "done", 3: "done", 4: "done" }))).toBe(1);
  });
});

describe("errorText", () => {
  it("prefers the server's detail sentence", () => {
    expect(errorText(new ApiError(409, "/p", JSON.stringify({ detail: "Stop the job first." })))).toBe("Stop the job first.");
  });
  it("shows a JSON body without a string detail as it is", () => {
    const body = JSON.stringify({ detail: [{ msg: "x" }] });
    expect(errorText(new ApiError(422, "/p", body))).toBe(body);
    expect(errorText(new ApiError(500, "/p", JSON.stringify({ other: 1 })))).toBe('{"other":1}');
  });
  it("shows a non-JSON body as it is", () => expect(errorText(new ApiError(502, "/p", "Bad gateway"))).toBe("Bad gateway"));
  it("falls back to the error message when the body is empty", () => {
    expect(errorText(new ApiError(500, "/p", ""))).toBe("500 from /p");
    expect(errorText(new Error("network down"))).toBe("network down");
  });
  it("stringifies anything else", () => {
    expect(errorText("plain")).toBe("plain");
    expect(errorText(undefined)).toBe("undefined");
    expect(errorText(null)).toBe("null");
  });
});

describe("rowText", () => {
  it("shows a string as it is", () => expect(rowText("hello")).toBe("hello"));
  it("shows an object as key: value lines, with non-strings as JSON", () => {
    expect(rowText({ prompt: "2+2", n: 4, tags: ["a"], none: null })).toBe('prompt: 2+2\nn: 4\ntags: ["a"]\nnone: null');
  });
  it("shows other values as JSON", () => {
    expect(rowText(5)).toBe("5");
    expect(rowText(null)).toBe("null");
    expect(rowText(undefined)).toBeUndefined();
    expect(rowText([1, 2])).toBe("0: 1\n1: 2"); // an array is listed by index
  });
});

describe("gateState", () => {
  const check = (ok: boolean | null) => ({ name: "c", detail: "d", ok });
  it("fails when the gate or any check failed", () => {
    expect(gateState({ ok: false, checks: [check(true)] })).toBe("failed");
    expect(gateState({ ok: true, checks: [check(true), check(false)] })).toBe("failed");
  });
  it("is unmeasured when no check measured anything", () => {
    expect(gateState({ ok: true, checks: [check(null), check(null)] })).toBe("unmeasured");
    expect(gateState({})).toBe("unmeasured");
  });
  it("passes when nothing failed and something was measured", () => {
    expect(gateState({ ok: true, checks: [check(true), check(null)] })).toBe("passed");
    expect(gateState({ ok: true })).toBe("passed");
  });
});

import { describe, expect, it } from "vitest";
import { T } from "./copy";
import { hoursText, planBody, sizingView, type Plan } from "./logic";

const plan = (extra: Record<string, unknown> = {}) => ({ pieces_per_step: 4096, ...extra }) as unknown as Plan;

describe("hoursText", () => {
  it("says minutes, hours or days", () => {
    expect([0.25, 1.5, 12, 60].map(hoursText)).toEqual(["15 min", "1.5 h", "12 h", "2.5 days"]);
  });
});

describe("planBody", () => {
  const base = { version: null, datasetIds: ["a"], mode: "catch_up" as const, rows: "16" as const, hold: "5%" as const, length: "Medium" as const };
  it("sends the hours given and any chosen steps", () => {
    expect(planBody(base)).toMatchObject({ hours: 4, steps: null, passes: 2 });
    expect(planBody({ ...base, hours: "12 h", steps: 500 })).toMatchObject({ hours: 12, steps: 500 });
  });
});

describe("sizingView", () => {
  const sized = (extra: Record<string, unknown> = {}) =>
    plan({ steps: 1000, sizing: { verdict: "model_too_small", warning: "Too small.", measured: true, run_hours: 3, device: "cpu", gpu_note: "", suggestion: null, expected: null, ...extra } });
  it("is null without a sizing", () => expect(sizingView(plan())).toBeNull());
  it("names the time and the device", () => expect(sizingView(sized())!.time).toBe("These 1,000 steps take about 3.0 h on the CPU."));
  it("says the design is being timed until it is", () => expect(sizingView(sized({ measured: false, run_hours: null }))!.time).toBe(T.sizingTiming));
  it("reads the fitted curve", () => {
    const view = sizingView(sized({ expected: { basis: "fit", loss_planned: 1.53, loss_flat: 1.6, flat_steps: 129129, curve: [] } }));
    expect(view!.expected).toContain("by step 129,129");
  });
  it("offers the step count only for the same design, and a design change only for another", () => {
    const found = { steps: 500, text: "x", same_design: true };
    expect(sizingView(sized({ suggestion: found }))!.suggestion).toMatchObject({ useSteps: true, designChange: false });
    expect(sizingView(sized({ suggestion: { ...found, same_design: false } }))!.suggestion).toMatchObject({ useSteps: false, designChange: true });
    expect(sizingView(sized({ suggestion: { ...found, steps: 1000 } }))!.suggestion!.useSteps).toBe(false);
  });
});

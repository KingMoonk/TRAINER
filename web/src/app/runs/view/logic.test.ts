import { describe, expect, test } from "vitest";
import { round, run } from "@/app/cluster/training/test-data";
import { clock, dilocoShape, extendRequest, gapNow, gapText, gaps, overfit, overfitBad, percentOf, roundPairs, roundSeries, runViewHref, soloShape, stepSeries, type Curve } from "./logic";

const curve = (over: Partial<Curve> = {}): Curve => ({ steps: [], smooth: [], heldout: [], ...over });
/** A training line that keeps falling, one point per 100 steps. */
const falling = (n: number): number[][] => Array.from({ length: n }, (_, i) => [i * 100, 4 - i * 0.1]);

describe("runViewHref", () => {
  test("is the Run view with the id in the query, encoded", () => {
    expect(runViewHref("v5-3f2a")).toBe("/runs/view/?id=v5-3f2a");
    expect(runViewHref("a b/c")).toBe("/runs/view/?id=a%20b%2Fc");
  });
});

describe("shape lines", () => {
  test("a single job names its machine twice, as W and as the queue's", () => {
    expect(soloShape("BEAST")).toBe("pretrain · local · W = {BEAST} · queue job on BEAST");
  });
  test("a DiLoCo run lists the number of machines, the coordinator and the settings", () => {
    expect(dilocoShape(run({ owner: "mac", inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9 }))).toBe(
      "pretrain · DiLoCo · W = {2} · c = mac · H = 100 · outer lr 0.7 · μ 0.9",
    );
  });
});

describe("overfit", () => {
  test("says it is too early before four checks of L_val", () => {
    const result = overfit(curve({ smooth: falling(5), heldout: [[0, 4], [100, 3.9], [200, 3.8]] }));
    expect(result).toEqual({ state: "early", checks: 3 });
    expect(overfitBad(result)).toBe(false);
  });
  test("is fine while L_val keeps improving", () => {
    const result = overfit(curve({ smooth: falling(5), heldout: [[0, 4], [100, 3.9], [200, 3.8], [300, 3.7]] }));
    expect(result.state).toBe("ok");
  });
  test("fires when L_val has not improved for 3 checks while L_train fell", () => {
    const result = overfit(curve({ smooth: falling(5), heldout: [[0, 3.5], [100, 3.6], [200, 3.55], [300, 3.7]] }));
    expect(result.state).toBe("stalled");
    expect(overfitBad(result)).toBe(true);
  });
  test("fires when the gap has grown for 3 checks in a row, even if L_val is still a little better", () => {
    // L_val creeps down 0.01 a check while L_train falls 0.1: the gap widens each time.
    const result = overfit(curve({ smooth: falling(5), heldout: [[0, 4.5], [100, 4.49], [200, 4.48], [300, 4.47]] }));
    expect(result.state).toBe("widening");
  });
  test("a flat L_val with a flat L_train is not overfitting", () => {
    const flat = Array.from({ length: 5 }, (_, i) => [i * 100, 3.5]);
    const result = overfit(curve({ smooth: flat, heldout: [[0, 3.6], [100, 3.6], [200, 3.6], [300, 3.6]] }));
    expect(overfitBad(result)).toBe(false);
  });
});

describe("gap", () => {
  test("is L_val minus the training loss at or before each check", () => {
    const c = curve({ smooth: [[0, 4], [100, 3.8], [200, 3.6]], heldout: [[100, 3.9], [200, 3.7]] });
    expect(gaps(c).map(([, g]) => Number(g.toFixed(2)))).toEqual([0.1, 0.1]);
    expect(gapText(gapNow(c))).toBe("+0.10");
  });
  test("reads from the raw line when there is no smoothed one, and is empty without a check", () => {
    expect(gapNow(curve({ steps: [[0, 4]], heldout: [[0, 3.9]] }))).toBeCloseTo(-0.1);
    expect(gapNow(curve({ smooth: falling(3) }))).toBeNull();
    expect(gapText(-0.02)).toBe("-0.02");
  });
});

describe("series", () => {
  test("a single job draws L_train solid and L_val dashed with a dot at each check", () => {
    const series = stepSeries(curve({ smooth: [[0, 4]], heldout: [[0, 4.1]] }));
    expect(series.map((s) => s.id)).toEqual(["train", "val"]);
    expect(series[0].dash).toBeUndefined();
    expect(series[1]).toMatchObject({ dash: "5 4", dots: true });
  });
  test("a missing curve or an empty line draws nothing", () => {
    expect(stepSeries(undefined)).toEqual([]);
    expect(stepSeries(curve())).toEqual([]);
  });
  test("a DiLoCo run draws its loss by round, counted from one, leaving a round without a number out", () => {
    const history = [round(0), round(1, { val_loss: null })];
    expect(roundSeries(history).map((s) => [s.id, s.points.length])).toEqual([["loss", 2], ["val_loss", 1]]);
    expect(roundPairs(history, (r) => r.loss)[0][0]).toBe(1);
  });
});

describe("small helpers", () => {
  test("percentOf stays between 0 and 100 and is 0 for an unknown total", () => {
    expect(percentOf(1, 4)).toBe(25);
    expect(percentOf(9, 4)).toBe(100);
    expect(percentOf(1, 0)).toBe(0);
  });
  test("clock reads minutes and seconds, and hours past an hour", () => {
    expect(clock(130)).toBe("2:10");
    expect(clock(3725)).toBe("1:02:05");
  });
});

describe("extendRequest", () => {
  const going = { rounds: 10, inner_steps: 100 };
  test("adds the rounds to the run's own and leaves H out when it is not changed", () => {
    expect(extendRequest(going, "40", "")).toEqual({ rounds: 50 });
    expect(extendRequest(going, "40", "100")).toEqual({ rounds: 50 });
  });
  test("sends a new H alone or with the rounds", () => {
    expect(extendRequest(going, "0", "200")).toEqual({ inner_steps: 200 });
    expect(extendRequest(going, "5", "50")).toEqual({ rounds: 15, inner_steps: 50 });
  });
  test("is null for what the API would refuse or what changes nothing", () => {
    for (const [add, steps] of [["0", ""], ["", ""], ["-1", ""], ["1.5", ""], ["abc", ""], ["5", "0"], ["5", "2.5"], ["100000", ""]]) expect(extendRequest(going, add, steps)).toBeNull();
  });
});

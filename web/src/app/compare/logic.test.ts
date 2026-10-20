import { describe, expect, it } from "vitest";
import { COMPARE } from "./copy";
import {
  axisFor,
  axisPoints,
  curveSeries,
  deltaCell,
  finalCell,
  letter,
  matchingText,
  queueBlock,
  variantName,
  type CurvePoint,
  type VariantStats,
} from "./logic";

const stats = (extra: Partial<VariantStats> = {}): VariantStats => ({
  variant: 1, bpc: [2.5, 2.7], mean: 2.6, min: 2.5, max: 2.7, spread: 0.2, delta: -0.1, inside_spread: false, pieces_per_s: 1234.4, peak_mb: 512.6, curve: [], ...extra,
});

// [step, bpc, tokens, seconds]
const curve: CurvePoint[] = [[0, 4, 0, 0], [10, 3, 1000, 120], [20, 2, 2000, 240]];

describe("axisPoints", () => {
  it("reads steps, tokens and minutes from their columns", () => {
    expect(axisPoints(curve, "steps")).toEqual([{ x: 0, y: 4 }, { x: 10, y: 3 }, { x: 20, y: 2 }]);
    expect(axisPoints(curve, "tokens")?.map((p) => p.x)).toEqual([0, 1000, 2000]);
    expect(axisPoints(curve, "time")?.map((p) => p.x)).toEqual([0, 2, 4]);
  });
  it("is null when any point lacks the column, or the error value", () => {
    expect(axisPoints([[0, 4], [1, 3]], "tokens")).toBeNull();
    expect(axisPoints([[0, 4, null, 1]], "tokens")).toBeNull();
    expect(axisPoints([[0, 4, 1, 1], [1, null, 1, 1]], "steps")).toBeNull();
    expect(axisPoints([[0, 4, 1, 1], [1, 3, 1]], "time")).toBeNull();
  });
  it("keeps zero values and an empty curve", () => {
    expect(axisPoints([[0, 0, 0, 0]], "time")).toEqual([{ x: 0, y: 0 }]);
    expect(axisPoints([], "tokens")).toEqual([]);
  });
});

describe("axisFor", () => {
  it("keeps the asked-for axis when every curve has it", () => {
    expect(axisFor([curve, curve], "tokens")).toEqual({ axis: "tokens", missing: null });
  });
  it("falls back to steps, naming the missing axis, when any curve lacks it", () => {
    expect(axisFor([curve, [[0, 4]]], "time")).toEqual({ axis: "steps", missing: "time" });
  });
  it("never falls back from steps", () => expect(axisFor([[[0, null]]], "steps")).toEqual({ axis: "steps", missing: null }));
  it("accepts no curves", () => expect(axisFor([], "tokens")).toEqual({ axis: "tokens", missing: null }));
});

describe("letter and variantName", () => {
  it("letters variants from A", () => expect([0, 1, 25].map(letter)).toEqual(["A", "B", "Z"]));
  it("prefers the saved name, then Base for the base and Variant n for the rest", () => {
    expect(variantName(["Tiny", "Wide"], 1)).toBe("Wide");
    expect(variantName(undefined, 0)).toBe("Base");
    expect(variantName([], 2)).toBe("Variant 2");
    expect(variantName(["", ""], 1)).toBe("Variant 1");
  });
});

describe("curveSeries", () => {
  const curves = new Map<number, CurvePoint[]>([[1, curve], [0, curve]]);
  it("draws a series per variant in variant order, labelled with letter and name", () => {
    const series = curveSeries(curves, ["Base", "Wide"], "steps", undefined);
    expect(series.map((s) => [s.id, s.label])).toEqual([["variant-A", "A · Base"], ["variant-B", "B · Wide"]]);
    expect(series[0].points).toEqual([{ x: 0, y: 4 }, { x: 10, y: 3 }, { x: 20, y: 2 }]);
  });
  it("leaves out a variant that lacks the axis", () => {
    const mixed = new Map<number, CurvePoint[]>([[0, curve], [1, [[0, 4], [1, 3]]]]);
    expect(curveSeries(mixed, undefined, "tokens", undefined).map((s) => s.id)).toEqual(["variant-A"]);
  });
  it("adds flat reference lines across the plotted range, using friendly names", () => {
    const series = curveSeries(curves, undefined, "tokens", { uniform: 6, other: 3 });
    expect(series.slice(2)).toEqual([
      { id: "reference-uniform", label: COMPARE.referenceNames.uniform, points: [{ x: 0, y: 6 }, { x: 2000, y: 6 }] },
      { id: "reference-other", label: "other", points: [{ x: 0, y: 3 }, { x: 2000, y: 3 }] },
    ]);
  });
  it("skips non-numeric references, and adds none when nothing is plotted", () => {
    expect(curveSeries(curves, undefined, "steps", { uniform: "x" as unknown as number })).toHaveLength(2);
    expect(curveSeries(new Map(), undefined, "steps", { uniform: 6 })).toEqual([]);
  });
});

describe("finalCell", () => {
  it("is a dash before results", () => expect(finalCell(undefined)).toBe("–"));
  it("shows the mean and spread over several seeds, then speed and peak memory", () => {
    expect(finalCell(stats())).toBe(`2.600 · ${COMPARE.spread} 0.200\n1,234 pieces/s · 513 MB peak`);
  });
  it("says one seed for a single run", () => {
    expect(finalCell(stats({ bpc: [2.6] }))).toBe(`2.600 · ${COMPARE.oneSeed}\n1,234 pieces/s · 513 MB peak`);
  });
});

describe("deltaCell", () => {
  it("is a dash before results and a long dash without a difference", () => {
    expect(deltaCell(undefined)).toBe("–");
    expect(deltaCell(stats({ delta: null }))).toBe("—");
    expect(deltaCell(stats({ delta: undefined }))).toBe("—");
  });
  it("signs the difference and says whether it is inside the spread", () => {
    expect(deltaCell(stats({ delta: -0.1234, inside_spread: false }))).toBe(`-0.123 · ${COMPARE.beyondSpread}`);
    expect(deltaCell(stats({ delta: 0.1234, inside_spread: true }))).toBe(`+0.123 · ${COMPARE.insideSpread}`);
    expect(deltaCell(stats({ delta: 0, inside_spread: true }))).toBe(`+0.000 · ${COMPARE.insideSpread}`);
  });
  it("says there is no spread to compare against with one seed", () => {
    expect(deltaCell(stats({ delta: 0.5, inside_spread: null }))).toBe(`+0.500 · ${COMPARE.oneSeedNoSpread}`);
    expect(deltaCell(stats({ delta: 0.5, inside_spread: undefined }))).toBe(`+0.500 · ${COMPARE.oneSeedNoSpread}`);
  });
});

describe("queueBlock", () => {
  it("asks for a variant first, else nothing", () => {
    expect(queueBlock(0)).toBe(COMPARE.needVariant);
    expect(queueBlock(2)).toBe("");
  });
});

describe("matchingText", () => {
  const plan = (extra = {}) => ({ variant: 1, parameters: 1_234_567, steps: 150, relative_error: 0.0123, adjustments: [], ...extra });
  it("is a dash for the base", () => expect(matchingText(plan({ variant: 0 }), 100, "params", 60, true)).toBe("—"));
  it("params: says whether the width was re-solved, and how close it landed", () => {
    expect(matchingText(plan(), 100, "params", 60, true)).toBe(`${COMPARE.noneNeeded("1.23M")} · within 1.2%`);
    expect(matchingText(plan({ adjustments: [{}] }), 100, "params", 60, true)).toBe(`${COMPARE.widthResolved("1.23M")} · within 1.2%`);
    expect(matchingText(plan({ relative_error: null }), 100, "params", 60, true)).toBe(`${COMPARE.noneNeeded("1.23M")} · within 0.0%`);
  });
  it("compute: shows the extra text it sees, or the same compute", () => {
    expect(matchingText(plan(), 100, "compute", 60, true)).toBe(COMPARE.stopsAtBase("50%"));
    expect(matchingText(plan({ steps: 100 }), 100, "compute", 60, true)).toBe(COMPARE.sameCompute);
    expect(matchingText(plan({ steps: 100.4 }), 100, "compute", 60, true)).toBe(COMPARE.sameCompute); // under 0.5%
    expect(matchingText(plan({ steps: 80 }), 100, "compute", 60, true)).toBe(COMPARE.sameCompute);
    expect(matchingText(plan({ steps: null }), 100, "compute", 60, true)).toBe(COMPARE.sameCompute);
    expect(matchingText(plan(), null, "compute", 60, true)).toBe(COMPARE.sameCompute);
  });
  it("tokens: shows the parameter count", () => expect(matchingText(plan(), 100, "tokens", 60, true)).toBe(COMPARE.parameters("1.23M")));
  it("any other mode: shows the time budget in minutes", () => {
    expect(matchingText(plan(), 100, "time", 90, true)).toBe(COMPARE.minBudget("1.5"));
    expect(matchingText(plan(), 100, "time", 600, true)).toBe(COMPARE.minBudget("10"));
  });
  it("adds the not-retuned note when the learning rate was not retuned", () => {
    expect(matchingText(plan(), 100, "tokens", 60, false)).toBe(COMPARE.parameters("1.23M") + COMPARE.notRetuned);
  });
});

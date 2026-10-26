import { describe, expect, it } from "vitest";
import { GATE_SEEDS, SWITCH, T, fill } from "./copy";
import { bits, chartSeries, last, phaseBands, progressLine, sampleRows, switchText, type Pair } from "./logic";

describe("sampleRows", () => {
  it("starts from the parent's weights, or from random ones", () => {
    expect(sampleRows([{ step: 100, text: "abc" }], 120, 3)[0]).toEqual({ label: T.step0, text: fill(T.identicalTo, { parent: "version 3" }) });
    expect(sampleRows([{ step: 100, text: "abc" }], 120, null)[0]).toEqual({ label: T.step0, text: T.randomStart });
    expect(sampleRows([{ step: 100, text: "abc" }], 120, 0)[0].text).toBe(T.randomStart);
  });
  it("lists each logged sample and marks the newest as now", () => {
    const rows = sampleRows([{ step: 100, text: "a" }, { step: 200, text: "b" }, { step: 300, text: "c" }], 350, null);
    expect(rows.map((r) => r.label)).toEqual([T.step0, "step 100", "step 200", fill(T.nowStep, { step: 300 })]);
    expect(rows.map((r) => r.text).slice(1)).toEqual(["a", "b", "c"]);
  });
  it("marks a single sample as now", () => {
    expect(sampleRows([{ step: 50, text: "a" }], null, null).map((r) => r.label)).toEqual([T.step0, fill(T.nowStep, { step: 50 })]);
  });
  it("before the first sample, adds a wait note at the current step", () => {
    expect(sampleRows([], 12345, null)[1]).toEqual({ label: fill(T.nowStep, { step: "12,345" }), text: T.noSampleYet });
    expect(sampleRows([], 0, null)[1].label).toBe(fill(T.nowStep, { step: "0" }));
  });
  it("labels the wait note plainly when the step is unknown", () => expect(sampleRows([], null, null)[1]).toEqual({ label: "now", text: T.noSampleYet }));
});

describe("bits and last", () => {
  it("shows three decimals, or a dash", () => {
    expect(bits(1.23456)).toBe("1.235");
    expect(bits(0)).toBe("0.000");
    expect(bits(null)).toBe("–");
    expect(bits(undefined)).toBe("–");
  });
  it("takes the value of the final point", () => {
    expect(last([[1, 5], [2, 7]])).toBe(7);
    expect(last([])).toBeNull();
  });
});

describe("chartSeries", () => {
  const labels = { train: "Train", raw: "Raw", lr: "Rate", held: "Held", uniform: "Uniform", unigram: "Unigram", best: "Best" };
  const pairs = (...ys: number[]): Pair[] => ys.map((y, i) => [i * 10, y]);
  const curve = { steps: pairs(5, 4, 3), smooth: pairs(5, 4, 3), heldout: [[10, 4.5], [30, 3.5]] as Pair[], lr: [] as Pair[] };

  it("draws the raw, smoothed and held-out curves", () => {
    const series = chartSeries(curve, {}, labels);
    expect(series.map((s) => [s.id, s.label])).toEqual([["steps", "Raw"], ["smooth", "Train"], ["held", "Held"]]);
    expect(series[2].points).toEqual([{ x: 10, y: 4.5 }, { x: 30, y: 3.5 }]);
  });
  it("draws each reference as a flat line across every curve's range", () => {
    const series = chartSeries(curve, { uniform_bpc: 6, unigram_bpc: 4.2, best_so_far_bpc: 3.1 }, labels);
    expect(series.slice(3)).toEqual([
      { id: "uniform", label: "Uniform", points: [{ x: 0, y: 6 }, { x: 30, y: 6 }] },
      { id: "unigram", label: "Unigram", points: [{ x: 0, y: 4.2 }, { x: 30, y: 4.2 }] },
      { id: "best", label: "Best", points: [{ x: 0, y: 3.1 }, { x: 30, y: 3.1 }] },
    ]);
  });
  it("skips missing references but keeps a reference of 0", () => {
    const series = chartSeries(curve, { uniform_bpc: null, unigram_bpc: 0 }, labels);
    expect(series.map((s) => s.id)).toEqual(["steps", "smooth", "held", "unigram"]);
  });
  it("adds the learning rate on its own scale only when there is one", () => {
    expect(chartSeries(curve, {}, labels).some((s) => s.id === "lr")).toBe(false);
    const series = chartSeries({ ...curve, lr: [[0, 0.001], [40, 0.0005]] }, {}, labels);
    expect(series.at(-1)).toMatchObject({ id: "lr", label: "Rate", axis: "right", points: [{ x: 0, y: 0.001 }, { x: 40, y: 0.0005 }] });
    // the learning rate's own x range stretches the reference lines
    expect(chartSeries({ ...curve, lr: [[0, 0.001], [40, 0.0005]] }, { uniform_bpc: 6 }, labels).find((s) => s.id === "uniform")?.points[1].x).toBe(40);
  });
  it("copes with an empty curve", () => {
    const series = chartSeries({ steps: [], smooth: [], heldout: [], lr: [] }, { uniform_bpc: 6 }, labels);
    expect(series.find((s) => s.id === "uniform")?.points).toEqual([{ x: 0, y: 6 }, { x: 0, y: 6 }]);
    expect(series.slice(0, 3).every((s) => s.points.length === 0)).toBe(true);
  });
});

describe("phaseBands", () => {
  it("turns each [from, to, label] phase into a band with a unique id", () => {
    expect(phaseBands([[0, 100, "warmup"], [100, 500, "steady"], [500, 600, "steady"]])).toEqual([
      { id: "warmup-0", label: "warmup", from: 0, to: 100 },
      { id: "steady-1", label: "steady", from: 100, to: 500 },
      { id: "steady-2", label: "steady", from: 500, to: 600 },
    ]);
  });
  it("drops phases that are malformed, empty or run backwards, keeping the index of the rest", () => {
    expect(phaseBands([["0", 5, "a"], [0, "5", "a"], [0, 5, 7], [5, 5, "flat"], [9, 3, "back"], [0, Number.NaN, "nan"], [0, Infinity, "inf"], [1, 2, "ok"]])).toEqual([
      { id: "ok-7", label: "ok", from: 1, to: 2 },
    ]);
    expect(phaseBands([])).toEqual([]);
    expect(phaseBands([[1, 2]])).toEqual([]);
  });
});

describe("switchText", () => {
  const note = { key: "gate_better", gap: "0.12", docs: 3, need: 5 };
  it("names the version, the earlier best and the gap", () => {
    expect(switchText(note, 3, 2, null, false)).toBe(fill(SWITCH.gate_better, { Version: "Version 3", version: "version 3", parent: "version 2", gap: "0.12" }));
  });
  it("compares against the gate's version when the run has ended, else the parent", () => {
    expect(switchText(note, 3, 2, 1, false)).toContain("beat version 1 on");
    expect(switchText(note, 3, 2, 1, true)).toContain("beat version 2 on");
  });
  it("uses generic names without a version or parent", () => {
    const text = switchText({ key: "held_back_inside" }, null, null, null, true);
    expect(text).toBe(fill(SWITCH.held_back_inside, { Version: "This version", version: "this version", parent: "the earlier best" }));
  });
  it("adds the seeds caveat to inconclusive verdicts only", () => {
    expect(switchText({ key: "gate_not_sure", gap: "0.01" }, 3, 2, null, false).endsWith(` ${GATE_SEEDS}`)).toBe(true);
    expect(switchText({ key: "gate_not_better", gap: "0.01" }, 3, 2, null, false).endsWith(` ${GATE_SEEDS}`)).toBe(true);
    expect(switchText(note, 3, 2, null, false)).not.toContain(GATE_SEEDS);
  });
  it("is empty for an unknown key", () => expect(switchText({ key: "nope" }, 1, null, null, false)).toBe(""));
  it("fills the document counts in where the sentence has them", () => {
    const key = Object.keys(SWITCH).find((k) => SWITCH[k].includes("{docs}"))!;
    expect(switchText({ key, docs: 7, need: 9 }, 2, 1, null, false)).toContain("only 7 held-back documents; at least 9 are needed");
  });
});

describe("progressLine", () => {
  const q = (extra = {}) => ({ step: 100, total: 1000, phase: "", lr: null, eta_s: null, stopping: false, blocked: null, params: {}, ...extra });
  it("shows the step and a pass count from the default sizes", () => {
    // 16 rows x 256 pieces per row: 100 steps = 409,600 pieces
    expect(progressLine(q())).toBe(`${T.step} 100 ${T.of} 1,000 · 409,600 ${T.piecesSeen} · ${T.lrAuto}`);
  });
  it("counts passes over the text from pieces_per_pass", () => {
    const line = progressLine(q({ params: { config: { batch_size: 10, context: 100 }, pieces_per_pass: 30_000 } }));
    // 100 x 10 x 100 = 100,000 pieces seen = pass 4 of 30,000
    expect(line).toContain(`${T.pass} 4 ${T.overText} · 100,000 ${T.piecesSeen}`);
  });
  it("says a checkpoint is being saved", () => {
    expect(progressLine(q({ saving: true })).endsWith(` · ${T.saving}`)).toBe(true);
    expect(progressLine(q({ saving: false }))).not.toContain(T.saving);
  });
  it("says starting without a total", () => expect(progressLine(q({ total: 0, step: 0 })).startsWith(T.starting)).toBe(true));
  it("adds the phase, learning rate and time left when known", () => {
    const line = progressLine(q({ phase: "steady", lr: 0.00123, eta_s: 600 }));
    expect(line).toContain(`${T.schedule} steady`);
    expect(line).toContain("learning rate: auto (0.0012)");
    expect(line.endsWith(`about 10 min ${T.left}`)).toBe(true);
  });
  it("never shows less than a minute left", () => expect(progressLine(q({ eta_s: 3 }))).toContain(`about 1 min ${T.left}`));
  it("adds the cool-down and what the job is waiting on, last", () => {
    expect(progressLine(q({ stopping: true, blocked: "battery" })).endsWith(` · ${T.coolingDown} · ${T.waiting} battery`)).toBe(true);
  });
});

import { describe, expect, it } from "vitest";
import { lossView, probability, showPiece, type LossResult } from "./logic";

const data = (extra: Partial<LossResult> = {}): LossResult => ({
  tokens: ["a"], pos: 0, probabilities: [0.1, 0.5, 0.25, 0.15], target: 2, pieces: ["a", "b", "cd", "e"], ...extra,
});

describe("lossView", () => {
  it("puts the target first, then the most likely others, with the gradient of each", () => {
    const view = lossView(data());
    expect(view.rows.map((r) => r.index)).toEqual([2, 1, 3, 0]);
    expect(view.rows[0]).toEqual({ index: 2, piece: "cd", probability: 0.25, target: 1, gradient: -0.75 });
    expect(view.rows[1]).toMatchObject({ piece: "b", target: 0, gradient: 0.5 });
  });
  it("measures the loss in nats and in bits per character of the target piece", () => {
    const view = lossView(data());
    expect(view.loss).toBeCloseTo(-Math.log(0.25), 10);
    expect(view.characters).toBe(2);
    expect(view.bits).toBeCloseTo(-Math.log(0.25) / (2 * Math.LN2), 10);
    expect(view.piece).toBe("cd");
    expect(view.probability).toBe(0.25);
  });
  it("counts code points, not UTF-16 units, and at least one character", () => {
    expect(lossView(data({ pieces: ["a", "b", "😀", "e"] })).characters).toBe(1);
    expect(lossView(data({ pieces: ["a", "b", "", "e"] })).characters).toBe(1);
  });
  it("reports the probability left out and how many pieces were not listed", () => {
    const view = lossView(data({ probabilities: [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4], pieces: Array.from({ length: 8 }, (_, i) => `p${i}`), target: 7 }));
    expect(view.rows).toHaveLength(5);
    expect(view.others).toBe(3);
    expect(view.remaining).toBeCloseTo(0.2, 10); // 0.05 + 0.05 + 0.1 unlisted
  });
  it("does not list the target twice when it is among the top five", () => {
    const view = lossView(data({ probabilities: [0.1, 0.5, 0.3, 0.1], target: 1 }));
    expect(view.rows.map((r) => r.index)).toEqual([1, 2, 0, 3]);
    expect(view.others).toBe(0);
  });
  it("keeps the loss finite when the target has probability 0", () => {
    const view = lossView(data({ probabilities: [0.5, 0.5, 0, 0], target: 2 }));
    expect(Number.isFinite(view.loss)).toBe(true);
    expect(view.loss).toBeGreaterThan(600);
  });
  it("never reports negative remaining probability", () => {
    expect(lossView(data({ probabilities: [1, 1, 1, 1], target: 0 })).remaining).toBe(0);
  });
  it.each([
    ["a target below range", { target: -1 }],
    ["a target past the end", { target: 4 }],
    ["a fractional target", { target: 1.5 }],
    ["pieces of the wrong length", { pieces: ["a"] }],
    ["a probability above 1", { probabilities: [2, 0, 0, 0] }],
    ["a negative probability", { probabilities: [-0.1, 0.5, 0.3, 0.3] }],
    ["a NaN probability", { probabilities: [Number.NaN, 0.5, 0.3, 0.3] }],
    ["an infinite probability", { probabilities: [Infinity, 0, 0, 0] }],
  ])("rejects %s", (_, extra) => expect(() => lossView(data(extra))).toThrow("invalid probability distribution"));
});

describe("showPiece", () => {
  it("makes spaces, newlines and tabs visible", () => expect(showPiece("a b\nc\td")).toBe("a·b↵c⇥d"));
  it("leaves other text alone", () => expect(showPiece("abc")).toBe("abc"));
});

describe("probability", () => {
  it("shows two decimals, or an exponent below a thousandth", () => {
    expect(probability(0.256)).toBe("0.26");
    expect(probability(0.001)).toBe("0.00");
    expect(probability(0.000123)).toBe("1.2e-4");
    expect(probability(0)).toBe("0.0e+0");
    expect(probability(1)).toBe("1.00");
  });
});

import { describe, expect, it } from "vitest";
import { BENCH } from "./copy";
import {
  CHARTS,
  SHOTS,
  againstChance,
  bandText,
  chartFor,
  designs,
  fewShotNote,
  fewShotRows,
  gainText,
  hasItems,
  pending,
  shotsText,
  statusText,
  tokensText,
  trendTables,
  valueText,
  versionName,
  type BenchmarkOut,
  type BenchmarkRow,
} from "./logic";

const row = (number: number, extra: Partial<BenchmarkRow> = {}): BenchmarkRow => ({ number, lineage: "a", tokens: number * 1000, result: {}, ...extra });
const withSizes: BenchmarkOut["sizes"] = { docs: 5, choice: 3, problems: 2 };

describe("tokensText", () => {
  it.each([
    [0, "0"],
    [999, "999"],
    [1000, "1K"],
    [1499, "1K"],
    [1500, "2K"],
    [999_999, "1000K"],
    [1e6, "1.0M"],
    [2_540_000, "2.5M"],
    [1e9, "1.0B"],
    [3.26e9, "3.3B"],
  ])("%d reads as %s", (n, text) => expect(tokensText(n)).toBe(text));
});

describe("valueText", () => {
  it("shows bits per character with two decimals", () => expect(valueText("bpc", 1.2345)).toBe("1.23"));
  it("shows the other checks as a rounded percentage", () => {
    expect(valueText("few_shot", 0.256)).toBe("26%");
    expect(valueText("problems", 0)).toBe("0%");
    expect(valueText("problems", 1)).toBe("100%");
  });
});

describe("versionName", () => {
  it("prefers the saved name", () => expect(versionName(row(3, { name: "Tiny" }))).toBe("Tiny"));
  it("falls back to the number for a missing or empty name", () => {
    expect(versionName(row(3))).toBe("Version 3");
    expect(versionName(row(4, { name: "" }))).toBe("Version 4");
  });
});

describe("designs", () => {
  it("letters designs by their first version, whatever order the rows come in", () => {
    const out = designs([row(5, { lineage: "b" }), row(1, { lineage: "a" }), row(3, { lineage: "b" }), row(2, { lineage: "c" })]);
    expect([...out]).toEqual([
      ["a", { letter: "A", numbers: [1] }],
      ["c", { letter: "B", numbers: [2] }],
      ["b", { letter: "C", numbers: [3, 5] }],
    ]);
  });
  it("groups rows without a lineage under one design", () => {
    const out = designs([row(2, { lineage: undefined }), row(1, { lineage: undefined })]);
    expect(out.get("")).toEqual({ letter: "A", numbers: [1, 2] });
  });
  it("is empty without rows and does not reorder its input", () => {
    expect(designs([]).size).toBe(0);
    const rows = [row(2), row(1)];
    designs(rows);
    expect(rows.map((r) => r.number)).toEqual([2, 1]);
  });
});

describe("pending", () => {
  it("lists the unchecked versions, oldest first", () => {
    const state: BenchmarkOut = { suite: "s", sizes: withSizes, rows: [row(3, { result: null }), row(1), row(2, { result: undefined })] };
    expect(pending(state)).toEqual([2, 3]);
  });
  it("is empty without a state, without held-back data, or with every size zero", () => {
    expect(pending(undefined)).toEqual([]);
    expect(pending({ suite: "s", rows: [row(1, { result: null })] })).toEqual([]);
    expect(pending({ suite: "s", sizes: { docs: 0, choice: 0 }, rows: [row(1, { result: null })] })).toEqual([]);
  });
  it("is empty when there are no rows", () => expect(pending({ suite: "s", sizes: withSizes })).toEqual([]));
  it("does not count an empty result object as missing", () => {
    expect(pending({ suite: "s", sizes: withSizes, rows: [row(1, { result: {} })] })).toEqual([]);
  });
});

describe("shotsText", () => {
  const fewShot = (used: number) => row(1, { result: { few_shot: { rows: [{ used: 0 }, { used }] } } });
  it("gives the number asked for when every question fit", () => {
    expect(shotsText({ suite: "s", rows: [fewShot(SHOTS)] })).toBe(String(SHOTS));
  });
  it("says 'up to' when a version fit fewer on average", () => {
    expect(shotsText({ suite: "s", rows: [fewShot(SHOTS), fewShot(2.5)] })).toBe(BENCH.upTo(SHOTS));
  });
  it("ignores rows without a few-shot table or a numeric 'used'", () => {
    const rows = [row(1, { result: null }), row(2, { result: {} }), row(3, { result: { few_shot: { rows: [] } } }), row(4, { result: { few_shot: null } })];
    expect(shotsText({ suite: "s", rows })).toBe(String(SHOTS));
    expect(shotsText({ suite: "s" })).toBe(String(SHOTS));
  });
});

describe("hasItems", () => {
  it("reads the size that belongs to each check", () => {
    const state: BenchmarkOut = { suite: "s", sizes: { docs: 1, choice: 0, problems: 4 } };
    expect(hasItems(state, "bpc")).toBe(true);
    expect(hasItems(state, "few_shot")).toBe(false);
    expect(hasItems(state, "problems")).toBe(true);
  });
  it("is false without sizes", () => expect(hasItems({ suite: "s" }, "bpc")).toBe(false));
});

describe("statusText", () => {
  it("says there are no versions first", () => {
    expect(statusText({ suite: "s", sizes: withSizes, rows: [] })).toBe(BENCH.noVersions);
    expect(statusText({ suite: "s" })).toBe(BENCH.noVersions);
  });
  it("says there is no data when rows exist but nothing is held back", () => {
    expect(statusText({ suite: "s", sizes: { docs: 0 }, rows: [row(1)] })).toBe(BENCH.noData);
  });
  it("counts the versions still waiting", () => {
    expect(statusText({ suite: "s", sizes: withSizes, rows: [row(1), row(2, { result: null }), row(3, { result: null })] })).toBe(BENCH.waiting(2, 3));
  });
  it("says everything is checked when no result is missing", () => {
    expect(statusText({ suite: "s", sizes: withSizes, rows: [row(1)] })).toBe(BENCH.allDone);
  });
});

describe("againstChance", () => {
  // 100 questions at 25%: expect 25, two standard deviations is 2 * sqrt(18.75), about 8.66.
  it.each([
    [34, BENCH.shotsAbove],
    [33, BENCH.shotsNear],
    [25, BENCH.shotsNear],
    [17, BENCH.shotsNear],
    [16, BENCH.shotsBelow],
  ])("%d right of 100 is %s", (correct, verdict) => expect(againstChance(correct, 100, 0.25)).toBe(verdict));
  it("treats chance of 0 or 1 as exact, with no spread", () => {
    expect(againstChance(1, 10, 0)).toBe(BENCH.shotsAbove);
    expect(againstChance(0, 10, 0)).toBe(BENCH.shotsNear);
    expect(againstChance(9, 10, 1)).toBe(BENCH.shotsBelow);
  });
});

describe("fewShotRows", () => {
  const table = { items: 20, tasks: 2, chance: 0.25, rows: [
    { shots: 0, used: 0, correct: 5, accuracy: 0.25 },
    { shots: 1, used: 1, correct: 6, accuracy: 0.3 },
    { shots: 4, used: 2.5, correct: 19, accuracy: null },
  ] };
  it("builds a row of five cells per number of examples", () => {
    expect(fewShotRows(table)).toEqual([
      [BENCH.shotsZero, "5 of 20", "25%", "25%", BENCH.shotsNear],
      [BENCH.shotsOne, "6 of 20", "30%", "25%", BENCH.shotsNear],
      [`4${BENCH.shotsFitted("2.5")}`, "19 of 20", "–", "25%", BENCH.shotsAbove],
    ]);
  });
  it("is empty when nothing was asked or the guess is unknown", () => {
    expect(fewShotRows(undefined)).toEqual([]);
    expect(fewShotRows({ ...table, items: 0 })).toEqual([]);
    expect(fewShotRows({ ...table, chance: null })).toEqual([]);
    expect(fewShotRows({ items: 5 })).toEqual([]);
  });
  it("keeps a chance of zero (not the same as unknown)", () => {
    expect(fewShotRows({ items: 4, chance: 0, rows: [{ shots: 2, used: 2, correct: 1, accuracy: 0.25 }] })).toEqual([["2", "1 of 4", "25%", "0%", BENCH.shotsAbove]]);
  });
  it("is empty rows when the table has no rows", () => expect(fewShotRows({ items: 5, chance: 0.5 })).toEqual([]));
});

describe("fewShotNote", () => {
  it("states what was asked and what guessing scores", () => {
    expect(fewShotNote({ items: 20, tasks: 2, chance: 0.256 })).toBe(`${BENCH.shotsAsked(20, 2)} ${BENCH.shotsChance("26")}`);
  });
  it("adds the training-set line only when explicitly not held back", () => {
    expect(fewShotNote({ items: 20, held_back: false, chance: 0.25 })).toBe([BENCH.shotsAsked(20, 1), BENCH.shotsTraining, BENCH.shotsChance("25")].join(" "));
    expect(fewShotNote({ items: 20, held_back: true, chance: 0.25 })).not.toContain(BENCH.shotsTraining);
    expect(fewShotNote({ items: 20, chance: 0.25 })).not.toContain(BENCH.shotsTraining);
  });
  it("defaults the tasks to 1 and an unknown chance to 0%", () => {
    expect(fewShotNote({ items: 3, chance: null })).toBe(`${BENCH.shotsAsked(3, 1)} ${BENCH.shotsChance("0")}`);
  });
});

describe("CHARTS", () => {
  it("has one chart per metric, in order", () => expect(CHARTS.map((c) => c.metric)).toEqual(["bpc", "few_shot", "problems"]));
  it("only the multiple-choice description mentions the shots", () => {
    const [bpc, fewShot, problems] = CHARTS;
    expect(bpc.desc("up to 4")).toBe(BENCH.bpcDesc);
    expect(problems.desc("up to 4")).toBe(BENCH.problemsDesc);
    expect(fewShot.desc("up to 4")).toBe(BENCH.fewShotDesc("up to 4"));
  });
});

describe("chartFor", () => {
  const rows = [
    row(1, { lineage: "a", tokens: 1000, result: { bpc: 3, few_shot: { accuracy: 0.3, chance: 0.25 }, problems: { rate: 0.1 } } }),
    row(2, { lineage: "b", tokens: 3000, result: { bpc: 2, few_shot: { accuracy: 0.35, chance: 0.25 } } }),
    row(3, { lineage: "a", tokens: 2000, result: { bpc: 2.5, problems: { rate: 0.2 } } }),
  ];
  const state: BenchmarkOut = { suite: "s", sizes: withSizes, rows };

  it("draws a line per design with its points sorted by tokens", () => {
    const { series, chance, unplaced } = chartFor(state, "bpc");
    expect(series.map((s) => [s.id, s.label])).toEqual([
      ["design-A", BENCH.design("A", [1, 3])],
      ["design-B", BENCH.design("B", [2])],
    ]);
    expect(series[0].points).toEqual([{ x: 1000, y: 3 }, { x: 2000, y: 2.5 }]);
    expect(chance).toBeNull();
    expect(unplaced).toEqual([]);
  });
  it("adds the guessing line across the chart for the multiple-choice check only", () => {
    const { series, chance } = chartFor(state, "few_shot");
    expect(chance).toBe(0.25);
    expect(series.at(-1)).toEqual({ id: "chance", label: BENCH.chance("25%"), points: [{ x: 0, y: 0.25 }, { x: 3000, y: 0.25 }] });
    expect(series).toHaveLength(3);
  });
  it("skips versions that have no number for the check", () => {
    const { series } = chartFor(state, "problems");
    expect(series.map((s) => s.id)).toEqual(["design-A"]);
    expect(series[0].points).toEqual([{ x: 1000, y: 0.1 }, { x: 2000, y: 0.2 }]);
  });
  it("leaves versions without a token count off the chart and reports them", () => {
    const { series, unplaced } = chartFor({ suite: "s", rows: [row(1, { tokens: null, result: { bpc: 1 } }), row(2, { tokens: undefined, result: { bpc: 1 } }), row(3, { result: { bpc: 1 } })] }, "bpc");
    expect(unplaced).toEqual([1, 2]);
    expect(series).toHaveLength(1);
    expect(series[0].points).toEqual([{ x: 3000, y: 1 }]);
  });
  it("keeps a token count of zero", () => {
    expect(chartFor({ suite: "s", rows: [row(1, { tokens: 0, result: { bpc: 1 } })] }, "bpc").series[0].points).toEqual([{ x: 0, y: 1 }]);
  });
  it("treats non-numeric values as missing", () => {
    const odd = { suite: "s", rows: [row(1, { result: { bpc: null } }), row(2, { result: { bpc: "1" as unknown as number } }), row(3, { result: null })] };
    expect(chartFor(odd, "bpc")).toEqual({ series: [], chance: null, unplaced: [] });
  });
  it("draws no guessing line when there are no series, even with a chance", () => {
    const only = { suite: "s", rows: [row(1, { tokens: null, result: { few_shot: { accuracy: 0.3, chance: 0.5 } } })] };
    const out = chartFor(only, "few_shot");
    expect(out.series).toEqual([]);
    expect(out.chance).toBeNull();
  });
  it("is empty without rows", () => expect(chartFor({ suite: "s" }, "bpc")).toEqual({ series: [], chance: null, unplaced: [] }));
});

describe("bandText", () => {
  it("adds the range when there is one", () => expect(bandText("bpc", 2.2, [2.15, 2.25])).toBe("2.20 (2.15 to 2.25)"));
  it("shows a share as a percentage with its range", () => expect(bandText("few_shot", 0.35, [0.21, 0.49])).toBe("35% (21% to 49%)"));
  it("leaves the range out when it is missing or not a number, and a dash without a value", () => {
    expect(bandText("bpc", 2.2, null)).toBe("2.20");
    expect(bandText("few_shot", 0.5, [Number.NaN, Number.NaN])).toBe("50%");
    expect(bandText("bpc", null, [1, 2])).toBe("–");
  });
});

describe("gainText", () => {
  it("is a dash for the first version of a lineage", () => {
    expect(gainText("bpc", null)).toBe("–");
    expect(gainText("bpc", undefined)).toBe("–");
  });
  it("calls it better only when the paired test says so", () => {
    expect(gainText("bpc", { diff: -0.4, better: true, docs: 30 })).toBe(BENCH.gainBetter("-0.40"));
    expect(gainText("bpc", { diff: -0.01, better: false, docs: 30 })).toBe(BENCH.gainUnsure("-0.01"));
  });
  it("writes an accuracy change in points", () => expect(gainText("few_shot", { diff: 0.05, better: false, items: 40 })).toBe(BENCH.gainUnsure("+5 points")));
  it("says when there are too few items to tell", () =>
    expect(gainText("few_shot", { diff: 0.2, better: false, items: 8, reason: "few_items" })).toBe(BENCH.gainFew("+20 points", 8)));
  it("says when it could not be measured", () => expect(gainText("bpc", { diff: Number.NaN, better: false })).toBe(BENCH.gainNone));
});

describe("trendTables", () => {
  const state: BenchmarkOut = {
    suite: "s",
    rows: [row(1, { lineage: "chain-1" }), row(2, { lineage: "chain-1" })],
    lineages: [
      {
        lineage: "chain-1",
        versions: [1, 2],
        points: [
          { number: 1, tokens: 200_000, bpc: 2.6, bpc_band: [2.5, 2.7], few_shot: 0.3, few_shot_band: [0.2, 0.4], dropped: { docs: 1, choice: 2 }, gain: null },
          {
            number: 2,
            tokens: 800_000,
            bpc: 2.2,
            bpc_band: [2.1, 2.3],
            few_shot: 0.35,
            few_shot_band: [0.25, 0.45],
            dropped: {},
            gain: { bpc: { diff: -0.4, better: true, docs: 24 }, few_shot: { diff: 0.05, better: false, items: 40 } },
          },
        ],
      },
    ],
  };
  it("is a row per version, with ranges, what the tests say and what was left out", () => {
    const [table] = trendTables(state);
    expect(table.title).toBe(BENCH.design("A", [1, 2]));
    expect(table.rows).toEqual([
      ["1", "200K", "2.60 (2.50 to 2.70)", "–", "30% (20% to 40%)", "–", "3"],
      ["2", "800K", "2.20 (2.10 to 2.30)", BENCH.gainBetter("-0.40"), "35% (25% to 45%)", BENCH.gainUnsure("+5 points"), "0"],
    ]);
  });
  it("is empty before any version is checked", () => expect(trendTables({ suite: "s" })).toEqual([]));
});

describe("statusText with versions checked under an older method", () => {
  it("says they will be checked again", () => {
    const state: BenchmarkOut = { suite: "s", sizes: withSizes, rows: [row(1, { result: null, stale: true }), row(2, { result: null }), row(3)] };
    expect(statusText(state)).toBe(BENCH.waiting(2, 3) + BENCH.waitingStale(1));
  });
});

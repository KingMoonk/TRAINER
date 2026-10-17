import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/api";
import { T, fill } from "./copy";
import {
  chosenIds,
  enoughView,
  errorText,
  lrLine,
  lrText,
  modeTexts,
  perStepLine,
  queueNote,
  selectedLine,
  specLr,
  stepsLine,
  type Plan,
  type RangeResult,
  type TextSet,
} from "./logic";

const plan = (extra: Record<string, unknown> = {}) => ({ pieces_per_step: 4096, pieces: 1_000_000, passes_over_text: 2.5, ...extra }) as unknown as Plan;
const sets = [{ id: "a", on: true }, { id: "b", on: false }, { id: "c", on: true }] as unknown as TextSet[];
const range = (extra: Record<string, unknown> = {}) =>
  ({ rates: [0.0001, 0.001, 0.01, 0.1], steepest: 1, suggestion: 0.0001, diverged: false, failed_at: null, cancelled: false, ...extra }) as unknown as RangeResult;

describe("chosenIds", () => {
  it("follows the server's defaults until the person ticks a set", () => expect(chosenIds(sets, {})).toEqual(["a", "c"]));
  it("lets the person's choice override the default either way", () => expect(chosenIds(sets, { a: false, b: true })).toEqual(["b", "c"]));
  it("is empty without sets", () => {
    expect(chosenIds(undefined, { a: true })).toEqual([]);
    expect(chosenIds([], {})).toEqual([]);
  });
});

describe("selectedLine", () => {
  it("counts the sources, singular or plural", () => {
    expect(selectedLine(1)).toBe("1 source selected");
    expect(selectedLine(3)).toBe("3 sources selected");
  });
  it("says nothing is selected for zero", () => expect(selectedLine(0)).toBe(T.nothingSelected));
});

describe("enoughView", () => {
  it("says it is estimating, with the plain tip, whatever the plan says", () => {
    const enough = { verdict: "plenty", headline: "H", ratio_text: "30" };
    expect(enoughView(plan({ enough }), true)).toEqual({ headline: T.estimating, tip: T.wordPiecesPlain });
  });
  it("shows the server's headline with the rule-of-thumb tip, filling in the ratio", () => {
    expect(enoughView(plan({ enough: { verdict: "short", headline: "Too little", ratio_text: "4.2" } }), false)).toEqual({
      headline: "Too little", tip: fill(T.wordPiecesRule, { ratio: "4.2" }),
    });
  });
  it("uses the repeating-not-needed tip for plenty", () => {
    const view = enoughView(plan({ enough: { verdict: "plenty", headline: "Plenty", ratio_text: "40" } }), false);
    expect(view.tip).toBe(fill(T.wordPiecesPlenty, { ratio: "40" }));
  });
  it("asks for some text without a plan or without a verdict", () => {
    expect(enoughView(undefined, false)).toEqual({ headline: T.textUnknown, tip: T.wordPiecesPlain });
    expect(enoughView(plan(), false)).toEqual({ headline: T.textUnknown, tip: T.wordPiecesPlain });
  });
});

describe("perStepLine", () => {
  it("is empty without a plan", () => expect(perStepLine(undefined, false)).toBe(""));
  it("shows the passes over the text, 'about' when estimated", () => {
    expect(perStepLine(plan(), false)).toBe(fill(T.perStepPieces, { pieces: "4,096", passes: "2.5" }));
    expect(perStepLine(plan({ estimated: true, passes_over_text: 1234 }), false)).toBe(fill(T.perStepPieces, { pieces: "4,096", passes: "about 1,234" }));
  });
  it("shows only the pieces while estimating or without a count of pieces or passes", () => {
    const only = fill(T.perStepOnly, { pieces: "4,096" });
    expect(perStepLine(plan(), true)).toBe(only);
    expect(perStepLine(plan({ pieces: 0 }), false)).toBe(only);
    expect(perStepLine(plan({ passes_over_text: 0 }), false)).toBe(only);
    expect(perStepLine(plan({ passes_over_text: null }), false)).toBe(only);
  });
});

describe("stepsLine", () => {
  it("shows steps and time", () => expect(stepsLine({ steps: 12000, minutes: 45 } as Plan, false)).toBe(fill(T.stepsTime, { steps: "12,000", minutes: "45" })));
  it("shows the steps alone before a first run has timed them", () => {
    expect(stepsLine({ steps: 12000, minutes: 0 } as Plan, false)).toBe(fill(T.stepsOnly, { steps: "12,000" }));
    expect(stepsLine({ steps: 12000 } as Plan, true)).toBe(fill(T.stepsOnly, { steps: "12,000" }));
  });
  it("says it is estimating, or asks for text, when there are no steps", () => {
    expect(stepsLine(undefined, true)).toBe(T.stepsEstimating);
    expect(stepsLine({ steps: 0, minutes: 5 } as Plan, true)).toBe(T.stepsEstimating);
    expect(stepsLine(undefined, false)).toBe(T.stepsNone);
  });
});

describe("lrLine", () => {
  const base = `${T.learningRate} ${T.lrAuto}`;
  it("shows only 'auto' without a learning rate", () => {
    expect(lrLine(null)).toBe(base);
    expect(lrLine(undefined)).toBe(base);
    expect(lrLine(0)).toBe(base);
  });
  it("adds the rate to two significant digits", () => {
    expect(lrLine(0.00123456)).toBe(`${base} ${fill(T.lrInfo, { lr: "0.0012" })}`);
    expect(lrLine(0.0003)).toBe(`${base} ${fill(T.lrInfo, { lr: "0.0003" })}`);
    expect(lrLine(1e-7)).toBe(`${base} ${fill(T.lrInfo, { lr: "1e-7" })}`);
  });
});

describe("modeTexts", () => {
  it("recommends continuing from the parent version", () => {
    const modes = modeTexts(plan({ parent: 2, version: 3 }));
    expect(modes.catch_up).toEqual({
      title: fill(T.continueFrom, { n: 2 }),
      note: fill(T.continueRecommended, { n: 2, version: "version 3" }),
    });
    expect(modes.from_scratch).toEqual({ title: T.fromScratch, note: T.fromScratchNote });
  });
  it("has generic wording without a parent or version", () => {
    expect(modeTexts(plan()).catch_up).toEqual({ title: T.continueFromEarlier, note: T.continueNone });
  });
  it("explains why a version must start over, replacing the recommendation", () => {
    const changed = modeTexts(plan({ parent: 2, version: 3, continue_blocked: "word_list_changed" })).catch_up;
    expect(changed.note).toBe(fill(T.mustStartOver, { why: T.wordListChanged, version: "version 3" }));
    expect(changed.title).toBe(fill(T.continueFrom, { n: 2 }));
    const none = modeTexts(plan({ continue_blocked: "no_parent" })).catch_up;
    expect(none.note).toBe(fill(T.mustStartOver, { why: T.noTrainedParent, version: T.thisVersion }));
  });
});

describe("queueNote", () => {
  it("names the parent and the version", () => expect(queueNote(plan({ parent: 1, version: 2 }))).toBe(fill(T.queueNote, { parent: "version 1", version: "version 2" })));
  it("falls back to generic words", () => expect(queueNote(plan())).toBe(fill(T.queueNote, { parent: "its current version", version: T.thisVersion })));
});

describe("lrText", () => {
  it("is empty before a test has run", () => expect(lrText(null)).toBe(""));
  it("reports a failure with its message", () => expect(lrText({ error: "no design" })).toBe(fill(T.lrFailed, { error: "no design" })));
  it("reports a stopped test before looking at the numbers", () => expect(lrText(range({ cancelled: true }))).toBe(T.lrStopped));
  it("suggests a rate, naming the steepest point and the current rate", () => {
    expect(lrText(range(), 0.0003)).toBe(fill(T.lrResult, { rate: "0.0001", steepest: "0.001", current: "0.0003" }));
  });
  it("uses the first rate as the steepest when the test gives none", () => {
    expect(lrText(range({ steepest: null }))).toBe(fill(T.lrResult, { rate: "0.0001", steepest: "0.0001", current: "its own default" }));
  });
  it("adds where the error blew up", () => {
    expect(lrText(range({ diverged: true, failed_at: 0.05 }), 0.001)).toBe(
      `${fill(T.lrResult, { rate: "0.0001", steepest: "0.001", current: "0.001" })} ${fill(T.lrBlewUp, { rate: "0.05" })}`,
    );
  });
  it("takes the last rate as the blow-up point when none is named", () => {
    expect(lrText(range({ diverged: true }))).toContain(fill(T.lrBlewUp, { rate: "0.1" }));
  });
  it("has no suggestion: blew up early, or flat", () => {
    expect(lrText(range({ suggestion: null, diverged: true, failed_at: 0.02 }))).toBe(fill(T.lrBlewUpEarly, { rate: "0.02" }));
    expect(lrText(range({ suggestion: null, diverged: true, rates: [] }))).toBe(T.lrFlat);
    expect(lrText(range({ suggestion: null }))).toBe(T.lrFlat);
    expect(lrText(range({ suggestion: 0 }))).toBe(T.lrFlat);
  });
  it("adds no blow-up note when the test did not diverge, even with a failure rate", () => {
    expect(lrText(range({ diverged: false, failed_at: 0.05 }))).not.toContain("blew up");
  });
});

describe("specLr", () => {
  it("reads the optimiser's learning rate", () => expect(specLr({ train: { optimizer: { config: { lr: 0.003 } } } })).toBe(0.003));
  it("is null when the spec does not name a numeric one", () => {
    expect(specLr(null)).toBeNull();
    expect(specLr(undefined)).toBeNull();
    expect(specLr({})).toBeNull();
    expect(specLr({ train: { optimizer: {} } })).toBeNull();
    expect(specLr({ train: { optimizer: { config: { lr: "0.1" } } } })).toBeNull();
  });
});

describe("errorText", () => {
  it("prefers the server's detail sentence", () => expect(errorText(new ApiError(409, "/p", JSON.stringify({ detail: "Busy." })))).toBe("Busy."));
  it("shows a JSON body without a string detail, or a non-JSON body, as it is", () => {
    expect(errorText(new ApiError(422, "/p", '{"detail":[1]}'))).toBe('{"detail":[1]}');
    expect(errorText(new ApiError(502, "/p", "Bad gateway"))).toBe("Bad gateway");
  });
  it("falls back to the message, or the string form", () => {
    expect(errorText(new ApiError(500, "/p", ""))).toBe("500 from /p");
    expect(errorText(new Error("offline"))).toBe("offline");
    expect(errorText("boom")).toBe("boom");
    expect(errorText(null)).toBe("null");
  });
});

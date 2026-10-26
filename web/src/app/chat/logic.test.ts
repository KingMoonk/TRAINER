import { describe, expect, it } from "vitest";
import { CHAT as T } from "./copy";
import {
  bestNumber,
  canRate,
  changesCheckpoints,
  checkpointStep,
  compareStart,
  fewShotNote,
  fewShotResult,
  fewShotRows,
  liveLabel,
  liveOnly,
  liveScore,
  otherVersion,
  scoreRows,
  trainedVersions,
  unfinished,
  verdictView,
  versionName,
  type LiveCheckpoint,
  type Version,
} from "./logic";

const version = (number: number, extra: Record<string, unknown> = {}) => ({ number, status: "done", best_checkpoint: `ckpt-${number}`, ...extra }) as unknown as Version;

const table = (correct: number, heldBack = true) => ({
  items: 20, tasks: 2, held_back: heldBack, chance: 0.25,
  rows: [{ shots: 0, used: 0, correct, accuracy: correct / 20 }, { shots: 4, used: 2.5, correct, accuracy: correct / 20 }],
});

describe("live checkpoints", () => {
  const live = (extra: Record<string, unknown> = {}) => ({ version: 2, step: 12000, total: 50000, saved_at: 1000, file: "best.npz", ...extra }) as unknown as LiveCheckpoint;
  it("labels the step and the age of the save", () => {
    expect(liveLabel(live(), 1000 + 180)).toBe("step 12,000 of 50,000 · saved 3 min ago");
    expect(liveLabel(live({ total: 0 }), 1000 + 5)).toBe("step 12,000 · saved just now");
    expect(liveLabel(live({ step: null }), 1000 + 7200)).toBe("step not recorded · saved 2 h ago");
  });
  it("keeps only versions without a trained checkpoint", () => {
    expect(liveOnly([live({ version: 3 }), live({ version: 1 }), live({ version: 2 })], [version(1)]).map((l) => l.version)).toEqual([2, 3]);
  });
  it("reads a score and refuses a malformed one", () => {
    expect(liveScore({ version: 2, step: 5, bpc: 2.5, documents: 4 })).toEqual({ version: 2, step: 5, bpc: 2.5, documents: 4 });
    expect(() => liveScore({ version: 2, step: 5, bpc: "x", documents: 4 })).toThrow(T.invalidLiveScore);
  });
});

describe("versionName", () => {
  it("names a version by its number", () => expect(versionName({ number: 7 })).toBe("Version 7"));
});

describe("trainedVersions", () => {
  it("keeps versions with a checkpoint, oldest first", () => {
    const all = [version(3), version(1, { best_checkpoint: null }), version(2), version(4, { best_checkpoint: "" })];
    expect(trainedVersions(all).map((v) => v.number)).toEqual([2, 3]);
  });
  it("is empty for no versions and does not reorder the input", () => {
    expect(trainedVersions([])).toEqual([]);
    const all = [version(2), version(1)];
    trainedVersions(all);
    expect(all.map((v) => v.number)).toEqual([2, 1]);
  });
});

describe("otherVersion", () => {
  it("prefers the nearest version before", () => expect(otherVersion([1, 2, 5, 7], 5)).toBe(2));
  it("falls back to the nearest one after", () => expect(otherVersion([3, 4, 9], 3)).toBe(4));
  it("is null when the version is alone", () => {
    expect(otherVersion([4], 4)).toBeNull();
    expect(otherVersion([], 1)).toBeNull();
  });
  it("treats no pick as before everything, so it takes the first version", () => expect(otherVersion([2, 3], null)).toBe(2));
});

describe("compareStart", () => {
  it("starts from the picked version and the one before it", () => expect(compareStart([1, 2, 3], 2)).toEqual([2, 1]));
  it("starts from the newest when nothing, or something untrained, is picked", () => {
    expect(compareStart([1, 2, 3], null)).toEqual([3, 2]);
    expect(compareStart([1, 2, 3], 9)).toEqual([3, 2]);
  });
  it("uses the next version when the pick is the oldest", () => expect(compareStart([1, 2, 3], 1)).toEqual([1, 2]));
  it("is null with fewer than two versions", () => {
    expect(compareStart([5], 5)).toBeNull();
    expect(compareStart([], null)).toBeNull();
  });
});

describe("scoreRows", () => {
  it("shows placeholders for a version with no scorecard", () => {
    expect(scoreRows(version(1))).toEqual([
      [T.errorHeldBack, T.valBpc],
      [T.answersPrompts, "not yet"],
      [T.checkedProblems, T.valKofn],
      [T.judgePreferred, T.valKofn],
      [T.sizeSpeed, T.valSizeSpeed],
    ]);
  });
  it("formats a full scorecard", () => {
    const card = { bpc: 1.2345, answers: true, problems: [3, 10], judge: [7, 9], params_m: 12.5, pieces_per_s: 840 };
    expect(scoreRows(version(1, { scorecard: card }))).toEqual([
      [T.errorHeldBack, "1.23 bits/char"],
      [T.answersPrompts, "yes"],
      [T.checkedProblems, "3 of 10"],
      [T.judgePreferred, "7 of 9"],
      [T.sizeSpeed, "12.5M · 840 pieces/s"],
    ]);
  });
  it("accepts 'yes' as an answer, and any other value as not yet", () => {
    expect(scoreRows(version(1, { scorecard: { answers: "yes" } }))[1][1]).toBe("yes");
    expect(scoreRows(version(1, { scorecard: { answers: "no" } }))[1][1]).toBe("not yet");
    expect(scoreRows(version(1, { scorecard: { answers: false } }))[1][1]).toBe("not yet");
  });
  it("keeps a bpc of 0 and shows size or speed alone", () => {
    expect(scoreRows(version(1, { scorecard: { bpc: 0 } }))[0][1]).toBe("0.00 bits/char");
    expect(scoreRows(version(1, { scorecard: { params_m: 3 } }))[4][1]).toBe("3M");
    expect(scoreRows(version(1, { scorecard: { pieces_per_s: 10 } }))[4][1]).toBe("10 pieces/s");
  });
  it("shows 'not checked yet' for a pair of the wrong shape", () => {
    expect(scoreRows(version(1, { scorecard: { problems: [1], judge: "3 of 4" } })).slice(2, 4).map((r) => r[1])).toEqual([T.valKofn, T.valKofn]);
  });
  it("ignores a non-numeric bpc", () => expect(scoreRows(version(1, { scorecard: { bpc: "1.0" } }))[0][1]).toBe(T.valBpc));
});

describe("unfinished", () => {
  it.each([["queued", true], ["training", true], ["done", false], ["failed", false], [undefined, false]])("%s is %s", (status, expected) => {
    expect(unfinished(version(1, { status }))).toBe(expected);
  });
});

describe("verdictView", () => {
  it("gives no verdict while either version is mid-schedule, naming the first unfinished one", () => {
    const view = verdictView(version(1), version(2, { status: "training" }));
    expect(view).toEqual({ line: T.noVerdictYet("version 2"), sub: T.noVerdictText, ok: false });
    expect(verdictView(version(1, { status: "queued" }), version(2, { status: "training" })).line).toBe(T.noVerdictYet("version 1"));
  });
  it("ignores the server's verdict while a run is open", () => {
    expect(verdictView(version(1), version(2, { status: "queued" }), { comparable: true, percent: 50 }).ok).toBe(false);
  });
  it("turns the server's percent into the share of training needed, whichever order the versions come in", () => {
    const expected = { line: T.verdictLine("Version 3", "Version 1", 60), sub: T.verdictSub, ok: true };
    expect(verdictView(version(1), version(3), { comparable: true, percent: 40 })).toEqual(expected);
    expect(verdictView(version(3), version(1), { comparable: true, percent: 40 })).toEqual(expected);
  });
  it("accepts a percent of 0", () => expect(verdictView(version(1), version(2), { comparable: true, percent: 0 }).ok).toBe(true));
  it("gives the server's reason when the runs are not comparable", () => {
    expect(verdictView(version(1), version(2), { comparable: false, reason: "different lengths" })).toEqual({
      line: "No verdict yet: different lengths", sub: T.noVerdictText, ok: false,
    });
  });
  it("falls back to a generic reason without a result, with no reason, or with a missing percent", () => {
    const line = "No verdict yet: the runs are not comparable.";
    expect(verdictView(version(1), version(2)).line).toBe(line);
    expect(verdictView(version(1), version(2), null).line).toBe(line);
    expect(verdictView(version(1), version(2), { comparable: false, reason: "" }).line).toBe(line);
    expect(verdictView(version(1), version(2), { comparable: true, percent: null }).line).toBe(line);
  });
});

describe("canRate", () => {
  it("needs two different replies and nothing streaming", () => {
    expect(canRate(["a", "b"], false)).toBe(true);
    expect(canRate(["a", "b"], true)).toBe(false);
    expect(canRate(["a", "a"], false)).toBe(false);
    expect(canRate(["", "b"], false)).toBe(false);
    expect(canRate(["a", ""], false)).toBe(false);
  });
});

describe("bestNumber", () => {
  it("is null for no best, else the version's number", () => {
    expect(bestNumber(null)).toBeNull();
    expect(bestNumber({ number: 3, best_checkpoint: "c" })).toBe(3);
  });
  it.each([
    ["undefined", undefined],
    ["a string", "x"],
    ["an array", []],
    ["no number", { best_checkpoint: "c" }],
    ["zero", { number: 0, best_checkpoint: "c" }],
    ["a fraction", { number: 1.5, best_checkpoint: "c" }],
    ["a negative", { number: -1, best_checkpoint: "c" }],
    ["a string number", { number: "1", best_checkpoint: "c" }],
    ["no checkpoint", { number: 1 }],
    ["an empty checkpoint", { number: 1, best_checkpoint: "" }],
    ["a non-string checkpoint", { number: 1, best_checkpoint: 4 }],
  ])("rejects %s", (_, value) => expect(() => bestNumber(value)).toThrow(T.invalidBest));
});

describe("checkpointStep", () => {
  it("accepts null and whole non-negative numbers", () => {
    expect(checkpointStep(null)).toBeNull();
    expect(checkpointStep(0)).toBe(0);
    expect(checkpointStep(500)).toBe(500);
  });
  it.each([[-1], [1.5], [Number.NaN], [Infinity], ["3"], [undefined], [{}]])("rejects %s", (value) => {
    expect(() => checkpointStep(value)).toThrow(T.invalidStep);
  });
});

describe("fewShotResult", () => {
  it("normalises a valid result, defaulting tasks to 0 and held_back to true", () => {
    expect(fewShotResult({ items: 4, chance: null, rows: [{ shots: 0, used: 0, correct: 1, accuracy: null, extra: 1 }] })).toEqual({
      items: 4, tasks: 0, chance: null, held_back: true, rows: [{ shots: 0, used: 0, correct: 1, accuracy: null }],
    });
  });
  it("keeps held_back false and the tasks it was given", () => {
    const out = fewShotResult(table(5, false));
    expect(out.held_back).toBe(false);
    expect(out.tasks).toBe(2);
    expect(out.rows).toHaveLength(2);
  });
  it.each([
    ["not an object", null],
    ["an array", []],
    ["no items", { chance: 0.5, rows: [] }],
    ["fractional items", { items: 1.5, chance: 0.5, rows: [] }],
    ["chance above 1", { items: 1, chance: 1.5, rows: [] }],
    ["negative chance", { items: 1, chance: -0.1, rows: [] }],
    ["undefined chance", { items: 1, rows: [] }],
    ["rows not an array", { items: 1, chance: 0.5, rows: {} }],
    ["bad tasks", { items: 1, chance: 0.5, rows: [], tasks: -1 }],
    ["bad held_back", { items: 1, chance: 0.5, rows: [], held_back: "yes" }],
    ["a row that is not an object", { items: 1, chance: 0.5, rows: [1] }],
    ["negative shots", { items: 5, chance: 0.5, rows: [{ shots: -1, used: 0, correct: 0, accuracy: null }] }],
    ["non-finite used", { items: 5, chance: 0.5, rows: [{ shots: 0, used: Infinity, correct: 0, accuracy: null }] }],
    ["negative used", { items: 5, chance: 0.5, rows: [{ shots: 0, used: -1, correct: 0, accuracy: null }] }],
    ["more correct than items", { items: 5, chance: 0.5, rows: [{ shots: 0, used: 0, correct: 6, accuracy: null }] }],
    ["accuracy above 1", { items: 5, chance: 0.5, rows: [{ shots: 0, used: 0, correct: 1, accuracy: 4 }] }],
    ["undefined accuracy", { items: 5, chance: 0.5, rows: [{ shots: 0, used: 0, correct: 1 }] }],
    ["repeated shot counts", { items: 5, chance: 0.5, rows: [{ shots: 1, used: 1, correct: 1, accuracy: 0.2 }, { shots: 1, used: 1, correct: 2, accuracy: 0.4 }] }],
  ])("rejects %s", (_, value) => expect(() => fewShotResult(value)).toThrow(T.invalidFewShot));
});

describe("fewShotRows", () => {
  it("builds the five cells per row, with the examples that fit", () => {
    expect(fewShotRows(fewShotResult(table(5)))).toEqual([
      { key: 0, cells: [T.fewShotZero, "5 of 20", "25%", "25%", T.fewShotNear] },
      { key: 4, cells: [`4${T.fewShotFitted("2.5")}`, "5 of 20", "25%", "25%", T.fewShotNear] },
    ]);
  });
  it("labels one shot and compares against guessing outside two standard deviations", () => {
    const rows = fewShotRows(fewShotResult({ items: 100, chance: 0.25, rows: [
      { shots: 1, used: 1, correct: 34, accuracy: 0.34 },
      { shots: 2, used: 2, correct: 16, accuracy: 0.16 },
      { shots: 3, used: 3, correct: 33, accuracy: 0.33 },
    ] }));
    expect(rows.map((r) => [r.cells[0], r.cells[4]])).toEqual([[T.fewShotOne, T.fewShotAbove], ["2", T.fewShotBelow], ["3", T.fewShotNear]]);
  });
  it("is empty when nothing was asked, and says 'not measured' for unknown values (empty-state spec)", () => {
    expect(fewShotRows(fewShotResult({ items: 0, tasks: 0, chance: null, held_back: true, rows: [] }))).toEqual([]);
    const rows = fewShotRows(fewShotResult({ ...table(0), chance: null, rows: [{ shots: 0, used: 0, correct: 0, accuracy: null }] }));
    expect(rows[0].cells).toEqual(["none", "0 of 20", "not measured", "not measured", "not measured"]);
    expect(() => fewShotResult({ ...table(0), rows: [{ shots: 0, used: 0, correct: 99, accuracy: 4 }] })).toThrow("invalid");
  });
  it("shows a measured chance of 0 as 0%", () => {
    expect(fewShotRows(fewShotResult({ items: 4, chance: 0, rows: [{ shots: 2, used: 2, correct: 1, accuracy: 0.25 }] }))[0].cells.slice(3)).toEqual(["0%", T.fewShotAbove]);
  });
});

describe("fewShotNote", () => {
  it("states the question count and the guessing score", () => {
    expect(fewShotNote(fewShotResult(table(5)))).toBe(`${T.fewShotAsked(20, 2)} ${T.fewShotChance("25")}`);
  });
  it("adds the training-set line when the problems were not held back", () => {
    expect(fewShotNote(fewShotResult(table(5, false)))).toBe([T.fewShotAsked(20, 2), T.fewShotTraining, T.fewShotChance("25")].join(" "));
  });
  it("says the guessing score was not measured when it is null", () => {
    expect(fewShotNote(fewShotResult({ ...table(5), chance: null }))).toBe(`${T.fewShotAsked(20, 2)} ${T.fewShotChanceUnknown}`);
  });
});

describe("changesCheckpoints", () => {
  it("only checkpoint changes and terminal training jobs invalidate metadata", () => {
    expect(changesCheckpoints({ kind: "versions", payload: null })).toBe(true);
    expect(changesCheckpoints({ kind: "checkpoint", payload: { phase: "done" } })).toBe(true);
    expect(changesCheckpoints({ kind: "job", payload: { job: "train", status: "done" } })).toBe(true);
    for (const status of ["done", "failed", "cancelled"]) expect(changesCheckpoints({ kind: "job", payload: { id: "read", kind: "versions.best", status } })).toBe(false);
    expect(changesCheckpoints({ kind: "job", payload: { job: "train", status: "running" } })).toBe(false);
  });
  it("covers every terminal status of a training job", () => {
    for (const status of ["done", "failed", "stopped", "cancelled"]) expect(changesCheckpoints({ kind: "job", payload: { job: "train", status } })).toBe(true);
    for (const status of ["queued", "paused", undefined]) expect(changesCheckpoints({ kind: "job", payload: { job: "train", status } })).toBe(false);
  });
  it("ignores a checkpoint that is not done", () => {
    expect(changesCheckpoints({ kind: "checkpoint", payload: { phase: "saving" } })).toBe(false);
  });
  it("ignores other kinds, non-object payloads and a job without a name", () => {
    expect(changesCheckpoints({ kind: "queue", payload: { job: "train", status: "done" } })).toBe(false);
    expect(changesCheckpoints({ kind: "job", payload: null })).toBe(false);
    expect(changesCheckpoints({ kind: "job", payload: "done" })).toBe(false);
    expect(changesCheckpoints({ kind: "job", payload: ["done"] })).toBe(false);
    expect(changesCheckpoints({ kind: "job", payload: { status: "done" } })).toBe(false);
    expect(changesCheckpoints({ kind: "job", payload: { job: 3, status: "done" } })).toBe(false);
    expect(changesCheckpoints({ kind: "checkpoint", payload: null })).toBe(false);
  });
});

import { describe, expect, test } from "vitest";
import { baseWarning, isKept } from "./kept-logic";

describe("isKept", () => {
  test("only a failed round with a checkpoint that was not adopted", () => {
    expect(isKept({ ok: false, checkpoint: "r1.npz" })).toBe(true);
    expect(isKept({ ok: true, checkpoint: "r1.npz" })).toBe(false);
    expect(isKept({ ok: false, checkpoint: "r1.npz", adopted: true })).toBe(false);
    expect(isKept({ ok: false, checkpoint: null })).toBe(false);
    expect(isKept(undefined)).toBe(false);
  });
});

describe("baseWarning", () => {
  test("names a Round 1 that failed, and one that has not run", () => {
    expect(baseWarning(2, { "1": { ok: false, checkpoint: "r1.npz" } }, "done", 3)).toMatch(/Round 1 failed its checks on version 3/);
    expect(baseWarning(4, {}, "idle", 3)).toMatch(/Round 1 has not run on version 3/);
  });
  test("stays quiet when Round 1 is live, adopted, on its way, or is the selected round", () => {
    expect(baseWarning(2, { "1": { ok: true } }, "done", 1)).toBeNull();
    expect(baseWarning(2, { "1": { ok: false, adopted: true } }, "done", 1)).toBeNull();
    expect(baseWarning(2, {}, "running", 1)).toBeNull();
    expect(baseWarning(2, {}, "queued", 1)).toBeNull();
    expect(baseWarning(1, {}, "idle", 1)).toBeNull();
    expect(baseWarning(2, {}, "idle", undefined)).toBeNull();
  });
});

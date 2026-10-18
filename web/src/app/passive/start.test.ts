import { describe, expect, test } from "vitest";
import { INITIAL_FORM, startBody, toggled } from "./start";

describe("startBody", () => {
  test("the defaults are the plan's defaults", () => {
    expect(startBody(INITIAL_FORM)).toEqual({
      body: {
        version: null,
        sources: null,
        duration: 8,
        unit: "hours",
        patience_rounds: 3,
        target_error: null,
        teacher: true,
        memory_fraction: 0.5,
        keep_awake: true,
        pause_on_battery: true,
      },
    });
  });

  test("typed values go through as numbers, ticked sets only when not all", () => {
    const sent = startBody({ ...INITIAL_FORM, version: "2", all: false, sources: ["a", "b"], duration: "1.5", unit: "days", patience: "0", target: "0.8", memory: "25", teacher: false });
    expect(sent).toEqual({
      body: expect.objectContaining({ version: 2, sources: ["a", "b"], duration: 1.5, unit: "days", patience_rounds: 0, target_error: 0.8, memory_fraction: 0.25, teacher: false }),
    });
  });

  test.each([
    [{ duration: "0" }, "time limit"],
    [{ duration: "soon" }, "time limit"],
    [{ patience: "1.5" }, "whole number"],
    [{ patience: "-1" }, "whole number"],
    [{ target: "low" }, "target test error"],
    [{ memory: "0" }, "memory share"],
    [{ memory: "150" }, "memory share"],
    [{ all: false, sources: [] }, "at least one set"],
  ])("%j is refused with a sentence", (change, words) => {
    const sent = startBody({ ...INITIAL_FORM, ...change });
    expect("problem" in sent && sent.problem).toContain(words);
  });

  test("toggled adds once and removes", () => {
    expect(toggled(["a"], "b", true)).toEqual(["a", "b"]);
    expect(toggled(["a"], "a", true)).toEqual(["a"]);
    expect(toggled(["a", "b"], "a", false)).toEqual(["b"]);
  });
});

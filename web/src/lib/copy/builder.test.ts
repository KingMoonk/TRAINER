import { describe, expect, test } from "vitest";
import { BUILDER, fill, presetAbout } from "./builder";

describe("fill", () => {
  test("fills each named hole, numbers included", () => {
    expect(fill("Save as {version}", { version: "version 3" })).toBe("Save as version 3");
    expect(fill("Saved versions {first}–{last}", { first: 2, last: 5 })).toBe("Saved versions 2–5");
  });

  test("fills a hole that repeats", () => {
    expect(fill("{n} and {n}", { n: 1 })).toBe("1 and 1");
  });

  test("leaves a hole it has no value for, and ignores values it does not use", () => {
    expect(fill("Saved {saved}; queued {queued}", { saved: "a" })).toBe("Saved a; queued {queued}");
    expect(fill("no holes", { extra: 1 })).toBe("no holes");
  });

  test("a zero or empty value is still a value", () => {
    expect(fill("{count} left{reason}", { count: 0, reason: "" })).toBe("0 left");
  });

  test("does not read substituted text as another hole", () => {
    expect(fill("{a}", { a: "{b}", b: "x" })).toBe("{b}");
  });

  test("works on the real templates", () => {
    expect(fill(BUILDER.about_multiplier, { mult: "2.3" })).toBe("about 2.3×");
    expect(fill(BUILDER.no_changes_text, { version: "version 4" })).toContain("Version version 4");
    expect(fill(BUILDER.split_partial, { saved: "3", queued: 2, total: 3, reason: "r" })).toBe("Saved 3; queued 2 of 3 jobs. r");
  });
});

describe("presetAbout", () => {
  test("gives the blurb of each preset the server lists", () => {
    expect(presetAbout("starter")).toBe(BUILDER.preset_starter);
    expect(presetAbout("gpt2_small_style")).toBe(BUILDER.preset_gpt2_small_style);
    expect(presetAbout("tiny_llama_style")).toBe(BUILDER.preset_tiny_llama_style);
    expect(presetAbout("gpt3_small_style")).toBe(BUILDER.preset_gpt3_small_style);
    expect(presetAbout("gpt3_medium_style")).toBe(BUILDER.preset_gpt3_medium_style);
  });

  test("a preset with no blurb gets an empty string, not undefined", () => {
    expect(presetAbout("not_a_preset")).toBe("");
  });
});

describe("BUILDER strings", () => {
  test("every string is non-empty and every {hole} is a plain word", () => {
    for (const [key, text] of Object.entries(BUILDER)) {
      expect(text.trim(), key).not.toBe("");
      for (const hole of text.match(/\{[^}]*\}/g) ?? []) expect(hole, key).toMatch(/^\{\w+\}$/);
    }
  });
});

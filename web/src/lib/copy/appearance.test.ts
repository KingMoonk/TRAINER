import { describe, expect, test } from "vitest";
import { ACCENTS, MODES } from "@/lib/theme";
import { APPEARANCE } from "./appearance";

// The labels are keyed by the ids in theme.ts; the Settings page renders one control per id and reads its label here.
describe("APPEARANCE", () => {
  test("has a label for every mode and no extra", () => {
    expect(Object.keys(APPEARANCE.modes).sort()).toEqual([...MODES].sort());
  });

  test("has a label for every accent and no extra", () => {
    expect(Object.keys(APPEARANCE.accents).sort()).toEqual([...ACCENTS].sort());
  });

  test("every label is a non-empty string", () => {
    for (const label of [...Object.values(APPEARANCE.modes), ...Object.values(APPEARANCE.accents)]) {
      expect(label.trim()).not.toBe("");
    }
  });
});

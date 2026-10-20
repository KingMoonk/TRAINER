import { describe, expect, test } from "vitest";
import { busyItems, fraction, queryLabel, toastOffset } from "./busy";

describe("busy items", () => {
  test("reads the event's list and drops what is not an item", () => {
    expect(busyItems({ items: [{ id: "a", label: "Loading", done: 1, total: 2 }, { id: 3 }, null, { id: "b", label: "Reading", done: 0, total: null }] })).toEqual([
      { id: "a", label: "Loading", done: 1, total: 2 },
      { id: "b", label: "Reading", done: 0, total: null },
    ]);
    expect(busyItems(null)).toEqual([]);
    expect(busyItems({ items: "x" })).toEqual([]);
  });

  test("a bar is filled to its real progress, or has none when its steps are unknown", () => {
    expect(fraction({ id: "a", label: "", done: 1, total: 4 })).toBe(0.25);
    expect(fraction({ id: "a", label: "", done: 9, total: 4 })).toBe(1);
    expect(fraction({ id: "a", label: "", done: 0, total: null })).toBeNull();
    expect(fraction({ id: "a", label: "", done: 0, total: 0 })).toBeNull();
  });
});

describe("slow read labels", () => {
  test("a known key has its own words and any other is spelled out from its key", () => {
    expect(queryLabel(["distributed", "runs"])).toBe("Loading runs");
    expect(queryLabel(["post-training", "import-models"])).toBe("Loading post training import models");
    expect(queryLabel(["design", "tower", 3])).toBe("Loading design tower");
    expect(queryLabel([])).toBe("Loading data");
  });
});

describe("toast offset", () => {
  test("toasts clear the card by more for every row", () => {
    expect(toastOffset(0)).toBe("");
    expect(parseFloat(toastOffset(2))).toBeGreaterThan(parseFloat(toastOffset(1)));
  });
});

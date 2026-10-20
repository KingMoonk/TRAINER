import { describe, expect, test } from "vitest";
import { busyItems, busyTitle, fraction, overall, queryLabel, toastOffset } from "./busy";

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

describe("the bar", () => {
  const item = (label: string, done: number, total: number | null) => ({ id: label, label, done, total });

  test("is filled to the mean progress, or indeterminate when anything has no steps", () => {
    expect(overall([item("a", 1, 4), item("b", 3, 4)])).toBe(0.5);
    expect(overall([item("a", 1, 4), item("b", 0, null)])).toBeNull();
    expect(overall([])).toBeNull();
  });

  test("its title names the first item, its steps, and how many more are busy", () => {
    expect(busyTitle([item("Reading", 1, 4)])).toBe("Reading (1 of 4)");
    expect(busyTitle([item("Loading runs", 0, null), item("b", 0, 1), item("c", 0, null)])).toBe("Loading runs +2 more");
    expect(busyTitle([])).toBe("");
  });

  test("toasts clear it while it shows", () => {
    expect(toastOffset(false)).toBe("");
    expect(parseFloat(toastOffset(true))).toBeGreaterThan(2.25);
  });
});

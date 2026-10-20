import { describe, expect, test } from "vitest";
import { historySeries, mfuText, peakNote, relativeText, secondsText, statusText, timeText, versionChanges } from "./mlperf-logic";

describe("training speed text", () => {
  test("times show the spread only when there were several attempts", () => {
    expect(timeText({ median: 42, min: 40.1, max: 44.9, n: 3 })).toBe("42.0 s (40.1 to 44.9 s, 3 attempts)");
    expect(timeText({ median: 42, min: 42, max: 42, n: 1 })).toBe("42.0 s");
    expect(timeText(null)).toBe("-");
    expect(secondsText(250.4)).toBe("250 s");
  });
  test("MFU is a percentage and says what it is a share of", () => {
    expect(mfuText(0.0009)).toBe("0.09%");
    expect(mfuText(0.317)).toBe("31.7%");
    expect(mfuText(null)).toBe("-");
    expect(peakNote("datasheet")).toContain("datasheet");
    expect(peakNote("measured_gemm")).toContain("matrix multiply");
  });
  test("statuses and relative times read plainly", () => {
    expect(statusText("did_not_converge")).toBe("Did not converge");
    expect(relativeText(1)).toBe("fastest");
    expect(relativeText(2)).toBe("2.00x slower");
  });
});

describe("history", () => {
  const points = [
    { id: "a", created_at: 1, app_version: "v1", status: "converged", median_s: 40 },
    { id: "b", created_at: 2, app_version: "v1", status: "did_not_converge", median_s: null },
    { id: "c", created_at: 3, app_version: "v2", status: "converged", median_s: 30 },
  ];
  test("the chart follows converged results in order", () => {
    expect(historySeries(points)[0].points).toEqual([
      { x: 1, y: 40 },
      { x: 2, y: 30 },
    ]);
    expect(historySeries([])).toEqual([]);
  });
  test("an app version is named where it first appears", () => {
    expect(versionChanges(points)).toEqual(["Result 1: v1", "Result 2: v2"]);
  });
});

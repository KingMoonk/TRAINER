import { describe, expect, it } from "vitest";
import { continueBody } from "./continue";

const base = { kind: "same" as const, steps: "5000", ids: [] as string[], width: "128", layers: "4" };

describe("continueBody", () => {
  it("asks only for steps to continue on the same data", () => expect(continueBody(base)).toEqual({ kind: "same", steps: 5000, dataset_ids: [], rows: 16, hold: 0.05 }));
  it("needs a whole number of steps", () => {
    for (const steps of ["", "0", "-5", "1.5", "many"]) expect(continueBody({ ...base, steps })).toBeNull();
  });
  it("needs text for other data", () => {
    expect(continueBody({ ...base, kind: "new_data" })).toBeNull();
    expect(continueBody({ ...base, kind: "new_data", ids: ["poems"] })).toEqual({ kind: "new_data", steps: 5000, dataset_ids: ["poems"], rows: 16, hold: 0.05 });
  });
  it("needs a size for a bigger design, and the text is the run's own unless some is ticked", () => {
    expect(continueBody({ ...base, kind: "grow" })).toEqual({ kind: "grow", steps: 5000, dataset_ids: [], rows: 16, hold: 0.05, width: 128, layers: 4 });
    expect(continueBody({ ...base, kind: "grow", width: "" })).toBeNull();
    expect(continueBody({ ...base, kind: "grow", layers: "x" })).toBeNull();
  });
});

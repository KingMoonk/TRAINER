import { describe, expect, test } from "vitest";
import { firstFailure, headline, ranAt, visible, type SelfTestRun } from "./logic";

const base: SelfTestRun = { id: "s", machines: ["box", "pc"], started_at: 1, finished_at: 100, state: "failed", steps: [] };
const step = (id: string, status: SelfTestRun["steps"][number]["status"]) => ({ id, title: id, machine: "", status, detail: "", fix: "", seconds: null });

describe("self-test logic", () => {
  test("the headline follows the state", () => {
    expect(headline(undefined)).toMatch(/Not run yet/);
    expect(headline({ ...base, state: "never" })).toMatch(/Not run yet/);
    expect(headline({ ...base, state: "running" })).toMatch(/Running now/);
    expect(headline({ ...base, state: "passed" })).toMatch(/^Passed/);
    expect(headline(base)).toMatch(/^Failed/);
  });

  test("when it last ran names the machines, and is empty while it runs or never ran", () => {
    expect(ranAt(base, (s) => `at ${s}`)).toBe("Last run at 100, with box, pc.");
    expect(ranAt({ ...base, state: "running", finished_at: null }, String)).toBe("");
    expect(ranAt(undefined, String)).toBe("");
  });

  test("the first failure is the first failed step; steps show once a test has run", () => {
    const run = { ...base, steps: [step("a", "pass"), step("b", "fail"), step("c", "fail")] };
    expect(firstFailure(run)?.id).toBe("b");
    expect(visible(run)).toHaveLength(3);
    expect(visible({ ...run, state: "never" })).toEqual([]);
  });
});

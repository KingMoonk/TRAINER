import { describe, expect, test } from "vitest";
import { config } from "@/test/collect";
import { approving, blocking, dropRows, isDirty, parseList, percent, statusLine, toDraft, toUpdate, type State } from "./logic";

const state = (patch: Partial<State> = {}): State =>
  ({ enabled: true, paused: false, running: true, status: "running", reason: "", stats: {}, ...patch }) as State;

describe("the form draft", () => {
  test("round-trips the settings and is not dirty until something changes", () => {
    const draft = toDraft(config);
    expect(isDirty(draft, config)).toBe(false);
    expect(isDirty({ ...draft, topics: ["science"] }, config)).toBe(true);
    const built = toUpdate(draft);
    expect("update" in built && built.update.limits?.disk_mb).toBe(4096);
    expect("update" in built && built.update.custom_topics).toEqual(["astronomy"]);
    expect("update" in built && built.update.web?.sites[0].host).toBe("docs.example.org");
  });

  test("text lists are split and trimmed", () => {
    expect(parseList(" a , b,, c ")).toEqual(["a", "b", "c"]);
    expect(parseList("")).toEqual([]);
  });

  test("a number that is not one is a sentence, not a request", () => {
    const draft = toDraft(config);
    const bad = toUpdate({ ...draft, limits: { ...draft.limits, disk_mb: "lots" } });
    expect(bad).toEqual({ error: "Disk (MB) must be a number." });
    expect(toUpdate({ ...draft, limits: { ...draft.limits, disk_mb: "" } })).toHaveProperty("error");
  });

  test("a blank website row is left out of the update", () => {
    const draft = toDraft(config);
    const built = toUpdate({ ...draft, web: [...draft.web, { host: " ", seeds: [], licence: "", pages: 200 }] });
    expect("update" in built && built.update.web?.sites).toHaveLength(1);
  });
});

describe("statusLine", () => {
  test("off, paused, starting, collecting, waiting and failed", () => {
    expect(statusLine(state({ enabled: false }))).toMatch(/^Off/);
    expect(statusLine(state({ paused: true }))).toMatch(/^Paused/);
    expect(statusLine(state({ running: false }))).toMatch(/Starting/);
    expect(statusLine(state())).toBe("Collecting.");
    expect(statusLine(state({ status: "waiting", reason: "Waiting for mains power." }))).toBe("Waiting for mains power.");
    expect(statusLine(state({ status: "error", reason: "boom" }))).toBe("Stopped by a problem: boom");
  });
});

describe("small helpers", () => {
  test("dropRows names every reason, in order", () => {
    const rows = dropRows({ licence: 1, quality: 2, pii: 3, duplicate: 4, near_duplicate: 5, decontam: 6, invalid: 7 });
    expect(rows.map((r) => r.n)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(rows[5].label).toBe("Overlaps held-out test text");
  });

  test("percent is capped and safe with no cap", () => {
    expect(percent(50, 200)).toBe(25);
    expect(percent(500, 200)).toBe(100);
    expect(percent(5, 0)).toBe(0);
  });

  test("approving and blocking change one id and keep the rest", () => {
    expect(approving({ ...config, approved: ["a"] }, "b")).toEqual(["a", "b"]);
    expect(approving({ ...config, approved: ["a"] }, "a")).toEqual(["a"]);
    expect(blocking({ ...config, blocked: ["a"] }, "b", true)).toEqual(["a", "b"]);
    expect(blocking({ ...config, blocked: ["a", "b"] }, "a", false)).toEqual(["b"]);
  });
});

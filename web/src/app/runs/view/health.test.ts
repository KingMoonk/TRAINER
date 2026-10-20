import { describe, expect, test } from "vitest";
import { bytesSeries, compressionText, lossSeries, percentText, rateText, speedupText, splitRows, throughputSeries, verdictTone, type RoundHealth, type RunHealth } from "./health";

const member = (over: Partial<RoundHealth["members"][string]> = {}) => ({ loss: 3, steps: 100, tokens: 1000, compute_s: 6, tokens_per_s: 200, wait_s: 0, idle_s: 0, sync_s: 2, ...over });
const round = (n: number, over: Partial<RoundHealth> = {}): RoundHealth => ({
  round: n, seconds: 10, tokens: 3000, slowest: "mac", val_loss: 4 - n * 0.1, loss: 3.9, lr: 0.7, update_norm: 0.05, bytes_up: 2 * 2 ** 20, bytes_down: 4 * 2 ** 20, dtype: "float16", compression: 2, sync_s: 4,
  members: { mac: member({ loss: 3.5 - n * 0.1 }), box: member({ compute_s: 4, tokens_per_s: 500, wait_s: 2, sync_s: 4, loss: 3.2 - n * 0.1 }) }, ...over,
});
const health = (over: Partial<RunHealth> = {}): RunHealth => ({
  verdict: { state: "working", text: "Working: 1.6× faster than box alone, loss falling." }, basis: "Tokens per second.", rounds: [round(0), round(1), round(2)], window: 3,
  aggregate_tokens_per_s: 800, baseline_node: "box", baseline_tokens_per_s: 500, speedup: 1.6, comm_share: 0.4, wait_node: "box", wait_share: 0.2, straggler: "mac", idle: {}, idle_node: "", idle_share: null, ...over,
});

describe("run health", () => {
  test("the loss chart has the outer loss and a line per member, counted from round 1", () => {
    const series = lossSeries(health());
    expect(series.map((s) => s.id)).toEqual(["outer", "mac", "box"]);
    expect(series[0].points[0]).toEqual({ x: 1, y: 4 });
    expect(series[1].points.map((p) => p.x)).toEqual([1, 2, 3]);
  });

  test("a member with no speed in a round has no point there", () => {
    const rounds = [round(0), round(1, { members: { mac: member({ tokens_per_s: null }), box: member() } })];
    const series = throughputSeries(health({ rounds }));
    expect(series.find((s) => s.id === "mac")?.points).toHaveLength(1);
    expect(series.find((s) => s.id === "box")?.points).toHaveLength(2);
  });

  test("bytes are in MB, up and down", () => {
    expect(bytesSeries(health()).map((s) => [s.id, s.points[0].y])).toEqual([["up", 2], ["down", 4]]);
  });

  test("the time split averages the members and adds up to the round", () => {
    const [row] = splitRows(health());
    expect(row).toEqual({ round: 1, compute: 5, wait: 1, sync: 3 });
    expect(row.compute + row.wait + row.sync).toBe(9);
  });

  test("the split shows only the latest rounds", () => {
    const rounds = Array.from({ length: 20 }, (_, n) => round(n));
    const rows = splitRows(health({ rounds }));
    expect(rows).toHaveLength(12);
    expect(rows[0].round).toBe(9);
  });

  test("readings say nothing when there is no measurement", () => {
    expect(speedupText(health())).toBe("1.60×");
    expect(speedupText(health({ speedup: null }))).toBe("");
    expect(speedupText(null)).toBe("");
    expect(percentText(0.456)).toBe("46%");
    expect(percentText(null)).toBe("");
    expect(rateText(5120)).toBe("5.1k tok/s");
    expect(rateText(640.2)).toBe("640 tok/s");
    expect(rateText(null)).toBe("");
    expect(compressionText(health())).toBe("2.0×");
    expect(compressionText(health({ rounds: [round(0, { compression: null })] }))).toBe("");
  });

  test("working is green, not enough data is quiet, anything else is red", () => {
    expect(verdictTone("working")).toContain("success");
    expect(verdictTone("not_enough_data")).toContain("muted");
    expect(verdictTone("not_helping")).toContain("destructive");
    expect(verdictTone("diverging")).toContain("destructive");
  });
});

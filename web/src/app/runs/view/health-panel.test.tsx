import { readFileSync } from "node:fs";
import { join } from "node:path";
import { screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { run } from "@/app/cluster/training/test-data";
import type { Run } from "@/app/cluster/training/logic";
import { renderPage } from "@/test/render";
import { HealthPanel } from "./health-panel";
import type { RunHealth } from "./health";

// The health panel against a run recorded over FakeServices (src/contract/fixtures) and against a measured two-member run.

const recorded = (JSON.parse(readFileSync(join(__dirname, "../../../contract/fixtures/distributed-run-demo-second-round.json"), "utf8")) as { body: Run }).body;

const member = (loss: number, rate: number, compute: number, wait: number) => ({ loss, steps: 100, tokens: rate * compute, compute_s: compute, tokens_per_s: rate, wait_s: wait, idle_s: wait, sync_s: 10 - compute - wait });
const round = (n: number) => ({
  round: n, seconds: 10, tokens: 6400, slowest: "mac-mini", val_loss: 4 - n * 0.2, loss: 4, lr: 0.7, update_norm: 0.05, bytes_up: 2 ** 21, bytes_down: 2 ** 22, dtype: "float16", sparsity_up: 0.05, sparsity_down: 0, compression: 2, sync_s: 2,
  members: { "mac-mini": member(4 - n * 0.2, 400, 8, 0), "gpu-box": member(3.8 - n * 0.2, 400, 8, 1) },
});
const measured: RunHealth = {
  verdict: { state: "working", text: "Working: 1.6× faster than gpu-box alone, loss falling." }, basis: "Tokens trained per second against the fastest member alone.",
  rounds: [round(0), round(1), round(2)], window: 3, aggregate_tokens_per_s: 640, baseline_node: "gpu-box", baseline_tokens_per_s: 400, speedup: 1.6, comm_share: 0.2, wait_node: "gpu-box", wait_share: 0.1, straggler: "mac-mini", idle: { "gpu-box": 1 }, idle_node: "gpu-box", idle_share: 0.1,
};

describe("the health panel", () => {
  test("a recorded run with too few rounds says so instead of guessing a speed-up", () => {
    renderPage(<HealthPanel run={recorded} />);
    expect(screen.getByTestId("verdict")).toHaveAttribute("data-state", "not_enough_data");
    expect(screen.getByTestId("verdict")).toHaveTextContent(/^Not enough data/);
    expect(screen.queryByTestId("health-speedup")).toBeNull();
    expect(screen.getByTestId("health-split")).toBeInTheDocument();
  });

  test("the link chart appears only when a transfer speed was measured", () => {
    const { unmount } = renderPage(<HealthPanel run={run({ health: measured })} />);
    expect(screen.queryByTestId("health-link")).toBeNull();
    unmount();
    const rounds = measured.rounds.map((r) => ({ ...r, members: { "mac-mini": { ...r.members["mac-mini"], down_mb_s: 2, up_mb_s: 1 }, "gpu-box": r.members["gpu-box"] } }));
    renderPage(<HealthPanel run={run({ health: { ...measured, rounds } })} />);
    expect(screen.getByTestId("health-link")).toHaveTextContent("Transfer speed of each member (MB/s)");
  });

  test("a measured run shows its verdict, speed-up, shares and every chart", () => {
    renderPage(<HealthPanel run={run({ health: measured })} />);
    expect(screen.getByTestId("verdict")).toHaveTextContent("Working: 1.6× faster than gpu-box alone, loss falling.");
    expect(screen.getByTestId("health-speedup")).toHaveTextContent("1.60×");
    expect(screen.getByTestId("health-baseline")).toHaveTextContent("gpu-box alone 400 tok/s");
    expect(screen.getByTestId("health-comm")).toHaveTextContent("20%");
    expect(screen.getByTestId("health-wait")).toHaveTextContent("gpu-box waits 10%");
    for (const id of ["health-loss", "health-throughput", "health-split", "health-bytes"]) expect(screen.getByTestId(id)).toBeInTheDocument();
    expect(within(screen.getByTestId("health-split")).getAllByRole("listitem").length).toBeGreaterThanOrEqual(3);
    expect(screen.getByTestId("health-bytes")).toHaveTextContent("2.0× smaller");
    expect(screen.getByTestId("health-sparsity")).toHaveTextContent("Sent up: the largest 5% of each tensor's entries. Down: all entries.");
  });

  test("a diverging verdict is marked", () => {
    const health = { ...measured, verdict: { state: "diverging" as const, text: "Diverging: the loss rose 3 rounds running." } };
    renderPage(<HealthPanel run={run({ health })} />);
    expect(screen.getByTestId("verdict")).toHaveAttribute("data-state", "diverging");
  });

  test("a run from an older computer has no health and shows nothing", () => {
    renderPage(<HealthPanel run={run({ health: null })} />);
    expect(screen.queryByTestId("health-panel")).toBeNull();
  });
});

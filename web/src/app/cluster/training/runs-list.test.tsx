import { screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { renderPage } from "@/test/render";
import { RunsList } from "./runs-list";
import { run } from "./test-data";

const health = (state: "working" | "not_helping" | "not_enough_data", speedup: number | null, text: string) => ({
  verdict: { state, text }, basis: "", rounds: [], window: 0, aggregate_tokens_per_s: null, baseline_node: "", baseline_tokens_per_s: null, speedup, comm_share: null, wait_node: "", wait_share: null, straggler: "", idle: {}, idle_node: "", idle_share: null,
});

describe("the other runs", () => {
  test("each run shows its verdict and speed-up, and none while there is not enough data", () => {
    const runs = [
      run({ id: "live" }),
      run({ id: "a", label: "A", health: health("working", 1.8, "Working: 1.8× faster than box alone, loss falling.") }),
      run({ id: "b", label: "B", health: health("not_helping", 0.6, "Not helping: box waits 70% of each round for mac.") }),
      run({ id: "c", label: "C", health: health("not_enough_data", null, "Not enough data") }),
      run({ id: "d", label: "D", health: null }),
    ];
    renderPage(<RunsList runs={runs} live={runs[0]} onSelect={() => {}} />);
    expect(screen.getByTestId("other-verdict-a")).toHaveTextContent("1.80× · Working: 1.8× faster than box alone, loss falling.");
    expect(screen.getByTestId("other-verdict-b")).toHaveTextContent("0.60× · Not helping: box waits 70% of each round for mac.");
    expect(screen.queryByTestId("other-verdict-c")).toBeNull();
    expect(screen.queryByTestId("other-verdict-d")).toBeNull();
  });
});

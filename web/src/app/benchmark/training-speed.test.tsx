import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { TrainingSpeed } from "./training-speed";

// jsdom has no layout, so give Recharts' container a fixed size.
vi.mock("recharts", async (original) => {
  const actual = await original<typeof import("recharts")>();
  const { cloneElement } = await import("react");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement<{ width?: number; height?: number }> }) => (
      <div style={{ width: 800, height: 400 }}>{cloneElement(children, { width: 800, height: 400 })}</div>
    ),
  };
});

const result = (id: string, label: string, median: number | null, extra = {}) => ({
  id,
  created_at: 1,
  version: 1,
  division: "single",
  status: median == null ? "did_not_converge" : "converged",
  label,
  app_version: "v1",
  time_to_target: median == null ? null : { median, min: median - 1, max: median + 1, n: 3 },
  tokens_per_s: 17139,
  mfu: 0.0009,
  peak_kind: "datasheet",
  peak_memory_bytes: 1.5 * 1024 ** 3,
  ...extra,
});

const board = {
  tasks: [{ id: "small", version: 1, title: "Small language model", target_bpc: 2, checksum: "abcdef0123456789", attempts: 3, step_cap: 400 }],
  board: {
    task: "small",
    version: 1,
    title: "Small language model",
    target_bpc: 2,
    divisions: { single: [result("a", "box · GPU", 30), result("b", "box · CPU", 300), result("c", "old · CPU", null)], cluster: [] },
    other_versions: [{ version: 0, results: 2 }],
  },
  history: {
    points: [
      { id: "b", created_at: 1, app_version: "v1", status: "converged", median_s: 300 },
      { id: "a", created_at: 2, app_version: "v2", status: "converged", median_s: 30 },
    ],
  },
  devices: ["cpu", "cuda"],
};

describe("Training speed section", () => {
  test("shows the leaderboard, the older versions held apart, and the history", async () => {
    mockApi({ "GET /mlperf/board": { body: board } });
    renderPage(<TrainingSpeed />);
    const table = await screen.findByRole("table");
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(4); // head + three results
    expect(rows[1]).toHaveTextContent("box · GPU");
    expect(rows[1]).toHaveTextContent("30.0 s (29.0 to 31.0 s, 3 attempts)");
    expect(rows[1]).toHaveTextContent("0.09%");
    expect(rows[3]).toHaveTextContent("Did not converge");
    expect(screen.getByText(/kept apart and not ranked/)).toHaveTextContent("v0 (2)");
    expect(screen.getByText(/Result 1: v1 · Result 2: v2/)).toBeInTheDocument();
  });

  test("a cluster result shows its scaling efficiency and communication share", async () => {
    const cluster = result("k", "cluster: box · GPU, old · CPU", 20, {
      division: "cluster",
      communication_share: 0.2,
      scaling: { baseline_label: "box · GPU", baseline_s: 30, speedup: 1.5, ideal_speedup: 2, ideal_basis: "machine count", efficiency: 0.75, machines: 2 },
    });
    mockApi({ "GET /mlperf/board": { body: { ...board, board: { ...board.board, divisions: { single: board.board.divisions.single, cluster: [cluster] } } } } });
    renderPage(<TrainingSpeed />);
    expect(await screen.findByText(/Leaderboard: Cluster/)).toBeInTheDocument();
    expect(screen.getByText(/75% scaling efficiency: 1.50x faster than the best single device \(box · GPU\)/)).toHaveTextContent("20% of the time went to communication");
  });

  test("compares the ticked results", async () => {
    const calls = mockApi({
      "GET /mlperf/board": { body: board },
      "POST /mlperf/compare": {
        body: { rows: [result("a", "box · GPU", 30, { relative_time: 1 }), result("b", "box · CPU", 300, { relative_time: 10 })] },
      },
    });
    const user = userEvent.setup();
    renderPage(<TrainingSpeed />);
    await user.click(await screen.findByLabelText("Compare box · GPU"));
    await user.click(screen.getByLabelText("Compare box · CPU"));
    await user.click(screen.getByRole("button", { name: "Compare 2 selected" }));
    const list = await screen.findByRole("list", { name: "Comparison" });
    expect(list).toHaveTextContent("fastest");
    expect(list).toHaveTextContent("10.00x slower");
    expect(calls.find((c) => c.path === "/mlperf/compare")?.body).toEqual({ ids: ["a", "b"] });
  });

  test("queues a run on the chosen device", async () => {
    const calls = mockApi({ "GET /mlperf/board": { body: board }, "POST /mlperf/run": { status: 201, body: { id: "j1", label: "Training speed: small on cuda" } } });
    const user = userEvent.setup();
    renderPage(<TrainingSpeed />);
    await user.click(await screen.findByRole("button", { name: "Run on this machine" }));
    await waitFor(() => expect(calls.find((c) => c.path === "/mlperf/run")?.body).toEqual({ task: "small", device: "cuda" }));
  });

  test("says so when there are no results", async () => {
    mockApi({ "GET /mlperf/board": { body: { ...board, board: { ...board.board, divisions: { single: [], cluster: [] } }, history: { points: [] } } } });
    renderPage(<TrainingSpeed />);
    expect(await screen.findByText(/No results for this task yet/)).toBeInTheDocument();
  });
});

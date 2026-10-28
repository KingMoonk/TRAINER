import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import TrainingRunPage from "./page";

// The Training run page against a mocked API (ported from the Playwright spec tests/training-run.spec.ts, now removed): it reads GET /training-run
// and /queue on load (so a run in progress, or one that has ended, shows without any event), and a `train` event makes it
// read again and redraw the chart in place.

// jsdom has no layout, so Recharts' ResponsiveContainer would draw nothing; give the chart a fixed size so its SVG renders.
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

const POINTS = [1, 2, 3, 4, 5, 6].map((x) => [x * 10, 3 - x * 0.2] as [number, number]);

const curve = (n: number) => ({
  steps: POINTS.slice(0, n),
  smooth: POINTS.slice(0, n),
  heldout: POINTS.slice(0, n).filter((_p, i) => i % 2 === 1),
  lr: POINTS.slice(0, n).map(([step]) => [step, step <= 20 ? step / 20 * 0.0003 : 0.0003 * (1 - (step - 20) / 80)]),
  phases: [[0, 10, "warm-up"], [10, 50, "steady"], [50, 60, "cool-down"]],
  total: 100,
  samples: n > 2 ? [{ step: 20, text: "The cat sat" }] : [],
  sample_every: 20,
  prompt: "Once upon",
});

const ENDED = { reason: "plateau", text: "Stopped at step 30 because the best validation loss (1.595) improved by only 0.19% over the last 6 steps.", step: 30 };

const runPage = (n: number, running: boolean, ended: typeof ENDED | null = null) => ({
  ended,
  version: 2,
  parent: 1,
  change: "6 blocks (was 4)",
  running,
  job_id: running ? "j1" : null,
  step: n * 10,
  total: 100,
  phase: n > 3 ? "steady" : "warm-up",
  status: "learning",
  curve: curve(n),
  references: { uniform_bpc: 8, unigram_bpc: 5, best_so_far_bpc: 2.9 },
  chat_version: 1,
  gate_vs: running ? null : 1,
  switch: running ? { key: "held_back_below", gap: null, docs: null, need: null } : { key: "gate_better", gap: "0.120", docs: null, need: null },
});

const queue = (n: number) => ({
  job: n
    ? { id: "j1", kind: "pretrain", label: "version 2", note: "", params: { config: { batch_size: 16, context: 256 }, pieces_per_pass: 100000 }, status: "running", eta_s: 120, created_at: 0, error: "" }
    : null,
  step: n * 10,
  total: n ? 100 : 0,
  phase: n > 3 ? "steady" : "warm-up",
  lr: 0.0003,
  eta_s: n ? 120 : null,
  paused: false,
  blocked: null,
  gate: { reason: null, resumes_at: null, supported: true },
  stopping: false,
  up_next: [],
  plan: {},
});

function server() {
  const state = { reads: 0, points: 2, running: true, paused: false };
  mockApi({
    "GET /training-run": () => {
      state.reads += 1;
      return { body: runPage(state.points, state.running) };
    },
    "GET /queue": () => ({ body: queue(state.running ? state.points : 0) }),
    "POST /queue/pause": () => {
      state.paused = true;
      return { body: { ...queue(state.points), paused: true } };
    },
  });
  return state;
}

const chart = () => document.querySelector("svg.recharts-surface") as SVGSVGElement;

describe("Training run page", () => {
  test("the chart separates raw loss, schedule bands and the learning-rate scale", async () => {
    const state = server();
    state.points = 6;
    renderPage(<TrainingRunPage />);
    await waitFor(() => expect(chart()).toBeTruthy());
    const svg = chart();
    await waitFor(() => expect(svg.querySelectorAll(".schedule-band")).toHaveLength(3));
    for (const phase of ["warm-up", "steady", "cool-down"]) expect(within(svg as unknown as HTMLElement).getByText(phase)).toBeInTheDocument();
    const raw = svg.querySelector(".series-steps .recharts-line-curve")!;
    expect(raw).toHaveAttribute("stroke-opacity", "0.2");
    expect(raw).toHaveAttribute("stroke-width", "1");
    expect(svg.querySelector(".series-smooth .recharts-line-curve")).toHaveAttribute("stroke-opacity", "1");
    expect(svg.querySelector(".series-lr .recharts-line-curve")).toHaveAttribute("stroke-dasharray", "4 3");
    // Rates of 0.00015-0.0003 sit on their own right-hand scale, not on the loss axis whose references reach 8 bits.
    expect(svg.querySelectorAll(".recharts-yAxis")).toHaveLength(2);
    expect(within(svg as unknown as HTMLElement).getByText("Learning rate")).toBeInTheDocument();
  });

  test("a run that ended itself on a plateau says why, and a run in progress does not", async () => {
    mockApi({
      "GET /training-run": () => ({ body: runPage(6, false, ENDED) }),
      "GET /queue": () => ({ body: queue(0) }),
    });
    renderPage(<TrainingRunPage />);
    const ended = await screen.findByTestId("ended");
    expect(ended).toHaveTextContent("Why this run ended");
    expect(ended).toHaveTextContent("Stopped at step 30 because the best validation loss (1.595) improved by only 0.19%");
  });

  test("a finished run can be continued, and the queued job says what it continues and from which step", async () => {
    const sent: unknown[] = [];
    mockApi({
      "GET /training-run": () => ({ body: runPage(6, false, ENDED) }),
      "GET /queue": () => ({ body: queue(0) }),
      "GET /pretraining/text_sets": { body: [{ id: "poems", name: "Poems", count: 10, held: false, library: false, on: true }] },
      "POST /versions/2/continue": ({ body }) => {
        sent.push(body);
        const kind = (body as { kind: string }).kind;
        const label = kind === "new_data" ? "Version 2 continued from step 60 on new data" : "Version 2 continued from step 60";
        return {
          status: 202,
          body: { id: "k1", kind: "versions.continue", status: "done", progress: "", fraction: null, result: { job_id: "j9", label, version: 2, kind, from_step: 60 }, error: "", created_at: 0, started_at: 0, finished_at: 1 },
        };
      },
    });
    const user = userEvent.setup();
    renderPage(<TrainingRunPage />);
    const card = await screen.findByTestId("continue-card");
    expect(within(card).getByRole("radio", { name: /More steps on the same data/ })).toHaveAttribute("aria-checked", "true");
    await user.click(within(card).getByRole("button", { name: "Add to the queue" }));
    expect(await screen.findByTestId("continued")).toHaveTextContent("Queued: Version 2 continued from step 60.");
    expect(sent).toEqual([{ kind: "same", steps: 1000, dataset_ids: [], rows: 16, hold: 0.05 }]);

    // Other text needs a set ticked before it can be queued.
    await user.click(within(card).getByRole("radio", { name: /Continue on other text/ }));
    expect(within(card).getByRole("button", { name: "Add to the queue" })).toBeDisabled();
    await user.click(await within(card).findByRole("checkbox", { name: "Poems" }));
    await user.click(within(card).getByRole("button", { name: "Add to the queue" }));
    await waitFor(() => expect(screen.getByTestId("continued")).toHaveTextContent("continued from step 60 on new data"));
    expect(sent.at(-1)).toEqual({ kind: "new_data", steps: 1000, dataset_ids: ["poems"], rows: 16, hold: 0.05 });
  });

  test("a run in progress cannot be continued", async () => {
    server();
    renderPage(<TrainingRunPage />);
    await screen.findByTestId("progress");
    expect(screen.queryByTestId("continue-card")).not.toBeInTheDocument();
  });

  test("a run in progress shows no ending", async () => {
    server();
    renderPage(<TrainingRunPage />);
    await screen.findByTestId("progress");
    expect(screen.queryByTestId("ended")).not.toBeInTheDocument();
  });

  test("a run in progress links to its Run view by the queue job id", async () => {
    server();
    renderPage(<TrainingRunPage />);
    expect(await screen.findByTestId("open-run-view")).toHaveAttribute("href", expect.stringMatching(/^\/runs\/view\/?\?id=j1$/));
  });

  test("a run in progress shows step k of n, the phase, the chart and the samples, and updates in place", async () => {
    const state = server();
    renderPage(<TrainingRunPage />);
    expect(await screen.findByRole("heading", { level: 1, name: "Training run · version 2 catching up" })).toBeVisible();
    expect(screen.getByText("Version 2 = version 1 with 6 blocks (was 4).", { exact: false })).toBeVisible();
    await waitFor(() => expect(screen.getByTestId("progress")).toHaveTextContent(/Step \d+ of 100 .* Schedule: (warm-up|steady) .* learning rate: auto \(0\.0003\)/));
    expect(screen.getByTestId("status")).toHaveTextContent("Learning");
    expect(screen.getByText("Chat still uses version 1.")).toBeVisible();
    await waitFor(() => expect(chart()).toBeTruthy());
    const kept = chart();
    kept.setAttribute("data-kept", "yes");
    expect(screen.getByTestId("progress")).toHaveTextContent("Step 20 of 100");
    const reads = state.reads;

    // A train event brings more points: the same chart element redraws, and the sample row appears.
    state.points = 6;
    emit("train", { job: "j1" });
    await waitFor(() => expect(screen.getByTestId("progress")).toHaveTextContent("Step 60 of 100"), { timeout: 5_000 });
    expect(screen.getByTestId("progress")).toHaveTextContent("Schedule: steady");
    expect(chart()).toBe(kept);
    expect(chart()).toHaveAttribute("data-kept", "yes");
    const samples = screen.getAllByTestId("sample");
    expect(samples.at(-1)).toHaveTextContent("The cat sat");
    expect(samples.at(-1)).toHaveTextContent("now · step 20");
    expect(state.reads).toBeGreaterThan(reads);
  });

  test("a page opened after the run ended shows the finished run and Chat's verdict", async () => {
    const state = server();
    state.points = 6;
    state.running = false;
    renderPage(<TrainingRunPage />);
    await waitFor(() => expect(screen.getByTestId("progress")).toHaveTextContent("Nothing is training."));
    expect(screen.getByTestId("switch")).toHaveTextContent("Version 2 beat version 1 on the same held-back passages by 0.120 bits per character");
    await waitFor(() => expect(chart()).toBeTruthy());
    expect(screen.getByRole("button", { name: "Pause" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Stop after a short cool-down" })).toBeDisabled();
  });

  test("a scored run shows measured assistant entries and their version trend", async () => {
    const entries = {
      choice_accuracy: { value: .5, band: [.3, .7], items: 40, method: "choice log-likelihood" },
      checked_accuracy: { value: .6, band: [.4, .8], items: 30, method: "rule-checked answers" },
      heldout_bpc: { value: 2.1, band: [2.0, 2.2], items: 24, method: "held-out text" },
      chat_win_rate: { value: .65, band: [.55, .75], items: 30, method: "head-to-head rule-checked replies" },
    };
    mockApi({
      "GET /training-run": () => ({ body: { ...runPage(6, false), report_card: { suite: "fixed", entries, seconds: 3.2, decontam: "checked" },
        report_card_trend: [{ version: 1, entries: { ...entries, choice_accuracy: { ...entries.choice_accuracy, value: .4 } } }, { version: 2, entries }] } }),
      "GET /queue": () => ({ body: queue(0) }),
    });
    renderPage(<TrainingRunPage />);
    const card = await screen.findByTestId("assistant-report-card");
    expect(card).toHaveTextContent("Multiple-choice accuracy");
    expect(card).toHaveTextContent("50.0%");
    expect(card).toHaveTextContent("Checked reply win rate vs previous version");
    expect(within(card).getByTestId("report-card-trend")).toHaveTextContent("40.0%");
  });

  test("Pause is sent to the queue", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<TrainingRunPage />);
    await user.click(await screen.findByRole("button", { name: "Pause" }));
    await waitFor(() => expect(state.paused).toBe(true));
    expect(await screen.findByRole("button", { name: "Resume" })).toBeVisible();
  });
});

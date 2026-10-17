import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { emit, mockApi, type Call } from "@/test/network";
import { renderPage } from "@/test/render";
import ComparePage from "./page";

// The Compare runs page against a mocked API (ported from the Playwright spec tests/compare.spec.ts, now removed).

// jsdom has no layout, so Recharts' ResponsiveContainer would draw nothing; give the chart a fixed size so its lines render.
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

// Radix Select asks for pointer capture, which jsdom lacks.
Element.prototype.hasPointerCapture ??= () => false;
Element.prototype.releasePointerCapture ??= () => {};
Element.prototype.setPointerCapture ??= () => {};

const job = (id: string, kind: string, result: unknown, status = "done") => ({
  id,
  kind,
  status,
  progress: "",
  fraction: null,
  result,
  error: "",
  created_at: 0,
  started_at: 0,
  finished_at: 0,
});

const VERSIONS = [
  { number: 1, name: "Version 1", change: "", spec: {}, status: "trained" },
  { number: 2, name: "Version 2", change: "wider", spec: {}, status: "trained" },
];

const curve = (offset: number) => [0, 100, 200, 300].map((step, i) => [step, 3 - 0.4 * i - offset, step * 64, step * 0.5]);
const stats = (variant: number, mean: number, delta: number | null, inside: boolean | null, bpc: number[]) => ({
  variant,
  seeds: bpc.map((_, i) => i),
  bpc,
  mean,
  min: Math.min(...bpc),
  max: Math.max(...bpc),
  spread: Math.max(...bpc) - Math.min(...bpc),
  delta,
  inside_spread: inside,
  pieces_per_s: 1500,
  peak_mb: 210,
  curve: curve(variant * 0.2),
});

const RUN = {
  base: {},
  variants: [{}],
  mode: "params",
  names: ["Base", "d_model 64 to 96"],
  question: "Does width help?",
  guess: "Better",
  finished: 1700000000,
  results: {
    variants: [stats(0, 1.8, null, null, [1.78, 1.82, 1.8]), stats(1, 1.6, -0.2, false, [1.58, 1.62, 1.6])],
    references: { uniform: 3.0 },
  },
};

const planVariant = (variant: number, name: string, over: object = {}) => ({
  variant,
  name,
  matched: true,
  reason: "",
  adjustments: [],
  steps: 300,
  parameters: 1,
  relative_error: 0,
  changes: [],
  ...over,
});
const simplePlan = {
  base: 2,
  n_runs: 2,
  n_sweeps: 0,
  wall_seconds: 120,
  variants: [planVariant(0, "Base"), planVariant(1, "wider", { changes: ["x"] })],
};

const lines = () => document.querySelectorAll(".recharts-line").length;
const bodies = (calls: Call[], path: string) => calls.filter((c) => c.method === "POST" && c.path === path).map((c) => c.body);

const base = (runs: () => unknown[], extra: Parameters<typeof mockApi>[0] = {}) =>
  mockApi({
    "GET /versions": { body: VERSIONS },
    "POST /benchmark/compare-runs": () => ({ status: 202, body: job("cr", "benchmark.compare_runs", runs()) }),
    ...extra,
  });

async function addVersionOne(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("combobox", { name: "Add a variant from a saved version" }));
  await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Version 1" }));
}

describe("Compare runs page", () => {
  test("two runs of a comparison draw on one chart, with the table and the guess's outcome", async () => {
    base(() => [RUN]);
    renderPage(<ComparePage />);
    expect(await screen.findByRole("heading", { level: 1, name: "Compare runs" })).toBeVisible();
    // Base, the variant and the measured guessing level are lines of the one shared chart.
    await waitFor(() => expect(lines()).toBe(3));
    expect(screen.getAllByText("A · Base").length).toBeGreaterThan(0);
    expect(screen.getAllByText("B · d_model 64 to 96").length).toBeGreaterThan(0);
    expect(screen.getByRole("cell", { name: /1\.800 · spread 0\.040/ })).toBeVisible();
    expect(screen.getByRole("cell", { name: "-0.200 · beyond the spread" })).toBeVisible();
    expect(screen.getByText("You guessed better; B came out better (beyond the spread).")).toBeVisible();
    expect(screen.getAllByText("Does width help?").length).toBeGreaterThan(0);
  });

  test("an axis the runs did not record falls back to steps and says so", async () => {
    const bare = { ...RUN, results: { ...RUN.results, variants: RUN.results.variants.map((v) => ({ ...v, curve: v.curve.map((p) => p.slice(0, 2)) })) } };
    base(() => [bare]);
    const user = userEvent.setup();
    renderPage(<ComparePage />);
    await user.click(await screen.findByRole("radio", { name: "Minutes" }));
    expect(screen.getByText(/These runs did not record minutes, so the curves are drawn against steps\./)).toBeVisible();
  });

  test("with nothing finished the page says so, and shows a skeleton while it reads", async () => {
    let release = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    mockApi({
      "GET /versions": { body: VERSIONS },
      "POST /benchmark/compare-runs": { status: 202, body: job("late", "benchmark.compare_runs", null, "queued") },
      "GET /jobs/late": async () => {
        await gate;
        return { body: job("late", "benchmark.compare_runs", []) };
      },
    });
    renderPage(<ComparePage />);
    expect(await screen.findByLabelText("Loading the finished comparisons")).toBeVisible();
    release();
    expect(await screen.findByText(/No comparison has finished yet\./)).toBeVisible();
  });

  test("a comparison is planned, needs a guess, and queues with the chosen seeds and mode", async () => {
    const calls = base(() => [], {
      "POST /benchmark/compare-plan": ({ body }) => {
        const { seeds, retune_lr } = body as { seeds: number; retune_lr: boolean };
        return {
          body: {
            base: 2,
            n_runs: 2 * seeds,
            n_sweeps: retune_lr ? 3 : 0,
            wall_seconds: 120,
            variants: [
              planVariant(0, "Base"),
              planVariant(1, "d_model 64 to 96", { adjustments: [{}], parameters: 101000, relative_error: 0.01, changes: ["dims.d_model: 64 → 96"] }),
            ],
          },
        };
      },
      "POST /benchmark/compare-queue": { status: 201, body: { id: "q1", label: "Compare runs: 6 runs + 3 short sweeps", outlook: "Starts now" } },
    });
    const user = userEvent.setup();
    renderPage(<ComparePage />);

    const queue = () => screen.getByRole("button", { name: /^Queue / });
    await screen.findByRole("combobox", { name: "Add a variant from a saved version" });
    expect(queue()).toBeDisabled();
    await addVersionOne(user);
    expect(await screen.findByText("dims.d_model: 64 → 96")).toBeVisible();
    expect(screen.getByText("width re-solved · 0.10M · within 1.0%")).toBeVisible();
    await waitFor(() => expect(queue()).toHaveTextContent("Queue 6 runs + 3 short sweeps"));
    expect(queue()).toBeDisabled(); // no guess yet
    expect(screen.getByText("Pick your guess first.")).toBeVisible();

    await user.click(screen.getByRole("radio", { name: "Better" }));
    await user.click(screen.getByRole("radio", { name: "Compute" }));
    await user.click(screen.getByRole("radio", { name: "5" }));
    await user.type(screen.getByLabelText("Question"), "Does width help?");
    await waitFor(() => expect(queue()).toHaveTextContent("Queue 10 runs + 3 short sweeps"));
    await user.click(queue());
    expect(await screen.findByText("Queued: Compare runs: 6 runs + 3 short sweeps. Starts now.", { selector: "p[role=status]" })).toBeVisible();
    expect(bodies(calls, "/benchmark/compare-queue")).toEqual([
      { base: null, variants: [1], mode: "compute", seeds: 5, retune_lr: true, question: "Does width help?", guess: "Better" },
    ]);
    expect(bodies(calls, "/benchmark/compare-plan").at(-1)).toMatchObject({ mode: "compute", seeds: 5, variants: [1] });
  });

  test("curves stream in while the queued runs train, and the finished comparison replaces them", async () => {
    let finished = false;
    base(() => (finished ? [RUN] : []), {
      "POST /benchmark/compare-plan": { body: simplePlan },
      "POST /benchmark/compare-queue": { status: 201, body: { id: "q1", label: "Compare runs: 2 runs", outlook: "Starts now" } },
    });
    const train = (variant: number, step: number, bpc: number) =>
      emit("train", { job: "q1", type: "compare", phase: "run", variant, seed: 7, step, total: 300, bpc, trained_tokens: step * 64, elapsed_seconds: step });
    const user = userEvent.setup();
    renderPage(<ComparePage />);
    await addVersionOne(user);
    await user.click(screen.getByRole("radio", { name: "Better" }));
    await user.click(await screen.findByRole("button", { name: /^Queue 2 runs/ }));
    // The page listens for the queued job's events once the queue call has answered.
    await screen.findByText(/^Queued: /, { selector: "p[role=status]" });
    for (const [variant, step, bpc] of [[0, 100, 2.5], [0, 200, 2.2], [1, 100, 2.4], [1, 200, 2.0]]) train(variant, step, bpc);
    expect(await screen.findByText("Running B, seed 7: step 200 of 300.")).toBeVisible();
    await waitFor(() => expect(lines()).toBe(2));
    expect(screen.getAllByText("A · Base").length).toBeGreaterThan(0);

    // The run ends: the finished comparison is read again and carries its own guess.
    finished = true;
    emit("train", { job: "q1", type: "compare_done" });
    expect(await screen.findByText("You guessed better; B came out better (beyond the spread).")).toBeVisible();
  });

  test("a queue refusal is shown in the page", async () => {
    base(() => [], {
      "POST /benchmark/compare-plan": { body: simplePlan },
      "POST /benchmark/compare-queue": { status: 422, body: { detail: "Your data has no plain text yet. Add some in Your data, then queue the runs." } },
    });
    const user = userEvent.setup();
    renderPage(<ComparePage />);
    await addVersionOne(user);
    await user.click(screen.getByRole("radio", { name: "Better" }));
    await user.click(await screen.findByRole("button", { name: /^Queue 2 runs/ }));
    expect(await screen.findByText(/Your data has no plain text yet\./, { selector: "p[role=status]" })).toBeVisible();
  });
});

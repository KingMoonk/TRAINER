import { expect, test, type Page } from "@playwright/test";

const job = (id: string, kind: string, result: unknown) => ({
  id,
  kind,
  status: "done",
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

async function mock(page: Page, runs: unknown[]) {
  await page.route("**/versions", (route) => route.fulfill({ json: VERSIONS }));
  await page.route("**/benchmark/compare-runs", (route) => route.fulfill({ status: 202, json: job("cr", "benchmark.compare_runs", runs) }));
}

test("two runs of a comparison draw on one chart, with the table and the guess's outcome", async ({ page }) => {
  await mock(page, [RUN]);
  await page.goto("/compare/");
  await expect(page.getByRole("heading", { level: 1, name: "Compare runs" })).toBeVisible();
  // Base, the variant and the measured guessing level are lines of the one shared chart.
  await expect(page.locator(".recharts-line")).toHaveCount(3);
  await expect(page.getByText("A · Base").first()).toBeVisible();
  await expect(page.getByText("B · d_model 64 to 96").first()).toBeVisible();
  await expect(page.getByRole("cell", { name: /1\.800 · spread 0\.040/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: "-0.200 · beyond the spread" })).toBeVisible();
  await expect(page.getByText("You guessed better; B came out better (beyond the spread).")).toBeVisible();
  await expect(page.getByText("Does width help?").first()).toBeVisible();
});

test("an axis the runs did not record falls back to steps and says so", async ({ page }) => {
  const bare = { ...RUN, results: { ...RUN.results, variants: RUN.results.variants.map((v) => ({ ...v, curve: v.curve.map((p) => p.slice(0, 2)) })) } };
  await mock(page, [bare]);
  await page.goto("/compare/");
  await page.getByRole("radio", { name: "Minutes" }).click();
  await expect(page.getByText("These runs did not record minutes, so the curves are drawn against steps.", { exact: false })).toBeVisible();
});

test("with nothing finished the page says so, and shows a skeleton while it reads", async ({ page }) => {
  await page.route("**/versions", (route) => route.fulfill({ json: VERSIONS }));
  let release = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  await page.route("**/benchmark/compare-runs", (route) => route.fulfill({ status: 202, json: { ...job("late", "benchmark.compare_runs", null), status: "queued" } }));
  await page.route("**/jobs/late", async (route) => {
    await gate;
    await route.fulfill({ json: job("late", "benchmark.compare_runs", []) });
  });
  await page.goto("/compare/");
  await expect(page.getByLabel("Loading the finished comparisons")).toBeVisible();
  release();
  await expect(page.getByText("No comparison has finished yet.", { exact: false })).toBeVisible();
});

test("a comparison is planned, needs a guess, and queues with the chosen seeds and mode", async ({ page }) => {
  await mock(page, []);
  const planned: Record<string, unknown>[] = [];
  const queued: Record<string, unknown>[] = [];
  await page.route("**/benchmark/compare-plan", (route) => {
    const body = route.request().postDataJSON();
    planned.push(body);
    return route.fulfill({
      json: {
        base: 2,
        n_runs: body.variants.length * 0 + 2 * body.seeds,
        n_sweeps: body.retune_lr ? 3 : 0,
        wall_seconds: 120,
        variants: [
          { variant: 0, name: "Base", matched: true, reason: "", adjustments: [], steps: 300, parameters: 100000, relative_error: 0, changes: [] },
          { variant: 1, name: "d_model 64 to 96", matched: true, reason: "", adjustments: [{}], steps: 300, parameters: 101000, relative_error: 0.01, changes: ["dims.d_model: 64 → 96"] },
        ],
      },
    });
  });
  await page.route("**/benchmark/compare-queue", (route) => {
    queued.push(route.request().postDataJSON());
    return route.fulfill({ status: 201, json: { id: "q1", label: "Compare runs: 6 runs + 3 short sweeps", outlook: "Starts now" } });
  });
  await page.goto("/compare/");

  const queue = page.getByRole("button", { name: /^Queue / });
  await expect(queue).toBeDisabled();
  await page.getByRole("combobox", { name: "Add a variant from a saved version" }).click();
  await page.getByRole("option", { name: "Version 1" }).click();
  await expect(page.getByText("dims.d_model: 64 → 96")).toBeVisible();
  await expect(page.getByText("width re-solved · 0.10M · within 1.0%")).toBeVisible();
  await expect(queue).toHaveText("Queue 6 runs + 3 short sweeps");
  await expect(queue).toBeDisabled(); // no guess yet
  await expect(page.getByText("Pick your guess first.")).toBeVisible();

  await page.getByRole("radio", { name: "Better" }).click();
  await page.getByRole("radio", { name: "Compute" }).click();
  await page.getByRole("radio", { name: "5" }).click();
  await page.getByLabel("Question").fill("Does width help?");
  await expect(queue).toHaveText("Queue 10 runs + 3 short sweeps");
  await queue.click();
  await expect(page.getByRole("status").filter({ hasText: "Queued: Compare runs: 6 runs + 3 short sweeps. Starts now." })).toBeVisible();
  expect(queued).toEqual([
    { base: null, variants: [1], mode: "compute", seeds: 5, retune_lr: true, question: "Does width help?", guess: "Better" },
  ]);
  expect(planned.at(-1)).toMatchObject({ mode: "compute", seeds: 5, variants: [1] });
});

test("curves stream in while the queued runs train, and the finished comparison replaces them", async ({ page }) => {
  let finished = false;
  await page.route("**/versions", (route) => route.fulfill({ json: VERSIONS }));
  await page.route("**/benchmark/compare-runs", (route) => route.fulfill({ status: 202, json: job("cr", "benchmark.compare_runs", finished ? [RUN] : []) }));
  await page.route("**/benchmark/compare-plan", (route) =>
    route.fulfill({
      json: {
        base: 2,
        n_runs: 2,
        n_sweeps: 0,
        wall_seconds: 120,
        variants: [
          { variant: 0, name: "Base", matched: true, reason: "", adjustments: [], steps: 300, parameters: 1, relative_error: 0, changes: [] },
          { variant: 1, name: "wider", matched: true, reason: "", adjustments: [], steps: 300, parameters: 1, relative_error: 0, changes: ["x"] },
        ],
      },
    }),
  );
  const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
  const train = (variant: number, step: number, bpc: number) => ({
    kind: "train",
    payload: { job: "q1", type: "compare", phase: "run", variant, seed: 7, step, total: 300, bpc, trained_tokens: step * 64, elapsed_seconds: step },
  });
  let streamed = "";
  await page.route("**/events", async (route) => {
    if (!streamed) return route.fulfill({ status: 404 });
    await route.fulfill({ status: 200, contentType: "text/event-stream", body: streamed });
  });
  await page.route("**/benchmark/compare-queue", (route) => {
    streamed = [train(0, 100, 2.5), train(0, 200, 2.2), train(1, 100, 2.4), train(1, 200, 2.0)].map(frame).join("");
    return route.fulfill({ status: 201, json: { id: "q1", label: "Compare runs: 2 runs", outlook: "Starts now" } });
  });
  await page.goto("/compare/");
  await page.getByRole("combobox", { name: "Add a variant from a saved version" }).click();
  await page.getByRole("option", { name: "Version 1" }).click();
  await page.getByRole("radio", { name: "Better" }).click();
  await page.getByRole("button", { name: /^Queue 2 runs/ }).click();
  await expect(page.getByText("Running B, seed 7: step 200 of 300.")).toBeVisible();
  await expect(page.locator(".recharts-line")).toHaveCount(2);
  await expect(page.getByText("A · Base").first()).toBeVisible();

  finished = true;
  streamed = "";
  await page.reload();
  await expect(page.getByText("You guessed better; B came out better (beyond the spread).")).toBeVisible(); // the saved run carries its own guess
});

test("a queue refusal is shown in the page", async ({ page }) => {
  await mock(page, []);
  await page.route("**/benchmark/compare-plan", (route) =>
    route.fulfill({ json: { base: 2, n_runs: 2, n_sweeps: 0, wall_seconds: 120, variants: [
      { variant: 0, name: "Base", matched: true, reason: "", adjustments: [], steps: 300, parameters: 1, relative_error: 0, changes: [] },
      { variant: 1, name: "v", matched: true, reason: "", adjustments: [], steps: 300, parameters: 1, relative_error: 0, changes: [] },
    ] } }),
  );
  await page.route("**/benchmark/compare-queue", (route) => route.fulfill({ status: 422, json: { detail: "Your data has no plain text yet. Add some in Your data, then queue the runs." } }));
  await page.goto("/compare/");
  await page.getByRole("combobox", { name: "Add a variant from a saved version" }).click();
  await page.getByRole("option", { name: "Version 1" }).click();
  await page.getByRole("radio", { name: "Better" }).click();
  await page.getByRole("button", { name: /^Queue 2 runs/ }).click();
  await expect(page.getByRole("status").filter({ hasText: "Your data has no plain text yet." })).toBeVisible();
});

import { expect, test, type Page } from "@playwright/test";

// The Training run page against a mocked API: it reads GET /training-run and /queue on load (so a run in progress, or one that
// has ended, shows without any event), and a `train` event makes it read again and redraw the chart in place.

const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

const POINTS = [1, 2, 3, 4, 5, 6].map((x) => [x * 10, 3 - x * 0.2] as [number, number]);

const curve = (n: number) => ({
  steps: POINTS.slice(0, n),
  smooth: POINTS.slice(0, n),
  heldout: POINTS.slice(0, n).filter((_p, i) => i % 2 === 1),
  lr: [],
  phases: [[0, 10, "warm-up"], [10, 50, "steady"], [50, 60, "cool-down"]],
  total: 100,
  samples: n > 2 ? [{ step: 20, text: "The cat sat" }] : [],
  sample_every: 20,
  prompt: "Once upon",
});

const runPage = (n: number, running: boolean) => ({
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

async function mockApi(page: Page, { events }: { events: boolean }) {
  const state = { reads: 0, points: 2, running: true, paused: false };
  await page.route("**/training-run", (route) => {
    state.reads += 1;
    return route.fulfill(json(runPage(state.points, state.running)));
  });
  await page.route("**/queue", (route) => route.fulfill(json(queue(state.running ? state.points : 0))));
  await page.route("**/queue/pause", (route) => {
    state.paused = true;
    return route.fulfill(json({ ...queue(state.points), paused: true }));
  });
  if (events) {
    // Each connection brings one train event and closes; the browser reconnects after `retry` ms, so the page reads again.
    await page.route("**/events", (route) => {
      state.points = Math.min(6, state.points + 1);
      return route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: `retry: 100\n\ndata: ${JSON.stringify({ kind: "train", payload: { job: "j1" } })}\n\n`,
      });
    });
  }
  return state;
}

test("a run in progress shows step k of n, the phase, the chart and the samples, and updates in place", async ({ page }) => {
  const state = await mockApi(page, { events: true });
  await page.goto("/training-run/");
  await expect(page.getByRole("heading", { level: 1, name: "Training run · version 2 catching up" })).toBeVisible();
  await expect(page.getByText("Version 2 = version 1 with 6 blocks (was 4).")).toBeVisible();
  await expect(page.getByTestId("progress")).toContainText(/Step \d+ of 100 .* Schedule: (warm-up|steady) .* learning rate: auto \(0\.0003\)/);
  await expect(page.getByTestId("status")).toHaveText("Learning");
  await expect(page.getByText("Chat still uses version 1.")).toBeVisible();

  const chart = page.locator("svg.recharts-surface").first();
  await expect(chart).toBeVisible();
  await chart.evaluate((el) => el.setAttribute("data-kept", "yes"));

  // A later event brings more points: the same chart element redraws, and the sample row appears.
  await expect(page.getByTestId("progress")).toContainText("Step 60 of 100", { timeout: 15_000 });
  await expect(page.getByTestId("progress")).toContainText("Schedule: steady");
  await expect(page.locator("svg.recharts-surface").first()).toHaveAttribute("data-kept", "yes");
  await expect(page.getByTestId("sample").last()).toContainText("The cat sat");
  await expect(page.getByTestId("sample").last()).toContainText("now · step 20");
  expect(state.reads).toBeGreaterThan(1);
});

test("a page opened after the run ended shows the finished run and Chat's verdict", async ({ page }) => {
  const state = await mockApi(page, { events: false });
  state.points = 6;
  state.running = false;
  await page.goto("/training-run/");
  await expect(page.getByTestId("progress")).toHaveText("Nothing is training.");
  await expect(page.getByTestId("switch")).toContainText("Version 2 beat version 1 on the same held-back passages by 0.120 bits per character");
  await expect(page.locator("svg.recharts-surface").first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Pause" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Stop after a short cool-down" })).toBeDisabled();
});

test("Pause is sent to the queue", async ({ page }) => {
  const state = await mockApi(page, { events: false });
  await page.goto("/training-run/");
  await page.getByRole("button", { name: "Pause" }).click();
  await expect.poll(() => state.paused).toBe(true);
  await expect(page.getByRole("button", { name: "Resume" })).toBeVisible();
});

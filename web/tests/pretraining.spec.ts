import { expect, test, type Page } from "@playwright/test";

test("Pretraining opens prose generation", async ({ page }) => {
  await mockApi(page);
  await page.goto("/pretraining/");
  await expect(page.getByRole("link", { name: /Make extra data/ })).toHaveAttribute("href", "/make-data/?kind=text");
});

// The Pretraining page against a mocked API (as chat.spec.ts does): the sets, the plan the server works out from the page's
// choices, the size preview and the learning-rate range test as jobs whose results arrive on /events, and Add to the queue.

const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

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
  finished_at: status === "done" ? 1 : null,
});

const PIECES = 2_000_000;
const preview = {
  pieces: PIECES,
  estimated: false,
  params: 1_000_000,
  vocab: 256,
  context: 256,
  first_rows: [
    [0, 0, 1, -1, 2, 2],
    [1, 1, 1, 1, -1, 0],
  ],
  tokens_per_second: null,
};

type PlanBody = { dataset_ids: string[]; mode: string; rows: number; hold: number; passes: number };

type State = {
  plans: PlanBody[];
  queued: unknown[];
  lr: unknown[];
  known: boolean;
  events: string[];
  release: () => void;
};

async function mockApi(page: Page, { known = true } = {}): Promise<State> {
  let release = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  const state: State = { plans: [], queued: [], lr: [], known, events: [], release };
  let streams = 0;

  await page.route("**/versions", (route) =>
    route.fulfill(
      json([
        { number: 1, name: "version 1", change: "", parent: null, spec: {}, status: "trained", best_checkpoint: "ck1", scorecard: {}, verdict: "", prediction: {} },
        { number: 2, name: "version 2", change: "", parent: 1, spec: { train: { optimizer: { config: { lr: 0.0006 } } } }, status: "saved", best_checkpoint: null, scorecard: {}, verdict: "", prediction: {} },
      ]),
    ),
  );
  await page.route("**/pretraining/text_sets", (route) =>
    route.fulfill(
      json([
        { id: "stories", name: "Short stories", count: 1200, held: false, library: false, on: true },
        { id: "catalog_wiki_validation", name: "Wiki validation", count: 300, held: true, library: true, on: false },
      ]),
    ),
  );
  await page.route("**/pretraining/plan", async (route) => {
    const body = route.request().postDataJSON() as PlanBody;
    state.plans.push(body);
    const ids = body.dataset_ids;
    const perStep = body.rows * 256;
    const steps = state.known && ids.length ? Math.ceil((PIECES * body.passes) / perStep) : null;
    const reason = !ids.length ? "Choose some text first." : !state.known ? "The size of the text is still being worked out." : "";
    await route.fulfill(
      json({
        version: 2,
        parent: 1,
        mode: body.mode,
        can_continue: true,
        continue_blocked: null,
        estimating: ids.length > 0 && !state.known,
        pieces: state.known && ids.length ? PIECES : null,
        estimated: false,
        params: 1_000_000,
        context: 256,
        pieces_per_step: perStep,
        passes_over_text: state.known && ids.length ? Math.round(PIECES / perStep) : null,
        steps,
        minutes: null,
        lr: 0.0006,
        enough: state.known && ids.length ? { ratio: 2, ratio_text: "2.00", headline: "This model could use about 10.0× more text than you have", verdict: "little" } : null,
        expectation: state.known && ids.length ? { headline: "Expect the error to fall to about 3.1 bits per character", details: "Based on the size of the model and the text." } : null,
        config: steps ? { steps, batch_size: body.rows, validation: body.hold, objective: "all" } : null,
        can_add: !reason,
        reason,
        job: reason
          ? null
          : {
              kind: "pretrain",
              label: `Version 2 ${body.mode === "catch_up" ? "catch-up" : "from scratch"}`,
              params: { config: { steps }, datasets: ids, mode: body.mode, init_from: body.mode === "catch_up" ? "ck1" : null, version: 2, heldout_fraction: body.hold },
            },
        outlook: "If you add it: starts now",
      }),
    );
  });
  await page.route("**/pretraining/preview**", async (route) => {
    if (route.request().method() === "GET") return route.fulfill(json(state.known ? preview : null));
    state.known = true; // the job has the answer; it arrives on /events
    state.events.push(frame({ kind: "job", payload: job("p1", "pretrain_preview", preview) }));
    return route.fulfill(json(job("p1", "pretrain_preview", null, "running"), 202));
  });
  await page.route("**/pretraining/lr_range_test", async (route) => {
    state.lr.push(route.request().postDataJSON());
    const rates = [1e-5, 1e-4, 1e-3, 1e-2];
    const result = { rates, losses: [5, 4.5, 3, 4], smoothed: [5, 4.5, 3, 4], suggestion: 1e-4, steepest: 2, diverged: false, cancelled: false, failed_at: null };
    state.events.push(frame({ kind: "job", payload: job("lr1", "lr_range_test", result) }));
    await route.fulfill(json(job("lr1", "lr_range_test", null, "running"), 202));
  });
  await page.route("**/jobs/*", (route) => route.fulfill(json(job("x", "x", null, "running"))));
  await page.route("**/queue/jobs", async (route) => {
    const body = route.request().postDataJSON() as { label: string };
    state.queued.push(body);
    await route.fulfill(json({ id: "q1", note: "", status: "waiting", eta_s: null, created_at: 0, error: "", ...body }, 201));
  });
  await page.route("**/queue/jobs/*/outlook", (route) => route.fulfill(json({ outlook: "Starts after Version 1 catch-up (about 3 min)" })));
  await page.route("**/data/collections/*", (route) =>
    route.fulfill(json([{ id: "tinystories", name: "TinyStories", size: "64 MB", records: "77,941 stories", state: "not downloaded", completed: 0, total: 0, error: "" }])),
  );
  await page.route("**/events", async (route) => {
    streams += 1;
    if (streams > 2) return route.abort(); // the shell opens the first stream, the page the second; neither is reopened
    await gate;
    return route.fulfill({ status: 200, contentType: "text/event-stream", body: state.events.join("") });
  });
  return state;
}

test("the choices give the plan the server works out, and a held-out set starts unticked", async ({ page }) => {
  const state = await mockApi(page);
  await page.goto("/pretraining/");
  await expect(page.getByRole("heading", { level: 1, name: "Pretraining" })).toBeVisible();
  await expect(page.getByTestId("model-line")).toHaveText("Version 2 · from the Network designer");
  await expect(page.getByTestId("set-stories")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("set-catalog_wiki_validation")).toHaveAttribute("aria-checked", "false");
  await expect(page.getByTestId("set-catalog_wiki_validation")).toContainText("held-out: kept for testing, not trained on");
  await expect(page.getByTestId("selected-line")).toHaveText("1 source selected");
  await expect(page.getByTestId("enough-headline")).toHaveText("This model could use about 10.0× more text than you have");
  await expect(page.getByTestId("expectation")).toContainText("Expect the error to fall");
  await expect(page.getByTestId("per-step")).toHaveText("4,096 pieces per step · one pass over your data ≈ 488 steps");
  await expect(page.getByTestId("steps-line")).toContainText("977 steps");
  await expect(page.getByTestId("rows-grid")).toBeVisible();
  await expect(page.getByTestId("mode-catch_up")).toContainText("Continue from version 1");
  await expect(page.getByText("Learning rate: auto (0.0006, set with width-scaled setup)")).toBeVisible();
  expect(state.plans.at(-1)).toMatchObject({ dataset_ids: ["stories"], mode: "catch_up", rows: 16, hold: 0.05, passes: 2 });

  // A changed choice is sent whole to the server, which answers with the new steps.
  await page.getByRole("radio", { name: "32" }).click();
  await page.getByRole("radio", { name: "Long" }).click();
  await page.getByRole("radio", { name: "10%" }).click();
  await page.getByTestId("set-catalog_wiki_validation").click();
  await expect(page.getByTestId("selected-line")).toHaveText("2 sources selected");
  await expect(page.getByTestId("steps-line")).toContainText("977 steps");
  expect(state.plans.at(-1)).toMatchObject({ dataset_ids: ["stories", "catalog_wiki_validation"], rows: 32, hold: 0.1, passes: 4 });

  // With nothing chosen the plan says why it cannot be added.
  await page.getByTestId("set-stories").click();
  await page.getByTestId("set-catalog_wiki_validation").click();
  await expect(page.getByTestId("selected-line")).toHaveText("nothing selected");
  await expect(page.getByTestId("reason")).toHaveText("Choose some text first.");
  await expect(page.getByRole("button", { name: "Add to the queue" })).toBeDisabled();
});

test("the size preview shows a skeleton, then the result its job delivers", async ({ page }) => {
  const state = await mockApi(page, { known: false });
  await page.goto("/pretraining/");
  await expect(page.getByLabel("Estimating how much text you have…").first()).toBeVisible();
  await expect(page.getByTestId("enough-headline")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Add to the queue" })).toBeDisabled();
  await expect(page.getByTestId("reason")).toHaveText("The size of the text is still being worked out.");

  await expect.poll(() => state.known).toBe(true); // the job was asked for
  state.release(); // its result arrives on /events
  await expect(page.getByTestId("enough-headline")).toHaveText("This model could use about 10.0× more text than you have");
  await expect(page.getByTestId("rows-grid")).toBeVisible();
  await expect(page.getByRole("button", { name: "Add to the queue" })).toBeEnabled();
});

test("Add to the queue posts the job the plan built and says when it starts", async ({ page }) => {
  const state = await mockApi(page);
  await page.goto("/pretraining/");
  await expect(page.getByTestId("queue-line")).toHaveText("If you add it: starts now.");
  await page.getByTestId("mode-from_scratch").click();
  await expect(page.getByTestId("mode-from_scratch")).toHaveAttribute("aria-checked", "true");
  await page.getByRole("button", { name: "Add to the queue" }).click();
  await expect(page.getByText("Added to the queue: Version 2 from scratch")).toBeVisible();
  expect(state.queued).toEqual([
    expect.objectContaining({ kind: "pretrain", label: "Version 2 from scratch", params: expect.objectContaining({ datasets: ["stories"], mode: "from_scratch", init_from: null, version: 2 }) }),
  ]);
  await expect(page.getByTestId("queue-line")).toHaveText("Queued. Starts after Version 1 catch-up (about 3 min).");
});

test("the learning-rate range test shows a skeleton, then its chart and the sentence that reads it", async ({ page }) => {
  const state = await mockApi(page);
  await page.goto("/pretraining/");
  await page.getByRole("button", { name: "Find a learning rate" }).click();
  await expect(page.getByLabel("Trying learning rates, smallest first. This takes a moment.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop" })).toBeVisible();
  expect(state.lr).toEqual([{ spec: { train: { optimizer: { config: { lr: 0.0006 } } } } }]);
  state.release();
  await expect(page.getByTestId("lr-text")).toContainText("Suggested learning rate: 0.0001. The error fell fastest near 0.001");
  await expect(page.getByTestId("lr-text")).toContainText("This design uses 0.0006.");
});

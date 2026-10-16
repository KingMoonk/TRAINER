import { expect, test, type Page } from "@playwright/test";

// The Post-training page against a mocked API: the page reads /versions, /post-training/options and /queue, runs the status
// job (its result arrives as a `job` event), queues a round, shows its progress and imports a model as a new version.

const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

const version = (number: number, trained: boolean) => ({
  number,
  name: `version ${number}`,
  change: number === 2 ? "Started from GPT-2 (124M)" : "First design",
  parent: null,
  spec: {},
  status: trained ? "trained" : "saved",
  best_checkpoint: trained ? `ck${number}` : null,
  scorecard: {},
  verdict: "",
  prediction: {},
});

const job = (id: string, result: unknown) => ({
  id,
  kind: "post_training.status",
  status: "done",
  progress: "",
  fraction: null,
  result,
  error: "",
  created_at: 0,
  started_at: 0,
  finished_at: 1,
});

const idle = {
  rounds: {},
  counts: { conversations: 12, corrections: 0, problems: 0, pairs: 0 },
  gates: {},
  overlap: {},
  example: { prompt: "What is 2 + 2?", answer: "4" },
};
const running = { ...idle, rounds: { "1": { status: "running", note: "" } } };

const queue = (step: number) => ({
  job: step ? { id: "j1", kind: "post_round", label: "Round 1 · Practice answers · LoRA rank 16", note: "", params: {}, status: "running", eta_s: 120, created_at: 0, error: "" } : null,
  step,
  total: step ? 100 : 0,
  phase: "",
  lr: null,
  eta_s: step ? 120 : null,
  paused: false,
  blocked: null,
  gate: { reason: null, resumes_at: null, supported: true },
  stopping: false,
  up_next: [],
  plan: {},
});

async function mockApi(page: Page) {
  const state = { versions: [version(1, true)], statusCalls: 0, rounds: [] as unknown[], imports: 0, queued: false };
  await page.route("**/versions", (route) => route.fulfill(json(state.versions)));
  await page.route("**/post-training/options**", (route) =>
    route.fulfill(
      json({
        leash: [{ name: "Medium", beta: 0.1, note: "A balance. A good default." }],
        lora_ranks: [4, 8, 16, 32],
        lora_blocker: null,
        lora_default: false,
        teachers: [],
        default_teacher: null,
      }),
    ),
  );
  await page.route("**/queue", (route) => route.fulfill(json(queue(state.queued ? 40 : 0))));
  await page.route("**/queue/outlook", (route) => route.fulfill(json({ outlook: "Starts now" })));
  await page.route("**/post-training/status", (route) => {
    state.statusCalls += 1;
    return route.fulfill(json({ ...job(`s${state.statusCalls}`, null), status: "queued", finished_at: null }, 202));
  });
  await page.route("**/jobs/*", (route) => route.fulfill(json({ ...job("x", null), status: "queued" })));
  await page.route("**/post-training/rounds", async (route) => {
    state.rounds.push(route.request().postDataJSON());
    state.queued = true;
    await route.fulfill(json({ id: "j1", label: "Round 1 · Practice answers · LoRA rank 16", round: 1, params: {} }));
  });
  await page.route("**/post-training/imports", async (route) => {
    state.imports += 1;
    state.versions = [...state.versions, version(2, false)];
    await route.fulfill(json(version(2, false)));
  });
  // Both status jobs finish on the stream: the first read shows an idle round 1, the read after queuing shows it running.
  await page.route("**/events", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: frame({ kind: "job", payload: job("s1", idle) }) + frame({ kind: "job", payload: job("s2", running) }),
    }),
  );
  return state;
}

test("a round with LoRA on is queued, shows its progress, and an imported version appears", async ({ page }) => {
  const state = await mockApi(page);
  await page.goto("/post-training/");
  await expect(page.getByRole("heading", { level: 1, name: "Post-training" })).toBeVisible();

  // The form is filled from /post-training/options, including the LoRA choice.
  await expect(page.getByText("Version 1 · best checkpoint from Pretraining")).toBeVisible();
  await expect(page.getByText("What it changes")).toBeVisible();
  await page.getByRole("switch", { name: "Train LoRA adapters" }).click();
  await expect(page.getByRole("radio", { name: "16" }).first()).toBeVisible();
  await page.getByRole("radiogroup", { name: "Adapter size (rank)" }).getByRole("radio", { name: "16" }).click();

  await page.getByRole("button", { name: "Add to the queue" }).click();
  await expect(page.getByText("Added to the queue: Round 1 · Practice answers · LoRA rank 16")).toBeVisible();
  expect(state.rounds).toEqual([expect.objectContaining({ round: 1, version: 1, lora: true, lora_rank: 16 })]);

  // The refreshed status says round 1 is running, and the queue's step count is its progress.
  await expect(page.getByTestId("step-line")).toContainText("step 40 of 100");
  await expect(page.getByRole("progressbar", { name: "Round progress" })).toHaveAttribute("aria-valuenow", "40");

  // Importing a model saves a new version, which is listed at once.
  await expect(page.getByTestId("version-2")).toHaveCount(0);
  await page.getByRole("button", { name: /^Import GPT-2/ }).click();
  await expect(page.getByTestId("version-2")).toContainText("Started from GPT-2 (124M)");
  expect(state.imports).toBe(1);
});

test("a round that has nothing to learn from yet says why and cannot be queued", async ({ page }) => {
  await mockApi(page);
  await page.goto("/post-training/");
  await page.getByRole("tab", { name: /Your votes/ }).click();
  await expect(page.getByText("Rate at least 2 pairs in Chat first.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Add votes round to the queue" })).toBeDisabled();
});

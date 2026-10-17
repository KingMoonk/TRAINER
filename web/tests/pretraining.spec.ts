import { expect, test, type Page } from "@playwright/test";

// Add to the queue against a mocked API (as chat.spec.ts does). The rest of the Pretraining page (sets, plan, size preview, learning-rate
// range test) is covered by src/app/pretraining/page.test.tsx.

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

type PlanBody = { dataset_ids: string[]; mode: string; rows: number };

async function mockApi(page: Page): Promise<{ queued: unknown[] }> {
  const state = { queued: [] as unknown[] };

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
    const ids = body.dataset_ids;
    const perStep = body.rows * 256;
    const steps = Math.ceil((PIECES * 2) / perStep);
    const reason = ids.length ? "" : "Choose some text first.";
    await route.fulfill(
      json({
        version: 2,
        parent: 1,
        mode: body.mode,
        can_continue: true,
        continue_blocked: null,
        estimating: false,
        pieces: PIECES,
        estimated: false,
        params: 1_000_000,
        context: 256,
        pieces_per_step: perStep,
        passes_over_text: Math.round(PIECES / perStep),
        steps,
        minutes: null,
        lr: 0.0006,
        enough: { ratio: 2, ratio_text: "2.00", headline: "This model could use about 10.0× more text than you have", verdict: "little" },
        expectation: { headline: "Expect the error to fall to about 3.1 bits per character", details: "Based on the size of the model and the text." },
        config: { steps, batch_size: body.rows, validation: 0.05, objective: "all" },
        can_add: !reason,
        reason,
        job: reason
          ? null
          : {
              kind: "pretrain",
              label: `Version 2 ${body.mode === "catch_up" ? "catch-up" : "from scratch"}`,
              params: { config: { steps }, datasets: ids, mode: body.mode, init_from: body.mode === "catch_up" ? "ck1" : null, version: 2, heldout_fraction: 0.05 },
            },
        outlook: "If you add it: starts now",
      }),
    );
  });
  await page.route("**/pretraining/preview**", (route) => route.fulfill(json(preview)));
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
  await page.route("**/events", (route) => route.abort());
  return state;
}

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

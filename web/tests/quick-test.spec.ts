import { expect, test, type Page } from "@playwright/test";
import { emit, fakeEvents, mockApi } from "./qp-mock";

const idle = { job: null, up_next: [], paused: false };
const current = { title: "Version 3", description: "4 blocks · 4 look-back heads · normal thinking room" };
const draft = { changes: ["more blocks (4 → 5)"], description: "5 blocks · 4 look-back heads · normal thinking room", cost: "1.2x bigger · 1.2x slower" };
const run = (steps: number, bpc: number) => ({ steps, bpc, curve: [[1, 4.2], [steps, bpc]], sample: `sample ${steps}` });
const results = {
  minutes: 3,
  shrink: false,
  dataset: "stories",
  verdict: "lower",
  delta: -0.2,
  threshold: 0.05,
  references: { uniform: 5.5, unigram: 3.9 },
  learned_context: true,
  checks: [
    { id: "starts_near_uniform", ok: true, detail: "5.4 vs 5.5" },
    { id: "finite", ok: true, detail: "ok" },
    { id: "memorises_one_batch", ok: true, detail: "4.2 -> 0.1" },
    { id: "error_falls", ok: false, detail: "3.0 -> 3.0" },
  ],
  broken: null,
  current: run(40, 3.1),
  draft: run(36, 2.9),
  finished_at: 2000,
};
const queued = { status: 202, body: { job_id: "job1", outlook: "Starts now", started_at: 1500 } };

async function open(page: Page, draftStaged: boolean, handlers: Parameters<typeof mockApi>[1] = {}) {
  await fakeEvents(page);
  const calls = await mockApi(page, {
    "GET /quicktest": () => ({ body: { current, draft: draftStaged ? draft : null, next_number: 4, last: null } }),
    "GET /queue": () => ({ body: idle }),
    ...handlers,
  });
  await page.goto("/quick-test/");
  await expect(page.getByRole("heading", { level: 1, name: "Quick test" })).toBeVisible();
  return calls;
}

test("quick test: shows the current design and the staged change with its cost", async ({ page }) => {
  await open(page, true);
  await expect(page.getByText("Version 3", { exact: true })).toBeVisible();
  await expect(page.getByText("Version 3 with your change")).toBeVisible();
  await expect(page.getByText("more blocks (4 → 5)")).toBeVisible();
  await expect(page.getByText("1.2x bigger · 1.2x slower")).toBeVisible();
  await expect(page.getByText("Nothing here is saved as a version")).toBeVisible();
  await page.getByText("How quick tests work").click();
  await expect(page.getByText("Fair by time, not steps.")).toBeVisible();
});

test("quick test: asks for a staged change and a guess before it runs", async ({ page }) => {
  const calls = await open(page, false);
  await expect(page.getByText("No change staged in the designer yet.")).toBeVisible();
  await page.getByRole("button", { name: "Run quick test" }).click();
  await expect(page.getByTestId("quicktest-status")).toHaveText("Stage a change in the designer first.");
  await page.unrouteAll();
  const staged = await mockApi(page, {
    "GET /quicktest": () => ({ body: { current, draft, next_number: 4, last: null } }),
    "GET /queue": () => ({ body: idle }),
  });
  await page.reload();
  await page.getByRole("button", { name: "Run quick test" }).click();
  await expect(page.getByTestId("quicktest-status")).toHaveText("Pick your guess first.");
  expect([...calls, ...staged].some((c) => c.startsWith("POST /quicktest/run"))).toBe(false);
});

test("quick test: runs with the chosen options, follows progress and shows the result", async ({ page }) => {
  let sent: unknown;
  const queue = { current: idle as object };
  await open(page, true, {
    "GET /queue": () => ({ body: queue.current }),
    "POST /quicktest/run": (route) => {
      sent = route.request().postDataJSON();
      queue.current = { ...idle, job: { id: "job1", kind: "quicktest", params: { guess: "Lower" } } };
      return queued;
    },
  });
  await page.getByRole("radio", { name: "Practice problems" }).click();
  await page.getByRole("radio", { name: "1 min" }).click();
  await page.getByRole("radio", { name: "Shrunk to ¼" }).click();
  await expect(page.getByText("Faster; a hint, not proof.")).toBeVisible();
  await page.getByRole("radio", { name: "Lower" }).click();
  await page.getByRole("button", { name: "Run quick test" }).click();

  await expect(page.getByTestId("quicktest-status")).toHaveText("Queued. Starts now.");
  expect(sent).toEqual({ dataset: "problems", minutes: 1, shrink: true, guess: "Lower" });
  await expect(page.getByRole("button", { name: "Running…" })).toBeDisabled();

  await emit(page, "train", { job: "other", type: "quicktest", which: "current", step: 9, total: 10 });
  await emit(page, "train", { job: "job1", type: "quicktest", which: "current", step: 10, total: 40 });
  await expect(page.getByRole("progressbar", { name: "Version 3" })).toHaveAttribute("aria-valuenow", "25");
  await expect(page.getByText("step 10")).toBeVisible();
  await expect(page.getByRole("progressbar", { name: "Your idea" })).toHaveAttribute("aria-valuenow", "0");

  queue.current = idle;
  await emit(page, "train", { job: "job1", type: "quicktest_done", results });
  await expect(page.getByText("After the same 3 minutes, your idea's error is lower than Version 3's. You guessed lower.")).toBeVisible();
  await expect(page.getByText("done · 40 steps")).toBeVisible();
  await expect(page.getByText("done · 36 steps")).toBeVisible();
  const checks = page.locator("[data-check]");
  await expect(checks.nth(0)).toContainText("✓");
  await expect(checks.nth(3)).toContainText("✗");
  await expect(checks.nth(3)).toContainText("Error falls");
  await expect(page.getByText("sample 40")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run again" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Save idea as version 4" })).toBeVisible();
});

test("quick test: a failed check names the part to look at, and a stopped run keeps nothing", async ({ page }) => {
  await open(page, true, { "POST /quicktest/run": () => queued });
  await page.getByRole("radio", { name: "Same" }).click();
  await page.getByRole("button", { name: "Run quick test" }).click();
  await expect(page.getByTestId("quicktest-status")).toHaveText("Queued. Starts now.");
  const broken = {
    ...results,
    verdict: null,
    broken: "memorises_one_batch",
    current: null,
    draft: null,
    checks: results.checks.map((c) => ({ ...c, ok: c.id === "memorises_one_batch" ? false : null })),
  };
  await emit(page, "train", { job: "job1", type: "quicktest_done", results: broken });
  await expect(page.getByText("A check failed: Can memorize one batch. Look at the parts you changed")).toBeVisible();
  await emit(page, "train", { job: "job1", type: "quicktest_done", results: { minutes: 3, stopped: true } });
  await expect(page.getByText("The quick test was stopped before it finished; nothing was kept.")).toBeVisible();
});

test("quick test: a refused start shows the server's reason, and a failed job says why", async ({ page }) => {
  let attempts = 0;
  await open(page, true, {
    "POST /quicktest/run": () => {
      attempts += 1;
      return attempts === 1
        ? { status: 400, body: { detail: "Your data has no plain text yet. Add some in Your data, or pick another tiny dataset." } }
        : queued;
    },
  });
  await page.getByRole("radio", { name: "A slice of your data" }).click();
  await page.getByRole("radio", { name: "Higher" }).click();
  await page.getByRole("button", { name: "Run quick test" }).click();
  await expect(page.getByTestId("quicktest-status")).toContainText("Your data has no plain text yet.");
  await page.getByRole("button", { name: "Run quick test" }).click();
  await expect(page.getByTestId("quicktest-status")).toHaveText("Queued. Starts now.");
  await emit(page, "job", { job: "job1", status: "failed", error: "out of memory" });
  await expect(page.getByTestId("quicktest-status")).toHaveText("The quick test failed: out of memory");
  await expect(page.getByRole("button", { name: "Run quick test" })).toBeEnabled();
});

test("quick test: Save makes the next version and Discard clears the idea, each opening the designer", async ({ page }) => {
  const handlers = {
    "POST /quicktest/run": () => queued,
    "POST /quicktest/save": () => ({ status: 201, body: { number: 4 } }),
    "DELETE /quicktest/draft": () => ({ status: 204 }),
  };
  for (const [button, call] of [
    ["Save idea as version 4", "POST /quicktest/save"],
    ["Discard idea", "DELETE /quicktest/draft"],
  ] as const) {
    const calls = await open(page, true, handlers);
    await page.getByRole("radio", { name: "Same" }).click();
    await page.getByRole("button", { name: "Run quick test" }).click();
    await expect(page.getByTestId("quicktest-status")).toHaveText("Queued. Starts now.");
    await emit(page, "train", { job: "job1", type: "quicktest_done", results });
    await page.getByRole("button", { name: button }).click();
    await expect(page).toHaveURL(/\/builder\/$/);
    expect(calls).toContain(call);
    await page.unrouteAll();
  }
});

test("quick test: a result that finished while another page was open is shown on return", async ({ page }) => {
  await open(page, true, {
    "POST /quicktest/run": () => queued,
    "GET /quicktest": (route) => {
      const finished = new URL(route.request().url()).searchParams.get("since") === "1500";
      return { body: { current, draft, next_number: 4, last: finished ? results : null } };
    },
  });
  await page.getByRole("radio", { name: "Lower" }).click();
  await page.getByRole("button", { name: "Run quick test" }).click();
  await expect(page.getByTestId("quicktest-status")).toHaveText("Queued. Starts now.");
  await page.reload();
  await expect(page.getByText("After the same 3 minutes, your idea's error is lower than Version 3's. You guessed lower.")).toBeVisible();
});

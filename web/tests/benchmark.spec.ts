import { expect, test, type Page } from "@playwright/test";

// The API is mocked at the network edge: every benchmark read is a job, so each POST answers with a job record.
const job = (id: string, kind: string, status: string, result: unknown = null, error = "") => ({
  id,
  kind,
  status,
  progress: "",
  fraction: null,
  result,
  error,
  created_at: 0,
  started_at: 0,
  finished_at: 0,
  ...(status === "queued" ? { started_at: null, finished_at: null } : {}),
});

const suite = (version: number, bpc: number, accuracy: number, rate: number) => ({
  version,
  bpc,
  problems: { passed: Math.round(30 * rate), total: 30, rate },
  few_shot: { items: 40, chance: 0.3, accuracy, rows: [{ shots: 4, used: 4, correct: 12, accuracy }] },
});

const track = (checked3: boolean) => ({
  suite: "fake",
  sizes: { docs: 24, choice: 40, problems: 30 },
  dropped: {},
  rows: [
    { number: 1, name: "Version 1", lineage: "a", tokens: 200_000, result: suite(1, 2.6, 0.3, 0.1) },
    { number: 2, name: "Version 2", lineage: "a", tokens: 800_000, result: suite(2, 2.2, 0.35, 0.2) },
    { number: 3, name: "Version 3", lineage: "b", tokens: 500_000, result: checked3 ? suite(3, 2.4, 0.33, 0.15) : null },
  ],
});

async function mockRead(page: Page, rows: () => unknown) {
  await page.route("**/benchmark/read", (route) => route.fulfill({ status: 202, json: job("r1", "benchmark.read", "done", rows()) }));
}

test("a benchmark reads the track, checks the unchecked version and charts every design on its own chart", async ({ page }) => {
  let checked = false;
  await mockRead(page, () => track(checked));
  const runs: unknown[] = [];
  await page.route("**/benchmark/run", async (route) => {
    runs.push(route.request().postDataJSON());
    checked = true;
    await route.fulfill({ status: 202, json: job("run3", "benchmark.run", "done", suite(3, 2.4, 0.33, 0.15)) });
  });
  await page.goto("/benchmark/");

  await expect(page.getByRole("heading", { level: 1, name: "Benchmark" })).toBeVisible();
  await expect(page.getByText("1 of 3 trained versions have not been checked yet.")).toBeVisible();
  await expect(page.getByText("Every version gets the same 24 passages, 40 questions and 30 problems", { exact: false })).toBeVisible();
  // Three checks, each drawn by the shared chart, one line per design.
  for (const title of ["Error on held-back text", "Multiple-choice questions", "Checked problems"]) await expect(page.getByText(title, { exact: true })).toBeVisible();
  await expect(page.locator(".recharts-line")).toHaveCount(1 + 2 + 1); // only design A is checked: one line per chart, and the guessing line on the multiple-choice one
  await page.getByRole("button", { name: "Check the rest" }).click();
  await expect(page.getByText("Every trained version has been checked.")).toBeVisible();
  await expect(page.locator(".recharts-line")).toHaveCount(2 + 3 + 2); // both designs now: A and B on each chart, plus the guessing line
  await expect(page.getByText("Design A: versions 1, 2").first()).toBeVisible();
  await expect(page.getByText("Guessing gets 30%").first()).toBeVisible();
  expect(runs).toEqual([{ version: 3 }]);
  await expect(page.getByRole("button", { name: "Check the rest" })).toHaveCount(0);
});

test("the page shows a skeleton until the job's result event, and reads the job once in case the event came first", async ({ page }) => {
  await page.route("**/benchmark/read", (route) => route.fulfill({ status: 202, json: job("late", "benchmark.read", "queued") }));
  let release = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  await page.route("**/jobs/late", async (route) => {
    await gate;
    await route.fulfill({ json: job("late", "benchmark.read", "done", track(true)) });
  });
  await page.goto("/benchmark/");
  await expect(page.getByLabel("Loading the benchmark")).toBeVisible();
  release();
  await expect(page.getByText("Every trained version has been checked.")).toBeVisible();
  await expect(page.getByLabel("Loading the benchmark")).toHaveCount(0);
});

test("the result event ends the skeleton", async ({ page }) => {
  await page.route("**/benchmark/read", (route) => route.fulfill({ status: 202, json: job("ev", "benchmark.read", "queued") }));
  await page.route("**/jobs/ev", (route) => route.fulfill({ json: job("ev", "benchmark.read", "running") }));
  const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
  await page.route("**/events", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: frame({ kind: "job", payload: job("ev", "benchmark.read", "done", track(true)) }) }),
  );
  await page.goto("/benchmark/");
  await expect(page.getByText("Every trained version has been checked.")).toBeVisible();
});

test("a failed read says why and can be tried again", async ({ page }) => {
  let fail = true;
  await page.route("**/benchmark/read", (route) =>
    route.fulfill({ status: 202, json: fail ? job("bad", "benchmark.read", "failed", null, "OSError: held-back data is unreadable") : job("ok", "benchmark.read", "done", track(true)) }),
  );
  await page.goto("/benchmark/");
  await expect(page.getByText("The benchmark could not be read.")).toContainText("held-back data is unreadable");
  fail = false;
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByText("Every trained version has been checked.")).toBeVisible();
});

test("with no trained version the page says so and draws no chart", async ({ page }) => {
  await mockRead(page, () => ({ suite: "fake", sizes: { docs: 24, choice: 40, problems: 30 }, dropped: {}, rows: [] }));
  await page.goto("/benchmark/");
  await expect(page.getByText("No trained version yet.", { exact: false })).toBeVisible();
  await expect(page.locator(".recharts-line")).toHaveCount(0);
});

test("a version is asked the multiple-choice questions and the table compares it with guessing", async ({ page }) => {
  await mockRead(page, () => track(true));
  const asked: unknown[] = [];
  await page.route("**/benchmark/few-shot", (route) => {
    asked.push(route.request().postDataJSON());
    return route.fulfill({
      status: 202,
      json: job("fs", "benchmark.few_shot", "done", {
        items: 40,
        tasks: 2,
        held_back: false,
        chance: 0.3,
        rows: [
          { shots: 0, used: 0, correct: 11, accuracy: 0.275 },
          { shots: 4, used: 3.5, correct: 24, accuracy: 0.6 },
        ],
      }),
    });
  });
  await page.goto("/benchmark/");
  await expect(page.getByText("Not checked yet for this version.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Check it" }).click();
  await expect(page.getByRole("cell", { name: "24 of 40" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "4 (3.5 fit on average)" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "better", exact: true })).toBeVisible();
  await expect(page.getByText("There are no held-back multiple-choice problems yet", { exact: false })).toBeVisible();
  expect(asked).toEqual([{ version: 3 }]);
  await expect(page.getByRole("button", { name: "Check again" })).toBeVisible();
});

test("two versions get a verdict, or the reason there is none", async ({ page }) => {
  await mockRead(page, () => track(true));
  await page.route("**/benchmark/verdict**", (route) => {
    const url = new URL(route.request().url());
    const same = url.searchParams.get("a") === "1" && url.searchParams.get("b") === "2";
    return route.fulfill({ json: same ? { percent: 37.5, comparable: true, reason: "" } : { percent: null, comparable: false, reason: "The runs have different total lengths." } });
  });
  await page.goto("/benchmark/");
  await expect(page.getByText("Pick two versions to compare.")).toBeVisible();
  const pick = async (label: string, name: string) => {
    await page.getByRole("combobox", { name: label }).click();
    await page.getByRole("option", { name }).click();
  };
  await pick("Older version", "Version 1");
  await pick("Newer version", "Version 2");
  await expect(page.getByText("To reach the same error, Version 2 needs 63% of the training Version 1 needed.")).toBeVisible();
  await pick("Newer version", "Version 3");
  await expect(page.getByText("No verdict yet: The runs have different total lengths.")).toBeVisible();
});

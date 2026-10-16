import { expect, test, type Page } from "@playwright/test";

// Browser coverage is CI-only. Keep numeric expectations independent of lossView.
const prompt = "The largest planet in our solar system is Jupiter";

test("a version link uses that checkpoint without choosing the best", async ({ page }) => {
  const state = await mockLoss(page);
  await page.goto("/loss/?version=2");
  await expect.poll(() => state.requests.length).toBeGreaterThan(0);
  expect(state.requests[0].version).toBe(2);
  expect(state.bestRequests).toBe(0);
});

test("an unavailable version needs an explicit choice before falling back", async ({ page }) => {
  const state = await mockLoss(page);
  await page.goto("/loss/?version=999");
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Version 999 has no available trained checkpoint.");
  expect(state.requests).toHaveLength(0);
  expect(state.bestRequests).toBe(0);
  await page.getByRole("button", { name: "Choose another version" }).click();
  await expect.poll(() => state.requests.length).toBeGreaterThan(0);
});
const version = (number: number, checkpoint: string | null = `checkpoint-${number}`) => ({
  number, name: `version ${number}`, change: "Loss fixture", spec: {}, parent: null,
  status: checkpoint ? "trained" : "saved", best_checkpoint: checkpoint, scorecard: {}, verdict: "",
});
const prediction = () => ({
  tokens: ["The answer is ", "🪐é"], pos: 1, target: 5,
  pieces: ["first", "second", "third", "fourth", "fifth", "🪐é", "last"],
  probabilities: [0.30, 0.25, 0.18, 0.12, 0.08, 0.04, 0.03], grad_error: 1e-8,
});
type LossRequest = { version: number; text: string; pos: number; trained: boolean };
const job = (id: string, result: unknown, status = "done", error = "", kind = "inspect.loss") => ({
  id, kind, status, result, error, progress: "", fraction: null,
  created_at: 0, started_at: 0, finished_at: status === "queued" ? null : 1,
});
function gate() {
  let open!: () => void;
  const promise = new Promise<void>((resolve) => { open = resolve; });
  return { promise, open };
}

async function mockLoss(page: Page, versions = [version(1), version(2)]) {
  const state = {
    requests: [] as LossRequest[], bestRequests: 0, result: prediction(),
    failure: "", beforeLoss: undefined as (() => Promise<void>) | undefined,
  };
  const jobs = new Map<string, ReturnType<typeof job>>();
  await page.route("**/events", (route) => route.abort());
  await page.route("**/versions", (route) => route.fulfill({ json: versions }));
  await page.route("**/versions/best", (route) => {
    state.bestRequests += 1;
    const id = `best-${state.bestRequests}`;
    // Deliberately choose an older checkpoint, not the last item in /versions.
    jobs.set(id, job(id, versions.find((item) => item.best_checkpoint) ?? null, "done", "", "versions.best"));
    return route.fulfill({ status: 202, json: job(id, null, "queued", "", "versions.best") });
  });
  await page.route("**/inspect/loss", async (route) => {
    state.requests.push(route.request().postDataJSON() as LossRequest);
    const id = `loss-${state.requests.length}`;
    await state.beforeLoss?.();
    jobs.set(id, job(id, state.failure ? null : state.result, state.failure ? "failed" : "done", state.failure));
    await route.fulfill({ status: 202, json: job(id, null, "queued") });
  });
  await page.route("**/jobs/*", (route) => {
    const id = new URL(route.request().url()).pathname.split("/").at(-1)!;
    const record = jobs.get(id);
    return route.fulfill({
      status: record ? 200 : 404,
      json: route.request().method() === "DELETE" ? job(id, null, "cancelled") : record ?? { detail: "Unknown job" },
    });
  });
  return state;
}

const table = (page: Page) => page.getByRole("table", { name: "Next-token probabilities and gradients" });
const targetRow = (page: Page) => table(page).getByRole("row").filter({
  has: page.getByRole("rowheader", { name: "🪐é", exact: true }),
});

test("the best checkpoint shows a target outside the top five, one-hot gradients, and Unicode BPC", async ({ page }) => {
  const state = await mockLoss(page);
  await page.goto("/loss/");
  await expect(page.getByTestId("loss-nats")).toHaveText("3.22 nats");
  await expect(page.getByLabel("Version", { exact: true })).toHaveValue("1");
  expect(state.bestRequests).toBe(1);
  expect(state.requests).toEqual([{ version: 1, text: prompt, pos: -1, trained: true }]);
  await expect(table(page).getByRole("row")).toHaveCount(8);
  await expect(targetRow(page).getByRole("cell")).toHaveText(["0.04", "1", "−0.96"]);
  const first = table(page).getByRole("row").filter({ has: page.getByRole("rowheader", { name: "first", exact: true }) });
  await expect(first.getByRole("cell")).toHaveText(["0.30", "0", "+0.30"]);
  await expect(table(page).getByRole("row").filter({ hasText: "all 1 others" }).getByRole("cell"))
    .toHaveText(["0.03", "0 each", "+p each"]);
  // 🪐é is two Unicode code points, but three UTF-16 code units.
  await expect(page.getByText(/This piece,.*2 characters:/)).toBeVisible();
  await expect(page.getByTestId("loss-bpc")).toHaveText("2.32 bits per character");
});

test("zero target probability produces a finite clamped loss", async ({ page }) => {
  const state = await mockLoss(page);
  state.result.probabilities = [0.40, 0.25, 0.18, 0.10, 0.04, 0, 0.03];
  await page.goto("/loss/");
  await expect(page.getByTestId("loss-nats")).toHaveText("690.78 nats");
  await expect(page.getByTestId("loss-bpc")).toHaveText("498.29 bits per character");
  await expect(targetRow(page).getByRole("cell")).toHaveText(["0.0e+0", "1", "−1.00"]);
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
});

test("random and trained requests preserve the chosen version and submitted prompt", async ({ page }) => {
  const state = await mockLoss(page);
  await page.goto("/loss/");
  await expect(page.getByTestId("loss-nats")).toHaveText("3.22 nats");
  await page.getByLabel("Text including the next piece to predict").fill("A second prediction 🪐é");
  await page.getByRole("button", { name: "Inspect prediction", exact: true }).click();
  await expect.poll(() => state.requests.length).toBe(2);
  await expect(page.getByRole("region", { name: "Prediction results" })).toHaveAttribute("aria-busy", "false");
  state.result.probabilities = state.result.pieces.map(() => 1 / 7);
  await page.getByRole("button", { name: "Step 0 (random weights)", exact: true }).click();
  await expect(page.getByTestId("loss-nats")).toHaveText("1.95 nats");
  expect(state.requests.at(-1)).toEqual({ version: 1, text: "A second prediction 🪐é", pos: -1, trained: false });
  await expect(page.getByRole("button", { name: "Step 0 (random weights)", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "After training", exact: true }).click();
  await expect(page.getByTestId("loss-nats")).toHaveText("3.22 nats");
  // The trained query may be cached; an explicit inspection must still send trained: true.
  state.result = prediction();
  await page.getByRole("button", { name: "Inspect prediction", exact: true }).click();
  await expect.poll(() => state.requests.length).toBe(4);
  expect(state.requests.at(-1)).toEqual({ version: 1, text: "A second prediction 🪐é", pos: -1, trained: true });
});

test("switching versions cancels a pending job and ignores its late result", async ({ page }) => {
  const state = await mockLoss(page);
  const oldRead = gate(), releaseOld = gate(), oldDelivered = gate(), cancelled = gate();
  const oldResult = prediction();
  oldResult.pieces[5] = "old-target";
  await page.route("**/jobs/loss-1", async (route) => {
    if (route.request().method() === "DELETE") {
      cancelled.open();
      await route.fulfill({ json: job("loss-1", null, "cancelled") });
      return;
    }
    oldRead.open();
    await releaseOld.promise;
    await route.fulfill({ json: job("loss-1", oldResult) });
    oldDelivered.open();
  });
  try {
    await page.goto("/loss/");
    await oldRead.promise;
    await expect(page.getByLabel("Loading prediction")).toBeVisible();
    state.result = { ...prediction(), probabilities: [0.28, 0.20, 0.15, 0.10, 0.04, 0.20, 0.03] };
    state.result.pieces[5] = "new-target";
    await page.getByLabel("Version", { exact: true }).selectOption("2");
    await expect(page.getByTestId("loss-nats")).toHaveText("1.61 nats");
    await cancelled.promise;
    releaseOld.open();
    await oldDelivered.promise;
    // Flush browser rendering after the obsolete response, without a timed sleep.
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    await expect(page.getByLabel("Version", { exact: true })).toHaveValue("2");
    await expect(page.getByTestId("loss-nats")).toHaveText("1.61 nats");
    await expect(table(page).getByRole("rowheader", { name: "new-target", exact: true })).toBeVisible();
    await expect(table(page).getByRole("rowheader", { name: "old-target", exact: true })).toHaveCount(0);
    expect(state.requests.map((request) => request.version)).toEqual([1, 2]);
  } finally {
    releaseOld.open();
    await page.unrouteAll({ behavior: "wait" });
  }
});

test("a failed inspection can be retried with the same input", async ({ page }) => {
  const state = await mockLoss(page);
  state.failure = "The checkpoint could not be read.";
  await page.goto("/loss/");
  await expect(page.getByRole("main").getByRole("alert")).toContainText(state.failure);
  await expect(page.getByTestId("loss-nats")).toHaveCount(0);
  state.failure = "";
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(page.getByTestId("loss-nats")).toHaveText("3.22 nats");
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  expect(state.requests).toHaveLength(2);
  expect(state.requests[1]).toEqual(state.requests[0]);
});

test("without a checkpoint there is a training link and no inspection job", async ({ page }) => {
  const state = await mockLoss(page, [version(1, null)]);
  await page.goto("/loss/");
  await expect(page.getByText("No trained checkpoint yet. Train a version first, then inspect its predictions here.")).toBeVisible();
  await expect(page.getByRole("link", { name: "Open Pretraining", exact: true })).toHaveAttribute("href", /\/pretraining\/?$/);
  await expect(page.getByRole("button", { name: "Inspect prediction", exact: true })).toHaveCount(0);
  expect(state.requests).toEqual([]);
  expect(state.bestRequests).toBe(0);
});

test("long token pieces do not overflow the document at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const state = await mockLoss(page);
  state.result.pieces = state.result.pieces.map((piece) => piece.repeat(60));
  await page.goto("/loss/");
  await expect(page.getByTestId("loss-nats")).toBeVisible();
  await expect(table(page)).toBeVisible();
  const widths = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    document: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }));
  expect(widths.document).toBeLessThanOrEqual(widths.viewport);
  expect(widths.body).toBeLessThanOrEqual(widths.viewport);
});

for (const event of [
  { kind: "checkpoint", payload: { version: 1 } },
  { kind: "versions", payload: version(1) },
  { kind: "job", payload: { job: "queue-pretraining-1", status: "done" } },
]) {
  test(`${event.kind} refresh keeps rows mounted and API completion does not loop`, async ({ page }) => {
    await page.clock.install();
    // A controllable event transport; live.spec.ts covers the real SSE connection.
    await page.addInitScript(() => {
      class Events extends EventTarget {
        onopen: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;
        private receive = (event: Event) => {
          this.onmessage?.(new MessageEvent("message", { data: JSON.stringify((event as CustomEvent).detail) }));
        };
        constructor() {
          super();
          window.addEventListener("loss-fixture-event", this.receive);
          queueMicrotask(() => this.onopen?.(new Event("open")));
        }
        close() { window.removeEventListener("loss-fixture-event", this.receive); }
      }
      window.EventSource = Events as unknown as typeof EventSource;
    });
    const state = await mockLoss(page);
    const release = gate(), started = gate();
    await page.goto("/loss/");
    await expect(page.getByTestId("loss-nats")).toHaveText("3.22 nats");
    const original = await targetRow(page).elementHandle();
    expect(original).not.toBeNull();
    state.result = { ...prediction(), probabilities: [0.28, 0.23, 0.16, 0.12, 0.08, 0.10, 0.03] };
    state.beforeLoss = async () => { started.open(); await release.promise; };
    try {
      await page.evaluate((value) => {
        window.dispatchEvent(new CustomEvent("loss-fixture-event", { detail: value }));
        window.dispatchEvent(new CustomEvent("loss-fixture-event", { detail: value }));
      }, event);
      await page.clock.fastForward(750);
      await started.promise;
      await expect(page.getByRole("region", { name: "Prediction results" })).toHaveAttribute("aria-busy", "true");
      await expect(page.getByTestId("loss-nats")).toHaveText("3.22 nats");
      expect(await original!.evaluate((element) => element.isConnected)).toBe(true);
      release.open();
      await expect(page.getByTestId("loss-nats")).toHaveText("2.30 nats");
      expect(await targetRow(page).evaluate((element, before) => element === before, original!)).toBe(true);
      await page.evaluate((payload) => {
        window.dispatchEvent(new CustomEvent("loss-fixture-event", { detail: { kind: "job", payload } }));
      }, job("loss-2", state.result));
      await page.clock.fastForward(1_000);
      expect(state.requests).toHaveLength(2);
      expect(state.bestRequests).toBe(2);
    } finally {
      release.open();
      await page.unrouteAll({ behavior: "wait" });
    }
  });
}

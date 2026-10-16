import { expect, test, type Page, type Route } from "@playwright/test";

for (const [path, endpoint] of [["trace", "trace"], ["value-map", "value-map"]]) {
  test(`${path}: carries the linked version into the inspection request`, async ({ page }) => {
    let body: { version: number } | undefined;
    await page.route("**/events", (route) => route.abort());
    await page.route(`**/inspect/${endpoint}`, (route) => {
      body = route.request().postDataJSON();
      return route.fulfill({ status: 202, json: job(`inspect.${endpoint}`, "done", path === "trace" ? TRACE : VALUE_MAP) });
    });
    await page.goto(`/${path}/?version=7`);
    await expect.poll(() => body?.version).toBe(7);
  });
}

// The Trace and Value map pages against mocked job endpoints: tests/serve-out.mjs has no API, so each test fulfils the POST, and
// the result reaches the page the way it does against the real server: on GET /events, or from GET /jobs/{id}.

const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
const job = (kind: string, status: string, result: unknown = null, error = "") => ({
  id: "j1",
  kind,
  status,
  progress: "",
  fraction: null,
  result,
  error,
  created_at: 0,
  started_at: null,
  finished_at: null,
});
const json = (route: Route, body: unknown, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

const TRACE = {
  tokens: ["The", " largest", " planet", " in", " our", " solar", " system", " is"],
  pos: 7,
  rows: [
    { stage: "embedding", kind: "embedding", norm: 1.0, rel_change: null, cosine: null, top_id: 1, top_token: " the", top_p: 0.1, approximate: true },
    { stage: "block 1 mixer", kind: "mixer", norm: 2.0, rel_change: 0.5, cosine: 0.9, top_id: 2, top_token: " big", top_p: 0.2, approximate: true },
    { stage: "block 1 channel", kind: "channel", norm: 3.0, rel_change: 0.25, cosine: 0.8, top_id: 3, top_token: " one", top_p: 0.3, approximate: true },
    { stage: "output", kind: "output", norm: 4.0, rel_change: 0.125, cosine: 0.7, top_id: 4, top_token: " Jupiter", top_p: 0.91, approximate: false },
  ],
};

const VALUE_MAP = { d_model: 256, heads: 4, blocks: { "0": 4, "1": 4 }, block: 0, head: 0, singular: [3, 1.5, 0.7, 0.1] };

async function expectNoSideScroll(page: Page) {
  const overflow = await page.locator("main").evaluate((el) => el.scrollWidth - el.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
}

test.describe("Trace", () => {
  test("shows a skeleton, then the result that arrives on /events", async ({ page }) => {
    await page.route("**/events", (route) =>
      route.fulfill({ status: 200, contentType: "text/event-stream", body: frame({ kind: "job", payload: job("inspect.trace", "done", TRACE) }) }),
    );
    let body: unknown;
    await page.route("**/inspect/trace", async (route) => {
      body = route.request().postDataJSON();
      await new Promise((resolve) => setTimeout(resolve, 400));
      await json(route, job("inspect.trace", "queued"), 202);
    });
    await page.goto("/trace/");
    await expect(page.getByRole("status", { name: "Loading the trace" })).toBeVisible();

    const rows = page.getByRole("row");
    await expect(page.getByRole("row", { name: /Embedding/ })).toBeVisible();
    await expect(page.getByRole("status", { name: "Loading the trace" })).toHaveCount(0);
    expect(body).toEqual({ text: "The largest planet in our solar system is", pos: -1 });
    await expect(page.getByLabel("Tokens")).toHaveText("The largest planet in our solar system is");
    // embedding has no "before": its change and cosine are dashes; every early guess carries the approximation mark
    await expect(page.getByRole("row", { name: /Embedding/ })).toContainText("—");
    await expect(page.getByRole("row", { name: /block 1 mixer/ })).toContainText("“ big”  0.20 ≈");
    await expect(page.getByRole("row", { name: /Output/ })).toContainText("“ Jupiter”  0.91");
    await expect(page.getByRole("row", { name: /Output/ })).not.toContainText("≈");
    await expect(rows).toHaveCount(1 + 4);
    await expect(page.getByText("Only the output row is the model's real prediction.")).toBeVisible();
  });

  test("reads GET /jobs/{id} when the event stream is down", async ({ page }) => {
    await page.route("**/inspect/trace", (route) => json(route, job("inspect.trace", "running"), 202));
    await page.route("**/jobs/j1", (route) => json(route, job("inspect.trace", "done", TRACE)));
    await page.goto("/trace/");
    await expect(page.getByRole("row", { name: /Output/ })).toContainText("“ Jupiter”  0.91");
  });

  test("a new prompt runs a new trace", async ({ page }) => {
    const texts: string[] = [];
    await page.route("**/inspect/trace", (route) => {
      texts.push(route.request().postDataJSON().text);
      return json(route, job("inspect.trace", "done", TRACE), 202);
    });
    await page.goto("/trace/");
    await expect(page.getByRole("row", { name: /Output/ })).toBeVisible();
    await page.getByLabel("Prompt").fill("Hello there");
    await page.getByRole("button", { name: "Trace" }).click();
    await expect.poll(() => texts).toEqual(["The largest planet in our solar system is", "Hello there"]);
  });

  test("a failed job says why, and an engine that cannot run says numbers appear later", async ({ page }) => {
    let calls = 0;
    await page.route("**/inspect/trace", (route) =>
      json(route, job("inspect.trace", "failed", null, ++calls === 1 ? "NotImplementedError: " : "ValueError: That prompt is empty."), 202),
    );
    await page.goto("/trace/");
    await expect(page.locator("main").getByRole("alert")).toHaveText("Numbers appear when the engine has run your prompt.");
    await page.getByLabel("Prompt").fill("again");
    await page.getByRole("button", { name: "Trace" }).click();
    await expect(page.locator("main").getByRole("alert")).toHaveText("That prompt is empty.");
  });

  test("stays readable on a 375 px screen", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.route("**/inspect/trace", (route) => json(route, job("inspect.trace", "done", TRACE), 202));
    await page.goto("/trace/");
    const output = page.getByRole("row", { name: /Output/ });
    await expect(output).toBeVisible();
    // the column heads are hidden, so each cell names itself
    await expect(output.getByText("Relative change ‖Δx‖ / ‖x‖")).toBeVisible();
    await expect(output.getByText("“ Jupiter”  0.91")).toBeVisible();
    await expectNoSideScroll(page);
    const box = (await output.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(375);
  });
});

test.describe("Value map", () => {
  test("shows a skeleton, then the funnel and the singular values", async ({ page }) => {
    let body: unknown;
    await page.route("**/inspect/value-map", async (route) => {
      body = route.request().postDataJSON();
      await new Promise((resolve) => setTimeout(resolve, 400));
      await json(route, job("inspect.value_map", "done", VALUE_MAP), 202);
    });
    await page.goto("/value-map/");
    await expect(page.getByRole("status", { name: "Loading the value map" })).toBeVisible();
    await expect(page.getByRole("heading", { level: 2 })).toHaveText("head_dim = 256 ÷ 4 = 64");
    expect(body).toEqual({ heads: 4, block: 0, head: 0 });
    await expect(page.getByText("d_model 256 · heads")).toBeVisible();
    await expect(page.getByText("Per head W_V and W_O hold 32,768 numbers; all 4 heads together hold 131,072 = 2 · 256², whatever the head count.")).toBeVisible();
    await expect(page.getByText("Measured on this model, block 1 head 1: the rank of W_V · W_O is 4, at most 64.")).toBeVisible();
    await expect(page.getByRole("img", { name: "4 singular values" })).toBeVisible();
    await expect(page.getByRole("radio", { name: "4", exact: true })).toBeChecked();
  });

  test("choosing heads, a block and a head asks the server again", async ({ page }) => {
    const bodies: unknown[] = [];
    await page.route("**/inspect/value-map", (route) => {
      const body = route.request().postDataJSON();
      bodies.push(body);
      // the model has 4 heads: asking for another count gives no singular values, as in the Tk page
      const result = body.heads === 4 ? VALUE_MAP : { ...VALUE_MAP, singular: null };
      return json(route, job("inspect.value_map", "done", result), 202);
    });
    await page.goto("/value-map/");
    await page.getByRole("radio", { name: "Block 2" }).click();
    await page.getByRole("radio", { name: "Head 3" }).click();
    await expect(page.getByText("block 2 head 3")).toBeVisible();
    await page.getByRole("radio", { name: "8", exact: true }).click();
    await expect(page.getByRole("heading", { level: 2 })).toHaveText("head_dim = 256 ÷ 8 = 32");
    await expect(page.getByText("Pick 4 heads to see this model's measured singular values; it has 4 heads per block.")).toBeVisible();
    expect(bodies).toEqual([
      { heads: 4, block: 0, head: 0 },
      { heads: 4, block: 1, head: 0 },
      { heads: 4, block: 1, head: 2 },
      { heads: 8, block: 0, head: 0 },
    ]);
  });

  test("a head count that does not divide d_model is said so", async ({ page }) => {
    await page.route("**/inspect/value-map", (route) => json(route, job("inspect.value_map", "done", { ...VALUE_MAP, d_model: 250 }), 202));
    await page.goto("/value-map/");
    await expect(page.locator("main").getByRole("alert")).toHaveText("250 does not split into 4 equal heads.");
  });

  test("stays readable on a 375 px screen", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.route("**/inspect/value-map", (route) => json(route, job("inspect.value_map", "done", VALUE_MAP), 202));
    await page.goto("/value-map/");
    await expect(page.getByText("Measured on this model")).toBeVisible();
    await expectNoSideScroll(page);
    // the diagram scrolls inside its own box instead of shrinking its labels to nothing
    const funnel = page.getByLabel("Funnel diagram");
    expect(await funnel.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
    const box = (await funnel.boundingBox())!;
    expect(box.x + box.width).toBeLessThanOrEqual(375);
  });
});

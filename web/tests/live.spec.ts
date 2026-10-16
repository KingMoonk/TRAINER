import { expect, test } from "@playwright/test";

test("Make data loads real rules, previews without saving, and stores checked problems", async ({ page, request }) => {
  const before = await (await request.get("/data/problems/rows?limit=1000")).json();
  await page.goto("/make-data/");
  await page.getByLabel("Rule", { exact: true }).selectOption("reverse");
  await page.getByRole("button", { name: "Try 5 first" }).click();
  await expect(page.getByText("Preview, nothing saved")).toBeVisible();
  const previewed = await (await request.get("/data/problems/rows?limit=1000")).json();
  expect(previewed.rows).toEqual(before.rows);
  await page.getByLabel("How many", { exact: true }).fill("2");
  const savedResponse = page.waitForResponse((response) => new URL(response.url()).pathname === "/data/problems/generate" && response.request().postDataJSON()?.store === true);
  await page.getByRole("button", { name: "Generate problems", exact: true }).click();
  const saved = await (await savedResponse).json();
  expect(saved.checked).toBe(saved.passed);
  expect(saved.added).toBeGreaterThan(0);
  const after = await (await request.get("/data/problems/rows?limit=1000")).json();
  expect(after.rows.length).toBe(before.rows.length + saved.added);
});

// Runs against `python -m studio_api --fake` serving web/out (the "live" project in playwright.config.ts; API_LIVE=1).
// The shell opens the real GET /events stream on the same origin, so this is the one test that crosses the wire.

test("the status strip goes from Connecting to Connected against the API", async ({ page }) => {
  // Hold the stream back so the first state is observable, then let the real server answer.
  await page.route("**/events", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    await route.continue();
  });
  await page.goto("/queue/");
  const strip = page.getByRole("status", { name: "Status" });
  await expect(strip).toHaveAttribute("data-connection", "connecting");
  await expect(strip).toContainText("Connecting");
  await expect(strip).toHaveAttribute("data-connection", "online");
  await expect(strip).toContainText("Connected");
});

test("the API serves the schema the client types come from", async ({ request }) => {
  const schema = await (await request.get("/openapi.json")).json();
  expect(schema.paths).toHaveProperty("/events");
  expect(schema.paths).toHaveProperty("/jobs/{job_id}");
});

test("Loss renders trained and random predictions from the real fake API", async ({ page, request }) => {
  const saved = await request.post("/versions", { data: { spec: {}, change: "Loss live smoke" } });
  expect(saved.status()).toBe(201);
  const version = await saved.json();
  // Plain --fake starts empty and never trains a checkpoint. Only make this real saved
  // version selectable in the browser; /versions/best, /inspect/loss, /jobs and /events
  // remain real server calls. FakeServices.inspect_loss does not require a checkpoint.
  await page.route("**/versions", async (route) => {
    const response = await route.fetch();
    const versions = await response.json();
    await route.fulfill({ response, json: versions.map((item: { number: number }) => item.number === version.number
      ? { ...item, best_checkpoint: "fake-live-checkpoint", status: "trained" } : item) });
  });
  const trainedResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/inspect/loss" && response.request().method() === "POST");
  await page.goto("/loss/");
  const trained = await trainedResponse;
  expect(trained.status()).toBe(202);
  expect(trained.request().postDataJSON()).toEqual({
    version: version.number, text: "The largest planet in our solar system is Jupiter", pos: -1, trained: true,
  });
  const submitted = await trained.json();
  expect(submitted.kind).toBe("inspect.loss");
  await expect(page.getByTestId("loss-nats")).toHaveText("0.25 nats");
  await expect(page.getByTestId("loss-bpc")).toHaveText("0.18 bits per character");
  const completed = await request.get(`/jobs/${submitted.id}`);
  expect(completed.ok()).toBe(true);
  const record = await completed.json();
  expect(record.status).toBe("done");
  expect(record.result.target).toBe(3);
  expect(record.result.probabilities[3]).toBeCloseTo(0.78);

  const randomResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/inspect/loss" && response.request().postDataJSON()?.trained === false);
  await page.getByRole("button", { name: "Step 0 (random weights)", exact: true }).click();
  const random = await randomResponse;
  expect(random.status()).toBe(202);
  expect(random.request().postDataJSON()).toMatchObject({ version: version.number, trained: false, pos: -1 });
  await expect(page.getByTestId("loss-nats")).toHaveText("2.48 nats");
  await expect(page.getByTestId("loss-bpc")).toHaveText("1.79 bits per character");
});

import { expect, test } from "@playwright/test";

test("Make data loads real rules, previews without saving, and stores checked problems", async ({ page, request }) => {
  const before = await (await request.get("/data/problems/rows?limit=1000")).json();
  await page.goto("/your-data/?kind=problems");
  await page.getByText("Make extra data", { exact: true }).click();
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

test("the Builder edits a part and saves a version against the API", async ({ page, request }) => {
  // The live specs share one workspace, so other specs may already have saved versions: the new one is the next number.
  const existing: { number: number }[] = await (await request.get("/versions")).json();
  const next = Math.max(0, ...existing.map((item) => item.number)) + 1;
  await page.goto("/builder/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Build your model");
  const tray = page.getByTestId("tray");
  await expect(tray).toContainText(existing.length ? "No changes" : "Not saved yet");

  // Pick a different position module: the picker's menu comes from /design/part-options.
  await page.getByRole("button", { name: "Advanced modules", exact: true }).click();
  const picker = page.getByTestId("picker-position");
  const before = await picker.textContent();
  await picker.click();
  const other = page.getByRole("option", { disabled: false }).filter({ hasNotText: before ?? "" }).first();
  await other.click();
  await expect(picker).not.toHaveText(before ?? "");
  await expect(tray).toContainText("Not saved · only this page sees these");

  await page.getByTestId("save").click();
  await expect(tray).toContainText("No changes");
  // The save makes the page read its design again, which can remount the closed history section: open it until the new version shows.
  const summary = page.locator("summary").filter({ hasText: "Version history & results" });
  await expect(async () => {
    if (!(await summary.evaluate((s) => (s.parentElement as HTMLDetailsElement).open))) await summary.click();
    await expect(page.getByTestId(`version-${next}`)).toBeVisible({ timeout: 1_000 });
  }).toPass();
});

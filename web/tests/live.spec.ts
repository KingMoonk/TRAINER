import { expect, test } from "@playwright/test";

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

test("the Builder edits a part and saves a version against the API", async ({ page }) => {
  await page.goto("/builder/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Build your model");
  const tray = page.getByTestId("tray");
  await expect(tray).toContainText("Not saved yet");
  await expect(tray).toContainText("Save as version 1");

  // Pick a different position module: the picker's menu comes from /design/part-options.
  const picker = page.getByTestId("picker-position");
  const before = await picker.textContent();
  await picker.click();
  const other = page.getByRole("option", { disabled: false }).filter({ hasNotText: before ?? "" }).first();
  await other.click();
  await expect(picker).not.toHaveText(before ?? "");
  await expect(tray).toContainText("Not saved · only this page sees these");

  await page.getByTestId("save").click();
  await expect(page.getByTestId("version-1")).toBeVisible();
  await expect(tray).toContainText("No changes");
});

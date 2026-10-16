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

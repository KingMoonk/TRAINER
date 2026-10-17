import { expect, test, type Page } from "@playwright/test";
import { ROUTES, areaHome } from "../src/lib/routes";

// A 375 px screen (iPhone SE/mini width): the document must not scroll sideways on any page.
test.use({ viewport: { width: 375, height: 812 } });

const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;

async function sideways(page: Page) {
  return page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
}

for (const route of ROUTES) {
  test(`${route.href}: the document does not scroll sideways at 375 px`, async ({ page }) => {
    await page.goto(`${route.href}/`);
    await expect(page.getByRole("status", { name: "Status" })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
    expect(await sideways(page)).toBe(0);
  });
}

test("a status strip full of long values still fits, and the connection light stays", async ({ page }) => {
  await page.route("**/events", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: frame({
        kind: "status",
        payload: {
          version: "a version with a rather long name indeed",
          queue: { state: "running", running: 1, waiting: 12 },
          resources: { device: "NVIDIA GeForce RTX 4090 Laptop GPU", diskFreeGb: 120 },
        },
      }),
    }),
  );
  await page.goto("/queue/");
  const strip = page.getByRole("status", { name: "Status" });
  await expect(strip).toContainText("a version with a rather long name indeed");
  await expect(strip).toContainText("RTX 4090");
  await expect(strip.getByText(/Connected|Offline/)).toBeVisible();
  expect(await sideways(page)).toBe(0);
});

test("the sidebar links stay reachable on a phone", async ({ page }) => {
  await page.goto("/queue/");
  const nav = page.getByRole("navigation", { name: "Studio" });
  await expect(nav.getByRole("link", { name: "Data", exact: true })).toBeVisible();
  await nav.getByRole("link", { name: "Data", exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`${areaHome("data").href}/?$`));
});

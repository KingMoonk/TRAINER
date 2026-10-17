import { expect, test } from "@playwright/test";
import { AREAS, ROUTES, areaHome } from "../src/lib/routes";

// The run pages come from #342 and #343; until they land these routes have no page yet.
const PENDING = ["runs", "new_run", "run_view"];

for (const route of ROUTES.filter((r) => !PENDING.includes(r.route))) {
  test(`${route.href}: shows its title and marks its area in the sidebar`, async ({ page }) => {
    await page.goto(`${route.href}/`);
    await expect(page.getByRole("heading", { level: 1, name: route.title })).toBeVisible();

    const nav = page.getByRole("navigation", { name: "Studio" });
    const home = areaHome(route.area);
    const area = AREAS.find((a) => a.id === route.area)!;
    await expect(nav.getByRole("link", { name: area.title, exact: true })).toHaveAttribute("aria-current", "page");
    await expect(nav.getByRole("link", { name: area.title, exact: true })).toHaveAttribute("href", new RegExp(`^${home.href}/?$`));
    await expect(nav.locator("a[aria-current=page]")).toHaveCount(1);
  });
}

test("the app opens on the Builder", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/builder\/$/);
});

test("the status strip says Offline when the event stream is unreachable", async ({ page }) => {
  await page.goto("/queue/");
  const strip = page.getByRole("status", { name: "Status" });
  await expect(strip).toContainText("Offline");
  await expect(strip).toContainText("Nothing is training");
});

test("status and problem_report events update the strip and raise a toast", async ({ page }) => {
  const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
  await page.route("**/events", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body:
        frame({
          kind: "status",
          payload: {
            version: "v3",
            queue: { state: "running", running: 1, waiting: 2 },
            resources: { device: "MPS", diskFreeGb: 120 },
          },
        }) +
        frame({ kind: "problem_report", payload: { items: [], added: 8, duplicates: 2, checked: 8, passed: 7 } }),
    }),
  );
  await page.goto("/queue/");
  const strip = page.getByRole("status", { name: "Status" });
  await expect(strip).toContainText("v3");
  await expect(strip).toContainText("Training, 2 waiting");
  await expect(strip).toContainText("MPS, 120 GB free");
  await expect(page.getByText("Added 8 problems; 2 already there; 1 of 8 checked did not pass their rule")).toBeVisible();
});

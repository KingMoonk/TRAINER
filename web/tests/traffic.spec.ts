import { expect, test } from "@playwright/test";
import { mockApi } from "./qp-mock";

const frame = (kind: string, payload: unknown) => `data: ${JSON.stringify({ kind, payload })}\n\n`;
const record = (id: number, over: object) => ({
  id, ts: 1_700_000_000 + id, node: "mac", direction: "in", remote: null, method: "GET", path: "/queue/jobs", route: "/queue/jobs", handler: "jobs",
  model: null, request_model: null, status: 200, ms: 12, state: "closed", streaming: false, error: "",
  request: { headers: {}, query: {}, body: null }, response: { headers: {}, body: null }, ...over,
});

test("the terminal button opens the traffic console on any page; it filters, expands a row and stays open across navigation", async ({ page }) => {
  const body = {
    type: "application/json", bytes: 40, json: { node: "BEAST", token: "••••••" },
  };
  // Playwright tries the newest route first, so mockApi's catch-all goes in before the stream route that must win over it.
  await mockApi(page, { "GET /traffic/rules": () => ({ body: { rules: [] } }) });
  await page.route("**/traffic", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body:
        frame("hello", { node: "mac", size: 2000, count: 3, started: 1_700_000_000 }) +
        frame("record", record(1, { path: "/queue/jobs" })) +
        frame("record", record(2, { method: "POST", path: "/collective/run1/join", status: 200, remote: "BEAST", request: { headers: {}, query: {}, body } })) +
        frame("record", record(3, { method: "POST", path: "/queue/jobs", status: 503 })),
    }),
  );

  await page.addInitScript(() => window.localStorage.setItem("settings.developerTools", "on")); // the button is hidden until Developer tools is on
  await page.goto("/queue/");
  await page.getByRole("button", { name: "API traffic console" }).locator("visible=true").click();
  const drawer = page.getByRole("region", { name: "API traffic" });
  await expect(drawer.getByTestId("traffic-row")).toHaveCount(3);

  await drawer.getByRole("button", { name: "5xx" }).click();
  await expect(drawer.getByTestId("traffic-row")).toHaveCount(1);
  await drawer.getByRole("button", { name: "All" }).click();

  await drawer.getByRole("button", { name: "POST /collective/run1/join" }).click();
  await expect(drawer.getByText("Request body")).toBeVisible();
  await expect(drawer.getByText("••••••")).toBeVisible();

  await page.getByRole("link", { name: "Settings" }).locator("visible=true").click();
  await expect(drawer).toBeVisible();

  await page.keyboard.press("Control+`");
  await expect(drawer).toBeHidden();
});

test("with Developer tools off (the default) there is no console button, and Ctrl+` still opens the console", async ({ page }) => {
  await mockApi(page, { "GET /traffic/rules": () => ({ body: { rules: [] } }) });
  await page.route("**/traffic", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: frame("hello", { node: "mac", size: 2000, count: 0, started: 1_700_000_000 }) }),
  );
  await page.goto("/queue/");
  await expect(page.getByRole("button", { name: "API traffic console" })).toHaveCount(0);

  await page.keyboard.press("Control+`");
  const drawer = page.getByRole("region", { name: "API traffic" });
  await expect(drawer).toBeVisible();
  await page.keyboard.press("Control+`");
  await expect(drawer).toBeHidden();
});

import { expect, test } from "@playwright/test";

// Real fake services and SSE, no intercepted responses. Setup uses the design
// API because the core editor is tracked separately in #148.
test("Builder follows a saved version into the queue and training without a reload", async ({ page, request }) => {
  expect((await request.post("/queue/pause")).ok()).toBeTruthy();
  const presetsResponse = await request.get("/design/presets");
  expect(presetsResponse.ok()).toBeTruthy();
  const presets = await presetsResponse.json();
  const saved = await request.post("/versions", { data: { spec: presets[0].spec, change: "Builder live fixture" } });
  expect(saved.ok()).toBeTruthy();
  const version = await saved.json();
  await page.goto("/builder/");
  const row = page.getByRole("list", { name: "Saved versions" }).getByRole("button", { name: new RegExp(`^Version ${version.number}\\b`) });
  await expect(row).toContainText("Saved");
  await row.evaluate((element) => element.setAttribute("data-original-row", "yes"));
  const enqueued = await request.post("/queue/jobs", { data: {
    kind: "pretrain", label: "Builder live training", params: { version: version.number, spec: presets[0].spec },
  } });
  expect(enqueued.ok()).toBeTruthy();
  await expect(row).toContainText("Queued");
  expect((await request.post("/queue/resume")).ok()).toBeTruthy();
  expect((await request.post("/queue/start")).ok()).toBeTruthy();
  await expect(row).toContainText("Training");
  await expect(row).toContainText("Trained");
  await expect(row).toHaveAttribute("data-original-row", "yes");
});

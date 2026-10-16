import { expect, test } from "@playwright/test";

// One real --fake API, no intercepted responses: the only fixture setup is an
// initial design (Builder belongs to Phase 3). All Phase 2 actions use the UI.
test("import text, queue pretraining, watch progress and see the trained version", async ({ page, request }, testInfo) => {
  const presetsResponse = await request.get("/design/presets");
  expect(presetsResponse.ok()).toBeTruthy();
  const presets = await presetsResponse.json();
  const saved = await request.post("/versions", { data: { spec: presets[0].spec, change: "Phase 2 browser fixture" } });
  expect(saved.ok()).toBeTruthy();
  const version = await saved.json();
  expect((await request.post("/queue/pause")).ok()).toBeTruthy();

  const source = `phase2-flow-${testInfo.retry}`;
  const text = `A small boat crossed the quiet lake for browser check ${testInfo.retry}.`;
  await page.goto("/your-data/");
  await page.getByRole("tab", { name: /^Text it reads/ }).click();
  await page.getByRole("button", { name: "Import files", exact: true }).click();
  await page.getByLabel("Choose files").setInputFiles({ name: `${source}.txt`, mimeType: "text/plain", buffer: Buffer.from(text) });
  await page.getByRole("button", { name: "Import 1 file", exact: true }).click();
  await expect(page.getByText(`${source}.txt: 1 added, 0 duplicates, 0 invalid`, { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Close", exact: true }).click();
  await expect(page.getByRole("main").getByText(text, { exact: true })).toBeVisible();

  await page.getByRole("navigation").getByRole("link", { name: "Pretraining", exact: true }).click();
  const sets = page.getByRole("list", { name: "Text sets" });
  const imported = sets.getByRole("checkbox", { name: new RegExp(source) });
  await expect(imported).toBeVisible();
  for (const checkbox of await sets.getByRole("checkbox").all()) {
    const selected = await checkbox.getAttribute("aria-checked") === "true";
    const wanted = (await checkbox.textContent())?.includes(source) ?? false;
    if (selected !== wanted) await checkbox.click();
  }
  await expect(imported).toHaveAttribute("aria-checked", "true");
  const add = page.getByRole("button", { name: "Add to the queue", exact: true });
  await expect(add).toBeEnabled();
  await add.click();
  await expect(page.getByText(/^Added to the queue:/)).toBeVisible();

  await page.getByRole("navigation").getByRole("link", { name: "Queue", exact: true }).click();
  await expect(page.getByRole("main").getByText(new RegExp(`Version ${version.number} from scratch`))).toBeVisible();
  await page.getByRole("main").getByRole("button", { name: "Resume", exact: true }).click();
  const start = page.getByRole("button", { name: "Start waiting jobs", exact: true });
  const progress = page.getByRole("progressbar", { name: "Job progress" });
  // A prior live test/retry may already have started the persistent fake runner.
  await expect.poll(async () => await start.isVisible() || await progress.isVisible()).toBeTruthy();
  if (await start.isVisible()) await start.click();
  await expect(progress).toBeVisible();
  const first = Number(await progress.getAttribute("aria-valuenow"));
  await expect.poll(async () => Number(await progress.getAttribute("aria-valuenow")), { intervals: [50] }).toBeGreaterThan(first);
  await expect(page.getByRole("main").getByText("Nothing is training.", { exact: true })).toBeVisible();
  await expect(page.getByRole("status", { name: "Status" })).toContainText(`${version.name} · trained`);
  await expect(page.getByRole("status", { name: "Status" })).toContainText("Nothing is training");
  await page.reload();
  await expect(page.getByRole("status", { name: "Status" })).toContainText(`${version.name} · trained`);
});

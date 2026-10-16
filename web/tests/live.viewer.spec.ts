import { expect, test, type Page } from "@playwright/test";

const sceneWith = (page: Page, expected: Record<string, string>) => page.waitForResponse((response) => {
  const url = new URL(response.url());
  return url.pathname === "/viewer/scene" && Object.entries(expected).every(([key, value]) => url.searchParams.get(key) === value);
});

test("Inside 3D draws a saved workspace design from the real viewer API", async ({ page, request }) => {
  const presets = await (await request.get("/design/presets")).json();
  const saved = await request.post("/versions", { data: { spec: presets[0].spec, change: "Viewer live fixture" } });
  expect(saved.ok()).toBeTruthy();
  const version = await saved.json();
  const sceneResponse = page.waitForResponse((response) => new URL(response.url()).pathname === "/viewer/scene");
  await page.goto(`/inside-3d/?version=${version.number}`);
  const response = await sceneResponse;
  expect(response.ok()).toBeTruthy();
  expect(new URL(response.url()).searchParams.get("version")).toBe(String(version.number));
  const scene = await response.json();
  await expect(page.getByText(scene.stats.info, { exact: true })).toBeVisible();
  await expect(page.getByRole("radio", { name: "Model", exact: true })).toBeChecked();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
});

test("Inside 3D asks the live viewer API again for each control and lists its snapshots", async ({ page, request }) => {
  const crashes: string[] = [];
  page.on("pageerror", (error) => crashes.push(error.message)); // a payload the page cannot draw throws inside the canvas
  const presets = await (await request.get("/design/presets")).json();
  const saved = await request.post("/versions", { data: { spec: presets[0].spec, change: "Viewer live controls" } });
  expect(saved.ok()).toBeTruthy();
  const version = String((await saved.json()).number);
  const snapshotsResponse = page.waitForResponse((response) => new URL(response.url()).pathname === "/viewer/snapshots");
  await page.goto(`/inside-3d/?version=${version}`);
  const snapshots = await snapshotsResponse;
  expect(snapshots.ok()).toBeTruthy();
  expect(new URL(snapshots.url()).searchParams.get("version")).toBe(version);
  expect(await snapshots.json()).toEqual([]); // --fake holds no snapshot files: there is nothing to replay
  await expect(page.getByText("No snapshots loaded.")).toBeVisible();

  const query = { version };
  const train = sceneWith(page, { ...query, mode: "Train" });
  await page.getByRole("radio", { name: "Train", exact: true }).click();
  const trained = await (await train).json();
  await expect(page.getByText(trained.stats.info, { exact: true })).toBeVisible();

  const head = sceneWith(page, { ...query, mode: "Train", level: "Head", headStage: "pattern" });
  await page.getByRole("radio", { name: "Head", exact: true }).click();
  await head;
  const values = sceneWith(page, { ...query, level: "Head", headStage: "values" });
  await page.getByRole("button", { name: "Next head step" }).click();
  expect((await (await values).json()).view.head_stage).toBe("values");

  const weights = sceneWith(page, { ...query, level: "Weights", tileKind: "weight" });
  await page.getByRole("radio", { name: "Weights", exact: true }).click();
  await weights;
  const gradient = sceneWith(page, { ...query, level: "Weights", tileKind: "gradient" });
  await page.getByRole("radio", { name: "gradient", exact: true }).click();
  expect((await (await gradient).json()).view.tile_kind).toBe("gradient");
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  await expect(page.locator("canvas")).toBeVisible();
  expect(crashes).toEqual([]);

  // The probe is a job on the real API: a version with no checkpoint fails it, and the page shows the server's reason.
  const probe = page.waitForRequest((request) => new URL(request.url()).pathname === "/viewer/probe");
  await page.getByRole("button", { name: "Run one real step", exact: true }).click();
  const posted = await probe;
  expect(posted.method()).toBe("POST");
  expect(posted.postDataJSON()).toEqual({ version: Number(version) });
  await expect(page.getByRole("main").getByRole("status")).toContainText("Train this version first");
  await expect(page.getByRole("button", { name: "Run one real step", exact: true })).toBeEnabled();
});

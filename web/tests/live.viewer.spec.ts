import { expect, test } from "@playwright/test";

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

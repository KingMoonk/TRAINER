import { expect, test } from "@playwright/test";
import fixture from "../src/fixtures/scene.json";
import { applyFrame, type SceneFrame, type ScenePayload } from "../src/lib/scene";
import { emit, fakeEvents, mockApi } from "./qp-mock";

const scenes = fixture.scenes as unknown as Record<string, { payload: ScenePayload; frames: Record<string, SceneFrame> }>;

test("viewer reads selected-version scenes, controls and a real probe job through the API", async ({ page }) => {
  await fakeEvents(page);
  const queries: URLSearchParams[] = [];
  let probeBody: unknown;
  let finished = false;
  await mockApi(page, {
    "GET /viewer/scene": (route) => {
      const query = new URL(route.request().url()).searchParams;
      queries.push(query);
      const level = query.get("level") ?? "Model";
      const variant = level === "Head" ? query.get("headStage") : level === "Weights" ? query.get("tileKind") : "-";
      const scene = scenes[`${query.get("snapshot") ?? 0}|${level}|${variant}`];
      const frame = scene.frames[query.get("mode") ?? "Build"];
      return { body: frame ? applyFrame(scene.payload, frame) : scene.payload };
    },
    "POST /viewer/probe": (route) => {
      probeBody = route.request().postDataJSON();
      return { status: 202, body: { id: "probe-7", kind: "viewer.probe", status: "queued", result: null } };
    },
    "GET /jobs/probe-7": () => ({ body: { id: "probe-7", kind: "viewer.probe", status: finished ? "done" : "running",
      result: finished ? { snapshot: 1, note: "API probe finished for version 7." } : null } }),
  });
  await page.goto("/inside-3d/?version=7");
  await expect.poll(() => queries.at(-1)?.get("version")).toBe("7");
  await page.getByRole("radio", { name: "Head", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Head step" })).toBeVisible();
  await page.getByRole("button", { name: "Next head step" }).click();
  await expect.poll(() => queries.at(-1)?.get("headStage")).toBe("values");
  await page.getByRole("radio", { name: "Train", exact: true }).click();
  await expect.poll(() => queries.at(-1)?.get("mode")).toBe("Train");
  await page.getByRole("button", { name: "Run one real step", exact: true }).click();
  await expect(page.getByRole("button", { name: "Running one step...", exact: true })).toBeDisabled();
  await expect.poll(() => probeBody).toMatchObject({ version: 7 });
  finished = true;
  await expect(page.getByRole("main").getByRole("status")).toContainText("API probe finished for version 7.");
  await expect.poll(() => queries.at(-1)?.get("snapshot")).toBe("1");
  const count = queries.length;
  await emit(page, "checkpoint", {});
  await expect.poll(() => queries.length).toBeGreaterThan(count);
});

test("an unavailable workspace shows the server's reason without substituting demo weights", async ({ page }) => {
  await fakeEvents(page);
  await mockApi(page, {
    "GET /viewer/scene": () => ({ status: 400, body: { detail: "There is no version 999." } }),
  });
  await page.goto("/inside-3d/?version=999");
  await expect(page.getByRole("main").getByRole("alert")).toHaveText("There is no version 999.");
  await expect(page.getByRole("button", { name: "Run one real step", exact: true })).toBeDisabled();
  await expect(page.locator("canvas")).toHaveCount(0);
});

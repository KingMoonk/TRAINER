import { expect, test } from "@playwright/test";
import type { Level, ScenePayload } from "../src/lib/scene";
import { emit, fakeEvents, mockApi } from "./qp-mock";

// The page is driven by mocked API replies here (the live run is live.viewer.spec.ts), so the scene is the smallest payload the
// page and canvas accept: the controls the level shows, the stats line that names the query, and no boxes.
const KEYS = ["activation", "channel", "token", "gate", "positive", "negative", "masked", "picked", "update", "mixer", "lane", "label", "label_selected", "background"];
const ROWS: Record<Level, string[]> = { Model: [], Window: [], Head: ["stage"], Weights: ["tile", "matrix"] };

function scene(query: URLSearchParams, count: number): ScenePayload {
  const level = (query.get("level") ?? "Model") as Level;
  const mode = query.get("mode") ?? "Build";
  const index = Number(query.get("snapshot") ?? 0);
  const rows = ["level", "mode", ...ROWS[level]];
  return {
    schema: 1, full: true, up: "z", removed_groups: [], cursor: {}, groups: {},
    palette: Object.fromEntries(KEYS.map((key) => [key, { color: "#6cb4ee", alpha: 1 }])),
    view: {
      level, mode, head_stage: query.get("headStage"), tile_kind: query.get("tileKind"), reduce_motion: false, show_numbers: true,
      levels: ["Model", "Window", "Head", "Weights"], modes: ["Build", "Train", "Ride a token"],
      head_stages: ["scores", "scale", "mask", "pattern", "values"], tile_kinds: ["weight", "gradient", "update"],
      controls: { rows, matrices: [], snapshots: { show: true, probe: "Run one real step", probe_enabled: true, note: "" }, toggles: [] },
    },
    camera: { target: [0, -3, 1], distance: 22, level },
    selection: { run: null, layer: 0, token: 0, head: 0, weight: null, row: 0, column: 0, block_id: "blk_000000", cell: [0, 0] },
    stats: { step: index, loss: null, info: `Mock | ${level} | ${mode} | snapshot ${index}`, training: {}, training_line: "", note: "" },
    design: { name: "Mock", d_model: 16, n_layer: 1, n_ctx: 32, gates: [] },
    playback: {
      show: count > 0, play: "Play", position: `Snapshot ${index + 1} of ${count}`, previous: index > 0, next: index + 1 < count,
      count, index, playing: false, pass_seconds: 3,
      pulse: { animate: false, blocks: 1, spacing: 3, speed: 2, z: 1, follow: false, token_x: 0, colors: { forward: [1, 1, 1, 1], backward: [1, 0.45, 0.08, 1] },
        idle_size: [0.3, 0.4, 0.3], sizes: {} },
    },
  } as unknown as ScenePayload;
}

const names = ["step_00000001.npz", "step_00000002.npz"];

test("viewer reads selected-version scenes, snapshots, controls and a real probe job through the API", async ({ page }) => {
  await fakeEvents(page);
  const queries: URLSearchParams[] = [];
  const snapshotQueries: URLSearchParams[] = [];
  let listed = names;
  let probeBody: unknown;
  await mockApi(page, {
    "GET /viewer/scene": (route) => {
      const query = new URL(route.request().url()).searchParams;
      queries.push(query);
      return { body: scene(query, listed.length) };
    },
    "GET /viewer/snapshots": (route) => {
      snapshotQueries.push(new URL(route.request().url()).searchParams);
      return { body: listed.map((name, index) => ({ index, name })) };
    },
    "POST /viewer/probe": (route) => {
      probeBody = route.request().postDataJSON();
      listed = [...names, "step_00000003.npz"];
      return { status: 202, body: { id: "probe-7", kind: "viewer.probe", status: "queued", result: null } };
    },
    // The job never reads as finished here: the page learns of its result from the /events message.
    "GET /jobs/probe-7": () => ({ body: { id: "probe-7", kind: "viewer.probe", status: "running", result: null } }),
  });
  await page.goto("/inside-3d/?version=7");
  await expect.poll(() => queries.at(-1)?.get("version")).toBe("7");
  await expect(page.getByText("Mock | Model | Build | snapshot 0", { exact: true })).toBeVisible();
  expect(snapshotQueries.at(-1)?.get("version")).toBe("7");

  await page.getByRole("radio", { name: "Window", exact: true }).click();
  await expect.poll(() => queries.at(-1)?.get("level")).toBe("Window");
  await page.getByRole("radio", { name: "Head", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Head step" })).toBeVisible();
  await page.getByRole("button", { name: "Next head step" }).click();
  await expect.poll(() => queries.at(-1)?.get("headStage")).toBe("values");
  await page.getByRole("radio", { name: "Weights", exact: true }).click();
  await page.getByRole("radio", { name: "gradient", exact: true }).click();
  await expect.poll(() => queries.at(-1)?.get("tileKind")).toBe("gradient");
  await page.getByRole("radio", { name: "Train", exact: true }).click();
  await expect.poll(() => queries.at(-1)?.get("mode")).toBe("Train");
  expect(queries.at(-1)?.get("version")).toBe("7");

  await page.getByRole("combobox", { name: "Snapshot" }).click();
  await page.getByRole("option", { name: names[1] }).click();
  await expect.poll(() => queries.at(-1)?.get("snapshot")).toBe("1");
  await expect(page.getByRole("combobox", { name: "Snapshot" })).toHaveText(names[1]);
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await expect(page.getByText("Mock | Weights | Train | snapshot 0", { exact: true })).toBeVisible(); // a scene read before may come from the cache

  await page.getByRole("button", { name: "Run one real step", exact: true }).click();
  await expect(page.getByRole("button", { name: "Running one step...", exact: true })).toBeDisabled();
  await expect.poll(() => probeBody).toEqual({ version: 7 }); // the server chooses the sentence
  const status = page.getByRole("main").getByRole("status");
  await expect(async () => {
    await emit(page, "job", { id: "probe-7", kind: "viewer.probe", status: "done", progress: "", error: "", created_at: 0,
      result: { snapshot: 2, note: "API probe finished for version 7." } });
    await expect(status).toContainText("API probe finished for version 7.", { timeout: 500 });
  }).toPass();
  await expect.poll(() => queries.at(-1)?.get("snapshot")).toBe("2");
  await expect(page.getByText("Snapshot 3 of 3")).toBeVisible();

  const count = queries.length;
  await emit(page, "checkpoint", {});
  await expect.poll(() => queries.length).toBeGreaterThan(count);
});

test("a failed probe job shows the server's reason", async ({ page }) => {
  await fakeEvents(page);
  await mockApi(page, {
    "GET /viewer/scene": (route) => ({ body: scene(new URL(route.request().url()).searchParams, 2) }),
    "GET /viewer/snapshots": () => ({ body: names.map((name, index) => ({ index, name })) }),
    "POST /viewer/probe": () => ({ status: 202, body: { id: "probe-8", kind: "viewer.probe", status: "queued", result: null } }),
    "GET /jobs/probe-8": () => ({ body: { id: "probe-8", kind: "viewer.probe", status: "running", result: null } }),
  });
  await page.goto("/inside-3d/?version=7");
  await page.getByRole("button", { name: "Run one real step", exact: true }).click();
  await expect(async () => {
    await emit(page, "job", { id: "probe-8", kind: "viewer.probe", status: "failed", progress: "", error: "There is no checkpoint to probe.", created_at: 0 });
    await expect(page.getByRole("main").getByRole("status")).toContainText("There is no checkpoint to probe.", { timeout: 500 });
  }).toPass();
});

test("an unavailable workspace shows the server's reason without substituting demo weights", async ({ page }) => {
  await fakeEvents(page);
  await mockApi(page, {
    "GET /viewer/scene": () => ({ status: 400, body: { detail: "There is no version 999." } }),
    "GET /viewer/snapshots": () => ({ status: 400, body: { detail: "There is no version 999." } }),
  });
  await page.goto("/inside-3d/?version=999");
  await expect(page.getByRole("main").getByRole("alert")).toHaveText("There is no version 999.");
  await expect(page.getByRole("button", { name: "Run one real step", exact: true })).toBeDisabled();
  await expect(page.locator("canvas")).toHaveCount(0);
});

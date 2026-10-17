import { expect, test } from "@playwright/test";
import type { Level, ScenePayload } from "../src/lib/scene";
import { emit, fakeEvents, mockApi } from "./qp-mock";

// The thermal look over a mocked scene: the query asks for it, the canvas draws it, the tile labels are on the page, and Follow
// training stays off until a run saves a snapshot. (viewer.spec.ts covers the controls; the live run is live.viewer.spec.ts.)
const KEYS = ["activation", "channel", "token", "gate", "positive", "negative", "masked", "picked", "update", "mixer", "lane", "connector", "label", "label_selected", "tile_title", "tile_dims", "tile_axis", "background"];
const names = ["step_00000001.npz", "step_00000002.npz", "step_00000003.npz"];

function array(shape: number[], fill: (i: number) => number) {
  return { shape, data: Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => fill(i)) };
}

function scene(query: URLSearchParams): ScenePayload {
  const level = (query.get("level") ?? "Model") as Level;
  const index = Number(query.get("snapshot") ?? 0);
  const thermal = query.get("thermal") === "true";
  const label = (text: string, kind: string, x: number, z: number) => ({ text, position: [x, 0, z], kind, selected: false, size: 0.2 });
  const tiles = thermal && level === "Head"
    ? [
        { id: "q", label: "Q", origin: [-3, 0, 0], size: [2, 3], selected: false, shape: [6, 4], range: [0, 1], values: array([6, 4], (i) => (i + index) / 30) },
        { id: "pattern", label: "A", origin: [2, 0, 0], size: [3, 3], selected: true, shape: [6, 6], range: [0, 1], values: array([6, 6], (i) => ((i % 6) + index) / 12), mask: array([6, 6], (i) => (i % 6 > Math.floor(i / 6) ? 1 : 0)) },
      ]
    : [];
  const labels = tiles.length ? [label("Q", "tile_title", -3, 2), label("6 × 4", "tile_dims", -3, -2), label("the", "tile_axis", 2, 2)] : [];
  return {
    schema: 1, full: true, up: "z", removed_groups: [], cursor: {},
    palette: Object.fromEntries(KEYS.map((key) => [key, { color: "#6cb4ee", alpha: 1 }])),
    view: {
      level, mode: "Build", head_stage: "pattern", tile_kind: "weight", thermal, reduce_motion: false, show_numbers: true,
      levels: ["Model", "Window", "Head", "Weights"], modes: ["Build", "Train", "Ride a token"],
      head_stages: ["scores", "scale", "mask", "pattern", "values"], tile_kinds: ["weight", "gradient", "update"],
      controls: { rows: ["level", "mode", ...(level === "Head" ? ["stage"] : [])], matrices: [], snapshots: { show: true, probe: "Run one real step", probe_enabled: true, note: "" }, toggles: ["numbers"] },
    },
    camera: { target: [0, 0, 0], distance: 20, level },
    selection: { run: null, layer: 0, token: 0, head: 0, weight: null, row: 0, column: 0, block_id: "blk_000000", cell: [0, 0] },
    stats: { step: index, loss: null, info: `Mock | ${level} | snapshot ${index}`, training: {}, training_line: "", note: "" },
    design: { name: "Mock", d_model: 16, n_layer: 1, n_ctx: 32, gates: [] },
    playback: {
      show: true, play: "Play", position: `Snapshot ${index + 1} of ${names.length}`, previous: index > 0, next: index + 1 < names.length,
      count: names.length, index, playing: false, pass_seconds: 3,
      pulse: { animate: false, blocks: 1, spacing: 3, speed: 2, z: 1, follow: false, token_x: 0, colors: { forward: [1, 1, 1, 1], backward: [1, 0.45, 0.08, 1] }, idle_size: [0.3, 0.4, 0.3], sizes: {} },
    },
    groups: tiles.length ? { matrix: { origin: [0, 0, 0], level: "Head", kind: "pattern", weight_name: null, shape: null, origin_cell: [0, 0], pitch: 0.4, caption: "colour = value | dark = masked", boxes: [], lines: [], labels, tiles } } : {},
  } as unknown as ScenePayload;
}

async function open(page: import("@playwright/test").Page) {
  await fakeEvents(page);
  const queries: URLSearchParams[] = [];
  await mockApi(page, {
    "GET /viewer/scene": (route) => {
      const query = new URL(route.request().url()).searchParams;
      queries.push(query);
      return { body: scene(query) };
    },
    "GET /viewer/snapshots": () => ({ body: names.map((name, index) => ({ index, name })) }),
  });
  return queries;
}

test("the thermal map is asked for by default, drawn with its labels and caption, and can be switched off", async ({ page }) => {
  const queries = await open(page);
  await page.goto("/inside-3d/?version=7");
  await expect.poll(() => queries.at(-1)?.get("thermal")).toBe("true");
  await page.getByRole("radio", { name: "Head", exact: true }).click();
  await expect(page.locator("canvas")).toBeVisible();
  await expect(page.getByText("6 × 4", { exact: true })).toBeAttached();
  await expect(page.getByText("colour = value | dark = masked")).toBeVisible();
  await page.getByRole("switch", { name: "Thermal map" }).click();
  await expect.poll(() => queries.at(-1)?.get("thermal")).toBe("false");
});

test("Follow training is off at load and shows the newest snapshot once on", async ({ page }) => {
  const queries = await open(page);
  await page.goto("/inside-3d/?version=7");
  await expect(page.getByText("Mock | Model | snapshot 0", { exact: true })).toBeVisible();
  const follow = page.getByRole("switch", { name: "Follow training" });
  await expect(follow).not.toBeChecked();
  await follow.click();
  await expect.poll(() => queries.at(-1)?.get("snapshot")).toBe("2");
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await expect(follow).not.toBeChecked(); // picking a snapshot ends it
  await expect.poll(() => queries.at(-1)?.get("snapshot")).toBe("1");
});

test("a run saving a snapshot turns Follow training on", async ({ page }) => {
  await open(page);
  await page.goto("/inside-3d/?version=7");
  await expect(page.getByRole("switch", { name: "Follow training" })).not.toBeChecked();
  await expect(async () => {
    await emit(page, "train", { job: "j1", type: "viz_snapshot", step: 50, path: "viz/step_00000050.npz", markers: [] });
    await expect(page.getByRole("switch", { name: "Follow training" })).toBeChecked({ timeout: 500 });
  }).toPass();
});

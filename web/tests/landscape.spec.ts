import { expect, test, type Page } from "@playwright/test";
import { ACCENTS } from "../src/lib/theme";

// The Loss landscape page against a mocked API. Browser coverage is CI-only.

const version = { number: 1, name: "version 1", change: "Landscape fixture", spec: {}, parent: null, status: "trained", best_checkpoint: "checkpoint-1", scorecard: {}, verdict: "" };
const job = (id: string, result: unknown, status = "done", error = "") => ({
  id, kind: "viewer.landscape", status, result, error, progress: "", fraction: null,
  created_at: 0, started_at: 0, finished_at: status === "queued" ? null : 1,
});

/** A bowl: loss rises with the distance from (0.3, -0.2), so the lowest cell is inside the picture unless `edge` pushes it to a corner. */
function landscape(grid: number, live: boolean, edge = false) {
  const axis = Array.from({ length: grid }, (_, i) => -1 + (2 * i) / (grid - 1));
  const centre = edge ? { x: 1, y: 1 } : { x: 0.3, y: -0.2 };
  const loss = axis.map((y) => axis.map((x) => 1.5 + 2 * ((x - centre.x) ** 2 + (y - centre.y) ** 2)));
  const flat = loss.flat();
  const lowest = Math.min(...flat);
  const at = flat.indexOf(lowest);
  return {
    mode: "random", unit: "filter", grid, span: 1, seed: 0, live, axis, loss,
    ceiling: [...flat].sort((a, b) => a - b)[Math.floor(0.95 * flat.length)],
    current: { x: 0, y: 0, loss: loss[(grid - 1) / 2][(grid - 1) / 2], step: 300 },
    minimum: { x: axis[at % grid], y: axis[Math.floor(at / grid)], loss: lowest, inside: !edge },
    trajectory: [{ step: 100, x: -0.8, y: 0.6, loss: 3.9 }, { step: 200, x: -0.3, y: 0.2, loss: 2.4 }, { step: 300, x: 0, y: 0, loss: loss[(grid - 1) / 2][(grid - 1) / 2] }],
    center: { step: 300, name: "best.npz" }, batch: { rows: 4, tokens: 124 }, cost: { seconds: 0.4, evaluations: grid * grid + 3 },
    note: "Held-back text, one fixed batch, so every frame is measured on the same words.",
  };
}

type Body = { version: number; mode: string; grid: number; span: number; seed: number; live: boolean; new_anchor: boolean };

async function mockLandscape(page: Page, options: { edge?: boolean; failure?: string; events?: string } = {}) {
  const state = { requests: [] as Body[] };
  const jobs = new Map<string, ReturnType<typeof job>>();
  await page.route("**/events", (route) => options.events
    ? route.fulfill({ status: 200, contentType: "text/event-stream", body: options.events })
    : route.abort());
  await page.route("**/versions", (route) => route.fulfill({ json: [version] }));
  await page.route("**/viewer/landscape", (route) => {
    const body = route.request().postDataJSON() as Body;
    state.requests.push(body);
    const id = `landscape-${state.requests.length}`;
    jobs.set(id, options.failure ? job(id, null, "failed", options.failure) : job(id, landscape(body.grid ?? 21, !!body.live, options.edge)));
    return route.fulfill({ status: 202, json: job(id, null, "queued") });
  });
  await page.route("**/jobs/*", (route) => {
    const id = new URL(route.request().url()).pathname.split("/").at(-1)!;
    const record = jobs.get(id);
    return route.fulfill({ status: record ? 200 : 404, json: record ?? { detail: "Unknown job" } });
  });
  return state;
}

test("the landscape draws as a 3D surface on a wide screen and reads out the numbers", async ({ page }) => {
  const state = await mockLandscape(page);
  await page.goto("/landscape/");
  await expect(page.getByRole("heading", { level: 1, name: "Loss landscape" })).toBeVisible();
  await expect(page.getByTestId("landscape-surface").locator("canvas")).toBeVisible();
  await expect(page.getByTestId("landscape-here")).toContainText("nats");
  await expect(page.getByTestId("landscape-lowest")).toContainText("1.50 nats");
  await expect(page.getByTestId("landscape-measured")).toContainText("loss evaluations");
  expect(state.requests[0]).toMatchObject({ version: 1, mode: "random", live: false });
});

test("the heat map is the other view of the same grid", async ({ page }) => {
  await mockLandscape(page);
  await page.goto("/landscape/");
  await page.getByRole("button", { name: "Heat map" }).click();
  await expect(page.getByTestId("landscape-heatmap")).toBeVisible();
  await expect(page.getByTestId("landscape-surface")).toHaveCount(0);
  await page.getByRole("button", { name: "Surface" }).click();
  await expect(page.getByTestId("landscape-surface")).toBeVisible();
});

test("a narrow screen starts on the heat map", async ({ page }) => {
  await mockLandscape(page);
  await page.setViewportSize({ width: 375, height: 800 });
  await page.goto("/landscape/");
  await expect(page.getByTestId("landscape-heatmap")).toBeVisible();
  await expect(page.getByTestId("landscape-surface")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("reduced motion starts on the heat map and says why", async ({ page }) => {
  await mockLandscape(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/landscape/");
  await expect(page.getByTestId("landscape-heatmap")).toBeVisible();
  await expect(page.getByText("less motion")).toBeVisible();
});

test("a minimum on the edge of the picture is flagged", async ({ page }) => {
  await mockLandscape(page, { edge: true });
  await page.goto("/landscape/");
  await expect(page.getByText("on the edge of the picture")).toBeVisible();
});

test("new directions ask the server for another seed", async ({ page }) => {
  const state = await mockLandscape(page);
  await page.goto("/landscape/");
  await expect.poll(() => state.requests.length).toBe(1);
  await page.getByRole("button", { name: "New random directions" }).click();
  await expect.poll(() => state.requests.at(-1)?.seed).toBe(1);
});

test("the training path directions are asked for by name", async ({ page }) => {
  const state = await mockLandscape(page);
  await page.goto("/landscape/");
  await page.getByRole("button", { name: "Training path" }).click();
  await expect.poll(() => state.requests.at(-1)?.mode).toBe("pca");
});

test("a failed measurement shows the server's sentence and can be retried", async ({ page }) => {
  const state = await mockLandscape(page, { failure: "No held-back text yet, so there is nothing to measure the error on." });
  await page.goto("/landscape/");
  await expect(page.getByRole("alert")).toContainText("No held-back text yet");
  await page.getByRole("button", { name: "Try again" }).click();
  await expect.poll(() => state.requests.length).toBeGreaterThan(1);
});

test("live updates are off until switched on, show their cost, and stay small", async ({ page }) => {
  const state = await mockLandscape(page);
  await page.goto("/landscape/");
  await expect(page.getByTestId("landscape-cost")).toContainText("Live updates are off");
  await expect(page.getByTestId("landscape-cost")).toContainText("loss evaluations");
  await expect.poll(() => state.requests.length).toBe(1);
  await page.getByRole("switch", { name: "Update while training" }).click();
  await expect.poll(() => state.requests.filter((request) => request.live).length).toBeGreaterThan(0);
  const frame = state.requests.find((request) => request.live)!;
  expect(frame.grid).toBeLessThanOrEqual(11);
  expect(frame.new_anchor).toBe(true);
  await expect(page.getByTestId("landscape-cost")).toContainText("quarter of one core");
});

test("a saved snapshot during training redraws the live landscape", async ({ page }) => {
  const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
  const state = await mockLandscape(page, { events: frame({ kind: "train", payload: { job: "j1", type: "viz_snapshot", step: 50, path: "viz/step_00000050.npz", markers: [] } }) });
  await page.goto("/landscape/");
  await page.getByRole("switch", { name: "Update while training" }).click();
  await expect.poll(() => state.requests.filter((request) => request.live).length).toBeGreaterThanOrEqual(2);
  expect(state.requests.filter((request) => request.live).slice(1).every((request) => !request.new_anchor)).toBe(true);
});

for (const mode of ["light", "dark"] as const) {
  for (const accent of ACCENTS) {
    test(`the heat map and its legend read the theme tokens in ${mode} with the ${accent} accent`, async ({ page }) => {
      await mockLandscape(page);
      await page.addInitScript(([m, a]) => {
        localStorage.setItem("theme", m);
        if (a !== "neutral") localStorage.setItem("groundwork-accent", a);
      }, [mode, accent]);
      await page.goto("/landscape/");
      await page.getByRole("button", { name: "Heat map" }).click();
      const canvas = page.getByTestId("landscape-heatmap").locator("canvas");
      await expect(canvas).toBeVisible();
      // The ramp ends are the theme's own tokens, and the canvas is not blank: its corner pixels differ.
      const ends = await page.evaluate(() => {
        const style = getComputedStyle(document.documentElement);
        return [style.getPropertyValue("--ramp-1").trim(), style.getPropertyValue("--ramp-9").trim()];
      });
      expect(ends.every((value) => value.length > 0)).toBe(true);
      const corners = await canvas.evaluate((element: HTMLCanvasElement) => {
        const context = element.getContext("2d")!;
        const last = element.width - 1;
        return [context.getImageData(0, 0, 1, 1).data.join(","), context.getImageData(last, last, 1, 1).data.join(",")];
      });
      expect(corners[0]).not.toBe(corners[1]);
      await expect(page.getByTestId("landscape-readout")).toBeVisible();
    });
  }
}

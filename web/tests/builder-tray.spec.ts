import { expect, test, type Page } from "@playwright/test";
import { emit, fakeEvents, mockApi } from "./qp-mock";

const version = (number: number, status = "saved") => ({
  number, name: `version ${number}`, status, change: `Design ${number}`, spec: {},
  parent: number > 1 ? number - 1 : null, best_checkpoint: status === "trained" ? `checkpoint-${number}` : null,
  scorecard: {}, prediction: {}, verdict: "",
});
const queued = (status = "waiting") => ({ id: "train-2", kind: "pretrain", label: "Version 2", status, params: { version: 2 } });

async function setup(page: Page) {
  await fakeEvents(page);
  const state = {
    versions: [version(1, "trained"), version(2)],
    queue: { job: null as ReturnType<typeof queued> | null, up_next: [] as ReturnType<typeof queued>[], paused: false },
    curves: {
      1: { steps: [[1, 3], [10, 2]], smooth: [], heldout: [[10, 2.1]], total: 10 },
      2: { steps: [] as number[][], smooth: [], heldout: [] as number[][], total: 0 },
    },
  };
  await mockApi(page, {
    "GET /versions": () => ({ body: state.versions }),
    "GET /queue": () => ({ body: state.queue }),
    "GET /pretraining/versions/1/curve": () => ({ body: state.curves[1] }),
    "GET /pretraining/versions/2/curve": () => ({ body: state.curves[2] }),
  });
  return state;
}

test("saved versions keep their rows through queue, hold, training and completion", async ({ page }) => {
  const state = await setup(page);
  await page.goto("/builder/");
  const row = page.getByRole("list", { name: "Saved versions" }).getByRole("button", { name: /Version 2/ });
  await expect(row).toContainText("Saved");
  await row.evaluate((element) => element.setAttribute("data-original-row", "yes"));
  state.queue.up_next = [queued()];
  await emit(page, "queue", {});
  await expect(row).toContainText("Queued");
  state.queue.up_next = [queued("held")];
  await emit(page, "queue", {});
  await expect(row).toContainText("Held");
  state.queue.up_next = [];
  state.queue.job = queued("running");
  await emit(page, "queue", {});
  await expect(row).toContainText("Training");
  state.queue.paused = true;
  await emit(page, "queue", {});
  await expect(row).toContainText("Training paused");
  state.queue.job = null;
  state.queue.paused = false;
  state.versions[1] = version(2, "trained");
  state.curves[2] = { steps: [[1, 1.9], [5, 1.5]], smooth: [], heldout: [[5, 1.6]], total: 5 };
  await emit(page, "job", { job: "train-2", status: "done" });
  await expect(row).toContainText("Trained");
  await expect(row).toHaveAttribute("data-original-row", "yes");
  await expect(page.getByText("Version 1 · held-back error: 2.100 bits per character", { exact: true })).toBeVisible();
  await expect(page.getByText("Version 2 · held-back error: 1.600 bits per character", { exact: true })).toBeVisible();
});

test("every inspector link follows the selected version without mobile overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page);
  await page.goto("/builder/");
  const history = page.getByRole("region", { name: "Version history" });
  const links = [
    ["Trace a position", "trace"], ["Head lab", "head-lab"], ["Inside a head", "head-zoom"],
    ["Loss", "loss"], ["Value map", "value-map"], ["Inside the network", "inside-3d"],
  ];
  for (const [name, path] of links) await expect(history.getByRole("link", { name, exact: false })).toHaveAttribute("href", `/${path}/?version=2`);
  await history.getByRole("button", { name: /Version 1/ }).click();
  for (const [name, path] of links) await expect(history.getByRole("link", { name, exact: false })).toHaveAttribute("href", `/${path}/?version=1`);
  await expect(history.getByRole("button", { name: /Version 1/ })).toHaveAttribute("aria-pressed", "true");
  expect(await page.locator("main").evaluate((element) => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(0);
});

test("an empty workspace explains where version history comes from", async ({ page }) => {
  const state = await setup(page);
  state.versions = [];
  await page.goto("/builder/");
  await expect(page.getByText(/No saved versions yet/)).toBeVisible();
  await expect(page.getByRole("list", { name: "Saved versions" })).toHaveCount(0);
});

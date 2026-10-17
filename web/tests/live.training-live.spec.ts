import { expect, test } from "@playwright/test";

// The Training tab's live view (#338) against the real --fake server, which opens on a simulated run (id demo-run): a MPS machine (mac-mini) and a CUDA
// machine (gpu-box) training in rounds of about six seconds. Nothing is intercepted. These specs only watch the run; they never stop it, since the
// other live specs share the workspace.

test("/cluster/ opens on the demo run: a banner, two machines of different kinds, and edges to the coordinator", async ({ page }) => {
  await page.goto("/cluster/");
  await expect(page).toHaveURL(/\/cluster\/training\/?$/);
  const card = page.getByTestId("run-demo-run");
  await expect(card).toBeVisible({ timeout: 15_000 });
  await expect(card.getByTestId("round-title")).toContainText(/Round \d+ of 400/);
  await expect(card.getByTestId("coordinator")).toContainText("Coordinator");
  await expect(card.getByTestId("device-mac-mini")).toHaveText("MPS");
  await expect(card.getByTestId("device-gpu-box")).toHaveText("CUDA");
  for (const node of ["mac-mini", "gpu-box"]) {
    await expect(card.getByTestId(`edge-down-${node}`)).toContainText("θ down");
    await expect(card.getByTestId(`edge-up-${node}`)).toContainText("Δ up");
  }
  await expect(page.getByRole("button", { name: "Start a run" })).toHaveAttribute("aria-expanded", "false"); // a run is going: the form is folded
});

test("the view moves on its own: progress, a phase, the countdown, the timeline and the log fill in without a reload", async ({ page }) => {
  await page.goto("/cluster/training/");
  const card = page.getByTestId("run-demo-run");
  const bar = card.getByRole("progressbar", { name: "Progress of gpu-box" });
  await expect(bar).toBeVisible({ timeout: 15_000 });
  // within a round the machine reaches its last step (the CUDA machine trains faster than the MPS one), so the bar is seen at more than one value
  const seen = new Set<string>();
  await expect
    .poll(async () => {
      seen.add((await bar.getAttribute("aria-valuenow")) ?? "");
      return seen.size;
    }, { timeout: 20_000, intervals: [300] })
    .toBeGreaterThan(1);
  await expect(card.getByTestId("topology").locator("[data-phase]").first()).toBeVisible();
  await expect(card.getByRole("list", { name: "Phases of a round" }).locator("[aria-current='step']")).toHaveCount(1, { timeout: 15_000 });
  await expect(card.getByRole("region", { name: "Round timeline" })).toBeVisible();
  await expect(card.getByTestId("lane-1-gpu-box")).toBeVisible();
  await expect(card.getByRole("region", { name: "Events" }).getByRole("listitem").first()).toBeVisible();
  await expect(card.getByRole("region", { name: "Loss over the rounds" })).toBeVisible();
  await expect(card.getByRole("region", { name: "Tokens per second" })).toBeVisible();
});

test("a round closes with an outer step, which the log lists newest first with its size", async ({ page }) => {
  await page.goto("/cluster/training/");
  const log = page.getByTestId("run-demo-run").getByRole("region", { name: "Events" });
  await expect(log.locator("[data-kind='outer_step']").first()).toBeVisible({ timeout: 30_000 });
  await expect(log.locator("[data-kind='outer_step']").first()).toContainText(/size/i);
  const seqs = await log.getByRole("listitem").evaluateAll((items) => items.map((li) => Number(li.getAttribute("data-testid")?.replace("event-", ""))));
  expect(seqs.length).toBeGreaterThan(1);
  expect(seqs).toEqual([...seqs].sort((a, b) => b - a)); // newest first
});

test("the machines can be stopped from their nodes, and the run from the coordinator", async ({ page }) => {
  await page.goto("/cluster/training/");
  const card = page.getByTestId("run-demo-run");
  await expect(card.getByRole("button", { name: "Stop run" })).toBeVisible({ timeout: 15_000 });
  await expect(card.getByTestId("coordinator").getByRole("button", { name: "Stop run" })).toBeVisible();
  await expect(card.getByTestId("node-gpu-box").getByRole("button", { name: "Stop gpu-box" })).toBeVisible();
  await expect(card.getByTestId("node-mac-mini").getByRole("button")).toHaveCount(0); // this computer is the coordinator's own member
});

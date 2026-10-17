import { expect, test } from "@playwright/test";

// The Cluster page's Training tab (#326) against the real --fake server: no intercepted responses. The machine chosen is a port nothing
// listens on, so the run starts without it and says why; this computer is the run's other member, trained by the fake coordinator.

const NOBODY = "127.0.0.1:9";

type Run = { id: string; state: string; members: { node: string; state: string; note: string }[] };

test.afterEach(async ({ request }) => {
  // the specs share one workspace: end any run still going, then empty the machines list
  const runs = (await (await request.get("/distributed/runs")).json()) as Run[];
  for (const run of runs.filter((r) => ["queued", "starting", "waiting", "running"].includes(r.state))) await request.post(`/distributed/runs/${run.id}/stop`);
  expect((await request.put("/cluster/nodes", { data: { addresses: [] } })).ok()).toBeTruthy();
});

test("starts a run with an unreachable machine, shows why it is out, then stops the run", async ({ page, request }) => {
  const presets = await (await request.get("/design/presets")).json();
  expect((await request.post("/versions", { data: { spec: presets[0].spec, change: "Training tab fixture" } })).ok()).toBeTruthy();
  expect((await request.put("/cluster/nodes", { data: { addresses: [NOBODY] } })).ok()).toBeTruthy();

  await page.goto("/cluster/");
  await expect(page).toHaveURL(/\/cluster\/training\/?$/); // Training is the default tab
  await expect(page.getByLabel("Rounds")).toHaveValue("10");
  await expect(page.getByText(/This computer coordinates/)).toBeVisible();

  await page.getByRole("checkbox", { name: new RegExp(NOBODY) }).check();
  await page.getByLabel("Rounds").fill("500"); // long enough that the run is still going when it is stopped
  const started = page.waitForResponse((r) => r.request().method() === "POST" && new URL(r.url()).pathname.replace(/\/$/, "") === "/distributed/runs");
  await page.getByRole("button", { name: "Start run" }).click();
  const id = ((await (await started).json()) as Run).id;

  const card = page.getByTestId(`run-${id}`);
  await expect(card).toBeVisible({ timeout: 10_000 });
  await expect(card).toContainText("of 500 rounds done");
  const unreachable = card.getByTestId(`member-${NOBODY}`);
  await expect(unreachable).toContainText("Could not be reached", { timeout: 10_000 });
  await expect(card.getByTestId(`reason-${NOBODY}`)).toContainText(NOBODY);
  await expect(card.getByText("This computer", { exact: true })).toBeVisible(); // the run goes on without the machine that did not answer

  await page.getByRole("button", { name: "Stop run" }).click();
  await expect(page.getByTestId(`run-state-${id}`)).toHaveText("Stopped", { timeout: 10_000 });
  await expect(page.getByRole("button", { name: "Stop run" })).toHaveCount(0);
});

test("the Queue links to the Training tab once other machines are listed", async ({ page, request }) => {
  expect((await request.put("/cluster/nodes", { data: { addresses: [NOBODY] } })).ok()).toBeTruthy();
  await page.goto("/queue/");
  await expect(page.getByTestId("queue-training-link")).toBeVisible({ timeout: 10_000 });
  await page.getByTestId("queue-training-link").click();
  await expect(page).toHaveURL(/\/cluster\/training\/?$/);
  await expect(page.getByRole("navigation", { name: "Cluster" }).getByRole("link", { name: "Training" })).toHaveAttribute("aria-current", "location");
});

import { expect, test, type Page } from "@playwright/test";

// The Queue page against a mocked API: it reads GET /queue, calls the hold/move/remove/pause/plan routes and keeps the running
// job's numbers current from the `train` events on /events. Live-fake specs wait for the fake server running jobs (#139).

const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

type Job = { id: string; kind: string; label: string; note: string; params: object; status: string; eta_s: number | null; created_at: number; error: string };
const job = (id: string, label: string, over: Partial<Job> = {}): Job => ({ id, kind: "pretrain", label, note: "", params: {}, status: "waiting", eta_s: 1800, created_at: 0, error: "", ...over });

const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };

async function mockApi(page: Page, start: { running?: boolean; reason?: string | null; supported?: boolean } = {}) {
  const calls: { method: string; path: string; body: unknown }[] = [];
  const state = {
    job: start.running === false ? null : (job("run1", "Version 3 pretraining", { status: "running", eta_s: null }) as Job | null),
    step: 100,
    total: 1000,
    paused: false,
    stopping: false,
    plan: { ...PLAN } as Record<string, unknown>,
    reason: start.reason ?? (null as string | null),
    supported: start.supported ?? true,
    up_next: [job("a", "First waiting"), job("b", "Second waiting", { note: "A note" }), job("c", "Third waiting")],
    connections: 0,
  };
  const view = () => ({
    job: state.job,
    step: state.step,
    total: state.job ? state.total : 0,
    phase: state.job ? "steady" : "",
    lr: state.job ? 0.0003 : null,
    eta_s: state.job ? 600 : null,
    paused: state.paused,
    blocked: state.reason,
    gate: { reason: state.reason, resumes_at: null, supported: state.supported },
    stopping: state.stopping,
    up_next: state.up_next,
    plan: state.plan,
  });
  const record = (route: import("@playwright/test").Route) => {
    const request = route.request();
    const body = request.postData() ? request.postDataJSON() : undefined;
    calls.push({ method: request.method(), path: new URL(request.url()).pathname, body });
    return body;
  };
  const jobIndex = (url: string) => state.up_next.findIndex((j) => j.id === url.split("/queue/jobs/")[1].split("/")[0]);

  await page.route("**/queue", (route) => route.fulfill(json(view())));
  await page.route("**/queue/jobs/*/outlook", (route) => route.fulfill(json({ outlook: "after about 30 min" })));
  await page.route("**/queue/jobs/*/hold", (route) => {
    const body = record(route) as { held: boolean };
    state.up_next[jobIndex(route.request().url())].status = body.held ? "held" : "waiting";
    return route.fulfill(json(view()));
  });
  await page.route("**/queue/jobs/*/move", (route) => {
    const { index } = record(route) as { index: number };
    const [moved] = state.up_next.splice(jobIndex(route.request().url()), 1);
    state.up_next.splice(index, 0, moved);
    return route.fulfill(json(view()));
  });
  await page.route("**/queue/jobs/*", (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    record(route);
    const id = route.request().url().split("/queue/jobs/")[1];
    if (state.job?.id === id) state.job = null;
    state.up_next = state.up_next.filter((j) => j.id !== id);
    return route.fulfill(json(view()));
  });
  await page.route("**/queue/pause", (route) => {
    record(route);
    state.paused = true;
    return route.fulfill(json(view()));
  });
  await page.route("**/queue/resume", (route) => {
    record(route);
    state.paused = false;
    return route.fulfill(json(view()));
  });
  await page.route("**/queue/start", (route) => {
    record(route);
    return route.fulfill(json(view()));
  });
  await page.route("**/queue/stop_after_cooldown", (route) => {
    record(route);
    state.stopping = true;
    return route.fulfill(json(view()));
  });
  await page.route("**/queue/checkpoint_now", (route) => {
    record(route);
    return route.fulfill(json({ started: true }));
  });
  await page.route("**/queue/plan", (route) => {
    const body = record(route) as Record<string, unknown>;
    state.plan = body;
    if (body.idle_only === false) state.reason = null;
    return route.fulfill(json(state.plan));
  });
  // Each (re)connection of the stream carries one training step: the page re-reads the queue, and the step has moved on.
  await page.route("**/events", (route) => {
    state.connections += 1;
    if (state.connections > 1) state.step += 50;
    return route.fulfill({ status: 200, contentType: "text/event-stream", body: frame({ kind: "train", payload: { type: "metric", step: state.step } }) });
  });
  return { state, calls };
}

const rows = (page: Page) => page.getByTestId("up-next").getByRole("listitem");

test("shows the running job, the jobs up next in order and the plan", async ({ page }) => {
  await mockApi(page);
  await page.goto("/queue/");
  await expect(page.getByRole("heading", { level: 1, name: "Queue" })).toBeVisible();

  await expect(page.getByTestId("state-line")).toHaveText("Version 3 pretraining · running");
  await expect(page.getByTestId("progress-line")).toContainText("step 100 of 1,000");
  await expect(page.getByTestId("progress-line")).toContainText("schedule: steady");
  await expect(page.getByTestId("progress-line")).toContainText("learning rate: auto (0.0003)");
  await expect(page.getByTestId("progress-line")).toContainText("about 10 min left");

  await expect(rows(page)).toHaveCount(3);
  await expect(rows(page).nth(1)).toContainText("Second waiting");
  await expect(rows(page).nth(1)).toContainText("A note");
  await expect(rows(page).nth(0)).toContainText("after about 30 min");
  await expect(page.getByText("all done in about 1.7 h at the current plan")).toBeVisible();

  await expect(page.getByRole("radio", { name: "Any time" })).toBeChecked();
  await expect(page.getByRole("radio", { name: "Most" })).toBeChecked();
  await expect(page.getByRole("switch", { name: /Only when I.m not using the computer/ })).toBeChecked();
});

test("hold, release, move, run next and remove a waiting job", async ({ page }) => {
  const { calls } = await mockApi(page);
  await page.goto("/queue/");
  await expect(rows(page)).toHaveCount(3);

  await page.getByRole("button", { name: "Hold: Second waiting" }).click();
  await expect(rows(page).nth(1)).toContainText("Second waiting · on hold");
  await page.getByRole("button", { name: "Release: Second waiting" }).click();
  await expect(rows(page).nth(1)).not.toContainText("on hold");

  await page.getByRole("button", { name: "Move up: Second waiting" }).click();
  await expect(rows(page).nth(0)).toContainText("Second waiting");
  await page.getByRole("button", { name: "Move down: Second waiting" }).click();
  await expect(rows(page).nth(1)).toContainText("Second waiting");
  await page.getByRole("button", { name: "Run next: Third waiting" }).click();
  await expect(rows(page).nth(0)).toContainText("Third waiting");
  await expect(page.getByRole("button", { name: "Move up: Third waiting" })).toBeDisabled();

  await page.getByRole("button", { name: "Remove: First waiting" }).click();
  await expect(rows(page)).toHaveCount(2);
  await expect(page.getByText("First waiting")).toHaveCount(0);

  expect(calls.map((c) => [c.method, c.path.replace("/queue/jobs/", ""), c.body])).toEqual([
    ["POST", "b/hold", { held: true }],
    ["POST", "b/hold", { held: false }],
    ["POST", "b/move", { index: 0 }],
    ["POST", "b/move", { index: 1 }],
    ["POST", "c/move", { index: 0 }],
    ["DELETE", "a", undefined],
  ]);
});

test("a waiting row is reordered by dragging it", async ({ page }) => {
  const { calls } = await mockApi(page);
  await page.goto("/queue/");
  await expect(rows(page)).toHaveCount(3);
  await rows(page).nth(0).dragTo(rows(page).nth(2));
  await expect(rows(page).nth(2)).toContainText("First waiting");
  expect(calls).toEqual([expect.objectContaining({ path: "/queue/jobs/a/move", body: { index: 2 } })]);
});

test("pause, resume, checkpoint now, stop after cool-down and stop", async ({ page }) => {
  const { calls } = await mockApi(page);
  await page.goto("/queue/");

  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByTestId("state-line")).toHaveText("Version 3 pretraining · paused");
  await page.getByRole("button", { name: "Resume" }).click();
  await expect(page.getByTestId("state-line")).toHaveText("Version 3 pretraining · running");

  await page.getByRole("button", { name: "Make a checkpoint now" }).click();
  await expect(page.getByText("Making a checkpoint from the last one.")).toBeVisible();

  await page.getByRole("button", { name: "Stop after a short cool-down" }).click();
  await expect(page.getByText("Asked it to stop. It stops at its next check.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop after a short cool-down" })).toBeDisabled();

  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(page.getByText("Nothing is training.")).toBeVisible();
  expect(calls.map((c) => `${c.method} ${c.path}`)).toEqual([
    "POST /queue/pause",
    "POST /queue/resume",
    "POST /queue/checkpoint_now",
    "POST /queue/stop_after_cooldown",
    "DELETE /queue/jobs/run1",
  ]);
});

test("with nothing running, jobs saved by an earlier session can be started", async ({ page }) => {
  const { calls } = await mockApi(page, { running: false });
  await page.goto("/queue/");
  await expect(page.getByText("Nothing is training.")).toBeVisible();
  await page.getByRole("button", { name: "Start the waiting jobs" }).click();
  await expect.poll(() => calls.map((c) => c.path)).toEqual(["/queue/start"]);
  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByText("Paused by you. Nothing starts until you resume.")).toBeVisible();
});

test("the plan is saved, and 'only when idle' can be turned off so training stops waiting", async ({ page }) => {
  const { calls } = await mockApi(page, { reason: "computer in use" });
  await page.goto("/queue/");

  await expect(page.getByTestId("gate")).toContainText("Training is waiting: computer in use");
  const idle = page.getByRole("switch", { name: /Only when I.m not using the computer/ });
  await expect(idle).toBeEnabled();
  await idle.click();
  await expect(idle).not.toBeChecked();
  await expect(page.getByTestId("gate")).toHaveCount(0);
  expect(calls[0]).toEqual(expect.objectContaining({ method: "PUT", path: "/queue/plan", body: expect.objectContaining({ idle_only: false, hours: "any", memory: "most" }) }));

  await page.getByRole("radio", { name: "Set hours" }).click();
  await expect(page.getByLabel("From hour")).toHaveValue("22");
  await page.getByLabel("From hour").fill("9");
  await page.getByLabel("From hour").press("Enter");
  await page.getByLabel("to hour").fill("99");
  await page.getByLabel("to hour").press("Enter");
  await expect(page.getByLabel("to hour")).toHaveValue("7");
  await page.getByLabel("to hour").fill("17");
  await page.getByLabel("to hour").press("Enter");
  await page.getByRole("radio", { name: "Half" }).click();
  await page.getByRole("switch", { name: "Keep the computer awake while training" }).click();
  await expect.poll(() => calls.at(-1)?.body).toEqual(
    expect.objectContaining({ hours: "set", start_hour: 9, end_hour: 17, memory: "half", keep_awake: true, idle_only: false }),
  );
});

test("idle detection the computer cannot do greys the idle switch", async ({ page }) => {
  await mockApi(page, { supported: false });
  await page.goto("/queue/");
  await expect(page.getByRole("switch", { name: /Only when I.m not using the computer/ })).toBeDisabled();
  await expect(page.getByText("Not available on this computer")).toBeVisible();
});

test("the running job's progress moves in place from /events while waiting rows stay mounted", async ({ page }) => {
  await mockApi(page);
  await page.goto("/queue/");
  await expect(rows(page)).toHaveCount(3);
  await rows(page).nth(1).evaluate((el) => ((el as HTMLElement).dataset.mounted = "yes"));
  await page.getByTestId("progress-line").evaluate((el) => ((el as HTMLElement).dataset.mounted = "yes"));

  await expect(page.getByTestId("progress-line")).toContainText(/step 1\d\d of 1,000/, { timeout: 15_000 });
  await expect(page.getByTestId("progress-line")).not.toContainText("step 100 of");
  await expect(rows(page).nth(1)).toHaveAttribute("data-mounted", "yes");
  await expect(page.getByTestId("progress-line")).toHaveAttribute("data-mounted", "yes");
  await expect(page.getByRole("progressbar", { name: "Job progress" })).not.toHaveAttribute("aria-valuenow", "10");
});

test("a failed call is a toast, not a broken page", async ({ page }) => {
  await mockApi(page);
  await page.route("**/queue/pause", (route) => route.fulfill(json({ detail: "The runner is not running" }, 409)));
  await page.goto("/queue/");
  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByText("The runner is not running")).toBeVisible();
  await expect(page.getByTestId("state-line")).toHaveText("Version 3 pretraining · running");
});

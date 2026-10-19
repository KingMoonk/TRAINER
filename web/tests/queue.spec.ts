import { expect, test, type Page } from "@playwright/test";

// The Queue page in a real browser against a mocked API: the list, hold/move/remove, and the two ways to reorder (drag and the
// arrow keys), which need real layout and drag events. The rest of the page (pause, plan, toasts, live progress) is
// src/app/queue/page.test.tsx under Vitest.

const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

type Job = { id: string; kind: string; label: string; note: string; params: object; status: string; eta_s: number | null; created_at: number; error: string };
const job = (id: string, label: string, over: Partial<Job> = {}): Job => ({ id, kind: "pretrain", label, note: "", params: {}, status: "waiting", eta_s: 1800, created_at: 0, error: "", ...over });

const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };

async function mockApi(page: Page) {
  const calls: { method: string; path: string; body: unknown }[] = [];
  const state = {
    job: job("run1", "Version 3 pretraining", { status: "running", eta_s: null }) as Job | null,
    step: 100,
    total: 1000,
    paused: false,
    stopping: false,
    plan: { ...PLAN } as Record<string, unknown>,
    reason: null as string | null,
    supported: true,
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
  await page.route("**/queue/jobs", (route) => route.fulfill(json([...(state.job ? [state.job] : []), ...state.up_next])));
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
  // Bring both ends into view before mouse-down. Scrolling the target after
  // mouse-down can move the grip away before Chromium starts the native drag.
  await rows(page).nth(2).scrollIntoViewIfNeeded();
  await rows(page).nth(0).getByRole("button", { name: "Drag to reorder: First waiting" }).dragTo(rows(page).nth(2));
  await expect(rows(page).nth(2)).toContainText("First waiting");
  expect(calls).toEqual([expect.objectContaining({ path: "/queue/jobs/a/move", body: { index: 2 } })]);
});

test("the reorder grip supports arrow keys and keeps focus on its job", async ({ page }) => {
  const { calls } = await mockApi(page);
  await page.goto("/queue/");
  const grip = page.getByRole("button", { name: "Drag to reorder: First waiting" });
  await grip.focus();
  await grip.press("ArrowUp");
  expect(calls).toEqual([]);
  await grip.press("ArrowDown");
  await expect(rows(page).nth(1)).toContainText("First waiting");
  await expect(grip).toBeFocused();
  await grip.press("ArrowUp");
  await expect(rows(page).nth(0)).toContainText("First waiting");
  expect(calls).toEqual([
    expect.objectContaining({ path: "/queue/jobs/a/move", body: { index: 1 } }),
    expect.objectContaining({ path: "/queue/jobs/a/move", body: { index: 0 } }),
  ]);
});


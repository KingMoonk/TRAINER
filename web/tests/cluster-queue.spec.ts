import { expect, test, type Page } from "@playwright/test";
import { emit, fakeEvents, mockApi } from "./qp-mock";

// The merged queue and merged versions (#165) against a mocked API: GET /cluster/merged lists the machines and every machine's jobs
// (the server polls the peers), GET /versions/published lists what each machine published. A cluster of one gets empty lists and
// the pages show nothing about machines. The live spec (live.cluster.spec.ts) runs the same page against two real --fake servers.

type Row = {
  id: string; kind: string; label: string; status: string; node: string; hostname: string; queued_on: string; queued_here: boolean;
  pinned: string; waiting_reason: string; stale: boolean;
};
const NODES = {
  me: { address: "local", hostname: "mac-mini", pin: "mac-mini", local: true, online: true, error: "", seen_at: 0 },
  gpu: { address: "gpu-box:8000", hostname: "gpu-box", pin: "gpu-box:8000", local: false, online: true, error: "", seen_at: 1_700_000_000 },
};
const job = (id: string, label: string, over: Record<string, unknown> = {}) => ({ id, kind: "pretrain", label, note: "", params: {}, status: "waiting", eta_s: null, created_at: 0, error: "", ...over });
const row = (id: string, label: string, over: Partial<Row> = {}): Row => ({
  id, kind: "pretrain", label, status: "waiting", node: "", hostname: "", queued_on: "mac-mini", queued_here: true, pinned: "", waiting_reason: "", stale: false, ...over,
});
const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };
const published = (number: number, over: Record<string, unknown> = {}) => ({
  key: `gpu-box:8000/${number}`, number, origin: "gpu-box", mine: false, local: false, local_number: null, name: `version ${number}`, change: `Design ${number}`, scorecard: {}, ...over,
});
const version = (number: number) => ({
  number, name: `version ${number}`, change: `change ${number}`, status: "trained", verdict: "", best_checkpoint: `ckpt-${number}`,
  parent: number > 1 ? number - 1 : null, scorecard: {}, spec: {}, prediction: {},
});
const doneJob = (result: unknown) => ({ id: "metadata", kind: "versions", status: "done", result, progress: "", error: "", created_at: 0 });

/** Two machines: this one runs "Version 3", has two jobs waiting; gpu-box runs one and has one waiting. */
function twoMachines() {
  const state = {
    nodes: [NODES.me, { ...NODES.gpu }],
    queue: {
      job: job("run1", "Version 3 pretraining", { status: "running" }),
      step: 100, total: 1000, phase: "steady", lr: 0.0003, eta_s: 600, paused: false, blocked: null, stopping: false,
      gate: { reason: null, resumes_at: null, supported: true },
      up_next: [job("w1", "First waiting"), job("w2", "Second waiting")],
      plan: PLAN,
    },
    jobs: [] as Row[],
    published: [published(7), published(2, { key: "mac-mini/2", origin: "mac-mini", mine: true, local: true, local_number: 2 })],
  };
  state.jobs = [
    row("run1", "Version 3 pretraining", { status: "running", node: "local", hostname: "mac-mini" }),
    row("w1", "First waiting"),
    row("w2", "Second waiting"),
    row("g1", "Version 9 pretraining", { status: "running", node: "gpu-box:8000", hostname: "gpu-box", queued_on: "gpu-box", queued_here: false }),
    row("g2", "Version 10 pretraining", { node: "gpu-box:8000", hostname: "gpu-box", queued_on: "gpu-box", queued_here: false, pinned: "gpu-box" }),
  ];
  return state;
}

async function setup(page: Page, state: ReturnType<typeof twoMachines> | null, clusterRoute = true) {
  await fakeEvents(page);
  const pins: { path: string; body: unknown }[] = [];
  const cluster = () => ({ body: state ? { nodes: state.nodes, jobs: state.jobs } : { nodes: [], jobs: [] } });
  const queue = () => ({ body: state?.queue ?? { job: null, up_next: [], paused: false, plan: PLAN, step: 0, total: 0, phase: "", lr: null, eta_s: null, blocked: null, stopping: false, gate: { reason: null, resumes_at: null, supported: true } } });
  const pin = (id: string) => (route: import("@playwright/test").Route) => {
    const body = route.request().postDataJSON() as { pinned: string | null };
    pins.push({ path: new URL(route.request().url()).pathname, body });
    if (state) {
      const mine = state.jobs.find((r) => r.id === id);
      const node = body.pinned ? state.nodes.find((n) => n.pin === body.pinned || n.hostname === body.pinned) : undefined;
      if (mine) Object.assign(mine, { pinned: body.pinned ?? "", node: node?.address ?? "", hostname: node?.hostname ?? (body.pinned ?? "") });
    }
    return queue();
  };
  await mockApi(page, {
    "GET /cluster/merged": () => (clusterRoute ? cluster() : undefined),
    "GET /queue": queue,
    "GET /queue/plan": () => ({ body: PLAN }),
    "GET /queue/jobs/w1/outlook": () => ({ body: { outlook: "after about 10 min" } }),
    "GET /queue/jobs/w2/outlook": () => ({ body: { outlook: "after about 20 min" } }),
    "POST /queue/jobs/w1/pin": pin("w1"),
    "POST /queue/jobs/w2/pin": pin("w2"),
    "GET /versions": () => ({ body: [version(1), version(2)] }),
    "GET /versions/published": () => (state ? { body: state.published } : { status: 409, body: { detail: "no store" } }),
    "GET /versions/best": () => ({ body: doneJob(version(2)) }),
    "GET /versions/1/checkpoint-step": () => ({ body: doneJob(100) }),
    "GET /versions/2/checkpoint-step": () => ({ body: doneJob(100) }),
  });
  return { pins };
}

test("the Queue page shows every machine's jobs with the node on each row", async ({ page }) => {
  const state = twoMachines();
  await setup(page, state);
  await page.goto("/queue/");

  await expect(page.getByTestId("now-place")).toHaveText("Running on mac-mini");
  await expect(page.getByTestId("place-w1")).toHaveText("Runs on any machine that fits it");
  await expect(page.getByTestId("place-w2")).toHaveText("Runs on any machine that fits it");

  const other = page.getByTestId("other-machines");
  await expect(other.getByTestId("node-local")).toContainText("mac-mini (this machine)");
  await expect(other.getByTestId("node-gpu-box:8000")).toContainText("gpu-box");
  await expect(other.getByTestId("other-g1")).toContainText("Version 9 pretraining");
  await expect(other.getByTestId("other-g1")).toContainText("Running on gpu-box");
  await expect(other.getByTestId("other-g2")).toContainText("Pinned to gpu-box");
  await expect(other.getByTestId("other-g2")).toContainText("queued on gpu-box");
  await expect(other.getByTestId("other-g1")).not.toHaveAttribute("data-stale", "true");
  // This machine's own jobs are not repeated in the other machines' card.
  await expect(other.getByTestId("other-w1")).toHaveCount(0);
});

test("pinning a waiting job to the other machine changes where its row says it will run", async ({ page }) => {
  const state = twoMachines();
  const { pins } = await setup(page, state);
  await page.goto("/queue/");
  await expect(page.getByTestId("place-w1")).toHaveText("Runs on any machine that fits it");

  const select = page.getByRole("combobox", { name: "Run on: First waiting" });
  await expect(select.getByRole("option")).toHaveText(["Any machine", "mac-mini (this machine)", "gpu-box"]);
  await select.selectOption({ label: "gpu-box" });
  await expect(page.getByTestId("place-w1")).toHaveText("Pinned to gpu-box: it runs there or waits");
  await expect(select).toHaveValue("gpu-box:8000");
  await expect(page.getByTestId("place-w2")).toHaveText("Runs on any machine that fits it");
  expect(pins).toEqual([{ path: "/queue/jobs/w1/pin", body: { pinned: "gpu-box:8000" } }]);

  await select.selectOption({ label: "Any machine" });
  await expect(page.getByTestId("place-w1")).toHaveText("Runs on any machine that fits it");
  expect(pins.at(-1)).toEqual({ path: "/queue/jobs/w1/pin", body: { pinned: null } });
});

test("two machines with the same hostname are told apart by their address", async ({ page }) => {
  const state = twoMachines();
  const twin = (address: string) => ({ ...NODES.gpu, address, pin: address });
  state.nodes = [NODES.me, twin("10.0.0.2:8000"), twin("10.0.0.3:8000")];
  // The server adds the address to a row's machine name when another listed machine has the same hostname.
  state.jobs = [
    row("w1", "First waiting", { node: "10.0.0.3:8000", hostname: "gpu-box (10.0.0.3:8000)", pinned: "10.0.0.3:8000" }),
    row("w2", "Second waiting"),
    row("g1", "Version 9 pretraining", { status: "running", node: "10.0.0.2:8000", hostname: "gpu-box (10.0.0.2:8000)", queued_on: "gpu-box (10.0.0.2:8000)", queued_here: false }),
    row("g2", "Version 10 pretraining", { node: "10.0.0.3:8000", hostname: "gpu-box (10.0.0.3:8000)", queued_on: "gpu-box (10.0.0.3:8000)", queued_here: false, pinned: "10.0.0.3:8000" }),
  ];
  await setup(page, state);
  await page.goto("/queue/");

  const other = page.getByTestId("other-machines");
  await expect(other.getByTestId("node-local")).toHaveText("mac-mini (this machine)");
  await expect(other.getByTestId("node-10.0.0.2:8000")).toHaveText("gpu-box (10.0.0.2:8000)");
  await expect(other.getByTestId("node-10.0.0.3:8000")).toHaveText("gpu-box (10.0.0.3:8000)");
  await expect(other.getByTestId("other-g1")).toContainText("Running on gpu-box (10.0.0.2:8000)");
  await expect(other.getByTestId("other-g1")).toContainText("queued on gpu-box (10.0.0.2:8000)");
  await expect(other.getByTestId("other-g2")).toContainText("Pinned to gpu-box (10.0.0.3:8000)");
  await expect(page.getByTestId("place-w1")).toHaveText("Pinned to gpu-box (10.0.0.3:8000): it runs there or waits");

  const select = page.getByRole("combobox", { name: "Run on: First waiting" });
  await expect(select.getByRole("option")).toHaveText(["Any machine", "mac-mini (this machine)", "gpu-box (10.0.0.2:8000)", "gpu-box (10.0.0.3:8000)"]);
  await expect(select).toHaveValue("10.0.0.3:8000");

  await expect(page.getByRole("status", { name: "Status" })).toContainText("mac-mini (this machine), gpu-box (10.0.0.2:8000), gpu-box (10.0.0.3:8000)");
});

test("a machine that stops answering keeps its last rows, greyed and labelled", async ({ page }) => {
  const state = twoMachines();
  await setup(page, state);
  await page.goto("/queue/");
  const other = page.getByTestId("other-machines");
  await expect(other.getByTestId("other-g1")).toBeVisible();
  await expect(other.getByRole("status")).toHaveCount(0);

  state.nodes[1] = { ...NODES.gpu, online: false, error: "no answer from gpu-box:8000 in 2 s" };
  for (const r of state.jobs.filter((r) => !r.queued_here)) r.stale = true;
  await emit(page, "queue", {});

  await expect(other.getByTestId("other-g1")).toHaveAttribute("data-stale", "true");
  await expect(other.getByTestId("other-g2")).toHaveAttribute("data-stale", "true");
  await expect(other.getByTestId("other-g1")).toContainText("last known");
  await expect(other.getByTestId("other-g1")).toContainText("Version 9 pretraining");
  await expect(other.getByRole("status")).toContainText("gpu-box is not answering");
  await expect(other.getByTestId("node-gpu-box:8000")).toHaveClass(/opacity-60/);
  // The "Run on" list still names the machine, marked.
  await expect(page.getByRole("combobox", { name: "Run on: First waiting" }).getByRole("option")).toHaveText(["Any machine", "mac-mini (this machine)", "gpu-box (offline)"]);
});

test("the status strip, Builder tray and Chat picker say where each version is", async ({ page }) => {
  const state = twoMachines();
  await setup(page, state);

  await page.goto("/queue/");
  const strip = page.getByRole("status", { name: "Status" });
  await expect(strip).toContainText("Machines");
  await expect(strip).toContainText("mac-mini (this machine), gpu-box");
  await expect(strip).toContainText("2 jobs on other machines, 1 version to pull");

  await page.goto("/builder/");
  const tray = page.getByRole("region", { name: "Version history" });
  await expect(tray.getByTestId("where-1")).toHaveText("On this machine");
  await expect(tray.getByTestId("where-2")).toHaveText("On this machine");
  const available = tray.getByRole("list", { name: "Available on other machines" });
  await expect(available).toContainText("Version 7");
  await expect(available).toContainText("available, not local");
  await expect(available).toContainText("from gpu-box");
  await expect(available.getByRole("button", { name: "Pull version 7 from gpu-box" })).toBeEnabled();
  // A version this machine already has is not offered again.
  await expect(available).not.toContainText("Version 2");

  await page.goto("/chat/");
  await expect(page.getByTestId("where-1")).toContainText("on this machine");
  const shelf = page.getByRole("list", { name: "Available on other machines" });
  await expect(shelf).toContainText("Version 7");
  await expect(shelf).toContainText("from gpu-box");
  await expect(shelf.getByRole("button", { name: "Pull version 7 from gpu-box" })).toBeVisible();
});

test("Pull asks the owner for the version through the existing pull job", async ({ page }) => {
  const state = twoMachines();
  await setup(page, state);
  const pulls: string[] = [];
  await page.route("**/versions/7/pull*", (route) => {
    pulls.push(new URL(route.request().url()).search);
    return route.fulfill({ status: 202, json: { id: "pull-7", kind: "pull", status: "running", progress: "", error: "", created_at: 0, result: null } });
  });
  await page.route("**/jobs/pull-7", (route) => route.fulfill({ json: { id: "pull-7", kind: "pull", status: "done", progress: "", error: "", created_at: 0, result: { number: 3 } } }));
  await page.goto("/builder/");
  await page.getByRole("button", { name: "Pull version 7 from gpu-box" }).click();
  await expect.poll(() => pulls).toEqual(["?origin=gpu-box"]);
});

test("a cluster of one shows nothing about machines on any page", async ({ page }) => {
  // The server answers with empty lists when no peer is configured.
  await setup(page, null);

  await page.goto("/queue/");
  await expect(page.getByRole("heading", { level: 1, name: "Queue" })).toBeVisible();
  await expect(page.getByText("Nothing is training.")).toBeVisible();
  await expect(page.getByTestId("other-machines")).toHaveCount(0);
  await expect(page.getByTestId("now-place")).toHaveCount(0);
  await expect(page.getByText(/ON OTHER MACHINES|Run on|machine/i)).toHaveCount(0);
  const strip = page.getByRole("status", { name: "Status" });
  await expect(strip).toContainText("Queue");
  await expect(strip).not.toContainText("Machines");
  await expect(strip).not.toContainText("Elsewhere");

  await page.goto("/builder/");
  await expect(page.getByRole("region", { name: "Version history" })).toBeVisible();
  await expect(page.getByTestId("where-1")).toHaveCount(0);
  await expect(page.getByText("Available on other machines")).toHaveCount(0);

  await page.goto("/chat/");
  await expect(page.getByRole("button", { name: /Version 1/ })).toBeVisible();
  await expect(page.getByTestId("where-1")).toHaveCount(0);
  await expect(page.getByText("Available on other machines")).toHaveCount(0);
});

test("a cluster of one that lists itself alone shows no node UI even with jobs", async ({ page }) => {
  const state = twoMachines();
  state.nodes = [NODES.me];
  state.jobs = state.jobs.filter((r) => r.queued_here);
  await setup(page, state);
  await page.goto("/queue/");
  await expect(page.getByTestId("place-w1")).toHaveCount(0);
  await expect(page.getByTestId("other-machines")).toHaveCount(0);
  await expect(page.getByRole("combobox")).toHaveCount(0);
});

test("a node without the merged-queue route is a cluster of one", async ({ page }) => {
  await setup(page, null, false);
  await page.goto("/queue/");
  await expect(page.getByText("Nothing is training.")).toBeVisible();
  await expect(page.getByTestId("other-machines")).toHaveCount(0);
  await expect(page.getByRole("status", { name: "Status" })).not.toContainText("Machines");
});

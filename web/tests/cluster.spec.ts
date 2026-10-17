import { expect, test, type Page } from "@playwright/test";
import { changed, cleanList, deviceText, jobRows, machineJobs, progressText, refusedUs, runPage } from "../src/app/cluster/logic";
import { fakeEvents, mockApi } from "./qp-mock";

// The Cluster page (#319) against a mocked API: GET and PUT /cluster/nodes, GET /cluster/merged, GET /queue, GET and PUT
// /cluster/access, DELETE /queue/jobs/{id} and POST /cluster/remote/{id}/stop. The live spec (live.cluster-page.spec.ts) runs it
// against a real --fake server.

type Node = {
  address: string; local: boolean; hostname: string; version: string; device: string; device_name: string; memory_bytes: number;
  busy: boolean; dtypes: string[]; online: boolean; error: string; tailnet_name: string;
};
const node = (address: string, over: Partial<Node> = {}): Node => ({
  address, local: false, hostname: "", version: "", device: "", device_name: "", memory_bytes: 0, busy: false, dtypes: [], online: false,
  error: "", tailnet_name: "", ...over,
});
const ME = node("local", {
  local: true, hostname: "mac-mini", version: "1.4.0", device: "mps", device_name: "Apple M2", memory_bytes: 16 * 2 ** 30, online: true,
  dtypes: ["float32"], tailnet_name: "mac-mini.tail1234.ts.net",
});
const GPU = node("gpu-box", { hostname: "gpu-box", version: "1.4.0", device: "cuda", device_name: "RTX 4090", memory_bytes: 24 * 2 ** 30, online: true, busy: true });
const DOWN = node("old-laptop", { error: "old-laptop did not answer within 2 seconds." });
const LOCKED = node("brother-pc", { error: "brother-pc answered 401 to GET /node: This node does not know the caller." });

type Row = {
  id: string; kind: string; label: string; status: "running" | "waiting" | "held"; node: string; hostname: string; queued_on: string; queued_here: boolean;
  queued_address: string; pinned: string; waiting_reason: string; stale: boolean; note: string; step: number; total: number; eta_s: number | null;
};
const row = (id: string, label: string, over: Partial<Row> = {}): Row => ({
  id, kind: "pretrain", label, status: "waiting", node: "", hostname: "", queued_on: "mac-mini", queued_here: true, queued_address: "local",
  pinned: "", waiting_reason: "", stale: false, note: "", step: 0, total: 0, eta_s: null, ...over,
});
const qnode = (n: Node, seen = 0) => ({ address: n.address, hostname: n.hostname, pin: n.address, local: n.local, online: n.online, error: n.error, seen_at: seen });
const job = (id: string, label: string, over: Record<string, unknown> = {}) => ({ id, kind: "pretrain", label, note: "", params: {}, status: "waiting", eta_s: null, created_at: 0, error: "", node: "", owner: "", epoch: 0, ...over });
const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };
const quiet = {
  job: null, up_next: [], paused: false, plan: PLAN, step: 0, total: 0, phase: "", lr: null, eta_s: null, blocked: null, stopping: false,
  gate: { reason: null, resumes_at: null, supported: true },
};

type Setup = { addresses: string[]; known?: Node[]; me?: Node; jobs?: Row[]; queue?: Record<string, unknown>; login?: string; users?: string[]; tags?: string[] };
/** The server: the list it keeps, what each address answers, and every write it was sent. */
async function setup(page: Page, opts: Setup) {
  await fakeEvents(page);
  const known = opts.known ?? [GPU, DOWN, LOCKED];
  const me = opts.me ?? ME;
  const state = {
    addresses: [...opts.addresses], puts: [] as string[][], access: [] as unknown[], stops: [] as string[], removes: [] as string[],
    users: opts.users ?? [], tags: opts.tags ?? ["tag:trainer"],
  };
  const nodes = () => [me, ...state.addresses.map((a) => known.find((n) => n.address === a) ?? node(a, { error: `${a} refused the connection.` }))];
  await mockApi(page, {
    "GET /cluster/nodes": () => ({ body: nodes() }),
    "PUT /cluster/nodes": (route) => {
      const { addresses: sent } = route.request().postDataJSON() as { addresses: string[] };
      state.puts.push(sent);
      const bad = sent.find((a) => a.includes(" "));
      if (bad) return { status: 400, body: { detail: `"${bad}" is not a machine address: it has spaces.` } };
      state.addresses = sent;
      return { body: { addresses: sent } };
    },
    "GET /cluster/merged": () => ({ body: state.addresses.length ? { nodes: nodes().map((n) => qnode(n, n.online && !n.local ? 1_700_000_000 : 0)), jobs: opts.jobs ?? [] } : { nodes: [], jobs: [] } }),
    "GET /queue": () => ({ body: opts.queue ?? quiet }),
    "GET /cluster/access": () => ({ body: { allow_users: state.users, allow_tags: state.tags, login: opts.login ?? "" } }),
    "PUT /cluster/access": (route) => {
      const body = route.request().postDataJSON() as { allow_users: string[]; allow_tags: string[] };
      state.access.push(body);
      state.users = body.allow_users;
      state.tags = body.allow_tags;
      return { body: { ...body, login: opts.login ?? "" } };
    },
    "DELETE /queue/jobs/run1": () => {
      state.removes.push("run1");
      return { body: quiet };
    },
    "DELETE /queue/jobs/w1": () => {
      state.removes.push("w1");
      return { body: quiet };
    },
    "POST /cluster/remote/g1/stop": (route) => {
      state.stops.push((route.request().postDataJSON() as { address: string }).address);
      return { status: 204 };
    },
  });
  await page.goto("/cluster/");
  return state;
}

test("a cluster of one shows this computer, what it runs and what waits, with no other machines", async ({ page }) => {
  await setup(page, {
    addresses: [],
    queue: { ...quiet, job: job("run1", "Version 3 pretraining", { status: "running" }), step: 100, total: 1000, eta_s: 600, up_next: [job("w1", "First waiting")] },
  });
  const me = page.getByTestId("machine-local");
  await expect(me).toContainText("mac-mini · This computer");
  await expect(me).toContainText("Apple M2 (mps, 16 GB) · version 1.4.0");
  await expect(me).toContainText("Online");
  await expect(page.getByTestId("progress-run1")).toHaveText("Step 100 of 1,000 · about 10 min left");
  await expect(page.getByTestId("run-link-run1")).toHaveAttribute("href", "/training-run/");
  await expect(page.getByTestId("job-w1")).toContainText("First waiting");
  await expect(page.getByTestId("job-w1")).toContainText("Runs on any machine that fits it");
  await expect(page.getByTestId("seen-local")).toHaveCount(0);
  await expect(page.locator("[data-testid^='machine-']:not([data-testid='machine-local']):not([data-testid^='machine-error'])")).toHaveCount(0);
});

test("two peers: one busy with its job and a waiting one, one offline with the reason and when it last answered", async ({ page }) => {
  await setup(page, {
    addresses: ["gpu-box", "old-laptop"],
    jobs: [
      row("g1", "Version 9 pretraining", { status: "running", node: "gpu-box", hostname: "gpu-box", queued_on: "gpu-box", queued_here: false, queued_address: "gpu-box", step: 40, total: 100 }),
      row("g2", "Version 10 pretraining", { node: "gpu-box", hostname: "gpu-box", pinned: "gpu-box", queued_on: "gpu-box", queued_here: false, queued_address: "gpu-box" }),
    ],
  });
  const gpu = page.getByTestId("machine-gpu-box");
  await expect(gpu).toContainText("Online");
  await expect(gpu).toContainText("Busy");
  await expect(gpu).toContainText("RTX 4090 (cuda, 24 GB)");
  await expect(gpu.getByTestId("progress-g1")).toHaveText("Step 40 of 100");
  await expect(gpu.getByTestId("job-g2")).toContainText("Pinned to gpu-box");
  await expect(gpu.getByTestId("job-g1")).toContainText("queued on gpu-box");
  await expect(gpu.getByTestId("run-link-g1")).toHaveCount(0); // a peer's run has no page here
  await expect(gpu.getByTestId("seen-gpu-box")).toContainText("Last answered");
  const down = page.getByTestId("machine-old-laptop");
  await expect(down).toHaveAttribute("data-online", "false");
  await expect(down).toContainText("Not answering");
  await expect(page.getByTestId("machine-error-old-laptop")).toHaveText("old-laptop did not answer within 2 seconds.");
  await expect(down.getByTestId("seen-old-laptop")).toHaveText("Has not answered yet");
  await expect(page.getByTestId("machine-refused")).toHaveCount(0);
});

test("a peer that answers 401 explains the login to add, using this computer's login", async ({ page }) => {
  await setup(page, { addresses: ["gpu-box", "brother-pc"], login: "me@example.com" });
  const hint = page.getByTestId("machine-refused");
  await expect(hint).toContainText("brother-pc refused this computer");
  await expect(hint).toContainText("me@example.com");
  await expect(hint).toContainText("tag:trainer");
  await expect(page.getByTestId("machine-error-brother-pc")).toContainText("answered 401");
});

test("a 401 with no known login says to tag this computer or add its login", async ({ page }) => {
  await setup(page, { addresses: ["brother-pc"], login: "" });
  await expect(page.getByTestId("machine-refused")).toContainText("Tag this computer tag:trainer");
});

test("adds and removes machines, and saves the list", async ({ page }) => {
  const server = await setup(page, { addresses: ["gpu-box", "old-laptop"] });
  const save = page.getByRole("button", { name: "Save", exact: true });
  await expect(page.getByTestId("machine-gpu-box")).toBeVisible();
  await expect(save).toBeDisabled();

  await page.getByRole("button", { name: "Remove old-laptop" }).click();
  await expect(page.getByTestId("machine-old-laptop")).toHaveCount(0);
  await page.getByLabel("Machine to add").fill("  windows-box ");
  await page.getByRole("button", { name: "Add", exact: true }).first().click();
  await expect(page.getByTestId("machine-windows-box")).toContainText("Not saved yet");
  await page.getByLabel("Machine to add").fill("gpu-box");
  await page.getByRole("button", { name: "Add", exact: true }).first().click();
  await expect(page.getByTestId("machines-problem")).toHaveText("That machine is already in the list.");

  await save.click();
  await expect(page.getByText("Saved the list of machines.")).toBeVisible();
  expect(server.puts).toEqual([["gpu-box", "windows-box"]]);
  await expect(page.getByTestId("machine-error-windows-box")).toHaveText("windows-box refused the connection.");
  await expect(save).toBeDisabled();
});

test("a refused list says why inline and saves nothing", async ({ page }) => {
  const server = await setup(page, { addresses: ["gpu-box"] });
  await page.getByLabel("Machine to add").fill("my box");
  await page.getByRole("button", { name: "Add", exact: true }).first().click();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByTestId("machines-problem")).toHaveText('"my box" is not a machine address: it has spaces.');
  expect(server.addresses).toEqual(["gpu-box"]);
  await page.getByRole("button", { name: "Undo changes" }).first().click();
  await expect(page.getByTestId("machine-my box")).toHaveCount(0);
  await expect(page.getByTestId("machines-problem")).toHaveCount(0);
});

test("shows this computer's tailnet name and copies it", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await setup(page, { addresses: [] });
  await expect(page.getByTestId("tailnet-name")).toHaveText("mac-mini.tail1234.ts.net");
  await page.getByRole("button", { name: "Copy" }).click();
  await expect(page.getByText("Copied this computer's name.")).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe("mac-mini.tail1234.ts.net");
});

test("says so when Tailscale has not named this computer", async ({ page }) => {
  await setup(page, { addresses: [], me: { ...ME, tailnet_name: "" } });
  await expect(page.getByTestId("tailnet-name-missing")).toContainText("MagicDNS");
  await expect(page.getByRole("button", { name: "Copy" })).toHaveCount(0);
});

test("edits who may call this computer, and says it counts after a restart", async ({ page }) => {
  const server = await setup(page, { addresses: [], login: "me@example.com", users: ["old@example.com"] });
  const access = page.getByTestId("access");
  await expect(page.getByTestId("access-login")).toContainText("me@example.com");
  await expect(access).toContainText("next time the app starts");
  const save = access.getByRole("button", { name: "Save who may call" });
  await expect(save).toBeDisabled();
  await access.getByRole("button", { name: "Remove old@example.com" }).click();
  await access.getByLabel("Login to allow").fill(" friend@example.com ");
  await access.getByTestId("access-users").getByRole("button", { name: "Add", exact: true }).click();
  await access.getByLabel("Tag to allow").fill("tag:lab");
  await access.getByTestId("access-tags").getByRole("button", { name: "Add", exact: true }).click();
  await save.click();
  await expect(page.getByText("Saved. It counts the next time the app starts.")).toBeVisible();
  expect(server.access).toEqual([{ allow_users: ["friend@example.com"], allow_tags: ["tag:trainer", "tag:lab"] }]);
  await expect(save).toBeDisabled();
});

test("stops a job queued here through this computer's queue, and one a peer queued through the peer", async ({ page }) => {
  const server = await setup(page, {
    addresses: ["gpu-box"],
    queue: { ...quiet, job: job("run1", "Version 3 pretraining", { status: "running" }), step: 5, total: 10 },
    jobs: [
      row("run1", "Version 3 pretraining", { status: "running", node: "local", hostname: "mac-mini", step: 5, total: 10 }),
      row("g1", "Version 9 pretraining", { status: "running", node: "gpu-box", hostname: "gpu-box", queued_on: "gpu-box", queued_here: false, queued_address: "gpu-box" }),
    ],
  });
  await page.getByRole("button", { name: "Stop Version 3 pretraining" }).click();
  await expect.poll(() => server.removes).toEqual(["run1"]);
  await page.getByRole("button", { name: "Stop Version 9 pretraining" }).click();
  await expect.poll(() => server.stops).toEqual(["gpu-box"]);
  await expect(page.getByText("Asked gpu-box to stop Version 9 pretraining.")).toBeVisible();
});

test("a job queued here and placed on a peer links to its run and is stopped through this computer's queue", async ({ page }) => {
  const server = await setup(page, {
    addresses: ["gpu-box"],
    jobs: [row("run1", "Version 3 pretraining", { status: "running", node: "gpu-box", hostname: "gpu-box" })],
  });
  await expect(page.getByTestId("machine-gpu-box").getByTestId("run-link-run1")).toHaveAttribute("href", "/training-run/");
  await page.getByRole("button", { name: "Stop Version 3 pretraining" }).click();
  await expect.poll(() => server.removes).toEqual(["run1"]);
});

test("cluster logic: device text, 401, cleaning, rows, per-machine jobs and progress", () => {
  expect(deviceText(ME)).toBe("Apple M2 (mps, 16 GB)");
  expect(deviceText(node("x", { device: "cpu" }))).toBe("cpu");
  expect(deviceText(DOWN)).toBe("");
  expect([GPU, DOWN, LOCKED].map(refusedUs)).toEqual([false, false, true]);
  expect(cleanList([" a ", "", "b", "a", "A"])).toEqual(["a", "b", "A"]);
  expect(changed(["a", "b"], ["a", "b"])).toBe(false);
  expect(changed(["b", "a"], ["a", "b"])).toBe(true);

  const rows = [
    row("a", "A", { status: "running", node: "gpu-box", queued_address: "local", step: 1, total: 4 }),
    row("b", "B", { status: "held", queued_address: "gpu-box" }),
    row("c", "C", { queued_address: "local" }),
  ];
  expect(machineJobs("gpu-box", rows).running.map((r) => r.id)).toEqual(["a"]);
  expect(machineJobs("gpu-box", rows).waiting.map((r) => r.id)).toEqual(["b"]);
  expect(machineJobs("local", rows).waiting.map((r) => r.id)).toEqual(["c"]);
  expect(progressText(rows[0])).toBe("Step 1 of 4");
  expect(progressText(row("z", "Z", { status: "running" }))).toBe("Starting");
  expect(runPage(rows[0])).toBe("/training-run");
  expect(runPage(rows[1])).toBeNull();
  expect(runPage({ ...rows[0], queued_here: false })).toBeNull();
  // A cluster of one has no merged rows: this machine's own queue stands in.
  const own = jobRows({ nodes: [], jobs: [] }, { ...quiet, job: job("r", "R", { status: "running" }), step: 2, total: 8, up_next: [job("w", "W", { status: "held" })] } as never);
  expect(own.map((r) => [r.id, r.status, r.node, r.queued_address])).toEqual([["r", "running", "local", "local"], ["w", "held", "local", "local"]]);
  expect(jobRows({ nodes: [], jobs: [] }, undefined)).toEqual([]);
});

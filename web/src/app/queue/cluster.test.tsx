import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import ChatPage from "@/app/chat/page";
import { VersionTray } from "@/app/builder/version-tray";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import QueuePage from "./page";

// The merged queue and merged versions (#165) against a mocked API (the component tier of tests/cluster-queue.spec.ts):
// GET /cluster/merged lists the machines and every machine's jobs (the server polls the peers), GET /versions/published lists what
// each machine published. A cluster of one gets empty lists and the pages show nothing about machines. The live spec
// (tests/live.cluster.spec.ts) runs the same page against two real --fake servers.

type Row = {
  id: string; kind: string; label: string; status: string; node: string; hostname: string; queued_on: string; queued_here: boolean;
  pinned: string; waiting_reason: string; stale: boolean; note: string;
};
const NODES = {
  me: { address: "local", hostname: "mac-mini", pin: "mac-mini", local: true, online: true, error: "", seen_at: 0 },
  gpu: { address: "gpu-box:8000", hostname: "gpu-box", pin: "gpu-box:8000", local: false, online: true, error: "", seen_at: 1_700_000_000 },
};
const job = (id: string, label: string, over: Record<string, unknown> = {}) => ({ id, kind: "pretrain", label, note: "", params: {}, status: "waiting", eta_s: null, created_at: 0, error: "", ...over });
const row = (id: string, label: string, over: Partial<Row> = {}): Row => ({
  id, kind: "pretrain", label, status: "waiting", node: "", hostname: "", queued_on: "mac-mini", queued_here: true, pinned: "", waiting_reason: "", stale: false, note: "", ...over,
});
const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };
const published = (number: number, over: Record<string, unknown> = {}) => ({
  key: `gpu-box:8000/${number}`, number, origin: "gpu-box", mine: false, local: false, local_number: null, name: `version ${number}`, change: `Design ${number}`, scorecard: {}, ...over,
});
const version = (number: number) => ({
  number, name: `version ${number}`, change: `change ${number}`, status: "trained", verdict: "", best_checkpoint: `ckpt-${number}`,
  parent: number > 1 ? number - 1 : null, scorecard: {}, spec: {}, prediction: {},
});
const doneJob = (id: string, result: unknown, kind = "versions") => ({ id, kind, status: "done", result, progress: "", error: "", created_at: 0 });

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

/** `state` null is a cluster of one: the server answers with empty lists, and 409 for the version store. */
function server(state: ReturnType<typeof twoMachines> | null, { clusterRoute = true, extra = {} }: { clusterRoute?: boolean; extra?: Parameters<typeof mockApi>[0] } = {}) {
  const pins: { path: string; body: unknown }[] = [];
  const cluster = () => ({ body: state ? { nodes: state.nodes, jobs: state.jobs } : { nodes: [], jobs: [] } });
  const queue = () => ({ body: state?.queue ?? { job: null, up_next: [], paused: false, plan: PLAN, step: 0, total: 0, phase: "", lr: null, eta_s: null, blocked: null, stopping: false, gate: { reason: null, resumes_at: null, supported: true } } });
  const pin = ({ path, body }: { path: string; body: unknown }) => {
    const { pinned } = body as { pinned: string | null };
    pins.push({ path, body });
    if (state) {
      const mine = state.jobs.find((r) => r.id === path.split("/")[3]);
      const node = pinned ? state.nodes.find((n) => n.pin === pinned || n.hostname === pinned) : undefined;
      if (mine) Object.assign(mine, { pinned: pinned ?? "", node: node?.address ?? "", hostname: node?.hostname ?? (pinned ?? "") });
    }
    return queue();
  };
  const calls = mockApi({
    "GET /cluster/merged": () => (clusterRoute ? cluster() : undefined),
    "GET /queue": queue,
    "GET /queue/plan": { body: PLAN },
    "GET /queue/jobs/w1/outlook": { body: { outlook: "after about 10 min" } },
    "GET /queue/jobs/w2/outlook": { body: { outlook: "after about 20 min" } },
    "POST /queue/jobs/*/pin": pin,
    "GET /versions": { body: [version(1), version(2)] },
    "GET /versions/published": () => (state ? { body: state.published } : { status: 409, body: { detail: "no store" } }),
    // Chat resolves the best version and each checkpoint's step through jobs that are already done.
    "POST /versions/best": { body: doneJob("best", version(2)) },
    "POST /versions/*/checkpoint-step": { body: doneJob("step", 100) },
    ...extra,
  });
  return { pins, calls };
}

const machines = (options: HTMLElement) => within(options).getAllByRole("option").map((o) => o.textContent);
const strip = () => screen.getByRole("status", { name: "Status" });

describe("Queue page with several machines", () => {
  test("shows every machine's jobs with the node on each row", async () => {
    server(twoMachines());
    renderPage(<QueuePage />);

    expect(await screen.findByTestId("now-place")).toHaveTextContent(/^Running on mac-mini$/);
    expect(await screen.findByTestId("place-w1")).toHaveTextContent(/^Runs on any machine that fits it$/);
    expect(screen.getByTestId("place-w2")).toHaveTextContent(/^Runs on any machine that fits it$/);

    const other = within(await screen.findByTestId("other-machines"));
    expect(other.getByTestId("node-local")).toHaveTextContent("mac-mini (this machine)");
    expect(other.getByTestId("node-gpu-box:8000")).toHaveTextContent("gpu-box");
    expect(other.getByTestId("other-g1")).toHaveTextContent("Version 9 pretraining");
    expect(other.getByTestId("other-g1")).toHaveTextContent("Running on gpu-box");
    expect(other.getByTestId("other-g2")).toHaveTextContent("Pinned to gpu-box");
    expect(other.getByTestId("other-g2")).toHaveTextContent("queued on gpu-box");
    expect(other.getByTestId("other-g1")).not.toHaveAttribute("data-stale", "true");
    // This machine's own jobs are not repeated in the other machines' card.
    expect(other.queryByTestId("other-w1")).toBeNull();
    // The card stays; each machine's state and its controls live on the Cluster page (#319).
    expect(other.getByTestId("queue-cluster-link")).toHaveAttribute("href", "/cluster");
  });

  test("a job running on another machine shows the note about its checkpoint in the other machines card, once", async () => {
    const state = twoMachines();
    const note = "The checkpoint was not stored: the bucket is full.";
    state.jobs.push(row("far1", "Version 4 pretraining", { status: "running", node: "gpu-box:8000", hostname: "gpu-box", note }));
    server(state);
    renderPage(<QueuePage />);

    const other = within(await screen.findByTestId("other-machines"));
    const far = await other.findByTestId("other-far1");
    expect(far).toHaveTextContent("Version 4 pretraining");
    expect(far).toHaveTextContent("Running on gpu-box");
    expect(far).toHaveTextContent(note);
    expect(screen.getAllByText(note)).toHaveLength(1);
    expect(other.getByTestId("other-g1")).not.toHaveTextContent("bucket");
  });

  test("a job still running on a machine that was removed from the list keeps its row and note", async () => {
    const state = twoMachines();
    const note = "The checkpoint was not stored: the bucket is full.";
    state.nodes = [NODES.me];
    state.jobs = [row("far1", "Version 4 pretraining", { status: "running", node: "", hostname: "gpu-box:8000", note })];
    server(state);
    renderPage(<QueuePage />);

    const far = await screen.findByTestId("other-far1");
    expect(far).toHaveTextContent("Running on gpu-box:8000");
    expect(far).toHaveTextContent(note);
  });

  test("pinning a waiting job to the other machine changes where its row says it will run", async () => {
    const state = twoMachines();
    const { pins } = server(state);
    const user = userEvent.setup();
    renderPage(<QueuePage />);
    expect(await screen.findByTestId("place-w1")).toHaveTextContent(/^Runs on any machine that fits it$/);

    const select = screen.getByRole("combobox", { name: "Run on: First waiting" });
    expect(machines(select)).toEqual(["Any machine", "mac-mini (this machine)", "gpu-box"]);
    await user.selectOptions(select, "gpu-box");
    await waitFor(() => expect(screen.getByTestId("place-w1")).toHaveTextContent(/^Pinned to gpu-box: it runs there or waits$/));
    await waitFor(() => expect(select).toHaveValue("gpu-box:8000"));
    expect(screen.getByTestId("place-w2")).toHaveTextContent(/^Runs on any machine that fits it$/);
    expect(pins).toEqual([{ path: "/queue/jobs/w1/pin", body: { pinned: "gpu-box:8000" } }]);

    await user.selectOptions(select, "Any machine");
    await waitFor(() => expect(screen.getByTestId("place-w1")).toHaveTextContent(/^Runs on any machine that fits it$/));
    expect(pins.at(-1)).toEqual({ path: "/queue/jobs/w1/pin", body: { pinned: null } });
  });

  test("two machines with the same hostname are told apart by their address", async () => {
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
    server(state);
    renderPage(<QueuePage />, { route: "/queue", shell: true });

    const other = within(await screen.findByTestId("other-machines"));
    expect(other.getByTestId("node-local")).toHaveTextContent(/^mac-mini \(this machine\)$/);
    expect(other.getByTestId("node-10.0.0.2:8000")).toHaveTextContent(/^gpu-box \(10\.0\.0\.2:8000\)$/);
    expect(other.getByTestId("node-10.0.0.3:8000")).toHaveTextContent(/^gpu-box \(10\.0\.0\.3:8000\)$/);
    expect(other.getByTestId("other-g1")).toHaveTextContent("Running on gpu-box (10.0.0.2:8000)");
    expect(other.getByTestId("other-g1")).toHaveTextContent("queued on gpu-box (10.0.0.2:8000)");
    expect(other.getByTestId("other-g2")).toHaveTextContent("Pinned to gpu-box (10.0.0.3:8000)");
    expect(await screen.findByTestId("place-w1")).toHaveTextContent(/^Pinned to gpu-box \(10\.0\.0\.3:8000\): it runs there or waits$/);

    const select = screen.getByRole("combobox", { name: "Run on: First waiting" });
    expect(machines(select)).toEqual(["Any machine", "mac-mini (this machine)", "gpu-box (10.0.0.2:8000)", "gpu-box (10.0.0.3:8000)"]);
    expect(select).toHaveValue("10.0.0.3:8000");

    await waitFor(() => expect(strip()).toHaveTextContent("mac-mini (this machine), gpu-box (10.0.0.2:8000), gpu-box (10.0.0.3:8000)"));
  });

  test("a machine that stops answering keeps its last rows, greyed and labelled", async () => {
    const state = twoMachines();
    server(state);
    renderPage(<QueuePage />);
    const other = within(await screen.findByTestId("other-machines"));
    expect(await other.findByTestId("other-g1")).toBeVisible();
    expect(other.queryAllByRole("status")).toHaveLength(0);

    state.nodes[1] = { ...NODES.gpu, online: false, error: "no answer from gpu-box:8000 in 2 s" };
    for (const r of state.jobs.filter((r) => !r.queued_here)) r.stale = true;
    act(() => emit("queue", {}));

    await waitFor(() => expect(other.getByTestId("other-g1")).toHaveAttribute("data-stale", "true"));
    expect(other.getByTestId("other-g2")).toHaveAttribute("data-stale", "true");
    expect(other.getByTestId("other-g1")).toHaveTextContent("last known");
    expect(other.getByTestId("other-g1")).toHaveTextContent("Version 9 pretraining");
    expect(other.getByRole("status")).toHaveTextContent("gpu-box is not answering");
    expect(other.getByTestId("node-gpu-box:8000")).toHaveClass("opacity-60");
    // The "Run on" list still names the machine, marked.
    expect(machines(screen.getByRole("combobox", { name: "Run on: First waiting" }))).toEqual(["Any machine", "mac-mini (this machine)", "gpu-box (offline)"]);
  });

  test("a cluster of one shows nothing about machines", async () => {
    // The server answers with empty lists when no peer is configured.
    server(null);
    renderPage(<QueuePage />, { route: "/queue", shell: true });

    expect(await screen.findByRole("heading", { level: 1, name: "Queue" })).toBeVisible();
    expect(await screen.findByText("Nothing is training.")).toBeVisible();
    await waitFor(() => expect(strip()).toHaveTextContent("Nothing is training"));
    expect(screen.queryByTestId("other-machines")).toBeNull();
    expect(screen.queryByTestId("now-place")).toBeNull();
    expect(screen.queryAllByText(/ON OTHER MACHINES|Run on|machine/i)).toHaveLength(0);
    expect(strip()).toHaveTextContent("Queue");
    expect(strip()).not.toHaveTextContent("Machines");
    expect(strip()).not.toHaveTextContent("Elsewhere");
  });

  // A machine that lists itself alone, with its own jobs in the merged list.
  async function aloneWithJobs() {
    const state = twoMachines();
    state.nodes = [NODES.me];
    state.jobs = state.jobs.filter((r) => r.queued_here);
    const { calls } = server(state);
    renderPage(<QueuePage />);
    // both reads have landed and the waiting jobs are drawn before the absences mean anything
    expect(await screen.findByText("First waiting")).toBeVisible();
    await waitFor(() => expect(calls.some((c) => c.path === "/cluster/merged")).toBe(true));
    await new Promise((resolve) => setTimeout(resolve, 50));
  }

  test("a cluster of one that lists itself alone shows no per-job machine UI even with jobs", async () => {
    await aloneWithJobs();
    expect(screen.queryByTestId("place-w1")).toBeNull();
    expect(screen.queryAllByRole("combobox")).toHaveLength(0);
  });

  // The card lists jobs that run on other machines, so a lone machine's own jobs do not show it (#316, fixed on the base branch).
  test("a cluster of one that lists itself alone shows no Other machines card", async () => {
    await aloneWithJobs();
    expect(screen.queryByTestId("other-machines")).toBeNull();
  });

  test("a node without the merged-queue route is a cluster of one", async () => {
    server(null, { clusterRoute: false });
    renderPage(<QueuePage />, { route: "/queue", shell: true });
    expect(await screen.findByText("Nothing is training.")).toBeVisible();
    await waitFor(() => expect(strip()).toHaveTextContent("Nothing is training"));
    expect(screen.queryByTestId("other-machines")).toBeNull();
    expect(strip()).not.toHaveTextContent("Machines");
  });
});

describe("the status strip, Builder tray and Chat picker in a cluster", () => {
  test("the status strip names the machines and what is elsewhere", async () => {
    server(twoMachines());
    renderPage(<QueuePage />, { route: "/queue", shell: true });
    await waitFor(() => expect(strip()).toHaveTextContent("Machines"));
    await waitFor(() => expect(strip()).toHaveTextContent("mac-mini (this machine), gpu-box"));
    await waitFor(() => expect(strip()).toHaveTextContent("2 jobs on other machines, 1 version to pull"));
  });

  test("the Builder version tray says where each version is and offers the ones this machine lacks", async () => {
    server(twoMachines());
    renderPage(<VersionTray />, { route: "/builder" });

    const tray = within(await screen.findByRole("region", { name: "Version history" }));
    expect(await tray.findByTestId("where-1")).toHaveTextContent(/^On this machine$/);
    expect(tray.getByTestId("where-2")).toHaveTextContent(/^On this machine$/);
    const available = within(await tray.findByRole("list", { name: "Available on other machines" }));
    expect(available.getByText("Version 7")).toBeVisible();
    expect(available.getByText("available, not local")).toBeVisible();
    expect(available.getByText("from gpu-box")).toBeVisible();
    expect(available.getByRole("button", { name: "Pull version 7 from gpu-box" })).toBeEnabled();
    // A version this machine already has is not offered again.
    expect(available.queryByText("Version 2")).toBeNull();
  });

  test("the Chat picker says where each version is and offers the ones this machine lacks", async () => {
    server(twoMachines());
    renderPage(<ChatPage />, { route: "/chat" });

    expect(await screen.findByTestId("where-1")).toHaveTextContent("on this machine");
    const shelf = within(await screen.findByRole("list", { name: "Available on other machines" }));
    expect(shelf.getByText("Version 7")).toBeVisible();
    expect(shelf.getByText("from gpu-box")).toBeVisible();
    expect(shelf.getByRole("button", { name: "Pull version 7 from gpu-box" })).toBeVisible();
  });

  test("Pull asks the owner for the version through the existing pull job", async () => {
    const { calls } = server(twoMachines(), {
      extra: {
        "POST /versions/7/pull": { status: 202, body: { id: "pull-7", kind: "pull", status: "running", progress: "", error: "", created_at: 0, result: null } },
        "GET /jobs/pull-7": { body: { id: "pull-7", kind: "pull", status: "done", progress: "", error: "", created_at: 0, result: { number: 3 } } },
      },
    });
    const user = userEvent.setup();
    renderPage(<VersionTray />, { route: "/builder" });
    await user.click(await screen.findByRole("button", { name: "Pull version 7 from gpu-box" }));
    expect(await screen.findByText("Pulled version 7 from gpu-box.")).toBeVisible();
    expect(calls.filter((c) => c.method === "POST" && c.path === "/versions/7/pull").map((c) => c.search)).toEqual(["?origin=gpu-box"]);
  });

  test("a cluster of one shows nothing about machines in the Builder tray or the Chat picker", async () => {
    server(null);
    const tray = renderPage(<VersionTray />, { route: "/builder" });
    const region = within(await screen.findByRole("region", { name: "Version history" }));
    expect(await region.findByText("Version 1")).toBeVisible();
    expect(region.queryByTestId("where-1")).toBeNull();
    expect(region.queryByText("Available on other machines")).toBeNull();
    tray.unmount();

    renderPage(<ChatPage />, { route: "/chat" });
    expect(await screen.findByRole("button", { name: /Version 1/ })).toBeVisible();
    expect(screen.queryByTestId("where-1")).toBeNull();
    expect(screen.queryByText("Available on other machines")).toBeNull();
  });
});

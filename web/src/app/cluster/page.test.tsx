import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import ClusterPage from "./page";

// The Cluster page (#319) against a mocked API: GET and PUT /cluster/nodes, GET /cluster/merged, GET /queue, GET and PUT
// /cluster/access, DELETE /queue/jobs/{id} and POST /cluster/remote/{id}/stop. The live spec (tests/live.cluster-page.spec.ts)
// runs it against a real --fake server.

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

type Setup = { addresses: string[]; me?: Node; jobs?: Row[]; queue?: Record<string, unknown>; login?: string; users?: string[]; tags?: string[] };
/** The server: the list it keeps, what each address answers, and every write it was sent. */
function server(opts: Setup) {
  const known = [GPU, DOWN, LOCKED];
  const me = opts.me ?? ME;
  const state = {
    addresses: [...opts.addresses], puts: [] as string[][], access: [] as unknown[], stops: [] as string[], removes: [] as string[],
    users: opts.users ?? [], tags: opts.tags ?? ["tag:trainer"],
  };
  const nodes = () => [me, ...state.addresses.map((a) => known.find((n) => n.address === a) ?? node(a, { error: `${a} refused the connection.` }))];
  mockApi({
    "GET /cluster/nodes": () => ({ body: nodes() }),
    "PUT /cluster/nodes": ({ body }) => {
      const { addresses: sent } = body as { addresses: string[] };
      state.puts.push(sent);
      const bad = sent.find((a) => a.includes(" "));
      if (bad) return { status: 400, body: { detail: `"${bad}" is not a machine address: it has spaces.` } };
      state.addresses = sent;
      return { body: { addresses: sent } };
    },
    "GET /cluster/merged": () => ({ body: state.addresses.length ? { nodes: nodes().map((n) => qnode(n, n.online && !n.local ? 1_700_000_000 : 0)), jobs: opts.jobs ?? [] } : { nodes: [], jobs: [] } }),
    "GET /queue": () => ({ body: opts.queue ?? quiet }),
    "GET /cluster/access": () => ({ body: { allow_users: state.users, allow_tags: state.tags, login: opts.login ?? "" } }),
    "PUT /cluster/access": ({ body }) => {
      const sent = body as { allow_users: string[]; allow_tags: string[] };
      state.access.push(sent);
      state.users = sent.allow_users;
      state.tags = sent.allow_tags;
      return { body: { ...sent, login: opts.login ?? "" } };
    },
    "DELETE /queue/jobs/*": ({ path }) => {
      state.removes.push(path.split("/").pop() ?? "");
      return { body: quiet };
    },
    "POST /cluster/remote/*/stop": ({ body }) => {
      state.stops.push((body as { address: string }).address);
      return { status: 204 };
    },
  });
  return state;
}

const text = (id: string) => screen.getByTestId(id);

describe("Cluster page", () => {
  test("a cluster of one shows this computer, what it runs and what waits, with no other machines", async () => {
    server({
      addresses: [],
      queue: { ...quiet, job: job("run1", "Version 3 pretraining", { status: "running" }), step: 100, total: 1000, eta_s: 600, up_next: [job("w1", "First waiting")] },
    });
    renderPage(<ClusterPage />, { route: "/cluster" });
    const me = await screen.findByTestId("machine-local");
    expect(me).toHaveTextContent("mac-mini · This computer");
    expect(me).toHaveTextContent("Apple M2 (mps, 16 GB) · version 1.4.0");
    expect(me).toHaveTextContent("Online");
    expect(await screen.findByTestId("progress-run1")).toHaveTextContent("Step 100 of 1,000 · about 10 min left");
    expect(text("run-link-run1")).toHaveAttribute("href", "/training-run");
    expect(text("job-w1")).toHaveTextContent("First waiting");
    expect(text("job-w1")).toHaveTextContent("Runs on any machine that fits it");
    expect(screen.queryByTestId("seen-local")).toBeNull();
    expect(screen.getAllByTestId(/^machine-(?!error|refused)/)).toHaveLength(1);
  });

  test("two peers: one busy with its job and a waiting one, one offline with the reason and when it last answered", async () => {
    server({
      addresses: ["gpu-box", "old-laptop"],
      jobs: [
        row("g1", "Version 9 pretraining", { status: "running", node: "gpu-box", hostname: "gpu-box", queued_on: "gpu-box", queued_here: false, queued_address: "gpu-box", step: 40, total: 100 }),
        row("g2", "Version 10 pretraining", { node: "gpu-box", hostname: "gpu-box", pinned: "gpu-box", queued_on: "gpu-box", queued_here: false, queued_address: "gpu-box" }),
      ],
    });
    renderPage(<ClusterPage />, { route: "/cluster" });
    const gpu = within(await screen.findByTestId("machine-gpu-box"));
    expect(await gpu.findByTestId("progress-g1")).toHaveTextContent(/^Step 40 of 100$/);
    expect(gpu.getByText("Online")).toBeInTheDocument();
    expect(gpu.getByText("Busy")).toBeInTheDocument();
    expect(gpu.getByTestId("device-gpu-box")).toHaveTextContent("RTX 4090 (cuda, 24 GB)");
    expect(gpu.getByTestId("job-g2")).toHaveTextContent("Pinned to gpu-box");
    expect(gpu.getByTestId("job-g1")).toHaveTextContent("queued on gpu-box");
    expect(gpu.queryByTestId("run-link-g1")).toBeNull(); // a peer's run has no page here
    expect(gpu.getByTestId("seen-gpu-box")).toHaveTextContent("Last answered");
    const down = text("machine-old-laptop");
    expect(down).toHaveAttribute("data-online", "false");
    expect(down).toHaveTextContent("Not answering");
    expect(text("machine-error-old-laptop")).toHaveTextContent(/^old-laptop did not answer within 2 seconds\.$/);
    expect(within(down).getByTestId("seen-old-laptop")).toHaveTextContent(/^Has not answered yet$/);
    expect(screen.queryByTestId("machine-refused")).toBeNull();
  });

  test("a peer that answers 401 explains the login to add, using this computer's login", async () => {
    server({ addresses: ["gpu-box", "brother-pc"], login: "me@example.com" });
    renderPage(<ClusterPage />, { route: "/cluster" });
    const hint = await screen.findByTestId("machine-refused");
    expect(hint).toHaveTextContent("brother-pc refused this computer");
    expect(hint).toHaveTextContent("me@example.com");
    expect(hint).toHaveTextContent("tag:trainer");
    expect(text("machine-error-brother-pc")).toHaveTextContent("answered 401");
  });

  test("a 401 with no known login says to tag this computer or add its login", async () => {
    server({ addresses: ["brother-pc"], login: "" });
    renderPage(<ClusterPage />, { route: "/cluster" });
    expect(await screen.findByTestId("machine-refused")).toHaveTextContent("Tag this computer tag:trainer");
  });

  test("adds and removes machines, and saves the list", async () => {
    const user = userEvent.setup();
    const state = server({ addresses: ["gpu-box", "old-laptop"] });
    renderPage(<ClusterPage />, { route: "/cluster" });
    const save = await screen.findByRole("button", { name: "Save" });
    await screen.findByTestId("machine-gpu-box");
    expect(save).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Remove old-laptop" }));
    expect(screen.queryByTestId("machine-old-laptop")).toBeNull();
    await user.type(screen.getByLabelText("Machine to add"), "  windows-box ");
    await user.click(screen.getAllByRole("button", { name: "Add" })[0]);
    expect(text("machine-windows-box")).toHaveTextContent("Not saved yet");
    await user.type(screen.getByLabelText("Machine to add"), "gpu-box");
    await user.click(screen.getAllByRole("button", { name: "Add" })[0]);
    expect(text("machines-problem")).toHaveTextContent("That machine is already in the list.");

    await user.click(save);
    expect(await screen.findByText("Saved the list of machines.")).toBeInTheDocument();
    expect(state.puts).toEqual([["gpu-box", "windows-box"]]);
    expect(await screen.findByTestId("machine-error-windows-box")).toHaveTextContent(/^windows-box refused the connection\.$/);
    await waitFor(() => expect(save).toBeDisabled());
  });

  test("a refused list says why inline and saves nothing", async () => {
    const user = userEvent.setup();
    const state = server({ addresses: ["gpu-box"] });
    renderPage(<ClusterPage />, { route: "/cluster" });
    await screen.findByTestId("machine-gpu-box");
    await user.type(screen.getByLabelText("Machine to add"), "my box");
    await user.click(screen.getAllByRole("button", { name: "Add" })[0]);
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByTestId("machines-problem")).toHaveTextContent('"my box" is not a machine address: it has spaces.');
    expect(state.addresses).toEqual(["gpu-box"]);
    await user.click(screen.getAllByRole("button", { name: "Undo changes" })[0]);
    expect(screen.queryByTestId("machine-my box")).toBeNull();
    expect(screen.queryByTestId("machines-problem")).toBeNull();
  });

  test("shows this computer's tailnet name and copies it", async () => {
    const user = userEvent.setup();
    server({ addresses: [] });
    renderPage(<ClusterPage />, { route: "/cluster" });
    expect(await screen.findByTestId("tailnet-name")).toHaveTextContent(/^mac-mini\.tail1234\.ts\.net$/);
    await user.click(screen.getByRole("button", { name: "Copy" }));
    expect(await screen.findByText("Copied this computer's name.")).toBeInTheDocument();
    expect(await navigator.clipboard.readText()).toBe("mac-mini.tail1234.ts.net");
  });

  test("says so when Tailscale has not named this computer", async () => {
    server({ addresses: [], me: { ...ME, tailnet_name: "" } });
    renderPage(<ClusterPage />, { route: "/cluster" });
    expect(await screen.findByTestId("tailnet-name-missing")).toHaveTextContent("MagicDNS");
    expect(screen.queryByRole("button", { name: "Copy" })).toBeNull();
  });

  test("edits who may call this computer, and says it counts after a restart", async () => {
    const user = userEvent.setup();
    const state = server({ addresses: [], login: "me@example.com", users: ["old@example.com"] });
    renderPage(<ClusterPage />, { route: "/cluster" });
    const access = within(await screen.findByTestId("access"));
    expect(await access.findByTestId("access-login")).toHaveTextContent("me@example.com");
    expect(screen.getByTestId("access")).toHaveTextContent("next time the app starts");
    const save = access.getByRole("button", { name: "Save who may call" });
    expect(save).toBeDisabled();
    await user.click(access.getByRole("button", { name: "Remove old@example.com" }));
    await user.type(access.getByLabelText("Login to allow"), " friend@example.com ");
    await user.click(within(access.getByTestId("access-users")).getByRole("button", { name: "Add" }));
    await user.type(access.getByLabelText("Tag to allow"), "tag:lab");
    await user.click(within(access.getByTestId("access-tags")).getByRole("button", { name: "Add" }));
    await user.click(save);
    expect(await screen.findByText("Saved. It counts the next time the app starts.")).toBeInTheDocument();
    expect(state.access).toEqual([{ allow_users: ["friend@example.com"], allow_tags: ["tag:trainer", "tag:lab"] }]);
    await waitFor(() => expect(save).toBeDisabled());
  });

  test("stops a job queued here through this computer's queue, and one a peer queued through the peer", async () => {
    const user = userEvent.setup();
    const state = server({
      addresses: ["gpu-box"],
      queue: { ...quiet, job: job("run1", "Version 3 pretraining", { status: "running" }), step: 5, total: 10 },
      jobs: [
        row("run1", "Version 3 pretraining", { status: "running", node: "local", hostname: "mac-mini", step: 5, total: 10 }),
        row("g1", "Version 9 pretraining", { status: "running", node: "gpu-box", hostname: "gpu-box", queued_on: "gpu-box", queued_here: false, queued_address: "gpu-box" }),
      ],
    });
    renderPage(<ClusterPage />, { route: "/cluster" });
    await user.click(await screen.findByRole("button", { name: "Stop Version 3 pretraining" }));
    await waitFor(() => expect(state.removes).toEqual(["run1"]));
    await user.click(screen.getByRole("button", { name: "Stop Version 9 pretraining" }));
    await waitFor(() => expect(state.stops).toEqual(["gpu-box"]));
    expect(await screen.findByText("Asked gpu-box to stop Version 9 pretraining.")).toBeInTheDocument();
  });

  test("a job queued here and placed on a peer links to its run and is stopped through this computer's queue", async () => {
    const user = userEvent.setup();
    const state = server({
      addresses: ["gpu-box"],
      jobs: [row("run1", "Version 3 pretraining", { status: "running", node: "gpu-box", hostname: "gpu-box" })],
    });
    renderPage(<ClusterPage />, { route: "/cluster" });
    const gpu = within(await screen.findByTestId("machine-gpu-box"));
    expect(await gpu.findByTestId("run-link-run1")).toHaveAttribute("href", "/training-run");
    await user.click(screen.getByRole("button", { name: "Stop Version 3 pretraining" }));
    await waitFor(() => expect(state.removes).toEqual(["run1"]));
  });

  test("the sidebar has a Cluster entry", async () => {
    server({ addresses: [] });
    renderPage(<ClusterPage />, { route: "/cluster", shell: true });
    expect(await screen.findByRole("link", { name: "Cluster" })).toHaveAttribute("href", "/cluster");
  });
});

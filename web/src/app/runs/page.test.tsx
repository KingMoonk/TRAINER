import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import RunsPage from "./page";

// The Runs page against a mocked API: GET /queue, /cluster/jobs, /distributed/runs and /cluster/nodes, merged into three sections.

const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };
const job = (id: string, label: string, over: Record<string, unknown> = {}) => ({ id, kind: "pretrain", label, note: "", params: {}, status: "waiting", eta_s: null, created_at: 0, error: "", ...over });
const node = (address: string, hostname: string, over: Record<string, unknown> = {}) => ({
  address, local: address === "local", hostname, version: "1", device: "mps", device_name: "", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "", ...over,
});
const member = (name: string, address: string) => ({ node: name, address, local: address === "local", device: "", device_name: "", state: "training", note: "", can_stop: false, steps: 0, rounds_done: 0, missed: 0, sent: true, last_delta_bytes: null, last_delta_dtype: "", tokens_per_second: null });
const diloco = (over: Record<string, unknown> = {}) => ({
  id: "v5run", label: "from scratch", version: 5, saved_version: null, state: "running", note: "", started_at: 5, owner: "mac", round: 7, rounds: 40, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16", can_stop: true,
  history: [{ round: 7, loss: 3.41, val_loss: 3.47, seconds: 600, step: 700, members: 2, contributed: 2, dropped: [], bytes_up: 0, bytes_down: 0 }],
  members: [member("mac", "local"), member("beast", "beast:8000")],
  ...over,
});

function server(runs: unknown[] = [diloco()], failed: unknown[] = [], extra: Parameters<typeof mockApi>[0] = {}) {
  const up_next = [job("w1", "v4 post", { kind: "post_training" }), job("w2", "v6 long", { params: { pinned: "beast:8000" } })];
  return mockApi({
    "GET /queue": () => ({ body: { job: job("v5run", "v5", { kind: "pretrain_distributed", status: "running" }), step: 0, total: 0, phase: "", lr: null, eta_s: null, paused: false, blocked: null, stopping: false, gate: { reason: null, supported: true }, up_next, failed, plan: PLAN } }),
    "POST /queue/jobs/*/restart": { body: { job: null, step: 0, total: 0, phase: "", lr: null, eta_s: null, paused: false, blocked: null, stopping: false, gate: { supported: true }, up_next, failed: [], plan: PLAN } },
    "GET /queue/jobs/*/outlook": { body: { outlook: "after v5" } },
    "GET /cluster/jobs": { body: [] },
    "GET /distributed/runs": () => ({ body: runs }),
    "GET /cluster/nodes": { body: [node("local", "mac"), node("beast:8000", "beast", { device: "cuda" })] },
    "POST /queue/jobs/*/hold": ({ body }) => ({ body: { job: null, step: 0, total: 0, phase: "", lr: null, eta_s: null, paused: false, blocked: null, stopping: false, gate: { supported: true }, up_next: up_next.map((j) => (j.id === "w1" ? { ...j, status: (body as { held: boolean }).held ? "held" : "waiting" } : j)), plan: PLAN } }),
    ...extra,
  });
}

describe("Runs page", () => {
  test("lists the active DiLoCo run with its roles, losses and a link to its view", async () => {
    server();
    renderPage(<RunsPage />);
    const card = await screen.findByTestId("active-v5run");
    expect(card).toHaveAttribute("href", expect.stringMatching(/^\/runs\/view\/?\?id=v5run$/));
    expect(within(card).getByText(/round 7\/40/)).toBeInTheDocument();
    expect(within(card).getByText("mac · c, w")).toBeInTheDocument();
    expect(within(card).getByText("beast · w")).toBeInTheDocument();
    expect(within(card).getByText("train 3.41")).toBeInTheDocument();
    expect(within(card).getByText("val 3.47")).toBeInTheDocument();
    expect(within(card).getByText(/ETA 5h 30m/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "New run" })).toHaveAttribute("href", expect.stringMatching(/^\/runs\/new\/?$/));
  });

  test("the Scheduled header links to the Queue", async () => {
    server();
    renderPage(<RunsPage />);
    expect(await screen.findByRole("link", { name: "Queue ›" })).toHaveAttribute("href", expect.stringMatching(/^\/queue\/?$/));
  });

  test("a running run another computer coordinates has a Join that asks it to add this one; this computer's own run has none", async () => {
    const runs = [diloco({ owner: "beast", can_join: true, members: [member("beast", "local")] }), diloco({ id: "own" })];
    const calls = server(runs, [], { "POST /distributed/runs/*/join": { body: diloco({ owner: "beast" }) } });
    renderPage(<RunsPage />);
    await screen.findByTestId("active-own");
    const join = await screen.findByRole("button", { name: /^Join: / });
    expect(screen.getAllByRole("button", { name: /^Join: / })).toHaveLength(1);
    await userEvent.click(join);
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/distributed/runs/v5run/join")).toBe(true));
  });

  test("a job running on a peer says why that peer was picked", async () => {
    const why = "beast measured 0.1 s per step for this design on its CUDA, 3.0x faster than the slowest free machine.";
    const far = { id: "far", kind: "pretrain", label: "v7 away", role: "owner", node: "beast:8000", epoch: 1, status: "running", lease: "held", waiting_reason: "", checkpoint_step: null, basis: why };
    server([diloco()], [], { "GET /cluster/jobs": { body: [far] } });
    renderPage(<RunsPage />);
    expect(await screen.findByTestId("basis-far")).toHaveTextContent(why);
    expect(screen.queryByTestId("basis-v5run")).toBeNull();
  });

  test("node chips show the device and the role in the active run", async () => {
    server();
    renderPage(<RunsPage />);
    expect(await screen.findByTestId("chip-local")).toHaveTextContent("mac · mps · c, w v5");
    expect(screen.getByTestId("chip-beast:8000")).toHaveTextContent("beast · cuda · w v5");
  });

  test("the scheduled table reads W and c from the pin, and Hold calls the queue", async () => {
    const calls = server();
    renderPage(<RunsPage />);
    const row = await screen.findByTestId("scheduled-w2");
    expect(within(row).getAllByText("beast")).toHaveLength(2);
    expect(within(screen.getByTestId("scheduled-w1")).getAllByText("any idle node")).toHaveLength(2);
    expect(await within(row).findByText("after v5")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Hold: v4 post" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/queue/jobs/w1/hold")).toBe(true));
    expect(calls.find((c) => c.path === "/queue/jobs/w1/hold")?.body).toEqual({ held: true });
  });

  test("a failed job is listed under Completed with its reason and a Restart; a finished run has none", async () => {
    const calls = server([diloco(), diloco({ id: "done", label: "v1", state: "finished", started_at: 1 })], [job("f1", "v3 broke", { status: "failed", error: "No room left on the disk." })]);
    renderPage(<RunsPage />);
    expect(await screen.findByTestId("completed-f1")).toHaveTextContent("failed: No room left on the disk.");
    expect(screen.queryByRole("button", { name: /^Restart: v1/ })).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /^Restart: .*v3 broke/ }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/queue/jobs/f1/restart")).toBe(true));
  });

  test("a single-machine job that finished done is listed under Completed and opens its Run view", async () => {
    server([diloco()], [], { "GET /queue/done": { body: [job("d1", "v2 solo", { status: "done", params: { version: 2 } })] } });
    renderPage(<RunsPage />);
    const row = await screen.findByTestId("completed-d1");
    expect(row).toHaveAttribute("href", expect.stringMatching(/^\/runs\/view\/?\?id=d1$/));
    expect(row).toHaveTextContent("v2 · v2 solo");
    expect(row).toHaveTextContent("finished");
    expect(screen.queryByRole("button", { name: /^Restart: .*v2 solo/ })).toBeNull();
  });

  test("a finished run is completed with its val loss, a failed one with its reason", async () => {
    server([diloco(), diloco({ id: "done", label: "v1", state: "finished", started_at: 1 }), diloco({ id: "bad", label: "ablation", state: "failed", note: "out of memory", started_at: 2 })]);
    renderPage(<RunsPage />);
    const done = await screen.findByTestId("completed-done");
    expect(done).toHaveTextContent("val 3.47");
    expect(screen.getByTestId("completed-bad")).toHaveTextContent("failed: out of memory");
  });

  test("Allowed hours opens the plan in a sheet", async () => {
    server();
    renderPage(<RunsPage />);
    await userEvent.click(await screen.findByRole("button", { name: "Allowed hours" }));
    expect(await screen.findByTestId("plan-card")).toBeInTheDocument();
  });
});

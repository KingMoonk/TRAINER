import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { event, lane, line, member, run } from "@/app/cluster/training/test-data";
import { mockApi, type Call } from "@/test/network";
import { renderPage } from "@/test/render";
import RunViewPage from "./page";

// The Run view against a mocked API: one page for a DiLoCo run (|W| = 2) and for a single job on this computer (|W| = 1).

const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };
const job = (id: string, over: Record<string, unknown> = {}) => ({ id, kind: "pretrain", label: "v3 from scratch", note: "", params: {}, status: "running", eta_s: null, created_at: 0, error: "", ...over });
const queue = (over: Record<string, unknown> = {}) => ({
  job: job("j1"), step: 1200, total: 4000, phase: "training", lr: 0.0003, eta_s: 5400, paused: false, blocked: null, stopping: false, gate: { reason: null, supported: true }, up_next: [], plan: PLAN, ...over,
});
const node = (address: string, hostname: string, over: Record<string, unknown> = {}) => ({
  address, local: address === "local", hostname, version: "1", device: "mps", device_name: "", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "", ...over,
});
const NODES = [node("local", "mac-mini"), node("gpu-box", "gpu-box", { device: "cuda" })];

const calls = (all: Call[], method: string, path: string) => all.filter((c) => c.method === method && c.path === path);

const diloco = () =>
  run({
    phase: "collecting", deadline_s: 3.2, window_s: 8, contributed: 1, expected: 2, now_s: 30,
    members: [
      member("mac-mini", { address: "local", local: true, device: "mps", can_stop: false, phase: "training", step_in_round: 40, round_steps: 100 }),
      member("gpu-box", { phase: "uploading", step_in_round: 100, round_steps: 100 }),
    ],
    timeline: [line(0), line(1, { open: true, closed: null, deadline: 8, window: 8, lanes: [lane("mac-mini", { delta_sent: null, delta_received: null, counted: null }), lane("gpu-box")] })],
    events: [event(1, { text: "gpu-box joined." }), event(2, { kind: "stale", node: "gpu-box", round: 1, t: 28, text: "gpu-box sent a change for an old round; refused." })],
  });

function dilocoServer() {
  return mockApi({
    "GET /distributed/runs/r1": { body: diloco() },
    "GET /queue": { body: queue({ job: job("r1", { kind: "pretrain_distributed" }) }) },
    "GET /cluster/nodes": { body: NODES },
    "POST /distributed/runs/r1/stop": { body: { ...diloco(), state: "stopped" } },
    "POST /distributed/runs/r1/members/gpu-box/stop": { body: diloco() },
  });
}

const CURVE = {
  steps: [], smooth: [[0, 4], [100, 3.8], [200, 3.6], [300, 3.4], [400, 3.2]],
  heldout: [[100, 3.9], [200, 3.9], [300, 3.95], [400, 4.0]],
};

function soloServer(over: { queue?: Record<string, unknown>; curve?: unknown } = {}) {
  return mockApi({
    "GET /queue": { body: queue(over.queue) },
    "GET /cluster/nodes": { body: NODES },
    "GET /pretraining/jobs/j1/curve": { body: over.curve ?? CURVE },
    "POST /queue/pause": { body: queue({ paused: true }) },
    "POST /queue/stop_after_cooldown": { body: queue({ stopping: true }) },
    "DELETE /queue/jobs/j1": { body: queue({ job: null }) },
  });
}

describe("Run view: a DiLoCo run", () => {
  test("shows the shape line, the round and its progress, who does what, the timeline and the loss by round", async () => {
    dilocoServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    expect(await screen.findByTestId("run-title")).toHaveTextContent("Version 2 across machines");
    expect(screen.getByTestId("run-shape")).toHaveTextContent("pretrain · DiLoCo · W = {2} · c = mac-mini · H = 100 · outer lr 0.7 · μ 0.9");
    expect(screen.getByTestId("run-progress")).toHaveTextContent("Round 3/10");
    expect(screen.getByTestId("round-banner")).toBeInTheDocument();
    expect(screen.getByTestId("coordinator")).toBeInTheDocument();
    expect(screen.getByTestId("member-gpu-box")).toBeInTheDocument();
    expect(screen.getByTestId("lane-1-gpu-box")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Loss vs. round" })).toBeInTheDocument();
    expect(screen.getByTestId("readout-val")).toBeInTheDocument();
  });

  test("Remove from run… asks first, then stops only that machine; the coordinator has none", async () => {
    const all = dilocoServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    await screen.findByTestId("member-gpu-box");
    expect(screen.queryByRole("button", { name: "Remove mac-mini from run…" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Remove gpu-box from run…" }));
    expect(calls(all, "POST", "/distributed/runs/r1/members/gpu-box/stop")).toHaveLength(0);
    const dialog = await screen.findByRole("alertdialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(calls(all, "POST", "/distributed/runs/r1/members/gpu-box/stop")).toHaveLength(1));
  });

  test("Stop run… asks first and stops the whole run", async () => {
    const all = dilocoServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    await userEvent.click(await screen.findByTestId("stop-run"));
    expect(calls(all, "POST", "/distributed/runs/r1/stop")).toHaveLength(0);
    await userEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Stop run" }));
    await waitFor(() => expect(calls(all, "POST", "/distributed/runs/r1/stop")).toHaveLength(1));
  });

  test("Keep running in the dialog stops nothing", async () => {
    const all = dilocoServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    await userEvent.click(await screen.findByTestId("stop-run"));
    await userEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Keep running" }));
    await waitFor(() => expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument());
    expect(calls(all, "POST", "/distributed/runs/r1/stop")).toHaveLength(0);
  });

  test("after the run ends, a machine that never joined says why and the coordinator that finished is not red", async () => {
    const reason = "It was busy when the run was placed. It joins a later round if it is free by then.";
    mockApi({
      "GET /distributed/runs/r1": {
        body: {
          ...diloco(), state: "finished", phase: "finished", can_stop: false,
          members: [
            member("mac-mini", { address: "local", local: true, device: "mps", can_stop: false, state: "finished" }),
            member("gpu-box", { state: "never_joined", note: reason, reason, can_stop: false }),
          ],
        },
      },
      "GET /queue": { body: queue({ job: null }) },
      "GET /cluster/nodes": { body: NODES },
    });
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    await screen.findByTestId("run-title");
    expect(screen.getByTestId("state-gpu-box")).toHaveTextContent("Never joined");
    expect(screen.getByTestId("reason-gpu-box")).toHaveTextContent(reason);
    expect(screen.getByTestId("node-gpu-box")).not.toHaveClass("border-destructive");
    expect(screen.getByTestId("state-mac-mini")).toHaveTextContent("Finished");
    expect(screen.queryByTestId("reason-mac-mini")).not.toBeInTheDocument();
    expect(screen.getByTestId("node-mac-mini")).not.toHaveClass("border-destructive");
  });

  test("a run that cannot be stopped here offers no stop controls", async () => {
    mockApi({
      "GET /distributed/runs/r1": { body: { ...diloco(), can_stop: false, members: diloco().members.map((m) => ({ ...m, can_stop: false })) } },
      "GET /queue": { body: queue({ job: null }) },
      "GET /cluster/nodes": { body: NODES },
    });
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    await screen.findByTestId("run-title");
    expect(screen.queryByTestId("stop-run")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Remove .* from run/ })).not.toBeInTheDocument();
  });

  test("an event row opens a popover with the event's details", async () => {
    dilocoServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    const row = await screen.findByTestId("event-2");
    await userEvent.click(within(row).getByRole("button"));
    const details = await screen.findByLabelText("Event details");
    expect(details).toHaveTextContent("gpu-box sent a change for an old round; refused.");
    expect(details).toHaveTextContent("gpu-box");
    expect(details).toHaveTextContent("#2");
  });
});

describe("Run view: a single job on this computer", () => {
  test("shows the shape line, the step and ETA, one node and no round, timeline or transfer panels", async () => {
    soloServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=j1" });
    expect(await screen.findByTestId("run-title")).toHaveTextContent("v3 from scratch");
    expect(screen.getByTestId("run-shape")).toHaveTextContent("pretrain · local · W = {mac-mini} · queue job on mac-mini");
    expect(screen.getByTestId("run-progress")).toHaveTextContent("Step 1,200/4,000");
    expect(screen.getByTestId("run-status")).toHaveTextContent(/training · lr 3\.0e-4 · ETA/);
    expect(screen.getByTestId("solo-node")).toHaveTextContent("COORDINATOR + WORKER");
    expect(screen.queryByTestId("round-banner")).not.toBeInTheDocument();
    expect(screen.queryByText("Rounds done")).not.toBeInTheDocument();
    expect(screen.queryByText(/transfer|traffic/i)).not.toBeInTheDocument();
    expect(screen.getByText("A single job on this computer does not record events.")).toBeInTheDocument();
  });

  test("the loss chart reads L_train, L_val and the gap, and the overfit pill fires when L_val stopped improving", async () => {
    soloServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=j1" });
    expect(await screen.findByTestId("readout-train")).toHaveTextContent("L_train 3.20");
    expect(screen.getByTestId("readout-val")).toHaveTextContent("L_val 4.00");
    expect(screen.getByTestId("readout-gap")).toHaveTextContent("gap +0.80");
    expect(screen.getByTestId("overfit")).toHaveAttribute("data-state", "stalled");
  });

  test("says it is too early to call overfitting with fewer than four checks", async () => {
    soloServer({ curve: { steps: [], smooth: CURVE.smooth, heldout: CURVE.heldout.slice(0, 2) } });
    renderPage(<RunViewPage />, { route: "/runs/view?id=j1" });
    const pill = await screen.findByTestId("overfit");
    expect(pill).toHaveAttribute("data-state", "early");
    expect(pill).toHaveTextContent("2 of 4 checks");
  });

  test("Pause pauses the whole queue, and its Tooltip says so", async () => {
    const all = soloServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=j1" });
    const pause = await screen.findByRole("button", { name: "Pause" });
    await userEvent.hover(pause);
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Pauses the whole queue on this computer");
    await userEvent.click(pause);
    await waitFor(() => expect(calls(all, "POST", "/queue/pause")).toHaveLength(1));
    expect(await screen.findByRole("button", { name: "Resume" })).toBeInTheDocument();
  });

  test("Cool down & stop… asks first, then calls stop_after_cooldown", async () => {
    const all = soloServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=j1" });
    await userEvent.click(await screen.findByRole("button", { name: "Cool down & stop…" }));
    expect(calls(all, "POST", "/queue/stop_after_cooldown")).toHaveLength(0);
    await userEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Cool down & stop" }));
    await waitFor(() => expect(calls(all, "POST", "/queue/stop_after_cooldown")).toHaveLength(1));
  });

  test("Stop now… asks first, says the last checkpoint is kept, then removes the job", async () => {
    const all = soloServer();
    renderPage(<RunViewPage />, { route: "/runs/view?id=j1" });
    const stop = await screen.findByRole("button", { name: "Stop now…" });
    await userEvent.hover(stop);
    expect(await screen.findByRole("tooltip")).toHaveTextContent("last checkpoint stays on disk");
    await userEvent.click(stop);
    expect(calls(all, "DELETE", "/queue/jobs/j1")).toHaveLength(0);
    await userEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Stop now" }));
    await waitFor(() => expect(calls(all, "DELETE", "/queue/jobs/j1")).toHaveLength(1));
  });

  test("a job that is waiting has no controls and no curve request", async () => {
    const all = soloServer({ queue: { job: null, up_next: [job("j1", { status: "waiting" })] } });
    renderPage(<RunViewPage />, { route: "/runs/view?id=j1" });
    expect(await screen.findByTestId("run-title")).toHaveTextContent("v3 from scratch");
    expect(screen.queryByRole("button", { name: "Pause" })).not.toBeInTheDocument();
    expect(calls(all, "GET", "/pretraining/jobs/j1/curve")).toHaveLength(0);
  });
});

describe("Run view: ids that are not here", () => {
  test("an id that is neither a run nor a queued job says it is not on this computer", async () => {
    mockApi({ "GET /queue": { body: queue({ job: null }) }, "GET /cluster/nodes": { body: NODES } });
    renderPage(<RunViewPage />, { route: "/runs/view?id=nope" });
    expect(await screen.findByRole("heading", { name: "This run is not on this computer." })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Runs/ })).toHaveAttribute("href", expect.stringMatching(/^\/runs\/?$/));
  });

  test("no id asks to open a run from the Runs page", async () => {
    mockApi({ "GET /queue": { body: queue({ job: null }) }, "GET /cluster/nodes": { body: NODES } });
    renderPage(<RunViewPage />, { route: "/runs/view" });
    expect(await screen.findByRole("heading", { name: /No run is chosen/ })).toBeInTheDocument();
  });

  test("a failing read shows the problem", async () => {
    mockApi({
      "GET /distributed/runs/r1": { status: 500, body: { detail: "boom" } },
      "GET /queue": { body: queue({ job: null }) },
      "GET /cluster/nodes": { body: NODES },
    });
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not read the run");
  });
});

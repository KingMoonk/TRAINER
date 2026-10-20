import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import type { Run } from "./logic";
import TrainingPage from "./page";
import { event, lane, line, member, round, run } from "./test-data";

// The Training tab's live view against a mocked API: the topology and its edges, the round banner, the timeline, the charts, the event log, the
// list of other runs, the folded start form and the polling. The contract tier (src/contract) renders it from recorded --fake responses.

const DEFAULTS = { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16" };
const SETS = [{ id: "stories", name: "Stories", count: 1200, on: true, held: false, library: false }];
const node = (address: string, over: Record<string, unknown> = {}) => ({
  address, local: false, hostname: address, version: "1", device: "cuda", device_name: "RTX 4090", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "", ...over,
});
const NODES = [node("local", { local: true, hostname: "mac-mini", device: "mps", device_name: "Apple M2" }), node("gpu-box")];

function server(initial: Run[]) {
  let runs = initial;
  const calls = { runs: 0 };
  mockApi({
    "GET /distributed/defaults": { body: DEFAULTS },
    "GET /pretraining/text_sets": { body: SETS },
    "GET /versions": { body: [] },
    "GET /cluster/nodes": { body: NODES },
    "GET /distributed/runs": () => {
      calls.runs += 1;
      return { body: runs };
    },
    "POST /distributed/runs": () => {
      const made = run({ id: "r9", label: "Fresh run", started_at: 300, state: "queued", round: 0, history: [] });
      runs = [made, ...runs];
      return { status: 201, body: made };
    },
  });
  return calls;
}

// A run mid-round: the CUDA machine has sent its change, the MPS one is still training, and one round is already on the timeline.
const midRound = (over: Partial<Run> = {}): Run =>
  run({
    phase: "collecting", deadline_s: 3.2, window_s: 8, contributed: 1, expected: 2, theta_bytes: 6_000_000, now_s: 30,
    members: [
      member("mac-mini", {
        address: "local", local: true, device: "mps", device_name: "Apple M2", can_stop: false, phase: "training", step_in_round: 40, round_steps: 100, loss: 4.213,
        live_tokens_per_second: 2380, reported_ago: 1.2, shards: 2,
        samples: [{ step: 0, t: 1, loss: 4.4, tokens_per_second: 2300 }, { step: 20, t: 2, loss: 4.3, tokens_per_second: 2400 }, { step: 40, t: 3, loss: 4.213, tokens_per_second: 2380 }],
      }),
      member("gpu-box", {
        phase: "uploading", step_in_round: 100, round_steps: 100, loss: 4.1, live_tokens_per_second: 9800, reported_ago: 0.4, shards: 1, last_delta_bytes: 3_000_000, last_delta_dtype: "float16",
        samples: [{ step: 0, t: 1, loss: 4.3, tokens_per_second: 9700 }, { step: 100, t: 3, loss: 4.1, tokens_per_second: 9800 }],
      }),
    ],
    timeline: [
      line(0),
      line(1, {
        open: true, closed: null, deadline: 8, window: 8,
        lanes: [lane("mac-mini", { first_step: 0.6, delta_sent: null, delta_received: null, counted: null }), lane("gpu-box", { delta_sent: 3, delta_received: null, counted: false })],
      }),
    ],
    events: [
      event(1, { text: "mac-mini joined." }),
      event(2, { kind: "outer_step", text: "Outer step applied, size 0.42.", t: 12, round: 0 }),
      event(3, { kind: "checkpoint", text: "Round 0 checkpoint written.", t: 14, round: 0 }),
      event(4, { kind: "stale", text: "gpu-box sent a change for an old round; refused.", t: 28, round: 1 }),
    ],
    ...over,
  });

describe("Training tab: the live view", () => {
  test("a machine node shows its device, state, progress, speed, loss, shards and the age of its last report", async () => {
    server([midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const mac = within(await screen.findByTestId("node-mac-mini"));
    expect(mac.getByTestId("device-mac-mini")).toHaveTextContent("MPS");
    expect(mac.getByRole("progressbar", { name: "Progress of mac-mini" })).toHaveAttribute("aria-valuenow", "40");
    expect(mac.getByText("40 of 100 steps")).toBeInTheDocument();
    expect(mac.getByText("2,380 tokens/s")).toBeInTheDocument();
    expect(mac.getByText("Loss 4.213")).toBeInTheDocument();
    expect(mac.getByText("2 shards")).toBeInTheDocument();
    expect(mac.getByText("Reported 1 s ago")).toBeInTheDocument();
    expect(mac.getByTestId("phase-mac-mini")).toHaveTextContent("Training");
    const gpu = within(screen.getByTestId("node-gpu-box"));
    expect(gpu.getByTestId("device-gpu-box")).toHaveTextContent("CUDA");
    expect(gpu.getByTestId("phase-gpu-box")).toHaveTextContent("Uploading Δ");
    expect(gpu.getByText("1 shard")).toBeInTheDocument();
  });

  test("a run links to its Run view", async () => {
    server([midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByTestId("open-run-view-r1")).toHaveAttribute("href", expect.stringMatching(/^\/runs\/view\/?\?id=r1$/));
  });

  test("an edge moves while its transfer runs and says how many bytes in which format", async () => {
    server([midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const up = await screen.findByTestId("edge-up-gpu-box");
    expect(up).toHaveAttribute("data-active", "true");
    expect(up).toHaveTextContent("Δ up");
    expect(up).toHaveTextContent("2.9 MB, float16");
    expect(up.querySelector(".edge-flow")).not.toBeNull();
    expect(screen.getByTestId("edge-down-gpu-box")).toHaveAttribute("data-active", "false");
    expect(screen.getByTestId("edge-down-gpu-box")).toHaveTextContent("5.7 MB, float32");
    expect(screen.getByTestId("edge-up-mac-mini")).toHaveAttribute("data-active", "false");
  });

  test("the banner names the round and phase, how many changes are in, and counts down to the deadline", async () => {
    server([midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByTestId("round-title")).toHaveTextContent("Round 3 of 10");
    expect(screen.getByTestId("stage-collecting")).toHaveAttribute("aria-current", "step");
    expect(screen.getByTestId("stage-collecting")).toHaveTextContent("Collecting changes (1 of 2 in)");
    expect(screen.getByTestId("stage-training")).toHaveAttribute("data-active", "false");
    expect(screen.getByTestId("countdown")).toHaveTextContent("4 s to the deadline");
  });

  test("the timeline has a lane per machine, marks the deadline, and says a late machine counted for nothing", async () => {
    server([midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const open = within(await screen.findByTestId("timeline-round-2"));
    expect(open.getByTestId("lane-2-mac-mini")).toBeInTheDocument();
    expect(open.getByTestId("deadline-2-mac-mini")).toBeInTheDocument();
    expect(within(screen.getByTestId("lane-2-gpu-box")).getAllByLabelText(/Download θ|Train|Upload Δ/)).toHaveLength(3);
    expect(open.getByTestId("nothing-2-gpu-box")).toHaveTextContent("Counted for nothing this round");
    expect(open.queryByTestId("nothing-2-mac-mini")).toBeNull();
    expect(screen.queryByTestId("nothing-1-gpu-box")).toBeNull(); // the finished round was counted
  });

  test("the event log lists the newest first, and the charts are drawn", async () => {
    server([midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const log = within(await screen.findByRole("region", { name: "Journal" }));
    const items = log.getAllByRole("listitem");
    expect(items.map((li) => li.getAttribute("data-kind"))).toEqual(["stale", "checkpoint", "outer_step", "joined"]);
    expect(items[0]).toHaveTextContent("gpu-box sent a change for an old round; refused.");
    expect(items[0]).toHaveTextContent("2 s ago");
    for (const name of ["Error curve", "Loss inside the open round", "Tokens per second", "Bytes moved per round"]) {
      expect(screen.getByRole("region", { name })).toBeInTheDocument();
    }
  });

  test("a machine with nothing reported yet says so, without a speed or a loss", async () => {
    server([run({ phase: "waiting", history: [], members: [member("gpu-box", { state: "joining", can_stop: false })] })]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const card = within(await screen.findByTestId("node-gpu-box"));
    expect(card.getByText("No report yet")).toBeInTheDocument();
    expect(card.queryByText(/Loss/)).toBeNull();
    expect(screen.getByTestId("round-phase")).toHaveTextContent("Waiting for members");
    expect(screen.queryByTestId("countdown")).toBeNull();
  });

  test("a finished round row lists in the table of rounds", async () => {
    server([midRound({ history: [round(0), round(1)] })]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await userEvent.setup().click(await screen.findByText("Rounds"));
    expect(screen.getByTestId("round-1")).toHaveTextContent("2 of 2");
  });
});

describe("Training tab: more than one run", () => {
  const older = () => run({ id: "r0", label: "Older run", state: "finished", can_stop: false, started_at: 50, round: 10 });

  test("the one that is going is the live view and the others are listed below; choosing one makes it live", async () => {
    server([older(), midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByTestId("run-r1")).toBeInTheDocument();
    expect(screen.queryByTestId("run-r0")).toBeNull();
    const others = within(screen.getByTestId("other-runs"));
    expect(others.getByTestId("other-state-r0")).toHaveTextContent("Finished");
    await userEvent.setup().click(others.getByRole("button", { name: "Show Older run" }));
    expect(await screen.findByTestId("run-r0")).toBeInTheDocument();
    expect(screen.queryByTestId("run-r1")).toBeNull();
    expect(within(screen.getByTestId("other-runs")).getByTestId("other-r1")).toBeInTheDocument();
  });

  test("the form is folded behind Start a run while there is a run, and the button opens it with the machine checklist", async () => {
    server([midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const toggle = await screen.findByRole("button", { name: "Start a run" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByLabelText("Rounds")).toBeNull();
    await userEvent.setup().click(toggle);
    expect(await screen.findByLabelText("Rounds")).toHaveValue(10);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByRole("checkbox", { name: /gpu-box/ })).toBeInTheDocument();
  });

  test("a run that is started closes the form and becomes the live view", async () => {
    server([older()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Start a run" }));
    await user.click(await screen.findByRole("button", { name: "Start run" }));
    expect(await screen.findByTestId("run-r9")).toHaveTextContent("Fresh run");
    await waitFor(() => expect(screen.queryByLabelText("Rounds")).toBeNull());
    expect(screen.getByTestId("other-r0")).toBeInTheDocument();
  });

  test("more runs than fit are counted, not listed", async () => {
    const many = Array.from({ length: 11 }, (_, i) => run({ id: `x${i}`, label: `Run ${i}`, state: "finished", can_stop: false, started_at: 10 - i }));
    server(many);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await screen.findByTestId("other-runs");
    expect(within(screen.getByTestId("other-runs")).getAllByRole("listitem")).toHaveLength(8);
    expect(screen.getByText("2 older runs are not listed.")).toBeInTheDocument();
  });
});

describe("Training tab: live updates", () => {
  test("re-reads the runs when the coordinator says one changed, without polling", async () => {
    const calls = server([midRound()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await screen.findByTestId("run-r1");
    await waitFor(() => expect(calls.runs).toBeGreaterThanOrEqual(1));
    const before = calls.runs;
    await new Promise((resolve) => setTimeout(resolve, 1_500));
    expect(calls.runs).toBe(before);
    emit("run", { run: "r1" });
    await waitFor(() => expect(calls.runs).toBeGreaterThan(before));
  });

  test("a finished run names the version it saved; a stopped one says nothing", async () => {
    server([run({ state: "finished", can_stop: false, saved_version: 3 })]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const saved = await screen.findByTestId("run-saved-r1");
    expect(saved).toHaveTextContent("Weights saved as Version 3.");
    expect(within(saved).queryByRole("link")).toBeNull();
  });

  test("a run that saved nothing shows no saved version", async () => {
    server([run({ state: "stopped", can_stop: false })]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await screen.findByTestId("run-r1");
    expect(screen.queryByTestId("run-saved-r1")).toBeNull();
  });

  test("does not poll once no run is going", async () => {
    const calls = server([run({ state: "finished", can_stop: false })]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await screen.findByTestId("run-r1");
    await new Promise((resolve) => setTimeout(resolve, 1500));
    expect(calls.runs).toBe(1);
  });
});

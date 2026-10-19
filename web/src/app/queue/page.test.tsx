import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { emit, mockApi } from "@/test/network";
import { invite } from "@/test/invites";
import { renderPage } from "@/test/render";
import QueuePage from "./page";

// The Queue page against a mocked API (the component tier of tests/queue.spec.ts, which keeps the browser-only flows: drag and keyboard reorder).

type Job = { id: string; kind: string; label: string; note: string; params: object; status: string; eta_s: number | null; created_at: number; error: string };
const job = (id: string, label: string, over: Partial<Job> = {}): Job => ({ id, kind: "pretrain", label, note: "", params: {}, status: "waiting", eta_s: 1800, created_at: 0, error: "", ...over });

const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };

function server(start: { running?: boolean; reason?: string | null; supported?: boolean } = {}, extra: Parameters<typeof mockApi>[0] = {}) {
  const state = {
    job: start.running === false ? null : (job("run1", "Version 3 pretraining", { status: "running", eta_s: null }) as Job | null),
    step: 100,
    paused: false,
    stopping: false,
    plan: { ...PLAN } as Record<string, unknown>,
    reason: start.reason ?? (null as string | null),
    supported: start.supported ?? true,
    up_next: [job("a", "First waiting"), job("b", "Second waiting", { note: "A note" }), job("c", "Third waiting")],
  };
  const view = () => ({
    job: state.job,
    step: state.step,
    total: state.job ? 1000 : 0,
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
  const index = (path: string) => state.up_next.findIndex((j) => j.id === path.split("/queue/jobs/")[1].split("/")[0]);
  const reply = () => ({ body: view() });
  const calls = mockApi({
    "GET /queue": () => reply(),
    "GET /queue/jobs/*/outlook": { body: { outlook: "after about 30 min" } },
    "POST /queue/jobs/*/hold": ({ path, body }) => {
      state.up_next[index(path)].status = (body as { held: boolean }).held ? "held" : "waiting";
      return reply();
    },
    "POST /queue/jobs/*/move": ({ path, body }) => {
      const [moved] = state.up_next.splice(index(path), 1);
      state.up_next.splice((body as { index: number }).index, 0, moved);
      return reply();
    },
    "DELETE /queue/jobs/*": ({ path }) => {
      const id = path.split("/queue/jobs/")[1];
      if (state.job?.id === id) state.job = null;
      state.up_next = state.up_next.filter((j) => j.id !== id);
      return reply();
    },
    "POST /queue/pause": () => {
      state.paused = true;
      return reply();
    },
    "POST /queue/resume": () => {
      state.paused = false;
      return reply();
    },
    "POST /queue/start": () => reply(),
    "POST /queue/stop_after_cooldown": () => {
      state.stopping = true;
      return reply();
    },
    "POST /queue/checkpoint_now": { body: { started: true } },
    "GET /queue/plan": () => ({ body: state.plan }),
    "PUT /queue/plan": ({ body }) => {
      state.plan = body as Record<string, unknown>;
      if (state.plan.idle_only === false) {
        state.reason = null;
        setTimeout(() => emit("queue", {}), 0); // the server announces a changed gate on /events
      }
      return { body: state.plan };
    },
    ...extra,
  });
  return { state, calls };
}

const rows = () => within(screen.getByTestId("up-next")).getAllByRole("listitem");
const idleSwitch = () => screen.getByRole("switch", { name: /Only when I.m not using the computer/ });

describe("Queue page", () => {
  test("shows the running job, the jobs up next in order and the plan", async () => {
    server();
    renderPage(<QueuePage />);
    expect(await screen.findByRole("heading", { level: 1, name: "Queue" })).toBeVisible();

    await waitFor(() => expect(screen.getByTestId("state-line")).toHaveTextContent("Version 3 pretraining · running"));
    const progress = screen.getByTestId("progress-line");
    expect(progress).toHaveTextContent("step 100 of 1,000");
    expect(progress).toHaveTextContent("schedule: steady");
    expect(progress).toHaveTextContent("learning rate: auto (0.0003)");
    expect(progress).toHaveTextContent("about 10 min left");

    await waitFor(() => expect(rows()).toHaveLength(3));
    expect(rows()[1]).toHaveTextContent("Second waiting");
    expect(rows()[1]).toHaveTextContent("A note");
    await waitFor(() => expect(rows()[0]).toHaveTextContent("after about 30 min"));
    expect(screen.getByText("all done in about 1.7 h at the current plan")).toBeVisible();

    expect(screen.getByRole("radio", { name: "Any time" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Most" })).toBeChecked();
    expect(idleSwitch()).toBeChecked();
  });

  test("a member job shows the run it belongs to, its state, and the run's round and progress", async () => {
    const mine = invite({ job_id: "run1", state: "training", round: 6, rounds: 50, tokens_per_second: 1200 });
    server({}, { "GET /distributed/invites": { body: { invites: [mine], pause_requests: [] } } });
    renderPage(<QueuePage />);
    expect(await screen.findByTestId("member-of")).toHaveTextContent("Member of 'Version 7' from Michael's Mac");
    expect(within(screen.getByTestId("member-of")).getByText("Training")).toBeVisible();
    expect(screen.getByTestId("member-round")).toHaveTextContent("Round 6 of 50, 1,200 tok/s");
    expect(screen.getByRole("progressbar", { name: "Run progress" })).toHaveAttribute("aria-valuenow", "12");
    expect(screen.queryByTestId("progress-line")).toBeNull(); // the job's own step line gives way to the run's round
  });

  test("a member job whose owner has not reported a round yet says it is waiting for that report", async () => {
    server({}, { "GET /distributed/invites": { body: { invites: [invite({ job_id: "run1", state: "joining" })], pause_requests: [] } } });
    renderPage(<QueuePage />);
    expect(await screen.findByTestId("member-round")).toHaveTextContent("Waiting for Michael's Mac's first report on the round");
  });

  test("a job that is not a member job keeps its own step line", async () => {
    server({}, { "GET /distributed/invites": { body: { invites: [invite({ job_id: "other" })], pause_requests: [] } } });
    renderPage(<QueuePage />);
    expect(await screen.findByTestId("progress-line")).toHaveTextContent("step 100 of 1,000");
    expect(screen.queryByTestId("member-row")).toBeNull();
  });

  test("hold, release, move, run next and remove a waiting job", async () => {
    const { calls } = server();
    const user = userEvent.setup();
    renderPage(<QueuePage />);
    await waitFor(() => expect(rows()).toHaveLength(3));

    await user.click(screen.getByRole("button", { name: "Hold: Second waiting" }));
    await waitFor(() => expect(rows()[1]).toHaveTextContent("Second waiting · on hold"));
    await user.click(screen.getByRole("button", { name: "Release: Second waiting" }));
    await waitFor(() => expect(rows()[1]).not.toHaveTextContent("on hold"));

    await user.click(screen.getByRole("button", { name: "Move up: Second waiting" }));
    await waitFor(() => expect(rows()[0]).toHaveTextContent("Second waiting"));
    await user.click(screen.getByRole("button", { name: "Move down: Second waiting" }));
    await waitFor(() => expect(rows()[1]).toHaveTextContent("Second waiting"));
    await user.click(screen.getByRole("button", { name: "Run next: Third waiting" }));
    await waitFor(() => expect(rows()[0]).toHaveTextContent("Third waiting"));
    expect(screen.getByRole("button", { name: "Move up: Third waiting" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Remove: First waiting" }));
    await waitFor(() => expect(rows()).toHaveLength(2));
    expect(screen.queryByText("First waiting")).toBeNull();

    expect(calls.filter((c) => c.method !== "GET").map((c) => [c.method, c.path.replace("/queue/jobs/", ""), c.body])).toEqual([
      ["POST", "b/hold", { held: true }],
      ["POST", "b/hold", { held: false }],
      ["POST", "b/move", { index: 0 }],
      ["POST", "b/move", { index: 1 }],
      ["POST", "c/move", { index: 0 }],
      ["DELETE", "a", undefined],
    ]);
  });

  test("pause, resume, checkpoint now, stop after cool-down and stop", async () => {
    const { calls } = server();
    const user = userEvent.setup();
    renderPage(<QueuePage />);
    await waitFor(() => expect(screen.getByTestId("state-line")).toHaveTextContent("running"));

    await user.click(screen.getByRole("button", { name: "Pause" }));
    await waitFor(() => expect(screen.getByTestId("state-line")).toHaveTextContent("Version 3 pretraining · paused"));
    await user.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("state-line")).toHaveTextContent("Version 3 pretraining · running"));

    await user.click(screen.getByRole("button", { name: "Make a checkpoint now" }));
    expect(await screen.findByText("Making a checkpoint from the last one.")).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Stop after a short cool-down" }));
    expect(await screen.findByText("Asked it to stop. It stops at its next check.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Stop after a short cool-down" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Stop" }));
    expect(await screen.findByText("Nothing is training.")).toBeVisible();
    expect(calls.filter((c) => c.method !== "GET").map((c) => `${c.method} ${c.path}`)).toEqual([
      "POST /queue/pause",
      "POST /queue/resume",
      "POST /queue/checkpoint_now",
      "POST /queue/stop_after_cooldown",
      "DELETE /queue/jobs/run1",
    ]);
  });

  test("with nothing running, jobs saved by an earlier session can be started", async () => {
    const { calls } = server({ running: false });
    const user = userEvent.setup();
    renderPage(<QueuePage />);
    expect(await screen.findByText("Nothing is training.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Start the waiting jobs" }));
    await waitFor(() => expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/queue/start"]));
    await user.click(screen.getByRole("button", { name: "Pause" }));
    expect(await screen.findByText("Paused by you. Nothing starts until you resume.")).toBeVisible();
  });

  test("the plan is saved, and 'only when idle' can be turned off so training stops waiting", async () => {
    const { calls } = server({ reason: "computer in use" });
    const user = userEvent.setup();
    renderPage(<QueuePage />);

    await waitFor(() => expect(screen.getByTestId("gate")).toHaveTextContent("Training is waiting: computer in use"));
    expect(idleSwitch()).toBeEnabled();
    await user.click(idleSwitch());
    await waitFor(() => expect(idleSwitch()).not.toBeChecked());
    await waitFor(() => expect(screen.queryByTestId("gate")).toBeNull());
    const put = calls.find((c) => c.method === "PUT");
    expect(put).toMatchObject({ path: "/queue/plan", body: { idle_only: false, hours: "any", memory: "most" } });

    await user.click(screen.getByRole("radio", { name: "Set hours" }));
    const from = await screen.findByLabelText("From hour");
    expect(from).toHaveValue("22");
    const type = async (label: string, text: string) => {
      const input = screen.getByLabelText(label);
      await user.clear(input);
      await user.type(input, `${text}{Enter}`);
    };
    await type("From hour", "9");
    await type("to hour", "99");
    await waitFor(() => expect(screen.getByLabelText("to hour")).toHaveValue("7"));
    await type("to hour", "17");
    await user.click(screen.getByRole("radio", { name: "Half" }));
    await user.click(screen.getByRole("switch", { name: /Keep the computer awake while training/ }));
    await waitFor(() => {
      const last = calls.filter((c) => c.method === "PUT").at(-1);
      expect(last?.body).toMatchObject({ hours: "set", start_hour: 9, end_hour: 17, memory: "half", keep_awake: true, idle_only: false });
    });
  });

  test("idle detection the computer cannot do greys the idle switch", async () => {
    server({ supported: false });
    renderPage(<QueuePage />);
    await waitFor(() => expect(idleSwitch()).toBeDisabled());
    expect(screen.getByText("Not available on this computer")).toBeVisible();
  });

  test("the running job's progress moves in place from /events while waiting rows stay mounted", async () => {
    const { state } = server();
    renderPage(<QueuePage />);
    await waitFor(() => expect(rows()).toHaveLength(3));
    const row = rows()[1];
    const progress = screen.getByTestId("progress-line");

    state.step = 150;
    emit("train", { type: "metric", step: 150 });
    await waitFor(() => expect(screen.getByTestId("progress-line")).toHaveTextContent("step 150 of 1,000"), { timeout: 3000 });
    expect(rows()[1]).toBe(row);
    expect(screen.getByTestId("progress-line")).toBe(progress);
    expect(screen.getByRole("progressbar", { name: "Job progress" })).not.toHaveAttribute("aria-valuenow", "10");
  });

  test("a failed call is a toast, not a broken page", async () => {
    server({}, { "POST /queue/pause": { status: 409, body: { detail: "The runner is not running" } } });
    const user = userEvent.setup();
    renderPage(<QueuePage />);
    await waitFor(() => expect(screen.getByTestId("state-line")).toHaveTextContent("running"));
    await user.click(screen.getByRole("button", { name: "Pause" }));
    expect(await screen.findByText("The runner is not running")).toBeVisible();
    expect(screen.getByTestId("state-line")).toHaveTextContent("Version 3 pretraining · running");
  });
});

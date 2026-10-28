import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, test } from "vitest";
import { emit, mockApi, type Call } from "@/test/network";
import { renderPage } from "@/test/render";
import type { Plan, RunView } from "./logic";
import AutoRunPage from "./page";
import { plan, run, summary } from "./test-data";

// The Auto Run page against a mocked API: GET /versions, /auto-runs, /auto-runs/defaults, /auto-runs/{id}, and the controls.

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.setPointerCapture ??= () => {};
  Element.prototype.releasePointerCapture ??= () => {};
});

const VERSIONS = [
  { number: 1, name: "tiny", best_checkpoint: null },
  { number: 2, name: "small", best_checkpoint: "v2/best.npz" },
];

/** A server holding one run (or none): the controls replace it with `next[path]`, and every call is recorded. */
function server(initial: RunView | null, next: Record<string, RunView | { status: number; detail: string }> = {}) {
  const state = { run: initial };
  const calls: Call[] = mockApi({
    "GET /versions": { body: VERSIONS },
    "GET /auto-runs": () => ({ body: state.run ? [summary(state.run)] : [] }),
    "GET /auto-runs/defaults": ({ search }) => ({ body: plan({ version: Number(new URLSearchParams(search).get("version")) }) }),
    "POST /auto-runs": ({ body }) => {
      state.run = { ...run("draft", null), plan: { ...(body as { plan: Plan }).plan, id: "run1" } };
      return { status: 201, body: state.run };
    },
    "GET /auto-runs/*": () => ({ body: state.run }),
    "PUT /auto-runs/*/plan": ({ body }) => {
      state.run = { ...state.run!, plan: (body as { plan: Plan }).plan };
      return { body: state.run };
    },
    "POST /auto-runs/*/*": (call) => {
      const reply = next[call.path.split("/").pop()!];
      if (!reply) return { status: 409, body: { detail: "No reply set up." } };
      if ("status" in reply && typeof reply.status === "number") return { status: reply.status, body: { detail: reply.detail } };
      state.run = reply as RunView;
      return { body: state.run };
    },
  });
  return { state, calls };
}

const posted = (calls: Call[], path: string) => calls.filter((c) => c.method !== "GET" && c.path === path);

describe("Auto Run page: a new run", () => {
  test("starts from the default plan of the best trained version", async () => {
    const { calls } = server(null);
    renderPage(<AutoRunPage />);
    expect(await screen.findByTestId("plan-editor")).toBeVisible();
    expect(screen.getByLabelText("Version to train")).toHaveValue("2");
    expect(calls.some((c) => c.path === "/auto-runs/defaults" && c.search === "?version=2")).toBe(true);
    for (const kind of ["data", "pretrain", "midtrain", "sft", "rlvr", "distill", "dpo"]) expect(screen.getByTestId(`plan-${kind}`)).toBeVisible();
    expect(screen.getByRole("switch", { name: "Include Mid-training" })).not.toBeChecked(); // optional: off unless the person adds it
    expect(screen.getByLabelText("Overall time budget (hours)")).toHaveValue("12");
  });

  test("edits the plan, makes the draft, then starts it", async () => {
    const running = run("running", "data", { data: { status: "running" } });
    const { calls } = server(null, { start: running });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await screen.findByTestId("plan-editor");

    await user.click(screen.getByRole("switch", { name: "Include Round 4 · Prefer the better answer" }));
    await user.selectOptions(screen.getByLabelText("When it fails, Round 1 · Learn to answer"), "skip");
    await user.selectOptions(screen.getByLabelText("When it passes, Pretraining"), "pause");
    await user.type(screen.getByLabelText("Steps, Round 1 · Learn to answer"), "40");
    await user.clear(screen.getByLabelText("Overall time budget (hours)"));
    await user.type(screen.getByLabelText("Overall time budget (hours)"), "3");
    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(await screen.findByTestId("auto-run-view")).toBeVisible();
    const made = posted(calls, "/auto-runs")[0].body as { plan: Plan };
    expect(made.plan.budget).toEqual({ hours: 3 });
    expect(made.plan.version).toBe(2);
    const byKind = Object.fromEntries(made.plan.phases.map((p) => [p.kind, p]));
    expect(byKind.dpo.enabled).toBe(false);
    expect(byKind.sft).toMatchObject({ on_fail: "skip", settings: { steps: 40 } });
    expect(byKind.pretrain.on_pass).toBe("pause");
    expect(byKind.midtrain.enabled).toBe(false);
    expect(posted(calls, "/auto-runs/run1/start")).toHaveLength(1);
    expect(screen.getByTestId("auto-run-status")).toHaveTextContent("Running");
  });

  test("includes mid-training with its steps", async () => {
    const running = run("running", "data", { data: { status: "running" } });
    const { calls } = server(null, { start: running });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await screen.findByTestId("plan-editor");
    expect(screen.queryByLabelText("Steps, Mid-training")).toBeNull(); // off: no settings yet
    await user.click(screen.getByRole("switch", { name: "Include Mid-training" }));
    await user.type(screen.getByLabelText("Steps, Mid-training"), "60000");
    await user.click(screen.getByRole("button", { name: "Start" }));
    await screen.findByTestId("auto-run-view");
    const made = posted(calls, "/auto-runs")[0].body as { plan: Plan };
    expect(made.plan.phases.find((p) => p.kind === "midtrain")).toMatchObject({ enabled: true, settings: { steps: 60000 } });
  });

  test("says what to fix instead of sending a setting the server would refuse", async () => {
    const { calls } = server(null);
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await screen.findByTestId("plan-editor");
    await user.type(screen.getByLabelText("Steps, Round 1 · Learn to answer"), "0");
    await user.click(screen.getByRole("button", { name: "Start" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Steps for Round 1 · Learn to answer must be a whole number from 1 to 5000");
    expect(posted(calls, "/auto-runs")).toHaveLength(0);
  });

  test("a start that is refused leaves the draft open and says why", async () => {
    server(null, { start: { status: 409, detail: "A post-training round for this version is already queued." } });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await screen.findByTestId("plan-editor");
    await user.click(screen.getByRole("button", { name: "Start" }));
    expect(await screen.findByText(/already queued/)).toBeVisible();
    expect(await screen.findByTestId("auto-run-status")).toHaveTextContent("Draft");
  });

  test("tells a person with no version to build one", async () => {
    mockApi({ "GET /versions": { body: [] }, "GET /auto-runs": { body: [] } });
    renderPage(<AutoRunPage />);
    expect(await screen.findByText(/No versions yet/)).toBeVisible();
  });
});

describe("Auto Run page: a draft", () => {
  test("opens on the draft and saves a changed plan", async () => {
    const { calls } = server(run("draft", null));
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await screen.findByTestId("plan-editor");
    await user.click(screen.getByRole("switch", { name: "Include Pretraining" }));
    await user.click(screen.getByRole("button", { name: "Save the plan" }));
    await waitFor(() => expect(posted(calls, "/auto-runs/run1/plan")).toHaveLength(1));
    expect((posted(calls, "/auto-runs/run1/plan")[0].body as { plan: Plan }).plan.phases.find((p) => p.kind === "pretrain")!.enabled).toBe(false);
  });
});

describe("Auto Run page: a running run", () => {
  const running = run(
    "running",
    "pretrain",
    { data: { status: "passed" }, pretrain: { status: "running", job_id: "run1-1-1" } },
    { progress: { id: "run1-1-1", kind: "pretrain", label: "Version 2 pretraining", status: "running", phase: "steady", step: 300, total: 1200, fraction: 0.25, eta_s: 900 } },
  );

  test("shows each phase's status and the live job", async () => {
    server(running);
    renderPage(<AutoRunPage />);
    const timeline = await screen.findByRole("region", { name: "Timeline" });
    expect(within(screen.getByTestId("phase-data")).getByText("Passed")).toBeVisible();
    expect(within(screen.getByTestId("phase-pretrain")).getByText("Running")).toBeVisible();
    expect(screen.getByTestId("phase-pretrain")).toHaveAttribute("aria-current", "step");
    expect(within(screen.getByTestId("phase-sft")).getByText("Not started")).toBeVisible();
    expect(within(timeline).queryByTestId("phase-midtrain")).toBeNull();
    expect(screen.getByRole("progressbar", { name: "Job progress" })).toHaveAttribute("aria-valuenow", "25");
    expect(screen.getByText("step 300 of 1,200 · about 15 min left")).toBeVisible();
    expect(screen.getByRole("link", { name: "Watch it in Runs" })).toHaveAttribute("href", expect.stringMatching(/^\/runs\/?$/));
    expect(within(screen.getByTestId("phase-pretrain")).getByRole("link", { name: "Open Pretraining" })).toHaveAttribute("href", expect.stringMatching(/^\/pretraining\/?$/));
    expect(screen.getByRole("button", { name: "Pause" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();
  });

  test("follows the events: a phase that passes moves the timeline on", async () => {
    const { state } = server(running);
    renderPage(<AutoRunPage />);
    await screen.findByTestId("phase-pretrain");
    state.run = run("running", "sft", { data: { status: "passed" }, pretrain: { status: "passed", checkpoint: "v2/best.npz" }, sft: { status: "running" } });
    act(() => emit("auto_run", { id: "run1", status: "running", current: "sft", reason: "" }));
    await waitFor(() => expect(within(screen.getByTestId("phase-pretrain")).getByText("Passed")).toBeVisible());
    expect(within(screen.getByTestId("phase-sft")).getByText("Running")).toBeVisible();
    expect(screen.getByText(/v2\/best\.npz/)).toBeVisible();
    expect(screen.queryByTestId("auto-run-progress")).toBeNull();
  });

  test("ignores another run's events", async () => {
    const { state, calls } = server(running);
    renderPage(<AutoRunPage />);
    await screen.findByTestId("phase-pretrain");
    const before = calls.filter((c) => c.path === "/auto-runs/run1").length;
    state.run = run("paused", "pretrain");
    act(() => emit("auto_run", { id: "other", status: "paused", current: "sft", reason: "" }));
    await new Promise((r) => setTimeout(r, 50));
    expect(calls.filter((c) => c.path === "/auto-runs/run1")).toHaveLength(before);
    expect(screen.getByTestId("auto-run-status")).toHaveTextContent("Running");
  });

  test("Pause asks the server and shows the answer", async () => {
    const pausing = { ...running, state: { ...running.state, status: "pausing" as const, reason: "" } };
    const { calls } = server(running, { pause: pausing });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await user.click(await screen.findByRole("button", { name: "Pause" }));
    expect(await screen.findByText(/the phase that is running finishes first/)).toBeVisible();
    expect(screen.getByTestId("auto-run-status")).toHaveTextContent("Pausing");
    expect(posted(calls, "/auto-runs/run1/pause")).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "Pause" })).toBeNull();
  });

  test("Stop asks first, then stops the run", async () => {
    const { calls } = server(running, { stop: run("stopped", "pretrain", { pretrain: { status: "stopped" } }, { reason: "Stopped by you." }) });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await user.click(await screen.findByRole("button", { name: "Stop the run" }));
    expect(posted(calls, "/auto-runs/run1/stop")).toHaveLength(0);
    await user.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Stop the run" }));
    expect(await screen.findByText("Stopped by you.")).toBeVisible();
    expect(screen.getByTestId("auto-run-status")).toHaveTextContent("Stopped");
    expect(screen.queryByRole("button", { name: "Pause" })).toBeNull();
    expect(screen.getByRole("button", { name: "Start another Auto Run" })).toBeVisible();
  });
});

describe("Auto Run page: the parameters the run chose", () => {
  const tuned = (() => {
    const base = run("running", "rlvr", { sft: { status: "failed", error: "Loss diverged." }, rlvr: { status: "queued", job_id: "run1-4-1" } });
    return {
      ...base,
      state: {
        ...base.state,
        tuning: [
          { phase: "rlvr", attempt: 1, settings: { learning_rate: 5e-5 }, evidence: null, proposal: { action: "grow", why: "Continue with a bigger design." },
            choices: [{ param: "learning_rate", from: 0.0001, to: 5e-5, why: "The sft phase diverged, so the learning rate is halved from 0.0001." }] },
        ],
      },
    } as RunView;
  })();

  test("lists each phase's chosen parameters with the reason, and a bigger-design suggestion", async () => {
    server(tuned);
    renderPage(<AutoRunPage />);
    const chosen = await screen.findByTestId("tuning-rlvr");
    expect(chosen).toHaveTextContent("learning rate: 0.0001 → 0.00005");
    expect(chosen).toHaveTextContent("The sft phase diverged, so the learning rate is halved");
    expect(chosen).toHaveTextContent("Suggestion: Continue with a bigger design.");
    expect(screen.queryByTestId("tuning-sft")).toBeNull();
  });

  test("the limits are shown in the draft's plan and saved as edited", async () => {
    const draft = run("draft", null);
    draft.plan = { ...draft.plan, bounds: { learning_rate: { min: 1e-6, max: 0.01 }, steps: { min: 10, max: 100000 } } };
    const { calls } = server(draft);
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await screen.findByTestId("plan-editor");
    await user.click(screen.getByText("Limits on automatic choices"));
    expect(screen.getByLabelText("Learning rate, largest")).toHaveValue("0.01");
    await user.clear(screen.getByLabelText("Learning rate, largest"));
    await user.type(screen.getByLabelText("Learning rate, largest"), "0.001");
    await user.click(screen.getByRole("button", { name: "Save the plan" }));
    await waitFor(() => expect(posted(calls, "/auto-runs/run1/plan")).toHaveLength(1));
    expect((posted(calls, "/auto-runs/run1/plan")[0].body as { plan: Plan }).plan.bounds).toMatchObject({ learning_rate: { min: 1e-6, max: 0.001 } });
  });
});

describe("Auto Run page: a run that waits for the person", () => {
  test("a run paused as planned continues", async () => {
    const paused = run("paused", "pretrain", { data: { status: "passed" }, pretrain: { status: "passed" } }, { reason: "The pretrain phase passed; paused as planned." });
    const { calls } = server(paused, { resume: run("running", "sft", { pretrain: { status: "passed" }, sft: { status: "running" } }) });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    expect(await screen.findByText("The pretrain phase passed; paused as planned.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Try this phase again" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(posted(calls, "/auto-runs/run1/resume")).toHaveLength(1));
    await waitFor(() => expect(screen.getByTestId("auto-run-status")).toHaveTextContent("Running"));
  });

  test("a failed phase shows its checks and can be tried again or skipped", async () => {
    const result = { round: 2, ok: false, verdict: "The checks did not pass.", checks: [{ name: "Held-back loss", detail: "3.1 to 3.4", ok: false }, { name: "Sample answers", detail: "", ok: null }] };
    const failed = run("paused", "rlvr", { rlvr: { status: "failed", error: "Round 2 failed its checks.", result } }, { reason: "The rlvr phase failed: Round 2 failed its checks." });
    const { calls } = server(failed, { continue: run("running", "rlvr", { rlvr: { status: "running", attempt: 2 } }) });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    const phase = await screen.findByTestId("phase-rlvr");
    const checks = within(phase).getByRole("list", { name: "Checks, Round 2 · Practice with checks" });
    expect(within(checks).getByText(/Held-back loss/)).toHaveTextContent("failed, 3.1 to 3.4");
    expect(within(checks).getByText(/Sample answers/)).toHaveTextContent("not measured");
    expect(within(phase).getByText("The checks did not pass.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();
    expect(screen.queryByTestId("kept-card")).toBeNull(); // no checkpoint was kept
    await user.click(screen.getByRole("button", { name: "Try this phase again" }));
    await waitFor(() => expect(posted(calls, "/auto-runs/run1/continue")).toHaveLength(1));
    expect(posted(calls, "/auto-runs/run1/continue")[0].body).toEqual({ choice: "retry" });
    expect(await screen.findByText(/attempt 2/)).toBeVisible();
  });

  test("Skip sends the choice to go on without the phase", async () => {
    const failed = run("waiting_user", "data", { data: { status: "blocked", error: "There is no text to train on yet." } }, { reason: "There is no text to train on yet. Add text in Your data and mark it Good." });
    const { calls } = server(failed, { continue: run("running", "pretrain", { data: { status: "skipped" }, pretrain: { status: "running" } }) });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await user.click(await screen.findByRole("button", { name: "Skip this phase" }));
    await waitFor(() => expect(posted(calls, "/auto-runs/run1/continue")).toHaveLength(1));
    expect(posted(calls, "/auto-runs/run1/continue")[0].body).toEqual({ choice: "next" });
  });

  test("a kept failed round offers Try in chat and Adopt, and adopting goes through the run", async () => {
    const result = { round: 1, ok: false, verdict: "Answers got worse.", checks: [{ name: "Held-back loss", detail: "", ok: false }], checkpoint: "v2/round1.npz" };
    const kept = run("paused", "sft", { sft: { status: "failed", checkpoint: "v2/round1.npz", result } }, { reason: "The sft phase failed: Answers got worse." });
    const { calls } = server(kept, { continue: run("running", "rlvr", { sft: { status: "skipped" }, rlvr: { status: "running" } }) });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    const card = await screen.findByTestId("kept-card");
    expect(within(card).getByRole("button", { name: "Try in chat" })).toBeVisible();
    await user.click(within(card).getByRole("button", { name: "Adopt this checkpoint" }));
    await waitFor(() => expect(posted(calls, "/auto-runs/run1/continue")).toHaveLength(1));
    expect(posted(calls, "/auto-runs/run1/continue")[0].body).toEqual({ choice: "adopt" });
    expect(calls.some((c) => c.path.startsWith("/post-training/rounds"))).toBe(false);
    await waitFor(() => expect(screen.queryByTestId("kept-card")).toBeNull());
    expect(screen.getByTestId("auto-run-status")).toHaveTextContent("Running");
  });

  test("a refusal shows the server's sentence", async () => {
    server(run("paused", "pretrain", { pretrain: { status: "passed" } }), { resume: { status: 409, detail: "Only a paused run can be resumed." } });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await user.click(await screen.findByRole("button", { name: "Continue" }));
    expect(await screen.findByText("Only a paused run can be resumed.")).toBeVisible();
  });

  test("a paused run's plan can be changed and saved", async () => {
    const paused = run("paused", "sft", { pretrain: { status: "passed" } });
    const { calls } = server(paused);
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    await screen.findByTestId("phase-sft");
    await user.click(screen.getByRole("switch", { name: "Include Round 3 · Learn from a teacher" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(posted(calls, "/auto-runs/run1/plan")).toHaveLength(1));
    expect((posted(calls, "/auto-runs/run1/plan")[0].body as { plan: Plan }).plan.phases.find((p) => p.kind === "distill")!.enabled).toBe(false);
  });
});

describe("Auto Run page: earlier runs", () => {
  test("with nothing going, a plan for a new run is shown and an earlier run can be opened; a finished run offers another", async () => {
    const done = run("done", null, { data: { status: "passed" } }, { reason: "Every phase has finished." });
    const earlier = { ...summary(run("stopped", "sft")), id: "old", name: "First try" };
    mockApi({
      "GET /versions": { body: VERSIONS },
      "GET /auto-runs": { body: [summary(done), earlier] },
      "GET /auto-runs/defaults": { body: plan() },
      "GET /auto-runs/run1": { body: done },
    });
    const user = userEvent.setup();
    renderPage(<AutoRunPage />);
    const list = await screen.findByRole("region", { name: "Earlier Auto Runs" });
    expect(await screen.findByTestId("plan-editor")).toBeVisible(); // nothing is going: a plan for a new run
    expect(within(list).getByText(/First try · Version 1/)).toBeVisible();
    await user.click(within(list).getAllByRole("button", { name: "Open" })[0]);
    expect(await screen.findByText("Every phase has finished.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Start another Auto Run" }));
    expect(await screen.findByTestId("plan-editor")).toBeVisible();
  });
});

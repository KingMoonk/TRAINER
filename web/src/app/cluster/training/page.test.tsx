import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import type { Run, StartRun } from "./logic";
import TrainingPage from "./page";
import { member, run } from "./test-data";

// The Training tab against a mocked API: GET /distributed/defaults and /distributed/runs, POST /distributed/runs, the two stop routes,
// and the lists the form reads. The live spec (tests/live.training.spec.ts) runs the same flow against a real --fake server.

const DEFAULTS = { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16" };
const SETS = [
  { id: "stories", name: "Stories", count: 1200, on: true, held: false, library: false },
  { id: "notes", name: "Notes", count: 40, on: false, held: false, library: false },
];
const CHATS = [{ id: "chats", name: "Your conversations", count: 30 }, { id: "stories", name: "Stories", count: 5 }];
const VERSIONS = [{ number: 2, name: "Wider", change: "", best_checkpoint: null }];
const node = (address: string, over: Record<string, unknown> = {}) => ({
  address, local: false, hostname: address, version: "1", device: "cuda", device_name: "RTX 4090", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "", ...over,
});
const NODES = [node("local", { local: true, hostname: "mac-mini", device: "mps", device_name: "Apple M2" }), node("gpu-box"), node("old-laptop", { online: false, error: "old-laptop did not answer within 2 seconds." })];

type Setup = { runs?: Run[]; start?: (body: StartRun) => Run | { status: number; detail: string }; sets?: typeof SETS; versions?: typeof VERSIONS; chats?: typeof CHATS; judges?: { url: string; models: string[]; selected: string; error: string | null } };
function server(opts: Setup = {}) {
  const state = { runs: opts.runs ?? [], judged: 0, started: [] as StartRun[], stopped: [] as string[], membersStopped: [] as string[] };
  mockApi({
    "GET /distributed/defaults": { body: DEFAULTS },
    "GET /distributed/runs": () => ({ body: state.runs }),
    "POST /distributed/runs": ({ body }) => {
      state.started.push(body as StartRun);
      const made = opts.start?.(body as StartRun) ?? run({ state: "queued", round: 0, history: [], members: [member("mac-mini", { local: true, state: "invited", can_stop: false })] });
      if ("status" in made) return { status: made.status, body: { detail: made.detail } };
      state.runs = [made, ...state.runs];
      return { status: 201, body: made };
    },
    "POST /distributed/runs/*/stop": ({ path }) => {
      const id = path.split("/")[3];
      state.stopped.push(id);
      state.runs = state.runs.map((r) => (r.id === id ? { ...r, state: "stopped", can_stop: false } : r));
      return { body: state.runs.find((r) => r.id === id) };
    },
    "POST /distributed/runs/*/members/*/stop": ({ path }) => {
      const [, , , id, , node] = path.split("/");
      state.membersStopped.push(decodeURIComponent(node));
      state.runs = state.runs.map((r) => (r.id === id ? { ...r, members: r.members.map((m) => (m.node === decodeURIComponent(node) ? { ...m, state: "stopped" as const, can_stop: false, note: "Stopped by you." } : m)) } : r));
      return { body: state.runs.find((r) => r.id === id) };
    },
    "GET /pretraining/text_sets": { body: opts.sets ?? SETS },
    "GET /versions": { body: opts.versions ?? VERSIONS },
    "GET /cluster/nodes": { body: NODES },
    "GET /distributed/chat_sets": { body: opts.chats ?? CHATS },
    "GET /post-training/judges": () => {
      state.judged += 1;
      return { body: opts.judges ?? { url: "http://localhost:11434", models: ["judge:1b", "big:8b"], selected: "", error: null } };
    },
  });
  return state;
}

describe("Training tab: the start form", () => {
  test("shows the defaults, says who coordinates, and offers the listed machines", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByLabelText("Rounds")).toHaveValue(10);
    expect(screen.getByLabelText("Steps per round")).toHaveValue(100);
    expect(screen.getByLabelText("Outer learning rate")).toHaveValue(0.7);
    expect(screen.getByLabelText("Outer momentum")).toHaveValue(0.9);
    expect(screen.getByText(/This computer coordinates/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Stories/ })).toBeChecked(); // the sets that are on
    expect(screen.getByRole("checkbox", { name: /Notes/ })).not.toBeChecked();
    expect(await screen.findByRole("checkbox", { name: /gpu-box/ })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: /old-laptop/ })).toBeInTheDocument();
    expect(screen.getByText("Unreachable")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Version 2 · Wider/ })).toBeInTheDocument();
    expect(screen.getByText("No run has been started. Start one above.")).toBeInTheDocument();
  });

  test("starts from the larger steps per round, with the reason, once a machine is ticked, and not when typed over", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    const box = await screen.findByRole("checkbox", { name: /gpu-box/ });
    expect(screen.queryByText(/moving the model is a fixed cost/)).not.toBeInTheDocument();
    await user.click(box);
    expect(screen.getByLabelText("Steps per round")).toHaveValue(1500);
    expect(screen.getByText("H 1500: moving the model is a fixed cost per round, so a longer round hides it.")).toBeInTheDocument();
    await user.click(box);
    expect(screen.getByLabelText("Steps per round")).toHaveValue(100);
    await user.clear(screen.getByLabelText("Steps per round"));
    await user.type(screen.getByLabelText("Steps per round"), "300");
    await user.click(box);
    expect(screen.getByLabelText("Steps per round")).toHaveValue(300);
    expect(screen.queryByText(/moving the model is a fixed cost/)).not.toBeInTheDocument();
  });

  test("starts a run with what was typed and ticked, and shows it", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /gpu-box/ }));
    await user.click(screen.getByRole("checkbox", { name: /Notes/ }));
    await user.clear(screen.getByLabelText("Rounds"));
    await user.type(screen.getByLabelText("Rounds"), "4");
    await user.selectOptions(screen.getByLabelText("Design"), "2");
    expect(screen.getByRole("checkbox", { name: /Overlap communication/ })).toBeChecked();
    await user.click(screen.getByRole("checkbox", { name: /Overlap communication/ }));
    await user.selectOptions(screen.getByLabelText("Number format of the traffic"), "int8");
    expect(screen.getByRole("checkbox", { name: /Reviewed examples only/ })).not.toBeChecked();
    await user.click(screen.getByRole("checkbox", { name: /Reviewed examples only/ }));
    await user.type(screen.getByLabelText("At least"), "2");
    await user.type(screen.getByLabelText("At most"), "3");
    await user.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(state.started).toHaveLength(1));
    expect(state.started[0]).toEqual({
      version: 2, dataset_ids: ["stories", "notes"], machines: ["gpu-box"], rounds: 4, inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, min_members: 2, max_members: 3, overlap: false, time_box: null, delta_dtype: "int8", reviewed_only: true,
    });
    expect(await screen.findByTestId("run-r1")).toHaveTextContent("Version 2 across machines");
    expect(screen.getByTestId("run-state-r1")).toHaveTextContent("Waiting for its turn");
  });

  test("starts a run with a teacher, a polish cadence and a win-rate target", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await screen.findByRole("button", { name: "Start run" });
    expect(state.judged).toBe(0); // Ollama is not asked until the section is opened
    await user.click(screen.getByText("Teacher (optional)"));
    expect(screen.getByLabelText("Teacher")).toHaveValue("");
    expect(await screen.findByRole("option", { name: "judge:1b" })).toBeInTheDocument();
    expect(state.judged).toBe(1);
    await user.type(screen.getByLabelText("Polish every"), "3");
    expect(screen.getByRole("button", { name: "Start run" })).toBeDisabled(); // no teacher chosen yet
    expect(screen.getByText(/Choose a teacher for the polish/)).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Teacher"), "judge:1b");
    await user.type(screen.getByLabelText("Win-rate target (%)"), "60");
    await user.click(await screen.findByRole("checkbox", { name: /Your conversations/ })); // a set of conversations only
    expect(screen.queryByRole("checkbox", { name: /^Stories 5/ })).toBeNull(); // a set with text is ticked above, not offered twice
    await user.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(state.started).toHaveLength(1));
    expect(state.started[0]).toMatchObject({ dataset_ids: ["stories", "chats"], teacher: "judge:1b", polish_every: 3, target_win_rate: 0.6 });
  });

  test("keeps the conversation sets out of the request when no teacher is used", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByText("Teacher (optional)"));
    await user.click(await screen.findByRole("checkbox", { name: /Your conversations/ }));
    await user.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(state.started).toHaveLength(1));
    expect(state.started[0].dataset_ids).toEqual(["stories"]);
  });

  test("says Ollama has no models when it answers with none", async () => {
    server({ judges: { url: "http://localhost:11434", models: [], selected: "", error: null } });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByText("Teacher (optional)"));
    expect(await screen.findByText("Ollama has no models; pull one first.")).toBeInTheDocument();
  });

  test("leaves the teacher out of the request when the section is not used", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Start run" }));
    await waitFor(() => expect(state.started).toHaveLength(1));
    for (const key of ["teacher", "polish_every", "target_win_rate"]) expect(state.started[0]).not.toHaveProperty(key);
  });

  test("a teacher with no polish and no target says so beside Start, and keeps it off", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByText("Teacher (optional)"));
    await user.selectOptions(await screen.findByLabelText("Teacher"), "judge:1b");
    expect(screen.getByRole("button", { name: "Start run" })).toBeDisabled();
    const reason = screen.getByText("Pick how often to polish or a win-rate target, or clear the teacher.");
    expect(reason.closest("details")).toBeNull(); // shown beside Start, not hidden in the section
  });

  test("says beside Start when the polish would never run", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByText("Teacher (optional)"));
    await user.selectOptions(await screen.findByLabelText("Teacher"), "judge:1b");
    await user.type(screen.getByLabelText("Polish every"), "10"); // 10 rounds
    expect(screen.getByRole("button", { name: "Start run" })).toBeDisabled();
    expect(screen.getByText(/Polish every must be fewer than the number of rounds/).closest("details")).toBeNull();
  });

  test("says why when Ollama cannot be read", async () => {
    server({ judges: { url: "http://localhost:11434", models: [], selected: "", error: "connection refused" } });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByText("Teacher (optional)"));
    expect(await screen.findByText(/Ollama could not be read, so no teacher can be chosen: connection refused/)).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "No teacher" })).toBeInTheDocument();
  });

  test("offers the ticked machines as coordinator and sends the choice", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    expect(screen.queryByLabelText("Coordinator")).not.toBeInTheDocument(); // nothing ticked: this computer coordinates
    await user.click(await screen.findByRole("checkbox", { name: /gpu-box/ }));
    const pick = screen.getByLabelText("Coordinator");
    expect(pick).toHaveValue("");
    await user.selectOptions(pick, "gpu-box");
    await user.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(state.started).toHaveLength(1));
    expect(state.started[0]).toMatchObject({ machines: ["gpu-box"], coordinator: "gpu-box" });
  });

  test("cannot start with no text ticked", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /Stories/ }));
    expect(screen.getByRole("button", { name: "Start run" })).toBeDisabled();
  });

  test("shows the server's reason when the run is refused, and keeps the form", async () => {
    server({ start: () => ({ status: 409, detail: "Save a design in the Network designer first." }) });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await userEvent.setup().click(await screen.findByRole("button", { name: "Start run" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not start the run: Save a design in the Network designer first.");
    expect(screen.getByLabelText("Rounds")).toHaveValue(10);
  });

  test("with no machine listed it says the run uses this computer alone", async () => {
    server();
    mockApi({ "GET /distributed/defaults": { body: DEFAULTS }, "GET /distributed/runs": { body: [] }, "GET /pretraining/text_sets": { body: SETS }, "GET /versions": { body: [] }, "GET /cluster/nodes": { body: [NODES[0]] } });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByText(/No other machine is listed/)).toBeInTheDocument();
    expect(screen.getByText(/No design is saved yet/)).toBeInTheDocument();
  });
});

describe("Training tab: a run", () => {
  test("shows each machine with its device kind, speed and last change, and the loss and bytes of each round", async () => {
    server({ runs: [run()] });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const card = within(await screen.findByTestId("run-r1"));
    expect(card.getByTestId("run-state-r1")).toHaveTextContent("Training");
    expect(card.getByText("mac-mini coordinates this run.")).toBeInTheDocument();
    expect(card.getByText("2 of 10 rounds done")).toBeInTheDocument();
    const gpu = within(card.getByTestId("member-gpu-box"));
    expect(gpu.getByText("RTX 4090 (cuda)")).toBeInTheDocument();
    expect(gpu.getByTestId("device-gpu-box")).toHaveTextContent("CUDA");
    expect(gpu.getByText("5,120 tokens/s")).toBeInTheDocument();
    expect(gpu.getByText("3.8 MB, float16")).toBeInTheDocument(); // on the edge that carries the change up
    expect(gpu.getByText("Training")).toBeInTheDocument();
    const me = within(card.getByTestId("member-mac-mini"));
    expect(me.getByText("This computer")).toBeInTheDocument();
    expect(me.getByText("Apple M2 (mps)")).toBeInTheDocument(); // an Apple GPU is a machine like any other
    expect(me.getByTestId("device-mac-mini")).toHaveTextContent("MPS");
    expect(me.queryByRole("button")).toBeNull(); // this computer cannot be stopped on its own
    await userEvent.setup().click(card.getByText("Rounds"));
    expect(card.getByTestId("round-2")).toHaveTextContent("3.000");
    expect(card.getByTestId("round-2")).toHaveTextContent("2.0 MB");
    expect(card.getByTestId("round-2")).toHaveTextContent("2 of 2");
  });

  test("an unreachable machine and one that refused with 401 show a sentence each while the others still render", async () => {
    server({
      runs: [run({
        members: [
          member("mac-mini", { local: true, rounds_done: 2, steps: 200, can_stop: false }),
          member("gpu-box", { rounds_done: 2, steps: 200 }),
          member("old-laptop", { device: "", device_name: "", state: "unreachable", note: "old-laptop did not answer within 2 seconds.", can_stop: false }),
          member("brother-pc", { device: "", device_name: "", state: "unreachable", note: "brother-pc refused this computer (401). Add this computer's Tailscale login to the allowed users on brother-pc.", can_stop: false }),
        ],
      })],
    });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByTestId("reason-old-laptop")).toHaveTextContent(/^old-laptop did not answer within 2 seconds\.$/);
    expect(screen.getByTestId("reason-brother-pc")).toHaveTextContent("refused this computer (401)");
    expect(within(screen.getByTestId("member-old-laptop")).getByText("Unreachable")).toBeInTheDocument();
    expect(screen.getByTestId("member-gpu-box")).toHaveTextContent("Training");
    expect(screen.queryByTestId("reason-gpu-box")).toBeNull();
  });

  test("a dropped member and a late one say why, with a fallback when the server gave no note", async () => {
    server({
      runs: [run({ members: [member("mac-mini", { local: true, can_stop: false }), member("gpu-box", { state: "dropped", note: "Missed 3 rounds.", can_stop: false }), member("slow", { state: "behind", missed: 1 })] })],
    });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByTestId("reason-gpu-box")).toHaveTextContent("Missed 3 rounds.");
    expect(screen.getByTestId("reason-slow")).toHaveTextContent("Late with its last change.");
  });

  test("stops a run, and its state follows", async () => {
    const state = server({ runs: [run()] });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await userEvent.setup().click(await screen.findByRole("button", { name: "Stop run" }));
    await waitFor(() => expect(state.stopped).toEqual(["r1"]));
    await waitFor(() => expect(screen.getByTestId("run-state-r1")).toHaveTextContent("Stopped"));
    expect(screen.queryByRole("button", { name: "Stop run" })).toBeNull();
  });

  test("stops one member and leaves the run going", async () => {
    const state = server({ runs: [run()] });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await userEvent.setup().click(await screen.findByRole("button", { name: "Stop gpu-box" }));
    await waitFor(() => expect(state.membersStopped).toEqual(["gpu-box"]));
    await waitFor(() => expect(screen.getByTestId("member-gpu-box")).toHaveTextContent("Stopped"));
    expect(screen.getByTestId("reason-gpu-box")).toBeInTheDocument();
    expect(screen.getByTestId("run-state-r1")).toHaveTextContent("Training");
    expect(state.stopped).toEqual([]);
  });

  test("a finished run offers no stop, and a queue event re-reads the list", async () => {
    const state = server({ runs: [run({ state: "finished", can_stop: false })] });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByTestId("run-state-r1")).toHaveTextContent("Finished");
    expect(screen.queryByRole("button", { name: "Stop run" })).toBeNull();
    state.runs = [run({ id: "r2", label: "Second run", state: "waiting", can_stop: true, started_at: 200 }), ...state.runs];
    emit("queue", {});
    expect(await screen.findByTestId("run-r2")).toHaveTextContent("Waiting for members");
  });

  test("a run that cannot be read says why", async () => {
    mockApi({
      "GET /distributed/defaults": { body: DEFAULTS }, "GET /pretraining/text_sets": { body: SETS }, "GET /versions": { body: [] }, "GET /cluster/nodes": { body: NODES },
      "GET /distributed/runs": { status: 500, body: { detail: "The runs could not be read." } },
    });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByText("Could not read the runs: The runs could not be read.")).toBeInTheDocument();
  });
});

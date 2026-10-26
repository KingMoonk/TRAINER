import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { emit, json, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import PretrainingPage from "./page";

// The Pretraining page against a mocked API (the component tier of tests/pretraining.spec.ts, which keeps "Add to the queue"):
// the sets, the plan the server works out from the page's choices, the size preview and the learning-rate range test as jobs
// whose results arrive on /events.

const job = (id: string, kind: string, result: unknown, status = "done") => ({
  id,
  kind,
  status,
  progress: "",
  fraction: null,
  result,
  error: "",
  created_at: 0,
  started_at: 0,
  finished_at: status === "done" ? 1 : null,
});

const PIECES = 2_000_000;
const preview = {
  pieces: PIECES,
  estimated: false,
  params: 1_000_000,
  vocab: 256,
  context: 256,
  first_rows: [
    [0, 0, 1, -1, 2, 2],
    [1, 1, 1, 1, -1, 0],
  ],
  tokens_per_second: null,
};

type PlanBody = { dataset_ids: string[]; mode: string; rows: number; hold: number; passes: number; hours: number; steps: number | null; plateau: boolean | null; keep_my_settings: boolean; window_hours: number };

const SIZING = {
  verdict: "model_too_small",
  warning: "This model is too small for this much text. The plan reads 25k word pieces per parameter.",
  tokens_per_param: 25000,
  measured: true,
  seconds_per_step: 0.2275,
  tokens_per_second: 18000,
  device: "cpu",
  run_hours: 51.8,
  hours: 4,
  expected: {
    basis: "fit",
    loss_planned: 1.53,
    loss_flat: 1.6,
    flat_steps: 129129,
    curve: [
      [1000, 2.5],
      [50000, 1.7],
      [129129, 1.6],
      [819635, 1.53],
    ],
  },
  suggestion: { width: 192, layers: 4, params: 1879488, steps: 14007, hours: 4, tokens_per_param: 30.5, same_design: false, text: "For 4.0 h, a design about 192 wide and 4 layers deep fits best." },
  gpu_note: "A RTX 4090 would run this design about 10.0× faster than the CPU does.",
};

const HARDWARE = { device: "cuda", gpu_name: "Test GPU", vram_total: 8 * 2 ** 30, vram_free: 7 * 2 ** 30, ram_total: 32 * 2 ** 30, ram_free: 24 * 2 ** 30, disk_free: 300 * 2 ** 30, cpu_cores: 16, tokens_per_second: 14000, measured_at: 1 };
const SCALING = {
  notes: [
    { kind: "batch", reason: "vram", text: "Batch 16 -> 3 x 6 accumulation to fit the 7.0 GB of GPU memory free.", advisory: false },
    { kind: "context", reason: "vram", text: "Context 2048 -> 1024 tokens: the activations at 2048 do not fit the 7.0 GB of GPU memory free.", advisory: false },
    { kind: "tokens", reason: "time", text: "Token budget 33B -> 3.6B to finish in about 3.0 days at 14,000 tokens/s.", advisory: true },
  ],
  scaled: true,
  can_keep_mine: true,
  kept_mine: false,
  window_hours: 72,
  hours: 72,
  token_budget: 3_600_000_000,
  offload: false,
  blocker: null,
  hardware: HARDWARE,
};

function server({ known = true, sizing = SIZING as unknown, batchSize, scaling = null, refusal = "" }: { known?: boolean; sizing?: unknown; batchSize?: number; scaling?: unknown; refusal?: string } = {}) {
  const state = { plans: [] as PlanBody[], lr: [] as unknown[], known, sizing, rates: [] as unknown[], starts: [] as unknown[], scaling };
  mockApi({
    "GET /versions": {
      body: [
        { number: 1, name: "version 1", change: "", parent: null, spec: {}, status: "trained", best_checkpoint: "ck1", scorecard: {}, verdict: "", prediction: {} },
        { number: 2, name: "version 2", change: "", parent: 1, spec: { train: { optimizer: { config: { lr: 0.0006 } }, ...(batchSize ? { run: { batch_size: batchSize } } : {}) } }, status: "saved", best_checkpoint: null, scorecard: {}, verdict: "", prediction: {} },
      ],
    },
    "GET /pretraining/text_sets": {
      body: [
        { id: "stories", name: "Short stories", count: 1200, held: false, library: false, on: true },
        { id: "catalog_wiki_validation", name: "Wiki validation", count: 300, held: true, library: true, on: false },
      ],
    },
    "POST /pretraining/plan": ({ body }) => {
      const sent = body as PlanBody;
      state.plans.push(sent);
      const ids = sent.dataset_ids;
      const perStep = sent.rows * 256;
      const steps = state.known && ids.length ? Math.ceil((PIECES * sent.passes) / perStep) : null;
      const reason = !ids.length ? "Choose some text first." : !state.known ? "The size of the text is still being worked out." : refusal;
      return {
        body: {
          version: 2,
          parent: 1,
          mode: sent.mode,
          can_continue: true,
          continue_blocked: null,
          estimating: ids.length > 0 && !state.known,
          pieces: state.known && ids.length ? PIECES : null,
          estimated: false,
          params: 1_000_000,
          context: 256,
          pieces_per_step: perStep,
          passes_over_text: state.known && ids.length ? Math.round(PIECES / perStep) : null,
          steps,
          minutes: null,
          sizing: state.known && ids.length ? state.sizing : null,
          scaling: state.known && ids.length && state.scaling ? { ...(state.scaling as object), kept_mine: sent.keep_my_settings } : null,
          plateau: null,
          plateau_line: sent.plateau === false || !ids.length ? "" : "Stops by itself, keeping its best, once the best validation loss improves by less than 1% over 49 steps (not before step 98).",
          lr: 0.0006,
          readout: ids.length
            ? { characters: 8_000_000, pieces: state.known ? PIECES : 1_900_000, counted: state.known, chars_per_piece: 4, params: 1_000_000, ratio: 2, ratio_text: "2.00", verdict: "few", target_params: 100_000 }
            : null,
          enough: state.known && ids.length ? { ratio: 2, ratio_text: "2.00", headline: "This model could use about 10.0× more text than you have", verdict: "little" } : null,
          expectation: state.known && ids.length ? { headline: "Expect the error to fall to about 3.1 bits per character", details: "Based on the size of the model and the text." } : null,
          config: steps ? { steps, batch_size: sent.rows, validation: sent.hold, objective: "all" } : null,
          can_add: !reason,
          reason,
          job: reason ? null : { kind: "pretrain", label: `Version 2 ${sent.mode === "catch_up" ? "catch-up" : "from scratch"}`, params: { config: { steps }, datasets: ids, mode: sent.mode, version: 2 } },
          outlook: "If you add it: starts now",
        },
      };
    },
    "GET /pretraining/hardware": { body: HARDWARE },
    "GET /pretraining/preview": () => json(state.known ? preview : null), // mockApi turns a null body into {}, so a null answer is sent as a Response
    "POST /pretraining/preview": () => {
      state.known = true; // the job has the answer; it arrives on /events
      return { status: 202, body: job("p1", "pretrain_preview", null, "running") };
    },
    "POST /pretraining/step_rate": ({ body }) => {
      state.rates.push(body);
      return { status: 202, body: job("r1", "step_rate", null, "running") };
    },
    "POST /pretraining/start": ({ body }) => {
      state.starts.push(body);
      return { status: 202, body: { id: "q1", label: "Version 2, once the size of its text is known" } };
    },
    "GET /queue/jobs/*/outlook": { body: { outlook: "Starts after the run in progress" } },
    "POST /pretraining/lr_range_test": ({ body }) => {
      state.lr.push(body);
      return { status: 202, body: job("lr1", "lr_range_test", null, "running") };
    },
    "GET /jobs/*": { body: job("x", "x", null, "running") },
    "GET /data/collections/*": { body: [{ id: "tinystories", name: "TinyStories", size: "64 MB", records: "77,941 stories", state: "not downloaded", completed: 0, total: 0, error: "" }] },
  });
  return state;
}

describe("Pretraining page", () => {
  test("Pretraining opens prose generation", async () => {
    server();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByRole("link", { name: /Make extra data/ })).toHaveAttribute("href", expect.stringMatching(/^\/your-data\/?\?kind=text#make-extra-data$/));
  });

  test("the choices give the plan the server works out, and a held-out set starts unticked", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByRole("heading", { level: 1, name: "Pretraining" })).toBeVisible();
    await waitFor(() => expect(screen.getByTestId("model-line")).toHaveTextContent(/^Version 2 · from the Network designer$/));
    expect(await screen.findByTestId("set-stories")).toHaveAttribute("aria-checked", "true");
    expect(screen.getByTestId("set-catalog_wiki_validation")).toHaveAttribute("aria-checked", "false");
    expect(screen.getByTestId("set-catalog_wiki_validation")).toHaveTextContent("held-out: kept for testing, not trained on");
    expect(screen.getByTestId("selected-line")).toHaveTextContent(/^1 source selected$/);
    expect(await screen.findByTestId("enough-headline")).toHaveTextContent("This model could use about 10.0× more text than you have");
    expect(screen.getByTestId("expectation")).toHaveTextContent("Expect the error to fall");
    expect(screen.getByTestId("per-step")).toHaveTextContent(/^4,096 pieces per step · one pass over your data ≈ 488 steps$/);
    expect(screen.getByTestId("steps-line")).toHaveTextContent("977 steps");
    expect(screen.getByTestId("rows-grid")).toBeVisible();
    expect(screen.getByTestId("mode-catch_up")).toHaveTextContent("Continue from version 1");
    expect(screen.getByText("Learning rate: auto (0.0006, set with width-scaled setup)")).toBeVisible();
    expect(state.plans.at(-1)).toMatchObject({ dataset_ids: ["stories"], mode: "catch_up", rows: 16, hold: 0.05, passes: 2 });

    // A changed choice is sent whole to the server, which answers with the new steps.
    await user.click(screen.getByRole("radio", { name: "32" }));
    await user.click(screen.getByRole("radio", { name: "Long" }));
    await user.click(screen.getByRole("radio", { name: "10%" }));
    await user.click(screen.getByTestId("set-catalog_wiki_validation"));
    await waitFor(() => expect(screen.getByTestId("selected-line")).toHaveTextContent(/^2 sources selected$/));
    await waitFor(() => expect(state.plans.at(-1)).toMatchObject({ dataset_ids: ["stories", "catalog_wiki_validation"], rows: 32, hold: 0.1, passes: 4 }));
    await waitFor(() => expect(screen.getByTestId("steps-line")).toHaveTextContent("977 steps"));

    // With nothing chosen the plan says why it cannot be added.
    await user.click(screen.getByTestId("set-stories"));
    await user.click(screen.getByTestId("set-catalog_wiki_validation"));
    await waitFor(() => expect(screen.getByTestId("selected-line")).toHaveTextContent(/^nothing selected$/));
    expect(await screen.findByTestId("reason")).toHaveTextContent(/^Choose some text first\.$/);
    expect(screen.getByRole("button", { name: "Add to the queue" })).toBeDisabled();
  });

  test("rows per step start at the batch size the newest version saved, and the person can still pick another", async () => {
    const state = server({ batchSize: 24 });
    const user = userEvent.setup();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByRole("radio", { name: "24" })).toHaveAttribute("aria-checked", "true");
    await waitFor(() => expect(state.plans.at(-1)).toMatchObject({ rows: 24 }));
    expect(screen.getAllByRole("radio").filter((r) => ["8", "16", "24", "32"].includes(r.textContent ?? "")).map((r) => r.textContent)).toEqual(["8", "16", "24", "32"]);
    await user.click(screen.getByRole("radio", { name: "8" }));
    await waitFor(() => expect(state.plans.at(-1)).toMatchObject({ rows: 8 }));
  });

  test("the plan warns that the model is too small, shows the time and the expected curve, and offers a design that fits the hours", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByTestId("sizing-warning")).toHaveTextContent("This model is too small for this much text");
    expect(screen.getByTestId("sizing-warning")).toHaveAttribute("role", "alert");
    expect(screen.getByTestId("sizing-time")).toHaveTextContent("These 977 steps take about 2.2 days on the CPU.");
    expect(screen.getByTestId("sizing-expected")).toHaveTextContent("by step 129,129");
    expect(screen.getByTestId("expected-curve")).toBeVisible();
    expect(screen.getByTestId("sizing-suggestion")).toHaveTextContent("a design about 192 wide and 4 layers deep");
    expect(screen.getByRole("link", { name: /Open the Network designer/ })).toHaveAttribute("href", "/builder");
    expect(screen.getByTestId("sizing-gpu")).toHaveTextContent("RTX 4090");
    expect(state.plans.at(-1)).toMatchObject({ hours: 4, window_hours: 4, steps: null });
    expect(state.rates).toEqual([]); // measured already: no timing job

    await user.click(screen.getByRole("radio", { name: "12 h" }));
    await waitFor(() => expect(state.plans.at(-1)).toMatchObject({ hours: 12, window_hours: 12 })); // the scaling is asked again for the new window
  });

  test("the plan says how the run was scaled to this computer, and the override keeps the person's settings", async () => {
    const state = server({ scaling: SCALING });
    const user = userEvent.setup();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByTestId("hardware-line")).toHaveTextContent("This computer: Test GPU, 8.0 GB GPU memory (7.0 GB free), 24.0 GB RAM free, 300.0 GB disk free, 14,000 tokens/s measured.");
    expect(screen.getByTestId("scaling-lead")).toHaveTextContent("The run was scaled to fit this computer:");
    const notes = screen.getByTestId("scaling-notes");
    expect(notes).toHaveTextContent("Batch 16 -> 3 x 6 accumulation");
    expect(notes).toHaveTextContent("Context 2048 -> 1024 tokens");
    expect(notes).toHaveTextContent("Token budget 33B -> 3.6B");
    expect(state.plans.at(-1)).toMatchObject({ keep_my_settings: false });
    await user.click(screen.getByTestId("keep-mine"));
    await waitFor(() => expect(state.plans.at(-1)).toMatchObject({ keep_my_settings: true }));
    expect(await screen.findByText("Keeping your settings.")).toBeVisible();
    expect(screen.getByTestId("keep-mine")).toHaveTextContent("Scale it to fit");
  });

  test("a run that cannot fit is refused with the plain message and cannot be added", async () => {
    const message = "Not enough GPU memory: 0.0 GB free for this run, and even the smallest model needs about 0.2 GB.";
    server({ scaling: { ...SCALING, notes: [], scaled: false, can_keep_mine: false, blocker: { resource: "vram", needed: 200_000_000, available: 0 }, refusal: message }, refusal: message });
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByTestId("scaling-refused")).toHaveTextContent(message);
    expect(screen.getByTestId("scaling-lead")).toHaveTextContent("This run cannot start here:");
    expect(screen.queryByTestId("keep-mine")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add to the queue" })).toBeDisabled();
  });

  test("a device mismatch is refused with its sentence even without a blocker, and never says the run fits", async () => {
    const message = "This run is set to use the CUDA, but this computer has no CUDA GPU.";
    server({ scaling: { ...SCALING, notes: [], scaled: false, can_keep_mine: false, blocker: null, refusal: message }, refusal: message });
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByTestId("scaling-refused")).toHaveTextContent(message);
    expect(screen.getByTestId("scaling-lead")).toHaveTextContent("This run cannot start here:");
    expect(screen.queryByText("The run fits this computer as it is set.")).not.toBeInTheDocument();
  });

  test("a run that fits shows the machine and no changes", async () => {
    server({ scaling: { ...SCALING, notes: [], scaled: false, can_keep_mine: false } });
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByTestId("scaling-lead")).toHaveTextContent("The run fits this computer as it is set.");
    expect(screen.queryByTestId("scaling-notes")).not.toBeInTheDocument();
  });

  test("the plan says when the run will stop itself, and the person can turn that off", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByTestId("plateau-line")).toHaveTextContent("Stops by itself, keeping its best");
    expect(state.plans.at(-1)).toMatchObject({ plateau: null });
    await user.click(screen.getByRole("radio", { name: "Never" }));
    await waitFor(() => expect(state.plans.at(-1)).toMatchObject({ plateau: false }));
    await waitFor(() => expect(screen.queryByTestId("plateau-line")).not.toBeInTheDocument());
  });

  test("a same-design suggestion can be taken as the step count, and choosing a length gives it back", async () => {
    const state = server({ sizing: { ...SIZING, verdict: "balanced", suggestion: { ...SIZING.suggestion, same_design: true, steps: 500, text: "Keep this design and run 500 steps." } } });
    const user = userEvent.setup();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    await user.click(await screen.findByRole("button", { name: "Use 500 steps" }));
    await waitFor(() => expect(state.plans.at(-1)).toMatchObject({ steps: 500 }));
    expect(screen.queryByRole("link", { name: /Network designer/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: "Long" }));
    await waitFor(() => expect(state.plans.at(-1)).toMatchObject({ steps: null, passes: 4 }));
  });

  test("a design this computer has not timed is timed by a job, and the plan is asked again", async () => {
    const state = server({ sizing: { ...SIZING, measured: false, seconds_per_step: null, run_hours: null, suggestion: null, expected: null } });
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByTestId("sizing-time")).toHaveTextContent("Timing a few steps of this design on this computer…");
    await waitFor(() => expect(state.rates).toEqual([{ version: 2, dataset_ids: ["stories"], rows: 16, preview_rows: 6 }]));
    state.sizing = SIZING;
    act(() => emit("job", job("r1", "step_rate", { seconds_per_step: 0.2275, tokens_per_second: 18000, device: "cpu", device_name: "CPU", note: "", rows: 16, context: 256 })));
    await waitFor(() => expect(screen.getByTestId("sizing-time")).toHaveTextContent("about 2.2 days"));
  });

  test("the size preview shows a skeleton, then the result its job delivers", async () => {
    const state = server({ known: false });
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect((await screen.findAllByLabelText("Estimating how much text you have…")).length).toBeGreaterThan(0);
    expect(screen.queryByTestId("enough-headline")).not.toBeInTheDocument();
    expect(await screen.findByTestId("size-first")).toHaveTextContent("You can add the run now: it waits in the queue until the size is known.");
    expect(screen.getByRole("button", { name: "Add to the queue" })).toBeEnabled();

    await waitFor(() => expect(state.known).toBe(true)); // the job was asked for
    act(() => emit("job", job("p1", "pretrain_preview", preview))); // its result arrives on /events
    expect(await screen.findByTestId("enough-headline")).toHaveTextContent("This model could use about 10.0× more text than you have");
    expect(screen.getByTestId("rows-grid")).toBeVisible();
    await waitFor(() => expect(screen.getByRole("button", { name: "Add to the queue" })).toBeEnabled());
  });

  test("the quick readout shows while the size is estimated, then the counted pieces", async () => {
    const state = server({ known: false });
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect(await screen.findByTestId("readout-size")).toHaveTextContent("8.0M characters · about 1.9M word pieces · 1.0M parameters");
    expect(screen.getByTestId("readout-source")).toHaveTextContent("estimated at 4 characters each");
    expect(screen.getByTestId("readout-advice")).toHaveTextContent("Short on text: about 2.00 word pieces per parameter");
    expect(screen.getByTestId("readout-advice")).toHaveTextContent("about 100k parameters");

    await waitFor(() => expect(state.known).toBe(true));
    act(() => emit("job", job("p1", "pretrain_preview", preview)));
    await waitFor(() => expect(screen.getByTestId("readout-size")).toHaveTextContent("8.0M characters · 2.0M word pieces · 1.0M parameters"));
    expect(screen.getByTestId("readout-source")).toHaveTextContent("counted by the size preview");
    expect(screen.getByTestId("readout-advice")).toHaveTextContent("Short on text: 2.00 word pieces per parameter");
  });

  test("adding while the size is worked out puts the run in the queue at once", async () => {
    const state = server({ known: false });
    const user = userEvent.setup();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    await user.click(await screen.findByRole("button", { name: "Add to the queue" }));
    expect(state.starts).toMatchObject([{ version: 2, dataset_ids: ["stories"], rows: 16, preview_rows: 6 }]);
    expect(await screen.findByTestId("queue-line")).toHaveTextContent("Queued. Starts after the run in progress."); // no job to wait for
  });

  test("the learning-rate range test shows a skeleton, then its chart and the sentence that reads it", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    await waitFor(() => expect(screen.getByTestId("model-line")).toBeVisible());
    await user.click(await screen.findByRole("button", { name: "Find a learning rate" }));
    expect(await screen.findByLabelText("Trying learning rates, smallest first. This takes a moment.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Stop" })).toBeVisible();
    expect(state.lr).toEqual([{ spec: { train: { optimizer: { config: { lr: 0.0006 } } } } }]);

    const rates = [1e-5, 1e-4, 1e-3, 1e-2];
    const result = { rates, losses: [5, 4.5, 3, 4], smoothed: [5, 4.5, 3, 4], suggestion: 1e-4, steepest: 2, diverged: false, cancelled: false, failed_at: null };
    act(() => emit("job", job("lr1", "lr_range_test", result)));
    const text = await screen.findByTestId("lr-text");
    expect(text).toHaveTextContent("Suggested learning rate: 0.0001. The error fell fastest near 0.001");
    expect(text).toHaveTextContent("This design uses 0.0006.");
  });
});

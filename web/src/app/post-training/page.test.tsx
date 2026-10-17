import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import PostTrainingPage from "./page";

// The Post-training page against a mocked API (the component tier of tests/post-training.spec.ts): the page reads /versions,
// /post-training/options and /queue, runs the status job, queues a round, shows its progress and imports a model as a new version.

// Radix's Select asks for pointer capture and scrolling, which jsdom does not have.
beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.setPointerCapture ??= () => {};
  Element.prototype.releasePointerCapture ??= () => {};
});

const version = (number: number, trained: boolean) => ({
  number,
  name: `version ${number}`,
  change: number === 2 ? "Started from GPT-2 (124M)" : "First design",
  parent: null,
  spec: {},
  status: trained ? "trained" : "saved",
  best_checkpoint: trained ? `ck${number}` : null,
  scorecard: {},
  verdict: "",
  prediction: {},
});

const job = (id: string, result: unknown, status = "done") => ({
  id,
  kind: "post_training.status",
  status,
  progress: "",
  fraction: null,
  result,
  error: "",
  created_at: 0,
  started_at: 0,
  finished_at: status === "done" ? 1 : null,
});

const idle = {
  rounds: {},
  counts: { conversations: 12, corrections: 0, problems: 0, pairs: 0 },
  gates: {},
  overlap: {},
  example: { prompt: "What is 2 + 2?", answer: "4" },
};
const running = { ...idle, rounds: { "1": { status: "running", note: "" } } };

const queue = (step: number) => ({
  job: step ? { id: "j1", kind: "post_round", label: "Round 1 · Practice answers · LoRA rank 16", note: "", params: {}, status: "running", eta_s: 120, created_at: 0, error: "" } : null,
  step,
  total: step ? 100 : 0,
  phase: "",
  lr: null,
  eta_s: step ? 120 : null,
  paused: false,
  blocked: null,
  gate: { reason: null, resumes_at: null, supported: true },
  stopping: false,
  up_next: [],
  plan: {},
});

const MODELS = [
  { id: "gpt2", name: "GPT-2 (124M)", size: 548_000_000 },
  { id: "new-model", name: "New server model", size: 12_000_000 },
];

function server(extra: Parameters<typeof mockApi>[0] = {}) {
  const state = { versions: [version(1, true)], statusCalls: 0, rounds: [] as unknown[], imports: [] as unknown[], queued: false };
  const calls = mockApi({
    "GET /post-training/import-models": { body: MODELS },
    "GET /versions": () => ({ body: state.versions }),
    "GET /post-training/options": {
      body: {
        leash: [{ name: "Medium", beta: 0.1, note: "A balance. A good default." }],
        lora_ranks: [4, 8, 16, 32],
        lora_blocker: null,
        lora_default: false,
        teachers: [],
        default_teacher: null,
      },
    },
    "GET /queue": () => ({ body: queue(state.queued ? 40 : 0) }),
    "GET /queue/outlook": { body: { outlook: "Starts now" } },
    "POST /post-training/status": () => {
      state.statusCalls += 1;
      return { status: 202, body: job(`s${state.statusCalls}`, null, "queued") };
    },
    // The status job's result is read back from /jobs: the first read shows an idle round 1, the read after queuing shows it running.
    "GET /jobs/*": ({ path }) => ({ body: job(path.split("/").pop()!, path.endsWith("/s1") ? idle : running) }),
    "POST /post-training/rounds": ({ body }) => {
      state.rounds.push(body);
      state.queued = true;
      return { body: { id: "j1", label: "Round 1 · Practice answers · LoRA rank 16", round: 1, params: {} } };
    },
    "POST /post-training/imports": ({ body }) => {
      state.imports.push(body);
      state.versions = [...state.versions, version(2, false)];
      return { body: version(2, false) };
    },
    ...extra,
  });
  return { state, calls };
}

describe("Post-training page", () => {
  test("new server catalogue entries are selectable and imported by their ID", async () => {
    const { state } = server();
    const user = userEvent.setup();
    renderPage(<PostTrainingPage />);
    await user.click(await screen.findByRole("combobox", { name: "Pretrained model" }));
    await user.click(await screen.findByRole("option", { name: "New server model" }));
    expect(await screen.findByText("About 12 MB to download once.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Import New server model" }));
    await waitFor(() => expect(state.imports).toHaveLength(1));
    // the body is a multipart form: the model goes by its ID, not its display name
    expect(state.imports[0]).toContain("new-model");
  });

  test("a failed model list cannot fall back to a hardcoded import", async () => {
    server({ "GET /post-training/import-models": { status: 503, body: { detail: "Catalogue unavailable" } } });
    const user = userEvent.setup();
    renderPage(<PostTrainingPage />);
    expect(await screen.findByText(/Could not load importable models: Catalogue unavailable/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Import model" })).toBeDisabled();

    mockApi({
      "GET /post-training/import-models": { body: [] },
      "GET /versions": { body: [version(1, true)] },
      "GET /post-training/options": { body: { leash: [], lora_ranks: [], lora_blocker: null, lora_default: false, teachers: [], default_teacher: null } },
      "GET /queue": { body: queue(0) },
      "GET /queue/outlook": { body: { outlook: "Starts now" } },
    });
    await user.click(screen.getByRole("button", { name: "Retry model list" }));
    expect(await screen.findByText("This server has no importable models.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Import model" })).toBeDisabled();
  });

  test("a round with LoRA on is queued, shows its progress, and an imported version appears", async () => {
    const { state } = server();
    const user = userEvent.setup();
    renderPage(<PostTrainingPage />);
    expect(await screen.findByRole("heading", { level: 1, name: "Post-training" })).toBeVisible();

    // The form is filled from /post-training/options, including the LoRA choice.
    expect(await screen.findByText("Version 1 · best checkpoint from Pretraining")).toBeVisible();
    expect(await screen.findByText("What it changes")).toBeVisible();
    await user.click(screen.getByRole("switch", { name: "Train LoRA adapters" }));
    const rank = await screen.findByRole("radiogroup", { name: "Adapter size (rank)" });
    expect(within(rank).getByRole("radio", { name: "16" })).toBeVisible();
    await user.click(within(rank).getByRole("radio", { name: "16" }));

    await user.click(screen.getByRole("button", { name: "Add to the queue" }));
    expect(await screen.findByText("Added to the queue: Round 1 · Practice answers · LoRA rank 16")).toBeVisible();
    expect(state.rounds).toEqual([expect.objectContaining({ round: 1, version: 1, lora: true, lora_rank: 16 })]);

    // The refreshed status says round 1 is running, and the queue's step count is its progress.
    expect(await screen.findByTestId("step-line")).toHaveTextContent("step 40 of 100");
    expect(screen.getByRole("progressbar", { name: "Round progress" })).toHaveAttribute("aria-valuenow", "40");

    // Importing a model saves a new version, which is listed at once.
    expect(screen.queryByTestId("version-2")).toBeNull();
    await user.click(await screen.findByRole("button", { name: /^Import GPT-2/ }));
    expect(await screen.findByTestId("version-2")).toHaveTextContent("Started from GPT-2 (124M)");
    expect(state.imports).toHaveLength(1);
  });

  test("a round that has nothing to learn from yet says why and cannot be queued", async () => {
    server();
    const user = userEvent.setup();
    renderPage(<PostTrainingPage />);
    await user.click(await screen.findByRole("tab", { name: /Your votes/ }));
    expect(await screen.findByText("Rate at least 2 pairs in Chat first.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Add votes round to the queue" })).toBeDisabled();
  });
});

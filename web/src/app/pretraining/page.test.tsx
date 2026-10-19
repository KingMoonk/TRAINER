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

type PlanBody = { dataset_ids: string[]; mode: string; rows: number; hold: number; passes: number };

function server({ known = true } = {}) {
  const state = { plans: [] as PlanBody[], lr: [] as unknown[], known };
  mockApi({
    "GET /versions": {
      body: [
        { number: 1, name: "version 1", change: "", parent: null, spec: {}, status: "trained", best_checkpoint: "ck1", scorecard: {}, verdict: "", prediction: {} },
        { number: 2, name: "version 2", change: "", parent: 1, spec: { train: { optimizer: { config: { lr: 0.0006 } } } }, status: "saved", best_checkpoint: null, scorecard: {}, verdict: "", prediction: {} },
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
      const reason = !ids.length ? "Choose some text first." : !state.known ? "The size of the text is still being worked out." : "";
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
          lr: 0.0006,
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
    "GET /pretraining/preview": () => json(state.known ? preview : null), // mockApi turns a null body into {}, so a null answer is sent as a Response
    "POST /pretraining/preview": () => {
      state.known = true; // the job has the answer; it arrives on /events
      return { status: 202, body: job("p1", "pretrain_preview", null, "running") };
    },
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

  test("the size preview shows a skeleton, then the result its job delivers", async () => {
    const state = server({ known: false });
    renderPage(<PretrainingPage />, { route: "/pretraining/" });
    expect((await screen.findAllByLabelText("Estimating how much text you have…")).length).toBeGreaterThan(0);
    expect(screen.queryByTestId("enough-headline")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add to the queue" })).toBeDisabled();
    expect(await screen.findByTestId("reason")).toHaveTextContent(/^The size of the text is still being worked out\.$/);

    await waitFor(() => expect(state.known).toBe(true)); // the job was asked for
    act(() => emit("job", job("p1", "pretrain_preview", preview))); // its result arrives on /events
    expect(await screen.findByTestId("enough-headline")).toHaveTextContent("This model could use about 10.0× more text than you have");
    expect(screen.getByTestId("rows-grid")).toBeVisible();
    await waitFor(() => expect(screen.getByRole("button", { name: "Add to the queue" })).toBeEnabled());
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

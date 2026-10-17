import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import LossPage from "./page";

// The Loss page against a mocked API (ported from the Playwright spec tests/loss.spec.ts, now removed). Numeric expectations are written out
// here, independent of lossView.

const prompt = "The largest planet in our solar system is Jupiter";

const version = (number: number, checkpoint: string | null = `checkpoint-${number}`) => ({
  number, name: `version ${number}`, change: "Loss fixture", spec: {}, parent: null,
  status: checkpoint ? "trained" : "saved", best_checkpoint: checkpoint, scorecard: {}, verdict: "",
});
const prediction = () => ({
  tokens: ["The answer is ", "🪐é"], pos: 1, target: 5,
  pieces: ["first", "second", "third", "fourth", "fifth", "🪐é", "last"],
  probabilities: [0.30, 0.25, 0.18, 0.12, 0.08, 0.04, 0.03], grad_error: 1e-8,
});
type LossRequest = { version: number; text: string; pos: number; trained: boolean };
const job = (id: string, result: unknown, status = "done", error = "", kind = "inspect.loss") => ({
  id, kind, status, result, error, progress: "", fraction: null,
  created_at: 0, started_at: 0, finished_at: status === "queued" ? null : 1,
});
function gate() {
  let open!: () => void;
  const promise = new Promise<void>((resolve) => { open = resolve; });
  return { promise, open };
}

function server(versions = [version(1), version(2)], extra: Parameters<typeof mockApi>[0] = {}) {
  const state = {
    requests: [] as LossRequest[], bestRequests: 0, result: prediction(),
    failure: "", beforeLoss: undefined as (() => Promise<void>) | undefined,
  };
  const jobs = new Map<string, ReturnType<typeof job>>();
  mockApi({
    "GET /versions": { body: versions },
    "POST /versions/best": () => {
      state.bestRequests += 1;
      const id = `best-${state.bestRequests}`;
      // Deliberately choose an older checkpoint, not the last item in /versions.
      jobs.set(id, job(id, versions.find((item) => item.best_checkpoint) ?? null, "done", "", "versions.best"));
      return { status: 202, body: job(id, null, "queued", "", "versions.best") };
    },
    "POST /inspect/loss": async ({ body }) => {
      state.requests.push(body as LossRequest);
      const id = `loss-${state.requests.length}`;
      await state.beforeLoss?.();
      jobs.set(id, job(id, state.failure ? null : state.result, state.failure ? "failed" : "done", state.failure));
      return { status: 202, body: job(id, null, "queued") };
    },
    "GET /jobs/*": ({ path }) => {
      const record = jobs.get(path.split("/").at(-1)!);
      return record ? { body: record } : { status: 404, body: { detail: "Unknown job" } };
    },
    "DELETE /jobs/*": ({ path }) => ({ body: job(path.split("/").at(-1)!, null, "cancelled") }),
    ...extra,
  });
  return state;
}

const nats = () => screen.findByTestId("loss-nats");
const table = () => screen.getByRole("table", { name: "Next-token probabilities and gradients" });
const row = (header: string | RegExp) => within(table()).getByRole("rowheader", { name: header }).closest("tr")!;
const cells = (tr: HTMLElement) => within(tr).getAllByRole("cell").map((cell) => cell.textContent);
const versionSelect = () => screen.getByRole("combobox", { name: "Version" });
const mainAlert = () => within(screen.getByRole("main")).getByRole("alert");

describe("Loss page", () => {
  test("a version link uses that checkpoint without choosing the best", async () => {
    const state = server();
    renderPage(<LossPage />, { route: "/loss?version=2" });
    await waitFor(() => expect(state.requests.length).toBeGreaterThan(0));
    expect(state.requests[0].version).toBe(2);
    expect(state.bestRequests).toBe(0);
  });

  test("an unavailable version needs an explicit choice before falling back", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<LossPage />, { route: "/loss?version=999" });
    expect(await screen.findByText("Version 999 has no available trained checkpoint.")).toBeVisible();
    expect(mainAlert()).toHaveTextContent("Version 999 has no available trained checkpoint.");
    expect(state.requests).toHaveLength(0);
    expect(state.bestRequests).toBe(0);
    await user.click(screen.getByRole("button", { name: "Choose another version" }));
    await waitFor(() => expect(state.requests.length).toBeGreaterThan(0));
  });

  test("the best checkpoint shows a target outside the top five, one-hot gradients, and Unicode BPC", async () => {
    const state = server();
    renderPage(<LossPage />, { route: "/loss" });
    expect(await nats()).toHaveTextContent("3.22 nats");
    expect(versionSelect()).toHaveValue("1");
    expect(state.bestRequests).toBe(1);
    expect(state.requests).toEqual([{ version: 1, text: prompt, pos: -1, trained: true }]);
    expect(within(table()).getAllByRole("row")).toHaveLength(8);
    expect(cells(row("🪐é"))).toEqual(["0.04", "1", "−0.96"]);
    expect(cells(row("first"))).toEqual(["0.30", "0", "+0.30"]);
    expect(cells(row("all 1 others"))).toEqual(["0.03", "0 each", "+p each"]);
    // 🪐é is two Unicode code points, but three UTF-16 code units.
    expect(screen.getByText(/This piece,.*2 characters:/)).toBeVisible();
    expect(screen.getByTestId("loss-bpc")).toHaveTextContent("2.32 bits per character");
  });

  test("zero target probability produces a finite clamped loss", async () => {
    const state = server();
    state.result.probabilities = [0.40, 0.25, 0.18, 0.10, 0.04, 0, 0.03];
    renderPage(<LossPage />, { route: "/loss" });
    expect(await nats()).toHaveTextContent("690.78 nats");
    expect(screen.getByTestId("loss-bpc")).toHaveTextContent("498.29 bits per character");
    expect(cells(row("🪐é"))).toEqual(["0.0e+0", "1", "−1.00"]);
    expect(within(screen.getByRole("main")).queryByRole("alert")).not.toBeInTheDocument();
  });

  test("random and trained requests preserve the chosen version and submitted prompt", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<LossPage />, { route: "/loss" });
    expect(await nats()).toHaveTextContent("3.22 nats");
    const box = screen.getByRole("textbox", { name: "Text including the next piece to predict" });
    await user.clear(box);
    await user.type(box, "A second prediction 🪐é");
    await user.click(screen.getByRole("button", { name: "Inspect prediction" }));
    await waitFor(() => expect(state.requests).toHaveLength(2));
    await waitFor(() => expect(screen.getByRole("region", { name: "Prediction results" })).toHaveAttribute("aria-busy", "false"));

    state.result.probabilities = state.result.pieces.map(() => 1 / 7);
    await user.click(screen.getByRole("button", { name: "Step 0 (random weights)" }));
    await waitFor(() => expect(screen.getByTestId("loss-nats")).toHaveTextContent("1.95 nats"));
    expect(state.requests.at(-1)).toEqual({ version: 1, text: "A second prediction 🪐é", pos: -1, trained: false });
    expect(screen.getByRole("button", { name: "Step 0 (random weights)" })).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByRole("button", { name: "After training" }));
    await waitFor(() => expect(screen.getByTestId("loss-nats")).toHaveTextContent("3.22 nats"));
    // The trained query may be cached; an explicit inspection must still send trained: true.
    state.result = prediction();
    await user.click(screen.getByRole("button", { name: "Inspect prediction" }));
    await waitFor(() => expect(state.requests).toHaveLength(4));
    expect(state.requests.at(-1)).toEqual({ version: 1, text: "A second prediction 🪐é", pos: -1, trained: true });
  });

  test("switching versions cancels a pending job and ignores its late result", async () => {
    const oldRead = gate(), releaseOld = gate(), cancelled = gate();
    let oldDelivered = false;
    const oldResult = prediction();
    oldResult.pieces[5] = "old-target";
    const state = server(undefined, {
      "GET /jobs/loss-1": async () => {
        oldRead.open();
        await releaseOld.promise;
        oldDelivered = true;
        return { body: job("loss-1", oldResult) };
      },
      "DELETE /jobs/loss-1": () => {
        cancelled.open();
        return { body: job("loss-1", null, "cancelled") };
      },
    });
    const user = userEvent.setup();
    renderPage(<LossPage />, { route: "/loss" });
    await oldRead.promise;
    expect(await screen.findByLabelText("Loading prediction")).toBeVisible();
    state.result = { ...prediction(), probabilities: [0.28, 0.20, 0.15, 0.10, 0.04, 0.20, 0.03] };
    state.result.pieces[5] = "new-target";
    await user.selectOptions(versionSelect(), "2");
    expect(await nats()).toHaveTextContent("1.61 nats");
    await cancelled.promise;
    releaseOld.open();
    await waitFor(() => expect(oldDelivered).toBe(true));
    // Let the obsolete response reach the page before checking that it changed nothing.
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 50)); });
    expect(versionSelect()).toHaveValue("2");
    expect(screen.getByTestId("loss-nats")).toHaveTextContent("1.61 nats");
    expect(within(table()).getByRole("rowheader", { name: "new-target" })).toBeVisible();
    expect(within(table()).queryByRole("rowheader", { name: "old-target" })).not.toBeInTheDocument();
    expect(state.requests.map((request) => request.version)).toEqual([1, 2]);
  });

  test("a failed inspection can be retried with the same input", async () => {
    const state = server();
    state.failure = "The checkpoint could not be read.";
    const user = userEvent.setup();
    renderPage(<LossPage />, { route: "/loss" });
    await waitFor(() => expect(mainAlert()).toHaveTextContent(state.failure));
    expect(screen.queryByTestId("loss-nats")).not.toBeInTheDocument();
    state.failure = "";
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await nats()).toHaveTextContent("3.22 nats");
    expect(within(screen.getByRole("main")).queryByRole("alert")).not.toBeInTheDocument();
    expect(state.requests).toHaveLength(2);
    expect(state.requests[1]).toEqual(state.requests[0]);
  });

  test("without a checkpoint there is a training link and no inspection job", async () => {
    const state = server([version(1, null)]);
    renderPage(<LossPage />, { route: "/loss" });
    expect(await screen.findByText("No trained checkpoint yet. Train a version first, then inspect its predictions here.")).toBeVisible();
    expect(screen.getByRole("link", { name: "Open Pretraining" })).toHaveAttribute("href", expect.stringMatching(/\/pretraining\/?$/));
    expect(screen.queryByRole("button", { name: "Inspect prediction" })).not.toBeInTheDocument();
    expect(state.requests).toEqual([]);
    expect(state.bestRequests).toBe(0);
  });

  describe.each([
    { kind: "checkpoint", payload: { version: 1 } },
    { kind: "versions", payload: version(1) },
    { kind: "job", payload: { job: "queue-pretraining-1", status: "done" } },
  ])("a $kind event", (event) => {
    test("refreshes with the rows kept mounted, and the API's own completion does not loop", async () => {
      const state = server();
      renderPage(<LossPage />, { route: "/loss" });
      expect(await nats()).toHaveTextContent("3.22 nats");
      const original = row("🪐é");
      state.result = { ...prediction(), probabilities: [0.28, 0.23, 0.16, 0.12, 0.08, 0.10, 0.03] };
      const release = gate(), started = gate();
      state.beforeLoss = async () => { started.open(); await release.promise; };

      // Two events in a row are one refresh: the page waits for them to settle.
      emit(event.kind, event.payload);
      emit(event.kind, event.payload);
      await started.promise;
      await waitFor(() => expect(screen.getByRole("region", { name: "Prediction results" })).toHaveAttribute("aria-busy", "true"));
      expect(screen.getByTestId("loss-nats")).toHaveTextContent("3.22 nats");
      expect(original.isConnected).toBe(true);

      release.open();
      await waitFor(() => expect(screen.getByTestId("loss-nats")).toHaveTextContent("2.30 nats"));
      expect(row("🪐é")).toBe(original);

      // The inspection job's own completion event carries no `job` key, so it must not start another refresh.
      emit("job", job("loss-2", state.result));
      await act(async () => { await new Promise((resolve) => setTimeout(resolve, 700)); });
      expect(state.requests).toHaveLength(2);
      expect(state.bestRequests).toBe(2);
    });
  });
});

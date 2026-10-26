import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { KeptCard } from "./kept-card";
import PostTrainingPage from "./page";

// A round that failed its gate: its kept checkpoint can be tried and adopted, and Rounds 2-4 warn about a Round 1 that is not live.

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.setPointerCapture ??= () => {};
  Element.prototype.releasePointerCapture ??= () => {};
});

/** A POST /chat/reply answer: the frames as `data:` lines in a streamed body. */
function sse(...frames: { kind: string; payload: object }[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const f of frames) controller.enqueue(encoder.encode(`data: ${JSON.stringify(f)}\n\n`));
      controller.close();
    },
  });
  return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
}

describe("Kept card", () => {
  test("chats with the kept round's checkpoint, naming the round in the request", async () => {
    const calls = mockApi({ "POST /chat/reply": () => sse({ kind: "text", payload: { text: "Hel" } }, { kind: "done", payload: { text: "Hello from round 1" } }) });
    const user = userEvent.setup();
    renderPage(<KeptCard round={1} version={2} onAdopted={() => {}} />);
    await user.type(screen.getByLabelText("Message for Round 1's checkpoint"), "hi");
    await user.click(screen.getByRole("button", { name: "Try in chat" }));
    expect(await screen.findByText("Hello from round 1")).toBeVisible();
    const sent = calls.find((c) => c.path === "/chat/reply");
    expect(sent?.body).toMatchObject({ version: 2, round: 1, messages: [{ role: "user", content: "hi" }] });
  });

  test("shows the sentence when the kept checkpoint cannot answer", async () => {
    mockApi({ "POST /chat/reply": () => sse({ kind: "error", payload: { message: "Round 1 kept no checkpoint file on version 2 to use." } }) });
    const user = userEvent.setup();
    renderPage(<KeptCard round={1} version={2} onAdopted={() => {}} />);
    await user.type(screen.getByLabelText("Message for Round 1's checkpoint"), "hi");
    await user.click(screen.getByRole("button", { name: "Try in chat" }));
    expect(await screen.findByText("Round 1 kept no checkpoint file on version 2 to use.")).toBeVisible();
  });

  test("adopts the checkpoint for the version", async () => {
    let adopted = 0;
    const calls = mockApi({
      "POST /post-training/rounds/1/adopt": () => {
        adopted += 1;
        return { body: { version: 2, round: 1, checkpoint: "r1.npz" } };
      },
    });
    const user = userEvent.setup();
    let done = 0;
    renderPage(<KeptCard round={1} version={2} onAdopted={() => (done += 1)} />);
    await user.click(screen.getByRole("button", { name: "Adopt this checkpoint" }));
    await waitFor(() => expect(done).toBe(1));
    expect(adopted).toBe(1);
    expect(calls.find((c) => c.path === "/post-training/rounds/1/adopt")?.body).toEqual({ version: 2 });
  });
});

const version = { number: 1, name: "version 1", change: "First design", parent: null, spec: {}, status: "trained", best_checkpoint: "ck1", scorecard: {}, verdict: "", prediction: {} };
const job = (result: unknown) => ({ id: "s1", kind: "post_training.status", status: "done", progress: "", fraction: null, result, error: "", created_at: 0, started_at: 0, finished_at: 1 });
const queue = { job: null, step: 0, total: 0, phase: "", lr: null, eta_s: null, paused: false, blocked: null, gate: { reason: null, resumes_at: null, supported: true }, stopping: false, up_next: [], plan: {} };

function page(status: object) {
  mockApi({
    "GET /post-training/import-models": { body: [] },
    "GET /versions": { body: [version] },
    "GET /post-training/options": { body: { leash: [], lora_ranks: [8], lora_blocker: null, lora_default: false, teachers: [], default_teacher: null } },
    "GET /queue": { body: queue },
    "GET /queue/outlook": { body: { outlook: "Starts now" } },
    "GET /post-training/loops": { body: [] },
    "GET /post-training/judges": { body: { url: "", models: [], selected: "", error: null } },
    "POST /post-training/status": { status: 202, body: { ...job(null), status: "queued" } },
    "GET /jobs/*": { body: job(status) },
  });
}

const counts = { conversations: 12, corrections: 0, problems: 3, pairs: 0 };

describe("Post-training page and a failed Round 1", () => {
  test("offers the kept checkpoint on the round and warns on Round 2 until it is adopted", async () => {
    page({ rounds: { "1": { status: "done", note: "kept, not live" } }, counts, overlap: {}, gates: { "1": { ok: false, verdict: "v", checks: [], checkpoint: "r1.npz" } } });
    const user = userEvent.setup();
    renderPage(<PostTrainingPage />);
    expect(await screen.findByTestId("base-warning")).toHaveTextContent("Round 1 failed its checks on version 1"); // the open round is 2
    expect(screen.queryByTestId("kept-card")).toBeNull();
    await user.click(screen.getByRole("tab", { name: /Practice answers/ }));
    expect(await screen.findByTestId("kept-card")).toBeVisible();
    expect(screen.queryByTestId("base-warning")).toBeNull();
  });

  test("warns that Round 1 has not run when Round 2 is opened", async () => {
    page({ rounds: {}, counts, overlap: {}, gates: {} });
    const user = userEvent.setup();
    renderPage(<PostTrainingPage />);
    await user.click(await screen.findByRole("tab", { name: /Practice with checks/ }));
    expect(await screen.findByTestId("base-warning")).toHaveTextContent("Round 1 has not run on version 1");
  });

  test("no warning and no kept card once the checkpoint is adopted", async () => {
    page({ rounds: { "1": { status: "done", note: "adopted" } }, counts, overlap: {}, gates: { "1": { ok: false, adopted: true, checks: [], checkpoint: "r1.npz" } } });
    const user = userEvent.setup();
    renderPage(<PostTrainingPage />);
    await user.click(await screen.findByRole("tab", { name: /Practice with checks/ }));
    await screen.findByRole("tab", { name: /Practice with checks/, selected: true });
    expect(screen.queryByTestId("base-warning")).toBeNull();
    await user.click(screen.getByRole("tab", { name: /Practice answers/ }));
    expect(screen.queryByTestId("kept-card")).toBeNull();
  });
});

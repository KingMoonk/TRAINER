import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import Page from "./page";

// The Chat page against a mocked API (the component tier of tests/chat.spec.ts, which keeps the two flows that need a real
// streaming response in a browser: a reply arriving in pieces, and Stop aborting the request).

const frame = (kind: string, payload: object) => ({ kind, payload });
const version = (number: number, extra: object = {}) => ({
  number,
  name: `version ${number}`,
  change: `change ${number}`,
  status: "trained",
  verdict: "",
  best_checkpoint: `ckpt-${number}`,
  parent: number > 1 ? number - 1 : null,
  scorecard: {},
  ...extra,
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

function mockVersions(versions: object[], extra: Parameters<typeof mockApi>[0] = {}) {
  const trained = versions.filter((v) => "best_checkpoint" in v && v.best_checkpoint);
  const job = (result: unknown) => ({ id: "metadata", kind: "versions", status: "done", result, progress: "", error: "", created_at: 0 });
  return mockApi({
    "GET /versions": { body: versions },
    "POST /versions/best": { body: job(trained.at(-1) ?? null) },
    "POST /versions/*/checkpoint-step": { body: job(100) },
    ...extra,
  });
}

describe("Chat page", () => {
  test("with no trained version the page says so and the message box is off", async () => {
    mockVersions([version(1, { best_checkpoint: null, status: "draft" })]);
    renderPage(<Page />, { route: "/chat" });
    expect(await screen.findByText("No trained version yet.")).toBeVisible();
    expect(screen.getByRole("link", { name: /Train one from Build your model\./ })).toBeVisible();
    expect(screen.getByLabelText("Your message")).toBeDisabled();
  });

  test("an error event shows a toast and the reply says why", async () => {
    mockVersions([version(1)], { "POST /chat/reply": () => sse(frame("error", { message: "The model ran out of memory." })) });
    const user = userEvent.setup();
    renderPage(<Page />, { route: "/chat" });
    await waitFor(() => expect(screen.getByLabelText("Your message")).toBeEnabled());
    await user.type(screen.getByLabelText("Your message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));
    // once in the toast and once under the reply
    await waitFor(() => expect(screen.getAllByText("The model ran out of memory.")).toHaveLength(2));
  });

  test("scores sit in a closed disclosure, there is no retry while the best version loads fine, and no designer link", async () => {
    mockVersions([version(1)]);
    renderPage(<Page />, { route: "/chat" });
    await screen.findByText("Version 1 is best");
    expect(screen.queryByRole("button", { name: "Check best again" })).toBeNull();
    expect(screen.queryByRole("link", { name: /designer/i })).toBeNull();
    const summary = screen.getByText("Scores for this version");
    const details = summary.closest("details")!;
    expect(details.open).toBe(false);
    expect(details).toHaveTextContent("Answers prompts");
    expect(details).toHaveTextContent("Learning from examples in the question");
  });

  test("a failed best check offers a retry", async () => {
    mockVersions([version(1)], {
      "POST /versions/best": { body: { id: "b", kind: "versions", status: "failed", result: null, progress: "", error: "Scoring failed", created_at: 0 } },
    });
    renderPage(<Page />, { route: "/chat" });
    expect(await screen.findByRole("button", { name: "Check best again" })).toBeVisible();
  });

  test("a correction is saved with the conversation up to the last message", async () => {
    let saved: unknown;
    mockVersions([version(1)], {
      "POST /chat/reply": () => sse(frame("done", { text: "Paris is in Italy." })),
      "POST /chat/corrections": ({ body }) => {
        saved = body;
        return { status: 204 };
      },
    });
    const user = userEvent.setup();
    renderPage(<Page />, { route: "/chat" });
    await waitFor(() => expect(screen.getByLabelText("Your message")).toBeEnabled()); // the page has loaded its versions
    await user.click(screen.getByRole("button", { name: "Teach a correction" }));
    expect(await screen.findByText(/Send a message first/)).toBeVisible();
    await user.type(screen.getByLabelText("Your message"), "Where is Paris?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Paris is in Italy.")).toBeVisible();
    await user.type(screen.getByLabelText("Correction"), "Paris is in France.");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("Saved for Post-training round 3.")).toBeVisible();
    expect(saved).toEqual({
      version: 1,
      messages: [{ role: "user", content: "Where is Paris?" }],
      correction: "Paris is in France.",
      whole_answer: false,
    });
  });

  test("a run in progress offers its live checkpoint: chat answers from it and says which step, Score now measures it", async () => {
    const job = (result: unknown) => ({ id: "j1", kind: "versions.score_live", status: "done", result, progress: "", error: "", created_at: 0 });
    let asked: unknown;
    const saved = Date.now() / 1000 - 180;
    mockVersions([version(1, { best_checkpoint: null, status: "training" })], {
      "GET /versions/live": { body: [{ version: 1, step: 12000, total: 50000, saved_at: saved, file: "best.npz" }] },
      "POST /versions/1/score-live": { body: job({ version: 1, step: 12000, saved_at: saved, bpc: 2.345, documents: 5 }) },
      "POST /chat/reply": ({ body }) => {
        asked = body;
        return sse(frame("done", { text: "from the live one" }));
      },
    });
    const user = userEvent.setup();
    renderPage(<Page />, { route: "/chat" });
    expect(await screen.findByText(/step 12,000 of 50,000 · saved 3 min ago/)).toBeVisible();
    expect(screen.queryByText("No trained version yet.")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Chat with it" }));
    expect(await screen.findByText(/Not the final model/)).toBeVisible();
    await user.type(screen.getByLabelText("Your message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("from the live one")).toBeVisible();
    expect(asked).toEqual({ version: 1, messages: [{ role: "user", content: "hi" }] });
    await user.click(screen.getByRole("button", { name: "Score now" }));
    expect(await screen.findByText(/2\.35 bits\/char at step 12,000 on 5 held-back documents/)).toBeVisible();
  });

  test("compare asks both versions, shows a verdict, and saves the better reply as a rated pair", async () => {
    let pair: unknown;
    mockVersions([version(1), version(2)], {
      "GET /benchmark/verdict": { body: { comparable: true, percent: 80, reason: "" } },
      "POST /chat/reply": ({ body }) => sse(frame("done", { text: `answer from ${(body as { version: number }).version}` })),
      "POST /chat/pairs": ({ body }) => {
        pair = body;
        return { body: { record: null } };
      },
    });
    const user = userEvent.setup();
    renderPage(<Page />, { route: "/chat" });
    await waitFor(() => expect(screen.getByRole("button", { name: "Compare two versions" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Compare two versions" }));
    expect(await screen.findByText("To reach the same error, Version 2 needs 20% of the training Version 1 needed.")).toBeVisible();
    await user.type(screen.getByLabelText("Your message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("answer from 2")).toBeVisible();
    expect(await screen.findByText("answer from 1")).toBeVisible();
    await waitFor(() => expect(screen.getAllByRole("button", { name: "This one is better" })[1]).toBeEnabled());
    await user.click(screen.getAllByRole("button", { name: "This one is better" })[1]);
    expect(await screen.findByText("Saved: Version 1 was better. Used in Post-training round 4.")).toBeVisible();
    expect(pair).toEqual({
      messages: [{ role: "user", content: "hi" }],
      chosen: { version: 1, reply: "answer from 1" },
      rejected: { version: 2, reply: "answer from 2" },
    });
  });
});

import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import type { components } from "@/lib/api-schema";
import type { JobOut } from "@/lib/job-result";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import Page from "./page";

// The Chat page's per-version checks (best version, checkpoint step, example check) against a mocked API; the component tier of tests/chat-checks.spec.ts.

const version = (number: number): components["schemas"]["Version"] => ({
  number, name: `Version ${number}`, change: `Design ${number}`, status: "trained", verdict: "", best_checkpoint: `checkpoint-${number}`,
});
const job = (id: string, result: unknown = null, status = "done"): JobOut => ({
  id, kind: "test", status, result, error: "", progress: "", created_at: 0,
});
const table = (correct: number, heldBack = true) => ({
  items: 20, tasks: 2, held_back: heldBack, chance: 0.25,
  rows: [{ shots: 0, used: 0, correct, accuracy: correct / 20 }, { shots: 4, used: 2.5, correct, accuracy: correct / 20 }],
});
const card = (number: number) => screen.getByLabelText(`Learning from examples in the question: Version ${number}`);
const checkIt = (number: number) => userEvent.click(within(card(number)).getByRole("button", { name: "Check it" }));
const pick = (number: number) => userEvent.click(screen.getByRole("button", { name: new RegExp(`^Version ${number}`) }));
const SLOW = { timeout: 4_000 }; // a running job is polled once a second

function sse(...frames: { kind: string; payload: unknown }[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(`data: ${JSON.stringify(frame)}\n\n`));
      controller.close();
    },
  });
  return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
}

function server(extra: Parameters<typeof mockApi>[0] = {}) {
  const state = { versions: [version(1), version(2)], best: 1, bestCalls: 0, stepCalls: 0, step: 0, exampleCalls: [] as number[], chat: undefined as unknown };
  const calls = mockApi({
    "GET /versions": () => ({ body: state.versions }),
    "POST /versions/best": () => {
      state.bestCalls++;
      return { body: job(`best-${state.bestCalls}`, state.versions.find((v) => v.number === state.best) ?? null) };
    },
    "POST /versions/*/checkpoint-step": () => {
      state.stepCalls++;
      return { body: job(`step-${state.stepCalls}`, state.step) };
    },
    "POST /benchmark/few-shot": ({ body }) => {
      const number = (body as { version: number }).version;
      state.exampleCalls.push(number);
      return { body: job(`examples-${state.exampleCalls.length}`, table(number === 1 ? 15 : 3)) };
    },
    "GET /benchmark/verdict": { body: { comparable: false, reason: "Different schedules" } },
    "POST /chat/reply": ({ body }) => {
      state.chat = body;
      return sse({ kind: "done", payload: { text: "An answer" } });
    },
    ...extra,
  });
  return { state, calls };
}

describe("Chat page checks", () => {
  test("follows the actual best, displays zero checkpoint steps, and sends its explicit version", async () => {
    const { state } = server();
    const user = userEvent.setup();
    renderPage(<Page />, { route: "/chat" });
    expect(await screen.findByText("Version 1 is best")).toBeVisible();
    await waitFor(() => expect(screen.getAllByLabelText("Checkpoint for Version 1")).toHaveLength(2));
    await waitFor(() => expect(screen.getAllByLabelText("Checkpoint for Version 1")[0]).toHaveTextContent("checkpoint at step 0"));
    await waitFor(() => expect(screen.getByLabelText("Checkpoint for Version 2")).toHaveTextContent("checkpoint at step 0"));
    await user.type(screen.getByLabelText("Your message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("An answer")).toBeVisible();
    expect(state.chat).toEqual({ version: 1, messages: [{ role: "user", content: "hi" }] });
    expect(state.bestCalls).toBe(1);
    await user.click(screen.getByRole("button", { name: "Compare two versions" }));
    expect(within(screen.getByLabelText("Replies")).getAllByText("Version 1")[0]).toBeVisible();
    await waitFor(() => expect(screen.getAllByLabelText("Checkpoint for Version 2")).toHaveLength(2));
  });

  test("few-shot results and errors stay with their version and return from cache", async () => {
    const { state } = server();
    renderPage(<Page />, { route: "/chat" });
    await screen.findByText("Version 1 is best");
    await checkIt(1);
    expect((await within(card(1)).findAllByRole("cell", { name: "15 of 20" }))[0]).toBeVisible();
    await pick(2);
    await waitFor(() => expect(card(2)).toHaveTextContent("Version 2"));
    expect(within(card(2)).queryByRole("table")).toBeNull();
    await checkIt(2);
    expect((await within(card(2)).findAllByRole("cell", { name: "3 of 20" }))[0]).toBeVisible();
    await pick(1);
    expect((await within(card(1)).findAllByRole("cell", { name: "15 of 20" }))[0]).toBeVisible();
    expect(state.exampleCalls).toEqual([1, 2]);
  });

  test("polls a running check, preserves rows on Check again, and explains training data", async () => {
    let calls = 0;
    let finished = false;
    server({
      "POST /benchmark/few-shot": () => ({ body: ++calls === 1 ? job("first", table(15)) : job("recheck", null, "running") }),
      "GET /jobs/recheck": () => ({ body: finished ? job("recheck", table(16, false)) : job("recheck", null, "running") }),
    });
    renderPage(<Page />, { route: "/chat" });
    await screen.findByText("Version 1 is best");
    await checkIt(1);
    await waitFor(() => expect(within(card(1)).getAllByRole("row")[1]).toHaveTextContent("15 of 20"));
    const row = within(card(1)).getAllByRole("row")[1];
    await userEvent.click(within(card(1)).getByRole("button", { name: "Check again" }));
    expect(await within(card(1)).findByRole("status", { name: "Asking the multiple-choice questions" })).toBeVisible();
    expect(row).toHaveTextContent("15 of 20");
    finished = true;
    emit("job", job("recheck", table(16, false))); // the server reports the check finished
    await waitFor(() => expect(row).toHaveTextContent("16 of 20"));
    expect(row.isConnected).toBe(true);
    expect(card(1)).toHaveTextContent("come from the ones used for training");
    expect(card(1)).toHaveTextContent("4 (2.5 fit on average)");
  });

  test("switching versions cancels the old check and cannot display its late completion", async () => {
    let cancelled = false;
    let started = false;
    server({
      "POST /benchmark/few-shot": () => {
        started = true;
        return { body: job("old-check", null, "running") };
      },
      "DELETE /jobs/old-check": () => {
        cancelled = true;
        return { body: job("old-check", null, "cancelled") };
      },
      "GET /jobs/old-check": { body: job("old-check", null, "running") },
    });
    renderPage(<Page />, { route: "/chat" });
    await screen.findByText("Version 1 is best");
    await checkIt(1);
    await waitFor(() => expect(started).toBe(true));
    await pick(2);
    await waitFor(() => expect(cancelled).toBe(true));
    await waitFor(() => expect(card(2)).toHaveTextContent("Not checked yet for this version"));
    emit("job", { id: "old-check", kind: "benchmark.few_shot", status: "done", result: { items: 999 } });
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(within(card(2)).queryByRole("table")).toBeNull();
  });

  test("Cancel check calls DELETE and Check again starts a fresh job", async () => {
    let submitted = 0;
    let deletes = 0;
    server({
      "POST /benchmark/few-shot": () => ({ body: ++submitted === 1 ? job("cancel-me", null, "running") : job("fresh", table(12)) }),
      "DELETE /jobs/cancel-me": () => {
        deletes++;
        return { body: job("cancel-me", null, "running") };
      },
      "GET /jobs/cancel-me": { body: job("cancel-me", null, "running") },
    });
    renderPage(<Page />, { route: "/chat" });
    await screen.findByText("Version 1 is best");
    await checkIt(1);
    await userEvent.click(await within(card(1)).findByRole("button", { name: "Cancel check" }));
    await waitFor(() => expect(deletes).toBe(1));
    expect(await within(card(1)).findByText("Example check cancelled.")).toBeVisible();
    await userEvent.click(within(card(1)).getByRole("button", { name: "Check again" }));
    expect((await within(card(1)).findAllByRole("cell", { name: "12 of 20" }))[0]).toBeVisible();
  });

  test("job failures can retry and do not appear under another version", async () => {
    let attempts = 0;
    server({
      "POST /benchmark/few-shot": () => ({ body: ++attempts === 1 ? { ...job("bad", null, "failed"), error: "Could not read the questions" } : job("ok", table(5)) }),
    });
    renderPage(<Page />, { route: "/chat" });
    await screen.findByText("Version 1 is best");
    await checkIt(1);
    expect(await within(card(1)).findByRole("alert")).toHaveTextContent("Could not read the questions");
    await pick(2);
    await waitFor(() => expect(card(2)).toHaveTextContent("Version 2"));
    expect(within(card(2)).queryByRole("alert")).toBeNull();
    await pick(1);
    await userEvent.click(within(card(1)).getByRole("button", { name: "Check again" }));
    expect(await within(card(1)).findByRole("table")).toBeVisible();
  });

  test("best and checkpoint failures recover through their retry controls", async () => {
    let bestFails = true;
    let stepFails = true;
    server({
      "POST /versions/best": () => ({ body: bestFails ? { ...job("best-failed", null, "failed"), error: "Scoring failed" } : job("best-ok", version(1)) }),
      "POST /versions/1/checkpoint-step": () => ({ body: stepFails ? { ...job("step-failed", null, "failed"), error: "Checkpoint unreadable" } : job("step-ok", 73) }),
    });
    renderPage(<Page />, { route: "/chat" });
    expect(await screen.findByText(/Could not check the best version.*Scoring failed/)).toBeVisible();
    expect(screen.getByLabelText("Your message")).toBeDisabled();
    bestFails = false;
    await userEvent.click(screen.getByRole("button", { name: /Check best again/ }));
    expect(await screen.findByText("Version 1 is best")).toBeVisible();
    stepFails = false;
    await userEvent.click((await screen.findAllByRole("button", { name: /Retry checkpoint step/ }))[0]);
    await waitFor(() => expect(screen.getAllByLabelText("Checkpoint for Version 1")[0]).toHaveTextContent("checkpoint at step 73"));
  });

  test("checkpoint replacement at the same path refreshes metadata and clears old results without a job-event loop", async () => {
    const { state } = server();
    renderPage(<Page />, { route: "/chat" });
    await screen.findByText("Version 1 is best");
    await checkIt(1);
    expect(await within(card(1)).findByRole("table")).toBeVisible();
    state.step = 900;
    state.best = 2;
    emit("checkpoint", { job: "training", phase: "done" });
    emit("versions", null);
    emit("job", { job: "training", status: "done" });
    expect(await screen.findByText("Version 2 is best")).toBeVisible();
    await waitFor(() => expect(screen.getAllByLabelText("Checkpoint for Version 2")[0]).toHaveTextContent("checkpoint at step 900"));
    await pick(1);
    await waitFor(() => expect(card(1)).toHaveTextContent("Not checked yet for this version"));
    expect(within(card(1)).queryByRole("table")).toBeNull();
    const calls = state.bestCalls;
    emit("job", { id: "api-best", kind: "versions.best", status: "done" });
    await new Promise((resolve) => setTimeout(resolve, 500)); // exceeds the refresh debounce; an API read must not trigger another read
    expect(state.bestCalls).toBe(calls);
  });

  test("version-path replacement does not reuse a prior example check", async () => {
    const { state } = server();
    renderPage(<Page />, { route: "/chat" });
    await screen.findByText("Version 1 is best");
    await checkIt(1);
    expect(await within(card(1)).findByRole("table")).toBeVisible();
    state.versions = [{ ...version(1), best_checkpoint: "replacement" }, version(2)];
    emit("versions", null);
    await waitFor(() => expect(within(card(1)).queryByRole("table")).toBeNull());
    expect(within(card(1)).getByRole("button", { name: "Check it" })).toBeEnabled();
  });
});

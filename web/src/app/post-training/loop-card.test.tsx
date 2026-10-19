import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { LoopCard } from "./loop-card";

// The automated loop's card against a mocked API: start, the rounds and their evidence, stop, continue and adopt.

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.setPointerCapture ??= () => {};
  Element.prototype.releasePointerCapture ??= () => {};
});

const checks = [
  { name: "Held-back text: error did not rise", detail: "1.900 bits per character, was 2.000", ok: true },
  { name: "Something improved", detail: "more checked problems solved", ok: true },
];
const rounds = [
  { round: 1, status: "kept", reason: "Kept: nothing got worse and something improved.", pairs: { total: 6, verified: 4, comparison: 2, skipped: {} }, checks,
    before: { bpc: 2, pass_k: 5, pass_n: 10 }, after: { bpc: 1.9, pass_k: 7, pass_n: 10 }, judge: { win_rate: 0.8 }, length_ratio: 1.1, seconds: 42 },
  { round: 2, status: "rolled_back", reason: "Rolled back: held-back text: error did not rise.", pairs: { total: 5 },
    checks: [{ name: "Held-back text: error did not rise", detail: "2.500 bits per character, was 1.900: rose", ok: false }],
    before: { bpc: 1.9, pass_k: 7, pass_n: 10 }, after: { bpc: 2.5, pass_k: 7, pass_n: 10 }, judge: null },
];
const loop = (patch: Record<string, unknown> = {}) => ({
  id: "loop1", version: 1, status: "stopped", stop_reason: "Stopped.", round: 3, kept: 1, failed_streak: 1, audit_flagged: 0, elapsed_s: 3600,
  config: {}, anchor: "a.npz", current: "r1.npz", rounds, active: false, ...patch,
});

function server(loops: unknown[], extra: Parameters<typeof mockApi>[0] = {}) {
  return mockApi({
    "GET /post-training/loops": { body: loops },
    "GET /post-training/judges": { body: { url: "http://localhost:11434", models: ["qwen3:8b"], selected: "", error: null } },
    ...extra,
  });
}

describe("Loop card", () => {
  test("starts a loop with the form's stop rules and judge", async () => {
    const calls = server([], { "POST /post-training/loops": { body: { job: "j1", loop_id: "loop1", label: "RLHF loop · version 1" } } });
    const user = userEvent.setup();
    renderPage(<LoopCard version={1} hasCheckpoint />);
    expect(await screen.findByText("No loop has run for this version yet.")).toBeVisible();
    const field = screen.getByLabelText("Rounds at most");
    await user.clear(field);
    await user.type(field, "8");
    await user.click(screen.getByRole("combobox", { name: "Judge model" }));
    await user.click(await screen.findByRole("option", { name: "qwen3:8b" }));
    await user.click(screen.getByRole("button", { name: "Start the loop" }));
    expect(await screen.findByText("Added to the queue: RLHF loop · version 1")).toBeVisible();
    const post = calls.find((c) => c.method === "POST" && c.path === "/post-training/loops");
    expect(post?.body).toEqual({ version: 1, resume: null, judge_model: "qwen3:8b", config: { max_rounds: 8, patience: 4 } });
  });

  test("a wrong field is told before anything is sent", async () => {
    const calls = server([]);
    const user = userEvent.setup();
    renderPage(<LoopCard version={1} hasCheckpoint />);
    await screen.findByText("No loop has run for this version yet.");
    const field = screen.getByLabelText("Rounds at most");
    await user.clear(field);
    await user.type(field, "0");
    await user.click(screen.getByRole("button", { name: "Start the loop" }));
    expect(await screen.findByText("Rounds must be a whole number from 1 to 100,000.")).toBeVisible();
    expect(calls.some((c) => c.method === "POST")).toBe(false);
  });

  test("shows each round's evidence, kept or rolled back, and its checks when opened", async () => {
    server([loop()]);
    const user = userEvent.setup();
    renderPage(<LoopCard version={1} hasCheckpoint />);
    expect(await screen.findByTestId("loop-status")).toHaveTextContent("Stopped before round 3. 1 of 2 rounds kept.");
    const first = screen.getByTestId("loop-round-1");
    expect(within(first).getByText("kept")).toBeVisible();
    expect(first).toHaveTextContent("solved 7 of 10 (was 5) · judge preferred it 80% · 1.10x as long");
    const second = screen.getByTestId("loop-round-2");
    expect(within(second).getByText("rolled back")).toBeVisible();
    expect(second).toHaveTextContent("Rolled back: held-back text: error did not rise.");
    await user.click(within(second).getByRole("button", { name: /Round 2/ }));
    expect(within(second).getByText(/2.500 bits per character, was 1.900: rose/)).toBeVisible();
  });

  test("a running loop can be stopped, and a stopped one continued from its last kept round", async () => {
    const calls = server([loop({ active: true, status: "running", phase: "round 3: training" })], {
      "POST /post-training/loops/loop1/stop": { body: { stopped: true } },
    });
    const user = userEvent.setup();
    const { unmount } = renderPage(<LoopCard version={1} hasCheckpoint />);
    expect(await screen.findByTestId("loop-status")).toHaveTextContent("round 3: training. 1 of 2 rounds kept.");
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Stop the loop" }));
    await waitFor(() => expect(calls.some((c) => c.path === "/post-training/loops/loop1/stop")).toBe(true));
    unmount();

    const again = server([loop()], { "POST /post-training/loops": { body: { job: "j2", loop_id: "loop1", label: "RLHF loop · version 1" } } });
    renderPage(<LoopCard version={1} hasCheckpoint />);
    await user.click(await screen.findByRole("button", { name: "Continue" }));
    await waitFor(() => expect(again.find((c) => c.method === "POST" && c.path === "/post-training/loops")?.body).toMatchObject({ resume: "loop1" }));
  });

  test("adopting the loop's best reports the gate's verdict", async () => {
    const job = (status: string, result: unknown) => ({ id: "adopt1", kind: "post_training.adopt_loop", status, progress: "", fraction: null, result, error: "", created_at: 0, started_at: 0, finished_at: 1 });
    server([loop()], {
      "POST /post-training/loops/loop1/adopt": { status: 202, body: job("queued", null) },
      "GET /jobs/*": { body: job("done", { ok: true, verdict: "The loop's best, after 1 kept round(s), did not get worse on any held-back data." }) },
    });
    const user = userEvent.setup();
    renderPage(<LoopCard version={1} hasCheckpoint />);
    await user.click(await screen.findByRole("button", { name: "Use its best as this version" }));
    expect(await screen.findByText(/did not get worse on any held-back data/)).toBeVisible();
  });
});

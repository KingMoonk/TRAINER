import { expect, test, type Page } from "@playwright/test";
import { changesCheckpoints, fewShotResult, fewShotRows } from "../src/app/chat/logic";
import type { components } from "../src/lib/api-schema";
import type { JobOut } from "../src/lib/job-result";

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
const card = (page: Page, number: number) => page.getByLabel(`Learning from examples in the question: Version ${number}`, { exact: true });
const pick = (page: Page, number: number) => page.getByRole("button", { name: new RegExp(`^Version ${number} `) }).click();

async function setup(page: Page) {
  // Deliver precisely the same event envelopes as /events, without reconnects replaying old test events.
  await page.addInitScript(() => {
    class TestSource {
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      listener = (event: Event) => this.onmessage?.(new MessageEvent("message", { data: JSON.stringify((event as CustomEvent).detail) }));
      constructor() {
        window.addEventListener("test-studio-event", this.listener);
        queueMicrotask(() => this.onopen?.(new Event("open")));
      }
      close() { window.removeEventListener("test-studio-event", this.listener); }
    }
    Object.defineProperty(window, "EventSource", { value: TestSource });
  });
  const state = { versions: [version(1), version(2)], best: 1, bestCalls: 0, stepCalls: 0, step: 0, exampleCalls: [] as number[] };
  await page.route("**/versions", (route) => route.fulfill({ json: state.versions }));
  await page.route("**/versions/best", (route) => {
    state.bestCalls++;
    return route.fulfill({ json: job(`best-${state.bestCalls}`, state.versions.find((v) => v.number === state.best) ?? null) });
  });
  await page.route("**/versions/*/checkpoint-step", (route) => {
    state.stepCalls++;
    return route.fulfill({ json: job(`step-${state.stepCalls}`, state.step) });
  });
  await page.route("**/benchmark/few-shot", (route) => {
    const number = route.request().postDataJSON().version as number;
    state.exampleCalls.push(number);
    return route.fulfill({ json: job(`examples-${state.exampleCalls.length}`, table(number === 1 ? 15 : 3)) });
  });
  await page.route("**/benchmark/verdict**", (route) => route.fulfill({ json: { comparable: false, reason: "Different schedules" } }));
  return state;
}

test("follows the actual best, displays zero checkpoint steps, and sends its explicit version", async ({ page }) => {
  const state = await setup(page);
  let request: unknown;
  await page.route("**/chat/reply", (route) => {
    request = route.request().postDataJSON();
    return route.fulfill({ contentType: "text/event-stream", body: 'data: {"kind":"done","payload":{"text":"An answer"}}\n\n' });
  });
  await page.goto("/chat/");
  await expect(page.getByText("Version 1 is best", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Checkpoint for Version 1")).toHaveCount(2);
  await expect(page.getByLabel("Checkpoint for Version 1").first()).toContainText("checkpoint at step 0");
  await expect(page.getByLabel("Checkpoint for Version 2")).toContainText("checkpoint at step 0");
  await page.getByLabel("Your message").fill("hi");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByText("An answer", { exact: true })).toBeVisible();
  expect(request).toEqual({ version: 1, messages: [{ role: "user", content: "hi" }] });
  expect(state.bestCalls).toBe(1);
  await page.getByRole("button", { name: "Compare two versions" }).click();
  await expect(page.getByLabel("Replies").getByText("Version 1", { exact: true }).first()).toBeVisible();
  await expect(page.getByLabel("Checkpoint for Version 2")).toHaveCount(2);
});

test("few-shot results and errors stay with their version and return from cache", async ({ page }) => {
  const state = await setup(page);
  await page.goto("/chat/");
  await card(page, 1).getByRole("button", { name: "Check it", exact: true }).click();
  await expect(card(page, 1).getByRole("cell", { name: "15 of 20", exact: true }).first()).toBeVisible();
  await pick(page, 2);
  await expect(card(page, 2).getByRole("table")).toHaveCount(0);
  await card(page, 2).getByRole("button", { name: "Check it", exact: true }).click();
  await expect(card(page, 2).getByRole("cell", { name: "3 of 20", exact: true }).first()).toBeVisible();
  await pick(page, 1);
  await expect(card(page, 1).getByRole("cell", { name: "15 of 20", exact: true }).first()).toBeVisible();
  expect(state.exampleCalls).toEqual([1, 2]);
});

test("polls a running check, preserves rows on Check again, and explains training data", async ({ page }) => {
  await setup(page);
  let calls = 0;
  let finished = false;
  await page.route("**/benchmark/few-shot", (route) => route.fulfill({ json: ++calls === 1 ? job("first", table(15)) : job("recheck", null, "running") }));
  await page.route("**/jobs/recheck", (route) => route.fulfill({ json: finished ? job("recheck", table(16, false)) : job("recheck", null, "running") }));
  await page.goto("/chat/");
  await card(page, 1).getByRole("button", { name: "Check it", exact: true }).click();
  const row = card(page, 1).getByRole("row").nth(1);
  await expect(row).toContainText("15 of 20");
  const element = await row.elementHandle();
  await card(page, 1).getByRole("button", { name: "Check again", exact: true }).click();
  await expect(card(page, 1).getByRole("status", { name: "Asking the multiple-choice questions" })).toBeVisible();
  await expect(row).toContainText("15 of 20");
  finished = true;
  await expect(row).toContainText("16 of 20");
  expect(await element!.evaluate((node) => node.isConnected)).toBe(true);
  await expect(card(page, 1)).toContainText("come from the ones used for training");
  await expect(card(page, 1)).toContainText("4 (2.5 fit on average)");
});

test("switching versions cancels the old check and cannot display its late completion", async ({ page }) => {
  await setup(page);
  let cancelled = false;
  let started = false;
  await page.route("**/benchmark/few-shot", (route) => {
    started = true;
    return route.fulfill({ json: job("old-check", null, "running") });
  });
  await page.route("**/jobs/old-check", (route) => {
    if (route.request().method() === "DELETE") { cancelled = true; return route.fulfill({ json: job("old-check", null, "cancelled") }); }
    return route.fulfill({ json: job("old-check", null, "running") });
  });
  await page.goto("/chat/");
  await card(page, 1).getByRole("button", { name: "Check it", exact: true }).click();
  await expect.poll(() => started).toBe(true);
  await pick(page, 2);
  await expect.poll(() => cancelled).toBe(true);
  await expect(card(page, 2)).toContainText("Not checked yet for this version");
  await page.evaluate(() => window.dispatchEvent(new CustomEvent("test-studio-event", { detail: { kind: "job", payload: { id: "old-check", kind: "benchmark.few_shot", status: "done", result: { items: 999 } } } })));
  await expect(card(page, 2).getByRole("table")).toHaveCount(0);
});

test("Cancel check calls DELETE and Check again starts a fresh job", async ({ page }) => {
  await setup(page);
  let submitted = 0;
  let deletes = 0;
  await page.route("**/benchmark/few-shot", (route) => route.fulfill({ json: ++submitted === 1 ? job("cancel-me", null, "running") : job("fresh", table(12)) }));
  await page.route("**/jobs/cancel-me", (route) => {
    if (route.request().method() === "DELETE") deletes++;
    return route.fulfill({ json: job("cancel-me", null, "running") });
  });
  await page.goto("/chat/");
  await card(page, 1).getByRole("button", { name: "Check it", exact: true }).click();
  await card(page, 1).getByRole("button", { name: "Cancel check", exact: true }).click();
  await expect.poll(() => deletes).toBe(1);
  await expect(card(page, 1)).toContainText("Example check cancelled.");
  await card(page, 1).getByRole("button", { name: "Check again", exact: true }).click();
  await expect(card(page, 1).getByRole("cell", { name: "12 of 20", exact: true }).first()).toBeVisible();
});

test("job failures can retry and do not appear under another version", async ({ page }) => {
  await setup(page);
  let attempts = 0;
  await page.route("**/benchmark/few-shot", (route) => route.fulfill({ json: ++attempts === 1 ? { ...job("bad", null, "failed"), error: "Could not read the questions" } : job("ok", table(5)) }));
  await page.goto("/chat/");
  await card(page, 1).getByRole("button", { name: "Check it", exact: true }).click();
  await expect(card(page, 1).getByRole("alert")).toContainText("Could not read the questions");
  await pick(page, 2);
  await expect(card(page, 2).getByRole("alert")).toHaveCount(0);
  await pick(page, 1);
  await card(page, 1).getByRole("button", { name: "Check again", exact: true }).click();
  await expect(card(page, 1).getByRole("table")).toBeVisible();
});

test("best and checkpoint failures recover through their retry controls", async ({ page }) => {
  await setup(page);
  let bestFails = true;
  let stepFails = true;
  await page.route("**/versions/best", (route) => route.fulfill({ json: bestFails ? { ...job("best-failed", null, "failed"), error: "Scoring failed" } : job("best-ok", version(1)) }));
  await page.route("**/versions/1/checkpoint-step", (route) => route.fulfill({ json: stepFails ? { ...job("step-failed", null, "failed"), error: "Checkpoint unreadable" } : job("step-ok", 73) }));
  await page.goto("/chat/");
  await expect(page.getByText(/Could not check the best version.*Scoring failed/)).toBeVisible();
  await expect(page.getByLabel("Your message")).toBeDisabled();
  bestFails = false;
  await page.getByRole("button", { name: "Check best again" }).click();
  await expect(page.getByText("Version 1 is best", { exact: true })).toBeVisible();
  stepFails = false;
  await page.getByRole("button", { name: "Retry checkpoint step" }).first().click();
  await expect(page.getByLabel("Checkpoint for Version 1").first()).toContainText("checkpoint at step 73");
});

test("checkpoint replacement at the same path refreshes metadata and clears old results without a job-event loop", async ({ page }) => {
  const state = await setup(page);
  await page.goto("/chat/");
  await card(page, 1).getByRole("button", { name: "Check it", exact: true }).click();
  await expect(card(page, 1).getByRole("table")).toBeVisible();
  state.step = 900;
  state.best = 2;
  await page.evaluate(() => {
    const emit = (payload: object) => window.dispatchEvent(new CustomEvent("test-studio-event", { detail: payload }));
    emit({ kind: "checkpoint", payload: { job: "training", phase: "done" } });
    emit({ kind: "versions", payload: null });
    emit({ kind: "job", payload: { job: "training", status: "done" } });
  });
  await expect(page.getByText("Version 2 is best", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Checkpoint for Version 2").first()).toContainText("checkpoint at step 900");
  await pick(page, 1);
  await expect(card(page, 1).getByRole("table")).toHaveCount(0);
  const calls = state.bestCalls;
  await page.evaluate(() => window.dispatchEvent(new CustomEvent("test-studio-event", { detail: { kind: "job", payload: { id: "api-best", kind: "versions.best", status: "done" } } })));
  await page.waitForTimeout(500); // exceeds the refresh debounce; an API read must not trigger another read
  expect(state.bestCalls).toBe(calls);
});

test("version-path replacement does not reuse a prior example check", async ({ page }) => {
  const state = await setup(page);
  await page.goto("/chat/");
  await card(page, 1).getByRole("button", { name: "Check it", exact: true }).click();
  await expect(card(page, 1).getByRole("table")).toBeVisible();
  state.versions = [{ ...version(1), best_checkpoint: "replacement" }, version(2)];
  await page.evaluate(() => window.dispatchEvent(new CustomEvent("test-studio-event", { detail: { kind: "versions", payload: null } })));
  await expect(card(page, 1).getByRole("table")).toHaveCount(0);
  await expect(card(page, 1).getByRole("button", { name: "Check it", exact: true })).toBeEnabled();
});

test("few-shot empty and nullable measurements stay truthful", () => {
  expect(fewShotRows(fewShotResult({ items: 0, tasks: 0, chance: null, held_back: true, rows: [] }))).toEqual([]);
  const rows = fewShotRows(fewShotResult({ ...table(0), chance: null, rows: [{ shots: 0, used: 0, correct: 0, accuracy: null }] }));
  expect(rows[0].cells).toEqual(["none", "0 of 20", "not measured", "not measured", "not measured"]);
  expect(() => fewShotResult({ ...table(0), rows: [{ shots: 0, used: 0, correct: 99, accuracy: 4 }] })).toThrow("invalid");
});

test("only checkpoint changes and terminal training jobs invalidate metadata", () => {
  expect(changesCheckpoints({ kind: "versions", payload: null })).toBe(true);
  expect(changesCheckpoints({ kind: "checkpoint", payload: { phase: "done" } })).toBe(true);
  expect(changesCheckpoints({ kind: "job", payload: { job: "train", status: "done" } })).toBe(true);
  for (const status of ["done", "failed", "cancelled"]) expect(changesCheckpoints({ kind: "job", payload: { id: "read", kind: "versions.best", status } })).toBe(false);
  expect(changesCheckpoints({ kind: "job", payload: { job: "train", status: "running" } })).toBe(false);
});

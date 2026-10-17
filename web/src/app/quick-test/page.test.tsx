import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test } from "vitest";
import { emit, mockApi } from "@/test/network";
import { nav } from "@/test/navigation";
import { renderPage } from "@/test/render";
import QuickTestPage from "./page";

// The Quick test page against a mocked API (the component tier of tests/quick-test.spec.ts).

const idle = { job: null, up_next: [], paused: false };
const current = { title: "Version 3", description: "4 blocks · 4 look-back heads · normal thinking room" };
const draft = { changes: ["more blocks (4 → 5)"], description: "5 blocks · 4 look-back heads · normal thinking room", cost: "1.2x bigger · 1.2x slower" };
const run = (steps: number, bpc: number) => ({ steps, bpc, curve: [[1, 4.2], [steps, bpc]], sample: `sample ${steps}` });
const results = {
  minutes: 3,
  shrink: false,
  dataset: "stories",
  verdict: "lower",
  delta: -0.2,
  threshold: 0.05,
  references: { uniform: 5.5, unigram: 3.9 },
  learned_context: true,
  checks: [
    { id: "starts_near_uniform", ok: true, detail: "5.4 vs 5.5" },
    { id: "finite", ok: true, detail: "ok" },
    { id: "memorises_one_batch", ok: true, detail: "4.2 -> 0.1" },
    { id: "error_falls", ok: false, detail: "3.0 -> 3.0" },
  ],
  broken: null,
  current: run(40, 3.1),
  draft: run(36, 2.9),
  finished_at: 2000,
};
const queued = { status: 202, body: { job_id: "job1", outlook: "Starts now", started_at: 1500 } };
const signal = "After the same 3 minutes, your idea's error is lower than Version 3's. You guessed lower.";

// The page keeps its last run in sessionStorage, which the shared setup does not clear.
beforeEach(() => window.sessionStorage.clear());

async function open(draftStaged: boolean, handlers: Parameters<typeof mockApi>[0] = {}) {
  const calls = mockApi({
    "GET /quicktest": { body: { current, draft: draftStaged ? draft : null, next_number: 4, last: null } },
    "GET /queue": { body: idle },
    ...handlers,
  });
  const user = userEvent.setup();
  const view = renderPage(<QuickTestPage />);
  expect(await screen.findByRole("heading", { level: 1, name: "Quick test" })).toBeVisible();
  await screen.findByRole("button", { name: "Run quick test" });
  return { calls, user, view };
}

const status = () => screen.getByTestId("quicktest-status");
const posts = (calls: { method: string; path: string; body: unknown }[], path: string) => calls.filter((c) => c.method === "POST" && c.path === path);

describe("Quick test page", () => {
  test("shows the current design and the staged change with its cost", async () => {
    const { user } = await open(true);
    expect(screen.getByText("Version 3")).toBeVisible();
    expect(screen.getByText("Version 3 with your change")).toBeVisible();
    expect(screen.getByText("more blocks (4 → 5)")).toBeVisible();
    expect(screen.getByText("1.2x bigger · 1.2x slower")).toBeVisible();
    expect(screen.getByText(/Nothing here is saved as a version/)).toBeVisible();
    await user.click(screen.getByText("How quick tests work"));
    expect(screen.getByText(/Fair by time, not steps\./)).toBeVisible();
  });

  test("asks for a staged change before it runs", async () => {
    const { calls, user } = await open(false);
    expect(screen.getByText("No change staged in the designer yet.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Run quick test" }));
    expect(status()).toHaveTextContent("Stage a change in the designer first.");
    expect(posts(calls, "/quicktest/run")).toHaveLength(0);
  });

  test("asks for a guess before it runs", async () => {
    const { calls, user } = await open(true);
    await user.click(screen.getByRole("button", { name: "Run quick test" }));
    expect(status()).toHaveTextContent("Pick your guess first.");
    expect(posts(calls, "/quicktest/run")).toHaveLength(0);
  });

  test("runs with the chosen options, follows progress and shows the result", async () => {
    const queue = { current: idle as object };
    const { calls, user } = await open(true, {
      "GET /queue": () => ({ body: queue.current }),
      "POST /quicktest/run": () => {
        queue.current = { ...idle, job: { id: "job1", kind: "quicktest", params: { guess: "Lower" } } };
        return queued;
      },
    });
    await user.click(screen.getByRole("radio", { name: "Practice problems" }));
    await user.click(screen.getByRole("radio", { name: "1 min" }));
    await user.click(screen.getByRole("radio", { name: "Shrunk to ¼" }));
    expect(screen.getByText("Faster; a hint, not proof.")).toBeVisible();
    await user.click(screen.getByRole("radio", { name: "Lower" }));
    await user.click(screen.getByRole("button", { name: "Run quick test" }));

    await waitFor(() => expect(status()).toHaveTextContent("Queued. Starts now."));
    expect(posts(calls, "/quicktest/run").map((c) => c.body)).toEqual([{ dataset: "problems", minutes: 1, shrink: true, guess: "Lower" }]);
    expect(await screen.findByRole("button", { name: "Running…" })).toBeDisabled();

    act(() => {
      emit("train", { job: "other", type: "quicktest", which: "current", step: 9, total: 10 });
      emit("train", { job: "job1", type: "quicktest", which: "current", step: 10, total: 40 });
    });
    // only the watched job's step counts: 10 of 40, not the other job's 9 of 10
    await waitFor(() => expect(screen.getByRole("progressbar", { name: "Version 3" })).toHaveAttribute("aria-valuenow", "25"));
    expect(screen.getByText(/step 10/)).toBeVisible();
    expect(screen.getByRole("progressbar", { name: "Your idea" })).toHaveAttribute("aria-valuenow", "0");

    queue.current = idle;
    act(() => emit("train", { job: "job1", type: "quicktest_done", results }));
    expect(await screen.findByText(signal)).toBeVisible();
    expect(screen.getByText("done · 40 steps")).toBeVisible();
    expect(screen.getByText("done · 36 steps")).toBeVisible();
    const checks = document.querySelectorAll("[data-check]");
    expect(checks).toHaveLength(4);
    expect(checks[0]).toHaveTextContent("✓");
    expect(checks[3]).toHaveTextContent("✗");
    expect(checks[3]).toHaveTextContent(/Error falls/i);
    expect(screen.getByText("sample 40")).toBeVisible();
    expect(screen.getByRole("button", { name: "Run again" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Save idea as version 4" })).toBeVisible();
  });

  test("a failed check names the part to look at, and a stopped run keeps nothing", async () => {
    const { user } = await open(true, { "POST /quicktest/run": queued });
    await user.click(screen.getByRole("radio", { name: "Same" }));
    await user.click(screen.getByRole("button", { name: "Run quick test" }));
    await waitFor(() => expect(status()).toHaveTextContent("Queued. Starts now."));
    const broken = {
      ...results,
      verdict: null,
      broken: "memorises_one_batch",
      current: null,
      draft: null,
      checks: results.checks.map((c) => ({ ...c, ok: c.id === "memorises_one_batch" ? false : null })),
    };
    act(() => emit("train", { job: "job1", type: "quicktest_done", results: broken }));
    expect(await screen.findByText(/A check failed: Can memorize one batch\. Look at the parts you changed/)).toBeVisible();
    act(() => emit("train", { job: "job1", type: "quicktest_done", results: { minutes: 3, stopped: true } }));
    expect(await screen.findByText("The quick test was stopped before it finished; nothing was kept.")).toBeVisible();
  });

  test("a refused start shows the server's reason, and a failed job says why", async () => {
    let attempts = 0;
    const { user } = await open(true, {
      "POST /quicktest/run": () => {
        attempts += 1;
        return attempts === 1
          ? { status: 400, body: { detail: "Your data has no plain text yet. Add some in Your data, or pick another tiny dataset." } }
          : queued;
      },
    });
    await user.click(screen.getByRole("radio", { name: "A slice of your data" }));
    await user.click(screen.getByRole("radio", { name: "Higher" }));
    await user.click(screen.getByRole("button", { name: "Run quick test" }));
    await waitFor(() => expect(status()).toHaveTextContent("Your data has no plain text yet."));
    await user.click(screen.getByRole("button", { name: "Run quick test" }));
    await waitFor(() => expect(status()).toHaveTextContent("Queued. Starts now."));
    act(() => emit("job", { job: "job1", status: "failed", error: "out of memory" }));
    await waitFor(() => expect(status()).toHaveTextContent("The quick test failed: out of memory"));
    expect(screen.getByRole("button", { name: "Run quick test" })).toBeEnabled();
  });

  test.each([
    ["Save idea as version 4", "POST", "/quicktest/save"],
    ["Discard idea", "DELETE", "/quicktest/draft"],
  ])("%s calls the server and opens the designer", async (button, method, path) => {
    const { calls, user } = await open(true, {
      "POST /quicktest/run": queued,
      "POST /quicktest/save": { status: 201, body: { number: 4 } },
      "DELETE /quicktest/draft": { status: 204 },
    });
    await user.click(screen.getByRole("radio", { name: "Same" }));
    await user.click(screen.getByRole("button", { name: "Run quick test" }));
    await waitFor(() => expect(status()).toHaveTextContent("Queued. Starts now."));
    act(() => emit("train", { job: "job1", type: "quicktest_done", results }));
    await user.click(await screen.findByRole("button", { name: button }));
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/builder"));
    expect(calls.some((c) => c.method === method && c.path === path)).toBe(true);
  });

  test("a result that finished while another page was open is shown on return", async () => {
    let finishedAway = false; // the server finishes the run while this page is not open
    const { user, view } = await open(true, {
      "POST /quicktest/run": queued,
      "GET /quicktest": ({ search }) => {
        const finished = finishedAway && new URLSearchParams(search).get("since") === "1500";
        return { body: { current, draft, next_number: 4, last: finished ? results : null } };
      },
    });
    await user.click(screen.getByRole("radio", { name: "Lower" }));
    await user.click(screen.getByRole("button", { name: "Run quick test" }));
    await waitFor(() => expect(status()).toHaveTextContent("Queued. Starts now."));
    expect(screen.queryByText(signal)).toBeNull();

    // leaving the page and coming back: the run is remembered in sessionStorage and asked for by when it was queued
    view.unmount();
    finishedAway = true;
    renderPage(<QuickTestPage />);
    expect(await screen.findByText(signal)).toBeVisible();
  });
});

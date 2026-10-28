import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { emit, mockApi, type Call } from "@/test/network";
import { renderPage } from "@/test/render";
import BenchmarkPage from "./page";

// The Benchmark page against a mocked API (ported from the Playwright spec tests/benchmark.spec.ts, now removed).

// jsdom has no layout, so Recharts' ResponsiveContainer would draw nothing; give the chart a fixed size so its lines render.
vi.mock("recharts", async (original) => {
  const actual = await original<typeof import("recharts")>();
  const { cloneElement } = await import("react");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement<{ width?: number; height?: number }> }) => (
      <div style={{ width: 800, height: 400 }}>{cloneElement(children, { width: 800, height: 400 })}</div>
    ),
  };
});

// Radix Select asks for pointer capture, which jsdom lacks.
Element.prototype.hasPointerCapture ??= () => false;
Element.prototype.releasePointerCapture ??= () => {};
Element.prototype.setPointerCapture ??= () => {};

// Every benchmark read is a job, so each POST answers with a job record.
const job = (id: string, kind: string, status: string, result: unknown = null, error = "") => ({
  id,
  kind,
  status,
  progress: "",
  fraction: null,
  result,
  error,
  created_at: 0,
  started_at: status === "queued" ? null : 0,
  finished_at: status === "queued" ? null : 0,
});

const suite = (version: number, bpc: number, accuracy: number, rate: number) => ({
  version,
  bpc,
  problems: { passed: Math.round(30 * rate), total: 30, rate },
  few_shot: { items: 40, chance: 0.3, accuracy, rows: [{ shots: 4, used: 4, correct: 12, accuracy }] },
});

const track = (checked3: boolean) => ({
  suite: "fake",
  sizes: { docs: 24, choice: 40, problems: 30 },
  dropped: {},
  rows: [
    { number: 1, name: "Version 1", lineage: "a", tokens: 200_000, result: suite(1, 2.6, 0.3, 0.1) },
    { number: 2, name: "Version 2", lineage: "a", tokens: 800_000, result: suite(2, 2.2, 0.35, 0.2) },
    { number: 3, name: "Version 3", lineage: "b", tokens: 500_000, result: checked3 ? suite(3, 2.4, 0.33, 0.15) : null },
  ],
});

const read = (rows: () => unknown) => ({ "POST /benchmark/read": () => ({ status: 202, body: job("r1", "benchmark.read", "done", rows()) }) });
const lines = () => document.querySelectorAll(".recharts-line").length;
const bodies = (calls: Call[], method: string, path: string) => calls.filter((c) => c.method === method && c.path === path).map((c) => c.body);

describe("Benchmark page", () => {
  test("reads the track, checks the unchecked version and charts every design on its own chart", async () => {
    let checked = false;
    const calls = mockApi({
      ...read(() => track(checked)),
      "POST /benchmark/run": () => {
        checked = true;
        return { status: 202, body: job("run3", "benchmark.run", "done", suite(3, 2.4, 0.33, 0.15)) };
      },
    });
    const user = userEvent.setup();
    renderPage(<BenchmarkPage />);

    expect(await screen.findByRole("heading", { level: 1, name: "Benchmark" })).toBeVisible();
    expect(await screen.findByText("1 of 3 trained versions have not been checked yet.")).toBeVisible();
    expect(screen.getByText(/Every version gets the same 24 passages, 40 questions and 30 problems/)).toBeVisible();
    // Three checks, each drawn by the shared chart, one line per design.
    for (const title of ["Error on held-back text", "Multiple-choice questions", "Checked problems"]) expect(screen.getByText(title)).toBeVisible();
    await waitFor(() => expect(lines()).toBe(1 + 2 + 1)); // only design A is checked: one line per chart, and the guessing line on the multiple-choice one

    await user.click(screen.getByRole("button", { name: "Check the rest" }));
    expect(await screen.findByText("Every trained version has been checked.")).toBeVisible();
    await waitFor(() => expect(lines()).toBe(2 + 3 + 2)); // both designs now: A and B on each chart, plus the guessing line
    expect(screen.getAllByText("Design A: versions 1, 2").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Guessing gets 30%").length).toBeGreaterThan(0);
    expect(bodies(calls, "POST", "/benchmark/run")).toEqual([{ version: 3 }]);
    expect(screen.queryByRole("button", { name: "Check the rest" })).not.toBeInTheDocument();
  });

  test("does not show training speed (it lives on Machines)", async () => {
    const calls = mockApi(read(() => track(true)));
    renderPage(<BenchmarkPage />);
    expect(await screen.findByText("Every trained version has been checked.")).toBeVisible();
    expect(screen.queryByRole("region", { name: "Training speed" })).not.toBeInTheDocument();
    expect(calls.some((c) => c.path === "/mlperf/board")).toBe(false);
  });

  test("shows the suite as a trend across a lineage, with the paired test beside each step and what was left out", async () => {
    const point = (number: number, tokens: number, bpc: number, accuracy: number, extra: Record<string, unknown> = {}) => ({
      number,
      tokens,
      bpc,
      bpc_band: [bpc - 0.05, bpc + 0.05],
      few_shot: accuracy,
      few_shot_band: [accuracy - 0.1, accuracy + 0.1],
      problems: 0.1,
      dropped: {},
      gain: null,
      ...extra,
    });
    const trended = {
      ...track(true),
      sources: ["Books", "Forum"],
      lineages: [
        {
          lineage: "a",
          versions: [1, 2],
          points: [
            point(1, 200_000, 2.6, 0.3, { dropped: { docs: 2, choice: 1 } }),
            point(2, 800_000, 2.2, 0.35, { gain: { bpc: { diff: -0.4, better: true, docs: 24 }, few_shot: { diff: 0.05, better: false, items: 40 }, problems: null } }),
          ],
        },
        { lineage: "b", versions: [3], points: [point(3, 500_000, 2.4, 0.33)] },
      ],
    };
    mockApi({ ...read(() => trended) });
    renderPage(<BenchmarkPage />);

    expect(await screen.findByText("Does it keep getting better?")).toBeVisible();
    expect(screen.getByText("Passages come from: Books, Forum.")).toBeVisible();
    const table = screen.getByRole("table", { name: "Design A: versions 1, 2" });
    const rows = within(table).getAllByRole("row");
    expect(within(rows[1]).getAllByRole("cell").map((c) => c.textContent)).toEqual(["1", "200K", "2.60 (2.55 to 2.65)", "–", "30% (20% to 40%)", "–", "3"]);
    expect(within(rows[2]).getAllByRole("cell").map((c) => c.textContent)).toEqual([
      "2",
      "800K",
      "2.20 (2.15 to 2.25)",
      "better (-0.40)",
      "35% (25% to 45%)",
      "not sure (+5 points)",
      "0",
    ]);
    expect(screen.getByRole("table", { name: "Design B: versions 3" })).toBeVisible();
  });

  test("a version scored under an older method is reported as waiting to be checked again", async () => {
    const stale = { ...track(true), rows: track(true).rows.map((r) => (r.number === 2 ? { ...r, result: null, stale: true } : r)) };
    mockApi({ ...read(() => stale) });
    renderPage(<BenchmarkPage />);
    expect(await screen.findByText(/1 of 3 trained versions have not been checked yet\. One was checked before the scoring method changed/)).toBeVisible();
  });

  test("shows a skeleton until the job is done, and reads the job once in case the event came first", async () => {
    let release = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    mockApi({
      "POST /benchmark/read": { status: 202, body: job("late", "benchmark.read", "queued") },
      "GET /jobs/late": async () => {
        await gate;
        return { body: job("late", "benchmark.read", "done", track(true)) };
      },
    });
    renderPage(<BenchmarkPage />);
    expect(await screen.findByLabelText("Loading the benchmark")).toBeVisible();
    release();
    expect(await screen.findByText("Every trained version has been checked.")).toBeVisible();
    expect(screen.queryByLabelText("Loading the benchmark")).not.toBeInTheDocument();
  });

  test("the job's result event ends the skeleton", async () => {
    const calls = mockApi({
      "POST /benchmark/read": { status: 202, body: job("ev", "benchmark.read", "queued") },
      "GET /jobs/ev": { body: job("ev", "benchmark.read", "running") },
    });
    renderPage(<BenchmarkPage />);
    expect(await screen.findByLabelText("Loading the benchmark")).toBeVisible();
    // The page listens for the job once it has read it; deliver the result then.
    await waitFor(() => expect(calls.some((c) => c.path === "/jobs/ev")).toBe(true));
    emit("job", job("ev", "benchmark.read", "done", track(true)));
    expect(await screen.findByText("Every trained version has been checked.")).toBeVisible();
  });

  test("a failed read says why and can be tried again", async () => {
    let fail = true;
    mockApi({
      "POST /benchmark/read": () => ({
        status: 202,
        body: fail ? job("bad", "benchmark.read", "failed", null, "OSError: held-back data is unreadable") : job("ok", "benchmark.read", "done", track(true)),
      }),
    });
    const user = userEvent.setup();
    renderPage(<BenchmarkPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("The benchmark could not be read.");
    expect(screen.getByRole("alert")).toHaveTextContent("held-back data is unreadable");
    fail = false;
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Every trained version has been checked.")).toBeVisible();
  });

  test("with no trained version the page says so and draws no chart", async () => {
    mockApi(read(() => ({ suite: "fake", sizes: { docs: 24, choice: 40, problems: 30 }, dropped: {}, rows: [] })));
    renderPage(<BenchmarkPage />);
    expect(await screen.findByText(/No trained version yet\./)).toBeVisible();
    expect(lines()).toBe(0);
  });

  test("a version is asked the multiple-choice questions and the table compares it with guessing", async () => {
    const calls = mockApi({
      ...read(() => track(true)),
      "POST /benchmark/few-shot": {
        status: 202,
        body: job("fs", "benchmark.few_shot", "done", {
          items: 40,
          tasks: 2,
          held_back: false,
          chance: 0.3,
          rows: [
            { shots: 0, used: 0, correct: 11, accuracy: 0.275 },
            { shots: 4, used: 3.5, correct: 24, accuracy: 0.6 },
          ],
        }),
      },
    });
    const user = userEvent.setup();
    renderPage(<BenchmarkPage />);
    expect(await screen.findByText(/Not checked yet for this version\./)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Check it" }));
    expect(await screen.findByRole("cell", { name: "24 of 40" })).toBeVisible();
    expect(screen.getByRole("cell", { name: "4 (3.5 fit on average)" })).toBeVisible();
    expect(screen.getByRole("cell", { name: "better" })).toBeVisible();
    expect(screen.getByText(/There are no held-back multiple-choice problems yet/)).toBeVisible();
    expect(bodies(calls, "POST", "/benchmark/few-shot")).toEqual([{ version: 3 }]);
    expect(screen.getByRole("button", { name: "Check again" })).toBeVisible();
  });

  test("two versions get a verdict, or the reason there is none", async () => {
    mockApi({
      ...read(() => track(true)),
      "GET /benchmark/verdict": ({ search }) => {
        const query = new URLSearchParams(search);
        const same = query.get("a") === "1" && query.get("b") === "2";
        return { body: same ? { percent: 37.5, comparable: true, reason: "" } : { percent: null, comparable: false, reason: "The runs have different total lengths." } };
      },
    });
    const user = userEvent.setup();
    renderPage(<BenchmarkPage />);
    expect(await screen.findByText("Pick two versions to compare.")).toBeVisible();
    const pick = async (label: string, name: string) => {
      await user.click(screen.getByRole("combobox", { name: label }));
      await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name }));
    };
    await pick("Older version", "Version 1");
    await pick("Newer version", "Version 2");
    expect(await screen.findByText("To reach the same error, Version 2 needs 63% of the training Version 1 needed.")).toBeVisible();
    await pick("Newer version", "Version 3");
    expect(await screen.findByText("No verdict yet: The runs have different total lengths.")).toBeVisible();
  });
});

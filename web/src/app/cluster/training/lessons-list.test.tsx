import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import type { Lesson, Run } from "./logic";
import TrainingPage from "./page";
import { run } from "./test-data";

// The lessons list on the live view: what the teacher gave, with Drop and Keep, against a mocked API.

const NODES = [{ address: "local", local: true, hostname: "mac-mini", version: "1", device: "mps", device_name: "Apple M2", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "" }];

const item = (prompt: string, answer: string) => ({ prompt, reply: "no idea", answer });

function server(initialRun: Run, initial: Lesson[]) {
  const state = { run: initialRun, lessons: initial, posted: [] as string[] };
  mockApi({
    "GET /distributed/defaults": { body: { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16" } },
    "GET /pretraining/text_sets": { body: [] },
    "GET /versions": { body: [] },
    "GET /cluster/nodes": { body: NODES },
    "GET /distributed/runs": () => ({ body: [state.run] }),
    "GET /distributed/runs/*/lessons": () => ({ body: state.lessons }),
    "POST /distributed/runs/*/lessons/*/*": ({ path }) => {
      state.posted.push(path);
      const [, , , , , lesson, verb] = path.split("/");
      state.lessons = state.lessons.map((l) => (l.id === Number(lesson) ? { ...l, status: verb === "drop" ? "dropped" : "live", keepable: verb === "drop" } : l));
      const live = state.lessons.filter((l) => l.status === "live").reduce((n, l) => n + l.rows, 0);
      state.run = { ...state.run, lessons: live, lessons_left_out: 5 - live };
      return { body: state.lessons };
    },
  });
  return state;
}

describe("lessons list", () => {
  test("lists what the teacher gave, drops a lesson, then keeps it again", async () => {
    const state = server(run({ lessons: 5, lessons_left_out: 0 }), [
      { id: 4, round: 1, rows: 3, status: "live", keepable: false, items: [item("What is 1 plus 1?", "2 exactly."), item("What is 2 plus 2?", "4 exactly.")], more: 1 },
      { id: 5, round: 3, rows: 2, status: "live", keepable: false, items: [item("What is 9 plus 9?", "18 exactly.")], more: 0 },
    ]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    expect(await screen.findByTestId("run-lessons-r1")).toHaveTextContent("5 lessons from the teacher in the training text");
    const items = within(await screen.findByTestId("lessons-r1")).getAllByRole("listitem", { name: "" }).filter((li) => li.hasAttribute("data-status"));
    expect(items.map((li) => li.getAttribute("data-testid"))).toEqual(["lesson-5", "lesson-4"]); // newest first
    const first = screen.getByTestId("lesson-4");
    expect(first).toHaveTextContent("After round 2 · 3 answers · Used in training");
    expect(first).toHaveTextContent("What is 1 plus 1?");
    expect(first).toHaveTextContent("no idea");
    expect(first).toHaveTextContent("2 exactly.");
    expect(first).toHaveTextContent("and 1 more");

    await user.click(within(first).getByRole("button", { name: "Drop" }));
    await waitFor(() => expect(screen.getByTestId("lesson-4")).toHaveTextContent("Dropped by you"));
    expect(state.posted).toEqual(["/distributed/runs/r1/lessons/4/drop"]);
    await waitFor(() => expect(screen.getByTestId("run-lessons-r1")).toHaveTextContent("2 lessons from the teacher in the training text, 3 left out"));
    await user.click(within(screen.getByTestId("lesson-4")).getByRole("button", { name: "Keep" }));
    await waitFor(() => expect(screen.getByTestId("lesson-4")).toHaveTextContent("Used in training"));
    expect(state.posted).toEqual(["/distributed/runs/r1/lessons/4/drop", "/distributed/runs/r1/lessons/4/keep"]);
  }, 20_000);

  test("a lesson that cannot be kept says why instead of offering Keep", async () => {
    server(run({ lessons: 0, lessons_left_out: 2 }), [
      { id: 4, round: 1, rows: 2, status: "left_out", keepable: false, items: [], more: 0 },
      { id: 5, round: 2, rows: 2, status: "left_out", keepable: true, items: [], more: 0 },
    ]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const kept = await screen.findByTestId("lesson-5");
    expect(within(kept).getByRole("button", { name: "Keep" })).toBeEnabled();
    const lost = screen.getByTestId("lesson-4");
    expect(lost).toHaveTextContent("Left out when the run went back to an earlier save");
    expect(within(lost).queryByRole("button")).toBeNull();
    expect(lost).toHaveTextContent("Cannot be kept: this computer no longer holds its answers.");
  });

  test("lessons that cannot be reached are a quiet note, not an alert", async () => {
    mockApi({
      "GET /distributed/defaults": { body: { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16" } },
      "GET /pretraining/text_sets": { body: [] },
      "GET /versions": { body: [] },
      "GET /cluster/nodes": { body: NODES },
      "GET /distributed/runs": { body: [run({ lessons: 2 })] },
      "GET /distributed/runs/*/lessons": { status: 502, body: { detail: "box did not answer" } },
    });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByText("Lessons are not reachable right now.")).toBeInTheDocument();
    expect(screen.queryByText(/box did not answer/)).toBeNull();
    expect(screen.queryAllByRole("alert").filter((a) => a.textContent?.includes("Lessons"))).toEqual([]);
  });

  test("an ended run shows only the count, without reading its lessons", async () => {
    const calls = mockApi({
      "GET /distributed/defaults": { body: { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16" } },
      "GET /pretraining/text_sets": { body: [] },
      "GET /versions": { body: [] },
      "GET /cluster/nodes": { body: NODES },
      "GET /distributed/runs": { body: [run({ state: "stopped", can_stop: false, lessons: 4 })] },
    });
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    expect(await screen.findByTestId("run-lessons-r1")).toHaveTextContent("4 lessons");
    expect(screen.queryByLabelText("Lessons from the teacher")).toBeNull();
    expect(calls.filter((c) => c.path.endsWith("/lessons"))).toEqual([]);
  });
});

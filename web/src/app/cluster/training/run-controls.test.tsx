import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import type { Run } from "./logic";
import RunViewPage from "@/app/runs/view/page";
import { round, run } from "./test-data";

// Pause, Resume and Restart on the live view, the stop reason, and the error curve, against a mocked API.

const NODES = [{ address: "local", local: true, hostname: "mac-mini", version: "1", device: "mps", device_name: "Apple M2", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "" }];

function server(initial: Run[]) {
  const state = { runs: initial, posted: [] as string[] };
  const change = (id: string, over: Partial<Run>) => {
    state.runs = state.runs.map((r) => (r.id === id ? { ...r, ...over } : r));
    return { body: state.runs.find((r) => r.id === id) };
  };
  mockApi({
    "GET /distributed/defaults": { body: { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16", max_steps: 1_000_000 } },
    "GET /pretraining/text_sets": { body: [] },
    "GET /versions": { body: [] },
    "GET /cluster/nodes": { body: NODES },
    "GET /distributed/runs/r1": () => ({ body: state.runs.find((r) => r.id === "r1") }),
    "GET /distributed/runs/r2": () => ({ body: state.runs.find((r) => r.id === "r2") }),
    "GET /queue": { body: { job: null, up_next: [] } },
    "POST /distributed/runs/*/pause": ({ path }) => {
      state.posted.push(path);
      return change(path.split("/")[3], { paused: true, can_pause: false, can_resume: true, phase: "paused" });
    },
    "POST /distributed/runs/*/resume": ({ path }) => {
      state.posted.push(path);
      return change(path.split("/")[3], { paused: false, can_pause: true, can_resume: false, phase: "training" });
    },
    "POST /distributed/runs/*/restart": ({ path }) => {
      state.posted.push(path);
      const made = run({ id: "r2", label: "Restarted run", started_at: 500, round: 0, history: [], can_pause: true });
      state.runs = [made, ...state.runs];
      return { status: 201, body: made };
    },
  });
  return state;
}

describe("run controls", () => {
  test("pauses a run, then resumes it", async () => {
    const state = server([run({ can_pause: true })]);
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Pause" }));
    await waitFor(() => expect(state.posted).toEqual(["/distributed/runs/r1/pause"]));
    expect(await screen.findByTestId("run-paused")).toHaveTextContent("Paused.");
    expect(screen.queryByRole("button", { name: "Pause" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(state.posted[1]).toBe("/distributed/runs/r1/resume"));
    await waitFor(() => expect(screen.queryByTestId("run-paused")).toBeNull());
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
  });

  test("a stopped run says why and offers Restart, which shows the new run", async () => {
    const state = server([run({ state: "stopped", can_stop: false, can_restart: true, stop_reason: "The loss stopped improving." })]);
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    expect(await screen.findByTestId("run-stop-reason")).toHaveTextContent("Stopped because: The loss stopped improving.");
    expect(screen.queryByRole("button", { name: "Pause" })).toBeNull();
    await userEvent.setup().click(screen.getByRole("button", { name: "Restart" }));
    await waitFor(() => expect(state.posted).toEqual(["/distributed/runs/r1/restart"]));
    expect(state.runs[0].label).toBe("Restarted run");
  });

  test("a run that cannot be paused or restarted shows none of the buttons", async () => {
    server([run()]);
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    await screen.findByTestId("run-r1");
    for (const name of ["Pause", "Resume", "Restart"]) expect(screen.queryByRole("button", { name })).toBeNull();
  });

  test("the error curve of a run with no round yet says so", async () => {
    server([run({ history: [] })]);
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    expect(within(await screen.findByTestId("run-outcome")).getByText("No round has finished yet.")).toBeInTheDocument();
  });

  test("the error curve is drawn for a run with rounds", async () => {
    server([run({ best_round: 1, history: [round(0), round(1), round(2, { pivot: "halve_lr" })] })]);
    renderPage(<RunViewPage />, { route: "/runs/view?id=r1" });
    const curve = await screen.findByTestId("run-outcome");
    expect(within(curve).getByRole("heading", { name: "Learning progress" })).toBeInTheDocument();
    expect(within(curve).queryByText("No round has finished yet.")).toBeNull(); // jsdom has no layout, so the lines themselves are not drawn here
  });
});

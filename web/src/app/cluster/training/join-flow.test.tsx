import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import type { Run, StartRun } from "./logic";
import TrainingPage from "./page";
import { member, run } from "./test-data";

// The join flow on the machine that starts a run: the readiness roster, the dialog when a chosen machine is not ready, and the run page's alerts.

const DEFAULTS = { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16", max_steps: 1_000_000 };
const SETS = [{ id: "stories", name: "Stories", count: 1200, on: true, held: false, library: false }];
const node = (address: string, over: Record<string, unknown> = {}) => ({
  address, local: false, hostname: address, version: "1", device: "cuda", device_name: "RTX 4090", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "", ...over,
});
const NODES = [
  node("local", { local: true, hostname: "mac-mini" }),
  node("quiet"),
  node("BEAST", { busy: true }),
  node("old-laptop.tail1234.ts.net:8000", { hostname: "old-laptop", online: false, error: "old-laptop did not answer within 2 seconds." }),
];
const BEAST_JOB = { id: "g1", kind: "pretrain", label: "Version 1", node: "BEAST", hostname: "BEAST", status: "running", eta_s: 720, note: "", pinned: "", queued_address: "", queued_here: false, queued_on: "BEAST", stale: false };

function server(runs: Run[] = []) {
  const state = { started: [] as StartRun[], asked: [] as unknown[] };
  mockApi({
    "GET /distributed/defaults": { body: DEFAULTS },
    "GET /distributed/runs": () => ({ body: runs }),
    "POST /distributed/runs": ({ body }) => {
      state.started.push(body as StartRun);
      return { status: 201, body: run({ state: "queued", round: 0, history: [] }) };
    },
    "POST /distributed/peers/pause-request": ({ body }) => {
      state.asked.push(body);
      return { body: { id: "q1", at: 1, from_node: "mac-mini", run_label: "", blocking_job: null } };
    },
    "GET /pretraining/text_sets": { body: SETS },
    "GET /versions": { body: [] },
    "GET /cluster/nodes": { body: NODES },
    "GET /cluster/merged": { body: { jobs: [BEAST_JOB], nodes: [] } },
  });
  return state;
}

const START = "Start with the ready machines now, others join when free";

describe("the readiness roster", () => {
  test("shows nothing until a machine is chosen, then Ready, Busy and Unreachable lines", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await screen.findByRole("checkbox", { name: /quiet/ });
    expect(screen.queryByTestId("roster")).toBeNull();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox", { name: /quiet/ }));
    await user.click(screen.getByRole("checkbox", { name: /BEAST/ }));
    await user.click(screen.getByRole("checkbox", { name: /old-laptop/ }));
    expect(within(screen.getByTestId("roster-quiet")).getByText("Ready")).toBeInTheDocument();
    expect(screen.getByTestId("roster-quiet")).toHaveTextContent("quiet is free and will join as soon as the run starts.");
    expect(within(screen.getByTestId("roster-BEAST")).getByText("Busy")).toBeInTheDocument();
    expect(screen.getByTestId("roster-BEAST")).toHaveTextContent("BEAST is training 'Version 1', about 12 min left.");
    const down = screen.getByTestId("roster-old-laptop.tail1234.ts.net:8000");
    expect(within(down).getByText("Unreachable")).toBeInTheDocument();
    expect(down).toHaveTextContent("old-laptop did not answer: old-laptop did not answer within 2 seconds.");
  });

  test("a busy machine gets an alert whose button sends the pause request", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /BEAST/ }));
    const alert = within(screen.getByTestId("roster-alert-BEAST"));
    expect(alert.getByText("BEAST is busy with 'Version 1'")).toBeInTheDocument();
    await user.click(alert.getByRole("button", { name: "Ask BEAST to pause it" }));
    await waitFor(() => expect(state.asked).toEqual([{ address: "BEAST", run_label: "" }]));
    expect(await screen.findByText(/Asked BEAST to pause 'Version 1'/)).toBeInTheDocument();
  });

  test("an unreachable machine gets an alert whose button copies the https address", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /old-laptop/ }));
    const alert = within(screen.getByTestId("roster-alert-old-laptop.tail1234.ts.net:8000"));
    expect(alert.getByText("Can't reach old-laptop")).toBeInTheDocument();
    await user.click(alert.getByRole("button", { name: "Copy the address to add" }));
    await waitFor(async () => expect(await navigator.clipboard.readText()).toBe("https://old-laptop.tail1234.ts.net"));
    expect(await screen.findByText("Copied https://old-laptop.tail1234.ts.net.")).toBeInTheDocument();
  });
});

describe("starting with machines that are not ready", () => {
  test("Start is plain when every chosen machine is ready", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /quiet/ }));
    await user.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(state.started).toHaveLength(1));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(state.started[0].machines).toEqual(["quiet"]);
  });

  test("a busy machine opens a dialog; starting with the ready ones sends the run as chosen", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /quiet/ }));
    await user.click(screen.getByRole("checkbox", { name: /BEAST/ }));
    await user.click(screen.getByRole("button", { name: "Start run" }));
    const dialog = within(await screen.findByRole("alertdialog"));
    expect(dialog.getByText("Some chosen machines are not ready")).toBeInTheDocument();
    expect(dialog.getByText("Ready now: quiet. Not ready: BEAST.")).toBeInTheDocument();
    expect(state.started).toEqual([]);
    await user.click(dialog.getByRole("button", { name: START }));
    await waitFor(() => expect(state.started).toHaveLength(1));
    expect(state.started[0].machines).toEqual(["quiet", "BEAST"]);
    expect(state.started[0]).not.toHaveProperty("min_members");
  });

  test("Wait for all makes the run wait until this computer and every answering machine are free", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /quiet/ }));
    await user.click(screen.getByRole("checkbox", { name: /BEAST/ }));
    await user.click(screen.getByRole("checkbox", { name: /old-laptop/ }));
    await user.click(screen.getByRole("button", { name: "Start run" }));
    await user.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Wait for all" }));
    await waitFor(() => expect(state.started).toHaveLength(1));
    expect(state.started[0].min_members).toBe(3);
  });

  test("going back to the form starts nothing", async () => {
    const state = server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /BEAST/ }));
    await user.click(screen.getByRole("button", { name: "Start run" }));
    await user.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Back to the form" }));
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    expect(state.started).toEqual([]);
  });

  test("with only unreachable machines there is nothing to wait for", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("checkbox", { name: /old-laptop/ }));
    await user.click(screen.getByRole("button", { name: "Start run" }));
    const dialog = within(await screen.findByRole("alertdialog"));
    expect(dialog.queryByRole("button", { name: "Wait for all" })).toBeNull();
    expect(dialog.getByRole("button", { name: START })).toBeInTheDocument();
  });
});

describe("a run's page", () => {
  test("a waiting member gets a busy alert and an unreachable one gets an address to add", async () => {
    server([
      run({
        members: [
          member("mac-mini", { address: "local", local: true, can_stop: false }),
          member("BEAST", { address: "BEAST", state: "waiting", reason: "Busy with 'Version 1'.", blocking_job: { label: "Version 1", eta_s: 720 } }),
          member("old-laptop", { address: "old-laptop.tail1234.ts.net:8000", state: "unreachable", reason: "old-laptop did not answer. Check that it is on.", can_stop: false }),
        ],
      }),
    ]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    const busy = within(await screen.findByTestId("member-alert-BEAST"));
    expect(busy.getByText("BEAST is busy with 'Version 1'")).toBeInTheDocument();
    expect(busy.getByRole("button", { name: "Ask BEAST to pause it" })).toBeInTheDocument();
    const down = within(screen.getByTestId("member-alert-old-laptop"));
    expect(down.getByText("Can't reach old-laptop")).toBeInTheDocument();
    expect(down.getByText("old-laptop did not answer. Check that it is on.")).toBeInTheDocument();
    expect(down.getByRole("button", { name: "Copy the address to add" })).toBeInTheDocument();
  });

  test("a run whose members all train has no alerts", async () => {
    server([run()]);
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await screen.findByTestId("run-r1");
    expect(screen.queryByTestId("member-alerts-r1")).toBeNull();
  });
});

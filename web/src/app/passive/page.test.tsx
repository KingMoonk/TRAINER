import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import PassivePage from "./page";

// The Passive training page against a mocked API (the component tier of tests/passive.spec.ts).

const idle = { job: null, up_next: [], paused: false };
const running = {
  id: "p1",
  status: "running",
  created: "2026-10-01T22:00:00",
  stop: "Stops after 8 hours",
  trainer: "Running",
  watchdog: "Running",
  restarts: 2,
  heard: 4.5,
  error: "",
};
const stopped = { ...running, id: "p0", status: "stopped", trainer: "Stopped", watchdog: "Stopped", heard: null, error: "ran out of disk" };

describe("Passive training page", () => {
  test("lists sessions with facts, Stop asks first and then stops one", async () => {
    let sessions = [running, stopped];
    const calls = mockApi({
      "GET /passive": () => ({ body: { sessions, signin: true } }),
      "GET /queue": { body: idle },
      "POST /passive/p1/stop": () => {
        sessions = [{ ...running, status: "stopped", trainer: "Stopped", watchdog: "Stopped", heard: null }, stopped];
        return { body: { status: "stopped" } };
      },
    });
    const user = userEvent.setup();
    renderPage(<PassivePage />);
    expect(await screen.findByRole("heading", { level: 1, name: "Passive training" })).toBeVisible();

    const live = await screen.findByTestId("session-p1");
    expect(live).toHaveTextContent("Running · started 01 Oct 22:00");
    expect(live).toHaveTextContent("Stops after 8 hours");
    expect(live).toHaveTextContent("Trainer: Running · watchdog: Running · restarts: 2 · last heard under a minute ago");
    const old = screen.getByTestId("session-p0");
    expect(old).toHaveTextContent("Stopped · started");
    expect(old).toHaveTextContent("ran out of disk");
    expect(within(old).queryByRole("button", { name: "Stop" })).toBeNull();

    await user.click(within(live).getByRole("button", { name: "Stop" }));
    expect(screen.getByText("Stop this passive training? It saves where it got to and does not start again.")).toBeVisible();
    await user.click(within(live).getByRole("button", { name: "Cancel" }));
    expect(within(live).getByRole("button", { name: "Stop" })).toBeVisible();
    expect(calls.some((c) => c.path === "/passive/p1/stop")).toBe(false);

    await user.click(within(live).getByRole("button", { name: "Stop" }));
    await user.click(within(live).getByRole("button", { name: "Yes, stop it" }));
    await waitFor(() => expect(screen.getByTestId("session-p1")).toHaveTextContent("Stopped · started"));
    expect(screen.getByTestId("passive-note")).toBeVisible();
    expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/passive/p1/stop"]);
  });

  test("Keep it going shows only for an active session with no trainer and no watchdog, and starts the watchdog again", async () => {
    let sessions = [
      { ...running, trainer: "Stopped", watchdog: "Stopped", heard: null },
      { ...running, id: "p2", trainer: "Stopped", watchdog: "Running" },
      stopped,
    ];
    const calls = mockApi({
      "GET /passive": () => ({ body: { sessions, signin: false } }),
      "GET /queue": { body: idle },
      "POST /passive/p1/keep-going": () => {
        sessions = [{ ...running, trainer: "Restarting", watchdog: "Running" }, sessions[1], sessions[2]];
        return { body: { status: "running" } };
      },
    });
    const user = userEvent.setup();
    renderPage(<PassivePage />);
    const dead = await screen.findByTestId("session-p1");
    expect(within(screen.getByTestId("session-p2")).queryByRole("button", { name: "Keep it going" })).toBeNull();
    expect(within(screen.getByTestId("session-p0")).queryByRole("button", { name: "Keep it going" })).toBeNull();
    await user.click(within(dead).getByRole("button", { name: "Keep it going" }));
    await waitFor(() => expect(within(screen.getByTestId("session-p1")).queryByRole("button", { name: "Keep it going" })).toBeNull());
    expect(screen.getByTestId("passive-note")).toHaveTextContent("Started the watchdog again.");
    expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/passive/p1/keep-going"]);
  });

  test("Pause shows on a running session and Resume on a paused one, each asking the API", async () => {
    let sessions = [running, { ...running, id: "p3", status: "paused" }, stopped];
    const calls = mockApi({
      "GET /passive": () => ({ body: { sessions, signin: false } }),
      "GET /queue": { body: idle },
      "POST /passive/p1/pause": () => {
        sessions = [{ ...running, status: "paused" }, sessions[1], sessions[2]];
        return { body: { status: "paused" } };
      },
      "POST /passive/p3/resume": () => {
        sessions = [sessions[0], { ...running, id: "p3" }, sessions[2]];
        return { body: { status: "running" } };
      },
    });
    const user = userEvent.setup();
    renderPage(<PassivePage />);
    const live = await screen.findByTestId("session-p1");
    const held = screen.getByTestId("session-p3");
    const old = screen.getByTestId("session-p0");
    expect(within(live).queryByRole("button", { name: "Resume" })).toBeNull();
    expect(within(held).queryByRole("button", { name: "Pause" })).toBeNull();
    expect(within(old).queryByRole("button", { name: "Pause" })).toBeNull();

    await user.click(within(live).getByRole("button", { name: "Pause" }));
    expect(await within(screen.getByTestId("session-p1")).findByRole("button", { name: "Resume" })).toBeVisible();
    expect(screen.getByTestId("passive-note")).toHaveTextContent("Asked it to pause.");
    await user.click(within(screen.getByTestId("session-p3")).getByRole("button", { name: "Resume" }));
    expect(await within(screen.getByTestId("session-p3")).findByRole("button", { name: "Pause" })).toBeVisible();
    expect(screen.getByTestId("passive-note")).toHaveTextContent("Asked it to carry on.");
    expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/passive/p1/pause", "/passive/p3/resume"]);
  });

  test("Resume shows only on failed and stopped rows and puts the session back in the queue", async () => {
    const failed = { ...stopped, id: "p9", status: "failed", error: "gave up" };
    let sessions = [{ ...stopped, id: "p8", status: "finished" }, failed, { ...running, id: "p7", status: "paused" }];
    const calls = mockApi({
      "GET /passive": () => ({ body: { sessions, signin: false } }),
      "GET /queue": { body: idle },
      "POST /passive/p9/resume": () => {
        sessions = [sessions[0], { ...failed, status: "queued", error: "", trainer: "Running", watchdog: "Running" }, sessions[2]];
        return { body: { status: "queued" } };
      },
    });
    const user = userEvent.setup();
    renderPage(<PassivePage />);
    const ended = await screen.findByTestId("session-p9");
    expect(within(screen.getByTestId("session-p8")).queryByRole("button", { name: "Resume" })).toBeNull(); // finished
    expect(within(screen.getByTestId("session-p7")).getByRole("button", { name: "Resume" })).toBeVisible(); // paused
    await user.click(within(ended).getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("session-p9")).toHaveTextContent("Starting · started"));
    expect(screen.getByTestId("passive-note")).toHaveTextContent("Put it back in the queue.");
    expect(within(screen.getByTestId("session-p9")).queryByRole("button", { name: "Resume" })).toBeNull();
    expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/passive/p9/resume"]);
  });

  test("the New session form creates a session and the new row appears", async () => {
    let sessions: (typeof running)[] = [];
    const fresh = { ...running, id: "p5", status: "queued", trainer: "Stopped", watchdog: "Running", heard: null as unknown as number, restarts: 0 };
    const calls = mockApi({
      "GET /passive": () => ({ body: { sessions, signin: false } }),
      "GET /queue": { body: idle },
      "GET /versions": { body: [{ number: 1, name: "Tiny" }, { number: 2, name: "" }] },
      "GET /pretraining/text_sets": {
        body: [
          { id: "mine", name: "My notes", count: 1200, held: false, library: false, on: true },
          { id: "val", name: "Validation split", count: 10, held: true, library: true, on: false },
        ],
      },
      "POST /passive": () => {
        sessions = [fresh];
        return { status: 201, body: { id: "p5", status: "queued" } };
      },
    });
    const user = userEvent.setup();
    renderPage(<PassivePage />);
    await user.click(await screen.findByRole("button", { name: "New session" }));
    const form = await screen.findByRole("form", { name: "New passive session" });
    expect(within(form).getByRole("option", { name: "Version 1 · Tiny" })).toBeVisible();
    expect(within(form).queryByText("Validation split")).toBeNull();

    await user.selectOptions(within(form).getByLabelText("Version to train"), "2");
    await user.click(within(form).getByLabelText("All of my text")); // untick: choose sets
    await user.click(within(form).getByRole("button", { name: "Start passive training" }));
    expect(await within(form).findByTestId("passive-start-error")).toHaveTextContent("Tick at least one set");
    await user.click(within(form).getByLabelText(/My notes/));
    await user.clear(within(form).getByLabelText("Time limit"));
    await user.type(within(form).getByLabelText("Time limit"), "30");
    await user.selectOptions(within(form).getByLabelText("Time unit"), "minutes");
    await user.clear(within(form).getByLabelText("Rounds without a better checkpoint"));
    await user.type(within(form).getByLabelText("Rounds without a better checkpoint"), "5");
    await user.type(within(form).getByLabelText("Test error to reach"), "0.8");
    await user.click(within(form).getByLabelText(/Polish each checkpoint/));
    await user.click(within(form).getByLabelText("Pause on battery power"));
    await user.click(within(form).getByRole("button", { name: "Start passive training" }));

    expect(await screen.findByTestId("session-p5")).toHaveTextContent("Starting · started");
    expect(screen.getByTestId("passive-note")).toHaveTextContent("Started.");
    expect(screen.queryByRole("form", { name: "New passive session" })).toBeNull();
    expect(calls.find((c) => c.method === "POST" && c.path === "/passive")?.body).toEqual({
      version: 2,
      sources: ["mine"],
      duration: 30,
      unit: "minutes",
      patience_rounds: 5,
      target_error: 0.8,
      teacher: false,
      memory_fraction: 0.5,
      keep_awake: true,
      pause_on_battery: false,
    });
  }, 20_000); // many keystrokes: slow on a busy machine

  test("the form shows the server's sentence when the plan is refused", async () => {
    mockApi({
      "GET /passive": { body: { sessions: [], signin: false } },
      "GET /queue": { body: idle },
      "GET /versions": { body: [{ number: 1, name: "" }] },
      "GET /pretraining/text_sets": { body: [{ id: "mine", name: "My notes", count: 3, held: false, library: false, on: true }] },
      "POST /passive": { status: 409, body: { detail: "A passive training session is already running. Stop it before starting another." } },
    });
    const user = userEvent.setup();
    renderPage(<PassivePage />);
    await user.click(await screen.findByRole("button", { name: "New session" }));
    await user.click(await screen.findByRole("button", { name: "Start passive training" }));
    expect(await screen.findByTestId("passive-start-error")).toHaveTextContent("already running");
  });

  test("New session is disabled while a session is active", async () => {
    mockApi({ "GET /passive": { body: { sessions: [running], signin: false } }, "GET /queue": { body: idle } });
    renderPage(<PassivePage />);
    expect(await screen.findByRole("button", { name: "New session" })).toBeDisabled();
    expect(screen.getByText("Stop or wait for the running session before starting another.")).toBeVisible();
  });

  test("the sign-in start can be removed", async () => {
    let signin = true;
    const calls = mockApi({
      "GET /passive": () => ({ body: { sessions: [], signin } }),
      "GET /queue": { body: idle },
      "DELETE /passive/signin": () => {
        signin = false;
        return { status: 204 };
      },
    });
    const user = userEvent.setup();
    renderPage(<PassivePage />);
    expect(await screen.findByText("No passive training on this computer.")).toBeVisible();
    expect(screen.getByText("The watchdog starts when you sign in, so passive training carries on after a restart.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Don't start when I sign in" }));
    expect(await screen.findByText("Nothing starts when you sign in.")).toBeVisible();
    expect(screen.getByText("Removed. Passive training no longer starts when you sign in.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Don't start when I sign in" })).toBeNull();
    expect(calls.some((c) => c.method === "DELETE" && c.path === "/passive/signin")).toBe(true);
  });

  test("warns when a running session and a running queue job share the computer", async () => {
    mockApi({
      "GET /passive": { body: { sessions: [running], signin: false } },
      "GET /queue": { body: { ...idle, job: { id: "j1", kind: "pretrain" } } },
    });
    renderPage(<PassivePage />);
    expect(await screen.findByTestId("passive-warning")).toHaveTextContent("The queue is training too.");
  });

  test("no warning when the queue is paused", async () => {
    const calls = mockApi({
      "GET /passive": { body: { sessions: [running], signin: false } },
      "GET /queue": { body: { ...idle, job: { id: "j1", kind: "pretrain" }, paused: true } },
    });
    renderPage(<PassivePage />);
    expect(await screen.findByTestId("session-p1")).toBeVisible();
    await waitFor(() => expect(calls.some((c) => c.path === "/queue")).toBe(true));
    // the queue read must have landed (and had time to render) before the absence of the warning means anything
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByTestId("passive-warning")).toBeNull();
  });

  test("a failed read says so", async () => {
    mockApi({ "GET /passive": { status: 500, body: "boom" } });
    renderPage(<PassivePage />);
    expect(await screen.findByTestId("passive-error")).toHaveTextContent("Could not read the passive sessions.");
  });
});

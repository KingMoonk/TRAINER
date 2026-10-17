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

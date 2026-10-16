import { expect, test } from "@playwright/test";
import { fakeEvents, mockApi } from "./qp-mock";

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

test("passive: lists sessions with facts, Stop asks first and then stops one", async ({ page }) => {
  await fakeEvents(page);
  let sessions = [running, stopped];
  const calls = await mockApi(page, {
    "GET /passive": () => ({ body: { sessions, signin: true } }),
    "GET /queue": () => ({ body: idle }),
    "POST /passive/p1/stop": () => {
      sessions = [{ ...running, status: "stopped", trainer: "Stopped", watchdog: "Stopped", heard: null }, stopped];
      return { body: { status: "stopped" } };
    },
  });
  await page.goto("/passive/");
  await expect(page.getByRole("heading", { level: 1, name: "Passive training" })).toBeVisible();

  const live = page.getByTestId("session-p1");
  await expect(live).toContainText("Running · started 01 Oct 22:00");
  await expect(live).toContainText("Stops after 8 hours");
  await expect(live).toContainText("Trainer: Running · watchdog: Running · restarts: 2 · last heard under a minute ago");
  const old = page.getByTestId("session-p0");
  await expect(old).toContainText("Stopped · started");
  await expect(old).toContainText("ran out of disk");
  await expect(old.getByRole("button", { name: "Stop" })).toHaveCount(0);

  await live.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(page.getByText("Stop this passive training? It saves where it got to and does not start again.")).toBeVisible();
  await live.getByRole("button", { name: "Cancel" }).click();
  expect(calls).not.toContain("POST /passive/p1/stop");

  await live.getByRole("button", { name: "Stop", exact: true }).click();
  await live.getByRole("button", { name: "Yes, stop it" }).click();
  await expect(live).toContainText("Stopped · started");
  await expect(page.getByTestId("passive-note")).toBeVisible();
  expect(calls).toContain("POST /passive/p1/stop");
});

test("passive: the sign-in start can be removed", async ({ page }) => {
  await fakeEvents(page);
  let signin = true;
  const calls = await mockApi(page, {
    "GET /passive": () => ({ body: { sessions: [], signin } }),
    "GET /queue": () => ({ body: idle }),
    "DELETE /passive/signin": () => {
      signin = false;
      return { status: 204 };
    },
  });
  await page.goto("/passive/");
  await expect(page.getByText("No passive training on this computer.")).toBeVisible();
  await expect(page.getByText("The watchdog starts when you sign in, so passive training carries on after a restart.")).toBeVisible();
  await page.getByRole("button", { name: "Don't start when I sign in" }).click();
  await expect(page.getByText("Nothing starts when you sign in.")).toBeVisible();
  await expect(page.getByText("Removed. Passive training no longer starts when you sign in.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Don't start when I sign in" })).toHaveCount(0);
  expect(calls).toContain("DELETE /passive/signin");
});

test("passive: warns when a running session and a running queue job share the computer", async ({ page }) => {
  await fakeEvents(page);
  await mockApi(page, {
    "GET /passive": () => ({ body: { sessions: [running], signin: false } }),
    "GET /queue": () => ({ body: { ...idle, job: { id: "j1", kind: "pretrain" } } }),
  });
  await page.goto("/passive/");
  await expect(page.getByTestId("passive-warning")).toContainText("The queue is training too.");
});

test("passive: no warning when the queue is paused, and a failed read says so", async ({ page }) => {
  await fakeEvents(page);
  await mockApi(page, {
    "GET /passive": () => ({ body: { sessions: [running], signin: false } }),
    "GET /queue": () => ({ body: { ...idle, job: { id: "j1", kind: "pretrain" }, paused: true } }),
  });
  await page.goto("/passive/");
  await expect(page.getByTestId("session-p1")).toBeVisible();
  await expect(page.getByTestId("passive-warning")).toHaveCount(0);

  await page.unrouteAll();
  await mockApi(page, { "GET /passive": () => ({ status: 500, body: "boom" }) });
  await page.reload();
  await expect(page.getByTestId("passive-error")).toContainText("Could not read the passive sessions.", { timeout: 20_000 });
});

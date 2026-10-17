import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import StoragePage from "@/app/storage/page";
import PassivePage from "@/app/passive/page";
import QueuePage from "@/app/queue/page";
import { fetchJson } from "@/lib/api";
import { json, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";

// The contract tier: what the API answered over FakeServices (tests/web_fixtures.py records it into ./fixtures; `python -m tests.web_fixtures`
// rewrites it, and tests/test_web_fixtures.py fails CI while a fixture is stale) is replayed here through the client and the real pages.
// A response whose shape changes in a way the web app does not follow fails here, not in a browser.

type Fixture = { request: { method: string; path: string; body: unknown }; status: number; body: unknown };

const DIR = join(__dirname, "fixtures");
const fixtures: Record<string, Fixture> = Object.fromEntries(
  readdirSync(DIR)
    .filter((name) => name.endsWith(".json"))
    .map((name) => [name.replace(/\.json$/, ""), JSON.parse(readFileSync(join(DIR, name), "utf8")) as Fixture]),
);

/** The fixtures as handlers: "METHOD /path" with `{job1}` standing for any one path segment. */
function replay(names: string[], extra: Parameters<typeof mockApi>[0] = {}) {
  const handlers: Parameters<typeof mockApi>[0] = {};
  for (const name of names) {
    const { request, status, body } = fixtures[name];
    handlers[`${request.method} ${request.path.replace(/\{[^}]+\}/g, "*")}`] = () => json(body, status);
  }
  return mockApi({ ...handlers, ...extra });
}

describe("recorded API responses", () => {
  test("there are fixtures, each a request and its reply", () => {
    expect(Object.keys(fixtures).length).toBeGreaterThan(20);
    for (const [name, fixture] of Object.entries(fixtures)) {
      expect(fixture.request.method, name).toMatch(/^(GET|POST|PUT|DELETE)$/);
      expect(fixture.status, name).toBeGreaterThanOrEqual(200);
    }
  });

  test.each(Object.entries(fixtures).filter(([, f]) => f.request.method === "GET" && !f.request.path.includes("{")))(
    "%s: the client reads the reply it was recorded with",
    async (name, fixture) => {
      replay([name]);
      if (fixture.status >= 400) {
        await expect(fetchJson(fixture.request.path)).rejects.toMatchObject({ status: fixture.status });
      } else {
        expect(await fetchJson(fixture.request.path)).toEqual(fixture.body);
      }
    },
  );
});

describe("pages on recorded responses", () => {
  test("Queue lists the jobs the API reported, with the plan it reported", async () => {
    replay(["queue-two-waiting", "queue-job-outlook", "queue-outlook", "cluster-merged"]);
    renderPage(<QueuePage />);
    const state = fixtures["queue-two-waiting"].body as { up_next: { label: string }[] };
    for (const job of state.up_next) expect(await screen.findByText(job.label)).toBeVisible();
  });

  test("Queue with nothing queued says so", async () => {
    replay(["queue-empty", "queue-outlook", "cluster-merged"]);
    renderPage(<QueuePage />);
    expect(await screen.findByRole("heading", { level: 1, name: /queue/i })).toBeVisible();
    expect(screen.queryByText("Version 1 pretraining")).toBeNull();
  });

  test("Passive training reads an empty session list", async () => {
    replay(["passive", "queue-empty"]);
    renderPage(<PassivePage />);
    expect(await screen.findByRole("heading", { level: 1, name: "Passive training" })).toBeVisible();
  });

  test("Storage reads the free space and the use", async () => {
    replay(["storage", "storage-free", "storage-exports"]);
    renderPage(<StoragePage />, { route: "/storage/" });
    expect(await screen.findByRole("heading", { name: "Storage" })).toBeVisible();
    expect(await screen.findByText(/free on this computer's disk/)).toBeVisible();
  });
});

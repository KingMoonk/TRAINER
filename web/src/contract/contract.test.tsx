import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import TrainingPage from "@/app/cluster/training/page";
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

  test("Passive training lists the session the API started, and Resume puts a stopped one back in the queue", async () => {
    const started = fixtures["passive-started"].body as { sessions: { id: string; status: string }[] };
    expect(fixtures["passive-start"].body).toEqual({ id: started.sessions[0].id, status: "queued" });
    expect(fixtures["passive-resume"].body).toEqual({ status: "queued" });
    replay(["passive-started", "queue-empty"]);
    renderPage(<PassivePage />);
    const row = await screen.findByTestId(`session-${started.sessions[0].id}`);
    expect(row).toHaveTextContent("Starting · started");
  });

  test("Passive training opens the detail the API recorded for a session", async () => {
    const started = fixtures["passive-started"].body as { sessions: { id: string }[] };
    const id = started.sessions[0].id;
    replay(["passive-started", "queue-empty", "passive-detail", "passive-lessons"]);
    renderPage(<PassivePage />);
    await userEvent.setup().click(within(await screen.findByTestId(`session-${id}`)).getByRole("button", { name: "Details" }));
    expect(await screen.findByTestId("passive-facts")).toBeVisible();
    expect(screen.getByText("Nothing has been written yet.")).toBeVisible();
    expect(await screen.findByText("The teacher has not given this session any lessons.")).toBeVisible();
  });

  test("the traffic console lists the alert rules the API shipped, each with a switch", async () => {
    replay(["traffic-rules"]);
    renderPage(<main />, { shell: true });
    await userEvent.setup().click(screen.getAllByRole("button", { name: "API traffic console" })[0]);
    await userEvent.setup().click(await screen.findByRole("button", { name: "Alert rules" }));
    const rules = (fixtures["traffic-rules"].body as { rules: { label: string; enabled: boolean }[] }).rules;
    expect(rules.length).toBeGreaterThan(5);
    for (const rule of rules) expect(await screen.findByRole("switch", { name: rule.label })).toBeVisible();
  });

  test("Storage reads the free space and the use", async () => {
    replay(["storage", "storage-free", "storage-exports"]);
    renderPage(<StoragePage />, { route: "/storage/" });
    expect(await screen.findByRole("heading", { name: "Storage" })).toBeVisible();
    expect(await screen.findByText(/free on this computer's disk/)).toBeVisible();
  });
});

describe("the Training tab's live view on the recorded demo run", () => {
  // The run `studio_api --fake` opens on, read at three moments of its first rounds (tests/web_fixtures.py DEMO_MOMENTS). The page reads the list,
  // so each fixture is served as a list of one, with the rest of what the page reads recorded or empty.
  const moment = (name: string) => {
    const body = fixtures[name].body;
    replay(["distributed-defaults", "pretraining-text-sets", "versions", "cluster-nodes"], { "GET /distributed/runs": () => json([body]) });
    return body as {
      round: number; rounds: number; deadline_s: number | null; contributed: number; expected: number;
      members: { node: string; device: string; step_in_round: number }[]; timeline: { round: number; lanes: { node: string }[] }[]; events: { kind: string; text: string }[];
    };
  };

  test("while the members train: one CUDA and one MPS machine, each with progress, and a lane for each in the open round", async () => {
    const run = moment("distributed-run-demo-training");
    renderPage(<TrainingPage />, { route: "/cluster/training/" });
    expect(await screen.findByTestId("round-title")).toHaveTextContent(`Round ${run.round + 1} of ${run.rounds}`);
    expect(screen.getByTestId("stage-training")).toHaveAttribute("aria-current", "step");
    expect(screen.getByTestId("countdown")).toHaveTextContent(`${Math.ceil(run.deadline_s as number)} s to the deadline`);
    expect(run.members.map((m) => m.device).sort()).toEqual(["cuda", "mps"]);
    for (const m of run.members) {
      const node = within(screen.getByTestId(`node-${m.node}`));
      expect(node.getByTestId(`device-${m.node}`)).toHaveTextContent(m.device.toUpperCase());
      expect(node.getByRole("progressbar")).toHaveAttribute("aria-valuenow", String(m.step_in_round));
    }
    expect(screen.getByTestId("lane-1-mac-mini")).toBeInTheDocument();
    expect(screen.getByTestId("lane-1-gpu-box")).toBeInTheDocument();
  });

  test("while changes are collected: how many are in, and the finished machine's upload edge", async () => {
    const run = moment("distributed-run-demo-collecting");
    renderPage(<TrainingPage />, { route: "/cluster/training/" });
    expect(await screen.findByTestId("stage-collecting")).toHaveTextContent(`(${run.contributed} of ${run.expected} in)`);
    expect(screen.getByTestId("stage-collecting")).toHaveAttribute("aria-current", "step");
    expect(screen.getByTestId("edge-up-gpu-box")).toHaveTextContent("float16");
  });

  test("in the second round: the first is on the timeline, closed, and its outer step and checkpoint are in the log", async () => {
    const run = moment("distributed-run-demo-second-round");
    renderPage(<TrainingPage />, { route: "/cluster/training/" });
    expect(await screen.findByTestId("round-title")).toHaveTextContent(`Round ${run.round + 1} of ${run.rounds}`);
    expect(screen.getByTestId("timeline-round-1")).toHaveAttribute("data-open", "false");
    expect(screen.getByTestId("timeline-round-2")).toHaveAttribute("data-open", "true");
    const kinds = within(screen.getByRole("region", { name: "Events" })).getAllByRole("listitem").map((li) => li.getAttribute("data-kind"));
    expect(kinds).toEqual(run.events.map((e) => e.kind).reverse());
    expect(kinds.slice(0, 2)).toEqual(["checkpoint", "outer_step"]);
  });
});

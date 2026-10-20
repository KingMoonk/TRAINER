import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import CollectPage from "./page";
import { config as baseConfig } from "@/test/collect";
import type { Config, Options, State } from "./logic";

// The Collect page against a mocked API (the pure helpers are in logic.test.ts).

const options: Options = {
  topics: [
    { id: "science", label: "Science" },
    { id: "code", label: "Code" },
    { id: "law", label: "Law" },
  ],
  sources: [
    { id: "hub", label: "Hugging Face datasets", what: "Datasets found by topic.", terms: "Each dataset's own licence.", note: "" },
    { id: "stackexchange", label: "Stack Exchange dumps", what: "Official dumps.", terms: "CC BY-SA, attribution kept.", note: "Reading a .7z dump needs 7-Zip." },
    { id: "github", label: "GitHub (permissive repositories)", what: "Code and READMEs.", terms: "Only permissive licences.", note: "" },
    { id: "web", label: "Allowed websites", what: "Pages from sites you list.", terms: "The licence you read in the site's terms.", note: "" },
  ],
  excluded: [
    { host: "x.com", reason: "X forbids scraping and use of its content for machine learning without a licence." },
    { host: "quora.com", reason: "Quora forbids scraping and use of its content for machine learning without a licence." },
    { host: "reddit.com", reason: "Reddit forbids scraping and use of its content for machine learning without a licence." },
  ],
  licences: [
    { id: "permissive", label: "Permissive", default: true, opt_in: "" },
    { id: "unknown", label: "Unknown or not stated", default: false, opt_in: "unknown" },
  ],
  user_agent: "GroundworkTrainer-Collector/test (+https://example.test)",
};

const dropped = { licence: 4, quality: 7, pii: 1, duplicate: 30, near_duplicate: 9, decontam: 2, invalid: 0 };
const state = (patch: Partial<State> = {}): State => ({
  enabled: true,
  paused: false,
  running: true,
  status: "running",
  reason: "",
  heartbeat_age: 2,
  stats: { seen: 200, kept: 147, dropped, redactions: 12, bytes_kept: 5_000_000 },
  sets: [{ id: "collect_topic-science", group: "topic-science", name: "Collected · science", kept: 147, records: 1200, bytes: 5_000_000, seen: 200, redactions: 12, dropped }],
  plan: [
    { id: "acme/science-corpus", source: "hub", topic: "science", group: "topic-science", licence: "apache-2.0", licence_class: "permissive", bytes: 30e6, status: "planned", reason: "", score: 0.8, url: "u", files: 1, approved: false, blocked: false },
    { id: "big/science-dump", source: "hub", topic: "science", group: "topic-science", licence: "cc-by-4.0", licence_class: "permissive", bytes: 180e6, status: "needs_approval", reason: "180 MB to download: approve it to start", score: 0.7, url: "u", files: 2, approved: false, blocked: false },
    { id: "mystery/dump", source: "hub", topic: "science", group: "topic-science", licence: "other", licence_class: "unknown", bytes: 0, status: "skipped", reason: "licence other is unknown", score: 0, url: "u", files: 0, approved: false, blocked: false },
  ],
  sources: [{ key: "hub:acme/science-corpus", kind: "hub", name: "acme/science-corpus", topic: "science", status: "reading", detail: "file 1 of 3", downloaded: 0, done: false, error: "" }],
  disk_used: 400e6,
  disk_cap: 4096e6,
  bandwidth_today: 100e6,
  bandwidth_cap: 2048e6,
  bandwidth_total: 900e6,
  requests_today: { "huggingface.co": 12 },
  waiting_shards: 2,
  journal: [{ at: "2026-10-19T06:00:00+00:00", type: "plan", text: "Plan ready: 3 hub items" }],
  file: "/data/collect",
  ...patch,
});

function server(initial: { state?: State; config?: Config } = {}) {
  const live = { state: initial.state ?? state(), config: initial.config ?? baseConfig, put: [] as unknown[], posts: [] as string[], refuse: "" };
  mockApi({
    "GET /collect": () => ({ body: live.state }),
    "GET /collect/options": { body: options },
    "GET /collect/config": () => ({ body: live.config }),
    "PUT /collect/config": ({ body }) => {
      live.put.push(body);
      if (live.refuse) return { status: 400, body: { detail: live.refuse } };
      live.config = { ...live.config, ...(body as object) } as Config;
      return { body: live.config };
    },
    "POST /collect/pause": () => {
      live.posts.push("pause");
      live.config = { ...live.config, paused: true };
      live.state = state({ paused: true });
      return { body: live.config };
    },
    "POST /collect/resume": () => {
      live.posts.push("resume");
      live.config = { ...live.config, paused: false, enabled: true };
      live.state = state();
      return { body: live.config };
    },
  });
  return live;
}

const save = () => screen.getByRole("button", { name: "Save settings" });

describe("Collect page", () => {
  test("shows the topics and sources with their terms, and says which sites are never collected and why", async () => {
    server();
    renderPage(<CollectPage />, { route: "/collect/" });
    expect(await screen.findByRole("checkbox", { name: "Science" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Law" })).not.toBeChecked();
    expect(screen.getByText("CC BY-SA, attribution kept.")).toBeVisible();
    expect(screen.getByText("Reading a .7z dump needs 7-Zip.")).toBeVisible();
    const never = screen.getByText("NEVER COLLECTED").closest("div[data-slot='card']") as HTMLElement;
    for (const host of ["x.com", "quora.com", "reddit.com"]) expect(within(never).getByText(host)).toBeVisible();
    expect(within(never).getAllByText(/forbids scraping and use of its content for machine learning without a licence/)).toHaveLength(3);
  });

  test("topics and sources are chosen and saved", async () => {
    const live = server();
    const user = userEvent.setup();
    renderPage(<CollectPage />, { route: "/collect/" });
    await screen.findByRole("checkbox", { name: "Law" });
    expect(save()).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: "Law" }));
    await user.click(screen.getByRole("checkbox", { name: "Code" }));
    await user.click(screen.getByRole("checkbox", { name: /^Stack Exchange dumps/ }));
    await user.click(screen.getByRole("checkbox", { name: "Also keep: Unknown or not stated" }));
    expect(save()).toBeEnabled();
    await user.click(save());
    expect(await screen.findByText("Settings saved")).toBeVisible();
    const sent = live.put[0] as Config;
    expect(sent.topics).toEqual(["science", "law"]);
    expect(sent.sources).toEqual({ hub: true, stackexchange: true, github: false, web: false });
    expect(sent.licences.unknown).toBe(true);
    await waitFor(() => expect(save()).toBeDisabled());
  });

  test("limits are saved as numbers; a bad one is a sentence and sends nothing", async () => {
    const live = server();
    const user = userEvent.setup();
    renderPage(<CollectPage />, { route: "/collect/" });
    const disk = await screen.findByLabelText("Disk (MB)");
    await user.clear(disk);
    await user.type(disk, "lots");
    await user.click(save());
    expect(await screen.findByText("Disk (MB) must be a number.")).toBeVisible();
    expect(live.put).toEqual([]);
    await user.clear(disk);
    await user.type(disk, "512");
    await user.click(screen.getByRole("checkbox", { name: /^My network is not metered/ }));
    await user.click(save());
    await waitFor(() => expect(live.put).toHaveLength(1));
    const sent = live.put[0] as Config;
    expect(sent.limits.disk_mb).toBe(512);
    expect(sent.limits.bandwidth_mb_per_day).toBe(2048);
    expect(sent.idle.unmetered_confirmed).toBe(true);
  });

  test("the server's sentence is shown when it refuses a setting", async () => {
    const live = server();
    live.refuse = "old.reddit.com cannot be added. Reddit forbids scraping.";
    const user = userEvent.setup();
    renderPage(<CollectPage />, { route: "/collect/" });
    await user.click(await screen.findByRole("checkbox", { name: "Law" }));
    await user.click(save());
    expect(await screen.findByText("old.reddit.com cannot be added. Reddit forbids scraping.")).toBeVisible();
  });

  test("progress, what was kept and each reason for leaving records out", async () => {
    server();
    renderPage(<CollectPage />, { route: "/collect/" });
    expect(await screen.findByText("Collecting.")).toBeVisible();
    const drops = screen.getByLabelText("53 records left out");
    expect(within(drops).getByText("Duplicate").nextSibling).toHaveTextContent("30");
    expect(within(drops).getByText("Low quality").nextSibling).toHaveTextContent("7");
    expect(within(drops).getByText("Licence not allowed").nextSibling).toHaveTextContent("4");
    expect(within(drops).getByText("Overlaps held-out test text").nextSibling).toHaveTextContent("2");
    expect(screen.getByText("147")).toBeVisible();
    expect(screen.getByText(/Personal data and secrets removed from kept records: 12/)).toBeVisible();
    expect(screen.getByText("Collected · science")).toBeVisible();
    expect(screen.getByText("1,200 records in your data · 147 kept this time")).toBeVisible();
    expect(screen.getByText("2 finished batches wait to be added to your data.")).toBeVisible();
    expect(screen.getByText(/file 1 of 3/)).toBeVisible();
    expect(screen.getByText("Plan ready: 3 hub items")).toBeVisible();
  });

  test("the plan shows what was skipped and why, and a large download waits for approval", async () => {
    const live = server();
    const user = userEvent.setup();
    renderPage(<CollectPage />, { route: "/collect/" });
    expect(await screen.findByText("Skipped: licence other is unknown")).toBeVisible();
    expect(screen.getByText("Waiting for approval: 180 MB to download: approve it to start")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Approve big/science-dump" }));
    await waitFor(() => expect(live.put).toEqual([{ approved: ["big/science-dump"] }]));
    await user.click(screen.getByRole("button", { name: "Skip acme/science-corpus" }));
    await waitFor(() => expect(live.put[1]).toEqual({ blocked: ["acme/science-corpus"] }));
  });

  test("pause and resume call the API and the button follows", async () => {
    const live = server();
    const user = userEvent.setup();
    renderPage(<CollectPage />, { route: "/collect/" });
    await user.click(await screen.findByRole("button", { name: "Pause" }));
    await waitFor(() => expect(live.posts).toEqual(["pause"]));
    expect(await screen.findByText(/^Paused\./)).toBeVisible();
    await user.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(live.posts).toEqual(["pause", "resume"]));
    expect(await screen.findByText("Collecting.")).toBeVisible();
  });

  test("off: the button starts collecting; the waiting reason is shown while it waits", async () => {
    const live = server({ state: state({ enabled: false, running: false, status: "off" }), config: { ...baseConfig, enabled: false } });
    const user = userEvent.setup();
    renderPage(<CollectPage />, { route: "/collect/" });
    expect(await screen.findByText(/^Off\./)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Start collecting" }));
    await waitFor(() => expect(live.posts).toEqual(["resume"]));
  });

  test("waiting for mains power shows its reason, and a collect event refreshes the page", async () => {
    const live = server({ state: state({ status: "waiting", reason: "Waiting for mains power." }) });
    renderPage(<CollectPage />, { route: "/collect/" });
    expect(await screen.findByText("Waiting for mains power.")).toBeVisible();
    live.state = state();
    emit("collect", {});
    expect(await screen.findByText("Collecting.")).toBeVisible();
  });

  test("a website is added with the licence the person read", async () => {
    const live = server({ config: { ...baseConfig, sources: { ...baseConfig.sources, web: true }, web: { sites: [] } } });
    const user = userEvent.setup();
    renderPage(<CollectPage />, { route: "/collect/" });
    await user.click(await screen.findByRole("button", { name: "Add a site" }));
    await user.type(screen.getByLabelText("Site"), "docs.example.org");
    await user.type(screen.getByLabelText("Licence on its pages"), "cc-by-4.0");
    await user.click(save());
    await waitFor(() => expect(live.put).toHaveLength(1));
    expect((live.put[0] as Config).web.sites).toEqual([{ host: "docs.example.org", seeds: [], licence: "cc-by-4.0", pages: 200 }]);
  });

  test("an unreadable collector says so", async () => {
    mockApi({ "GET /collect": { status: 500, body: { detail: "x" } }, "GET /collect/options": { body: options }, "GET /collect/config": { body: baseConfig } });
    renderPage(<CollectPage />, { route: "/collect/" });
    expect(await screen.findByText("Could not read the collector's state.")).toBeVisible();
  });
});

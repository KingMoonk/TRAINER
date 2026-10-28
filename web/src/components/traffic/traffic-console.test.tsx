import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import type { Rule, TrafficRecord } from "@/lib/traffic";

let seq = 0;
function record(over: Partial<TrafficRecord> = {}): TrafficRecord {
  seq += 1;
  return {
    id: seq, ts: 1_700_000_000 + seq, node: "mac", direction: "in", remote: null, method: "GET", path: "/queue/jobs", route: "/queue/jobs", handler: "jobs",
    model: null, request_model: null, status: 200, ms: 12, state: "closed", streaming: false, error: "",
    request: { headers: {}, query: {}, body: null }, response: { headers: {}, body: null }, ...over,
  };
}

const RULES: Rule[] = [
  { id: "run.finished", label: "Run finished", severity: "success", enabled: true, source: "event", kind: "job", where: "status", equals: "done", new_min: false },
  { id: "api.5xx", label: "Server error (5xx)", severity: "destructive", enabled: true, source: "traffic", status: "5xx", new_min: false },
] as Rule[];

/** A /traffic response a test can push more frames into; `closed()` tells whether the page aborted the request. */
function stream(first: TrafficRecord[]) {
  const encoder = new TextEncoder();
  const frame = (kind: string, payload: unknown) => encoder.encode(`data: ${JSON.stringify({ kind, payload })}\n\n`);
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  let signal: AbortSignal | undefined;
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
      c.enqueue(frame("hello", { node: "mac", size: 2000, count: first.length, started: 1_700_000_000 }));
      for (const r of first) c.enqueue(frame("record", r));
    },
  });
  return {
    respond: (_call: unknown, request: Request) => {
      signal = request.signal;
      return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
    },
    push: (r: TrafficRecord) => controller.enqueue(frame("record", r)),
    closed: () => signal?.aborted === true,
  };
}

// the sidebar has one button for the phone bar and one for the desktop foot; CSS shows one of them
const openButton = () => screen.getAllByRole("button", { name: "API traffic console" })[0];
const console_ = () => screen.getByRole("region", { name: "API traffic" });

function setup(first: TrafficRecord[], extra: Record<string, unknown> = {}, developerTools = true) {
  const live = stream(first);
  const calls = mockApi({
    "GET /traffic": live.respond,
    "GET /traffic/rules": { body: { rules: RULES } },
    ...extra,
  });
  // the sidebar button is hidden until Developer tools is on; Ctrl+` works either way
  if (developerTools) localStorage.setItem("settings.developerTools", "on");
  renderPage(<main>page</main>, { shell: true });
  return { live, calls };
}

describe("the traffic console", () => {
  test("opens from the sidebar button, shows the backfill newest first, and closes the stream when closed", async () => {
    const first = [record({ path: "/a-old" }), record({ path: "/b-new", status: 404 })];
    const { live } = setup(first);
    const user = userEvent.setup();
    expect(screen.queryByRole("region", { name: "API traffic" })).not.toBeInTheDocument();
    await user.click(openButton());
    await waitFor(() => expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(2));
    const paths = within(console_()).getAllByTestId("traffic-row").map((row) => row.textContent ?? "");
    expect(paths[0]).toContain("/b-new");
    expect(paths[1]).toContain("/a-old");
    expect(within(console_()).getByText("2 / 2000")).toBeInTheDocument();
    await user.click(within(console_()).getByRole("button", { name: "Close the traffic console" }));
    expect(screen.queryByRole("region", { name: "API traffic" })).not.toBeInTheDocument();
    await waitFor(() => expect(live.closed()).toBe(true));
  });

  test("Ctrl+` toggles it from anywhere", async () => {
    setup([]);
    const user = userEvent.setup();
    await user.keyboard("{Control>}`{/Control}");
    expect(console_()).toBeInTheDocument();
    await user.keyboard("{Control>}`{/Control}");
    expect(screen.queryByRole("region", { name: "API traffic" })).not.toBeInTheDocument();
  });

  test("Ctrl+` works with Developer tools off, when the sidebar has no button", async () => {
    setup([], {}, false);
    const user = userEvent.setup();
    expect(screen.queryByRole("button", { name: "API traffic console" })).not.toBeInTheDocument();
    await user.keyboard("{Control>}`{/Control}");
    expect(console_()).toBeInTheDocument();
  });

  test("shows new records as they arrive, hides noise, and filters by status and search", async () => {
    const { live } = setup([record({ path: "/health" }), record({ path: "/queue/jobs" })]);
    const user = userEvent.setup();
    await user.click(openButton());
    await waitFor(() => expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(1));
    act(() => live.push(record({ path: "/collective/run1/join", method: "POST", status: 503, remote: "BEAST" })));
    await waitFor(() => expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(2));
    await user.click(within(console_()).getByRole("button", { name: "5xx" }));
    expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(1);
    expect(within(console_()).getByTestId("traffic-row")).toHaveTextContent("/collective/run1/join");
    await user.click(within(console_()).getByRole("button", { name: "All" }));
    await user.type(within(console_()).getByRole("textbox", { name: "Search" }), "beast");
    expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(1);
    await user.click(within(console_()).getByRole("switch", { name: /Hide health/ }));
    await user.clear(within(console_()).getByRole("textbox", { name: "Search" }));
    expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(3);
  });

  test("pause holds new records back until resumed; clear empties the list", async () => {
    const { live } = setup([record({ path: "/one" })]);
    const user = userEvent.setup();
    await user.click(openButton());
    await waitFor(() => expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(1));
    await user.click(within(console_()).getByRole("button", { name: "Pause" }));
    act(() => live.push(record({ path: "/two" })));
    await waitFor(() => expect(within(console_()).getByRole("button", { name: /Resume \(1 held\)/ })).toBeInTheDocument());
    expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(1);
    await user.click(within(console_()).getByRole("button", { name: /Resume/ }));
    expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(2);
    await user.click(within(console_()).getByRole("button", { name: "Clear" }));
    expect(within(console_()).queryAllByTestId("traffic-row")).toHaveLength(0);
  });

  test("expanding a row lays the body out by field with its schema type, and flags a body that does not fit", async () => {
    const body = { type: "application/json", bytes: 30, json: { lr: "fast", steps: 4 } };
    const { calls } = setup([record({ method: "POST", path: "/jobs", route: "/jobs", request_model: "JobRequest", request: { headers: {}, query: {}, body } })], {
      "GET /openapi.json": {
        body: {
          paths: { "/jobs": { post: { operationId: "create_job_jobs_post" } } },
          components: { schemas: { JobRequest: { type: "object", required: ["lr"], properties: { lr: { type: "number" }, steps: { type: "integer" } } } } },
        },
      },
    });
    const user = userEvent.setup();
    await user.click(openButton());
    await user.click(await screen.findByRole("button", { name: "POST /jobs" }));
    expect(await screen.findByText("create_job_jobs_post")).toBeInTheDocument();
    expect(screen.getByText("number")).toBeInTheDocument();
    expect(screen.getByText("integer")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("Does not match JobRequest: lr: expected number, got string");
    expect(calls.some((c) => c.path === "/openapi.json")).toBe(true);
    expect(screen.getByRole("button", { name: "Copy as curl" })).toBeInTheDocument();
  });

  test("an event rule raises an alert with the console closed, and a traffic rule raises one while it is open", async () => {
    const { live } = setup([]);
    const user = userEvent.setup();
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });  // the rules and the /events stream settle
    act(() => emit("job", { id: "read-1", kind: "design.notebook", status: "done", progress: "", error: "" }));
    expect(screen.queryByText("Run finished")).not.toBeInTheDocument();
    act(() => emit("job", { status: "done", job: "j1" }));
    expect(await screen.findByText("Run finished")).toBeInTheDocument();  // the toast
    await user.click(openButton());
    const aside = await screen.findByRole("complementary", { name: "Alerts and rules" });
    expect(within(aside).getByText("Run finished")).toBeInTheDocument();
    act(() => live.push(record({ method: "POST", path: "/x", status: 500 })));
    expect(await within(aside).findByText("Server error (5xx)")).toBeInTheDocument();
  });

  test("the rules list toggles a rule and saves the whole set", async () => {
    const { calls } = setup([], { "PUT /traffic/rules": (call: { body: unknown }) => ({ body: call.body }) });
    const user = userEvent.setup();
    await user.click(openButton());
    await user.click(within(console_()).getByRole("button", { name: "Alert rules" }));
    await user.click(await screen.findByRole("switch", { name: "Run finished" }));
    await waitFor(() => expect(calls.find((c) => c.method === "PUT")).toBeTruthy());
    const put = calls.find((c) => c.method === "PUT")!.body as { rules: Rule[] };
    expect(put.rules.find((r) => r.id === "run.finished")?.enabled).toBe(false);
    expect(put.rules.find((r) => r.id === "api.5xx")?.enabled).toBe(true);
  });

  test("the panel paints an opaque theme surface and isolates its stacking, so the page beneath never shows through", async () => {
    setup([record({ path: "/one" })]);
    const user = userEvent.setup();
    await user.click(openButton());
    await waitFor(() => expect(within(console_()).getAllByTestId("traffic-row")).toHaveLength(1));
    expect(console_()).toHaveClass("bg-popover", "text-popover-foreground", "isolate", "z-30");
    expect(console_()).not.toHaveClass("bg-card");
  });
});

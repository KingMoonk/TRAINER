import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { emit, eventsUnreachable, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import Page from "./page";

// The Trace page against mocked job endpoints (the component tier of tests/trace-value-map.spec.ts, which keeps the 375 px layout check).
// The result reaches the page the way it does against the real server: on /events, or from GET /jobs/{id}.

const job = (kind: string, status: string, result: unknown = null, error = "") => ({
  id: "j1",
  kind,
  status,
  progress: "",
  fraction: null,
  result,
  error,
  created_at: 0,
  started_at: null,
  finished_at: null,
});

const TRACE = {
  tokens: ["The", " largest", " planet", " in", " our", " solar", " system", " is"],
  pos: 7,
  rows: [
    { stage: "embedding", kind: "embedding", norm: 1.0, rel_change: null, cosine: null, top_id: 1, top_token: " the", top_p: 0.1, approximate: true },
    { stage: "block 1 mixer", kind: "mixer", norm: 2.0, rel_change: 0.5, cosine: 0.9, top_id: 2, top_token: " big", top_p: 0.2, approximate: true },
    { stage: "block 1 channel", kind: "channel", norm: 3.0, rel_change: 0.25, cosine: 0.8, top_id: 3, top_token: " one", top_p: 0.3, approximate: true },
    { stage: "output", kind: "output", norm: 4.0, rel_change: 0.125, cosine: 0.7, top_id: 4, top_token: " Jupiter", top_p: 0.91, approximate: false },
  ],
};

const done = () => ({ status: 202, body: job("inspect.trace", "done", TRACE) });
const main = () => within(screen.getByRole("main"));

describe("Trace", () => {
  test("carries the linked version into the inspection request", async () => {
    const calls = mockApi({ "POST /inspect/trace": done });
    renderPage(<Page />, { route: "/trace?version=7" });
    await waitFor(() => expect(calls.find((call) => call.path === "/inspect/trace")?.body).toMatchObject({ version: 7 }));
  });

  test("shows a skeleton, then the result that arrives on /events", async () => {
    let body: unknown;
    mockApi({
      "POST /inspect/trace": (call) => {
        body = call.body;
        setTimeout(() => emit("job", job("inspect.trace", "done", TRACE)), 100);
        return { status: 202, body: job("inspect.trace", "queued") };
      },
    });
    renderPage(<Page />, { route: "/trace" });
    expect(await screen.findByRole("status", { name: "Loading the trace" })).toBeVisible();

    expect(await screen.findByRole("row", { name: /Embedding/ })).toBeVisible();
    expect(screen.queryByRole("status", { name: "Loading the trace" })).toBeNull();
    expect(body).toEqual({ text: "The largest planet in our solar system is", pos: -1 });
    expect(screen.getByLabelText("Tokens").textContent).toBe("The largest planet in our solar system is");
    // embedding has no "before": its change and cosine are dashes; every early guess carries the approximation mark
    expect(screen.getByRole("row", { name: /Embedding/ })).toHaveTextContent("—");
    expect(screen.getByRole("row", { name: /block 1 mixer/ })).toHaveTextContent(/“ big”\s+0\.20 ≈/);
    expect(screen.getByRole("row", { name: /Output/ })).toHaveTextContent(/“ Jupiter”\s+0\.91/);
    expect(screen.getByRole("row", { name: /Output/ })).not.toHaveTextContent("≈");
    expect(screen.getAllByRole("row")).toHaveLength(1 + 4);
    expect(screen.getByText(/Only the output row is the model's real prediction\./)).toBeVisible();
  });

  test("reads GET /jobs/{id} when the event stream is down", async () => {
    eventsUnreachable();
    mockApi({
      "POST /inspect/trace": { status: 202, body: job("inspect.trace", "running") },
      "GET /jobs/j1": { body: job("inspect.trace", "done", TRACE) },
    });
    renderPage(<Page />, { route: "/trace" });
    expect(await screen.findByRole("row", { name: /Output/ })).toHaveTextContent(/“ Jupiter”\s+0\.91/);
  });

  test("a new prompt runs a new trace", async () => {
    const user = userEvent.setup();
    const calls = mockApi({ "POST /inspect/trace": done });
    renderPage(<Page />, { route: "/trace" });
    expect(await screen.findByRole("row", { name: /Output/ })).toBeVisible();
    await user.clear(screen.getByLabelText("Prompt"));
    await user.type(screen.getByLabelText("Prompt"), "Hello there");
    await user.click(screen.getByRole("button", { name: "Trace" }));
    await waitFor(() =>
      expect(calls.filter((call) => call.path === "/inspect/trace").map((call) => (call.body as { text: string }).text)).toEqual([
        "The largest planet in our solar system is",
        "Hello there",
      ]),
    );
  });

  test("a failed job says why, and an engine that cannot run says numbers appear later", async () => {
    const user = userEvent.setup();
    let calls = 0;
    mockApi({
      "POST /inspect/trace": () => ({
        status: 202,
        body: job("inspect.trace", "failed", null, ++calls === 1 ? "NotImplementedError: " : "ValueError: That prompt is empty."),
      }),
    });
    renderPage(<Page />, { route: "/trace" });
    expect(await main().findByRole("alert")).toHaveTextContent("Numbers appear when the engine has run your prompt.");
    await user.clear(screen.getByLabelText("Prompt"));
    await user.type(screen.getByLabelText("Prompt"), "again");
    await user.click(screen.getByRole("button", { name: "Trace" }));
    await waitFor(() => expect(main().getByRole("alert")).toHaveTextContent("That prompt is empty."));
  });
});

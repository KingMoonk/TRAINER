import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { json, mockApi } from "@/test/network";
import { TOKEN_KEY } from "./token";

// api.ts keeps `refused` (a 401 on the event stream) for the life of the page, so every test loads a fresh copy of it.
async function load() {
  vi.resetModules();
  const mod = await import("./api");
  const { toast } = await import("sonner");
  return { ...mod, toast };
}

const sse = (...events: { kind: string; payload: unknown }[]) =>
  new Response(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""), { status: 200, headers: { "Content-Type": "text/event-stream" } });

/** An EventSource the test drives directly (the shared fake only delivers well-formed `emit`s). */
function recordingEventSource() {
  const made: { url: string; onmessage?: (m: { data: string }) => void; onopen?: () => void; onerror?: (e: Event) => void; closed: boolean }[] = [];
  const original = globalThis.EventSource;
  globalThis.EventSource = class {
    closed = false;
    onmessage?: (m: { data: string }) => void;
    onopen?: () => void;
    onerror?: (e: Event) => void;
    constructor(readonly url: string) {
      made.push(this);
    }
    close() {
      this.closed = true;
    }
  } as unknown as typeof EventSource;
  return { made, restore: () => (globalThis.EventSource = original) };
}

afterEach(() => vi.useRealTimers());

// With fake timers vi.waitFor itself advances the clock, which would blur the 3-second reconnect pause, so wait on real I/O turns instead.
async function until(condition: () => boolean) {
  for (let turn = 0; turn < 500 && !condition(); turn++) await new Promise((resolve) => setImmediate(resolve));
  expect(condition()).toBe(true);
}

describe("fetchJson", () => {
  test("parses the JSON reply and asks for JSON", async () => {
    const { fetchJson } = await load();
    let accept: string | null = null;
    mockApi({ "GET /things": (_, request) => ((accept = request.headers.get("Accept")), { body: [{ id: 1 }] }) });
    await expect(fetchJson<{ id: number }[]>("/things")).resolves.toEqual([{ id: 1 }]);
    expect(accept).toBe("application/json");
  });

  test("sets Content-Type only when there is a body", async () => {
    const { fetchJson } = await load();
    const types: (string | null)[] = [];
    mockApi({ "POST /things": (_, request) => (types.push(request.headers.get("Content-Type")), { body: {} }), "GET /things": (_, request) => (types.push(request.headers.get("Content-Type")), { body: {} }) });
    await fetchJson("/things", { method: "POST", body: JSON.stringify({ a: 1 }) });
    await fetchJson("/things");
    expect(types[0]).toBe("application/json");
    expect(types[1]).toBeNull();
  });

  test("a caller's header overrides the defaults", async () => {
    const { fetchJson } = await load();
    let accept: string | null = null;
    mockApi({ "GET /things": (_, request) => ((accept = request.headers.get("Accept")), { body: {} }) });
    await fetchJson("/things", { headers: { Accept: "application/x-custom" } });
    expect(accept).toBe("application/x-custom");
  });

  test("sends the stored token", async () => {
    const { fetchJson } = await load();
    localStorage.setItem(TOKEN_KEY, "secret");
    let token: string | null = null;
    mockApi({ "GET /things": (_, request) => ((token = request.headers.get("X-Groundwork-Token")), { body: {} }) });
    await fetchJson("/things");
    expect(token).toBe("secret");
  });

  test("a 204 gives undefined without reading a body", async () => {
    const { fetchJson } = await load();
    mockApi({ "DELETE /things/1": { status: 204 } });
    await expect(fetchJson("/things/1", { method: "DELETE" })).resolves.toBeUndefined();
  });

  test("a non-2xx status throws ApiError with the response text", async () => {
    const { fetchJson, ApiError } = await load();
    mockApi({ "GET /things": { status: 409, body: { detail: "No store" } } });
    const error = await fetchJson("/things").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 409, path: "/things", body: '{"detail":"No store"}' });
  });
});

describe("unwrap", () => {
  test("gives the data of a 2xx reply", async () => {
    const { unwrap } = await load();
    await expect(unwrap(Promise.resolve({ data: { a: 1 }, response: new Response(null, { status: 200 }) }))).resolves.toEqual({ a: 1 });
  });

  test("a non-2xx reply throws ApiError with a JSON error body", async () => {
    const { unwrap, ApiError } = await load();
    const response = new Response(null, { status: 422 });
    Object.defineProperty(response, "url", { value: "http://node:8787/design/read" });
    const error = await unwrap(Promise.resolve({ error: { detail: "bad spec" }, response })).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 422, path: "/design/read", body: '{"detail":"bad spec"}' });
  });

  test("a string error body is kept as it is", async () => {
    const { unwrap } = await load();
    const response = new Response(null, { status: 500 });
    await expect(unwrap(Promise.resolve({ error: "plain text", response }))).rejects.toMatchObject({ status: 500, body: "plain text" });
  });
});

describe("api (the typed client)", () => {
  test("calls the route, with the path parameter filled and the query serialised", async () => {
    const { api } = await load();
    const calls = mockApi({ "GET /data/problems/rows": { body: [] }, "GET /jobs/*": { body: { id: "j" } } });
    await api.GET("/jobs/{job_id}", { params: { path: { job_id: "j" } } });
    await api.GET("/data/{kind}/rows", { params: { path: { kind: "problems" }, query: { limit: 3 } } });
    expect(calls.map((c) => `${c.method} ${c.path}${c.search}`)).toEqual(["GET /jobs/j", "GET /data/problems/rows?limit=3"]);
  });

  test("adds the token header from storage on every request, and none without one", async () => {
    const { api } = await load();
    const seen: (string | null)[] = [];
    mockApi({ "GET /jobs/*": (_, request) => (seen.push(request.headers.get("X-Groundwork-Token")), { body: {} }) });
    await api.GET("/jobs/{job_id}", { params: { path: { job_id: "j" } } });
    localStorage.setItem(TOKEN_KEY, "secret");
    await api.GET("/jobs/{job_id}", { params: { path: { job_id: "j" } } });
    expect(seen).toEqual([null, "secret"]);
  });

  test("sends a JSON body", async () => {
    const { api } = await load();
    const calls = mockApi({ "POST /design/resize": { body: {} } });
    await api.POST("/design/resize", { body: { choices: {}, n: 2 } as never });
    expect(calls[0].body).toEqual({ choices: {}, n: 2 });
  });
});

describe("subscribe without a token (EventSource)", () => {
  let source: ReturnType<typeof recordingEventSource>;
  beforeEach(() => {
    source = recordingEventSource();
  });
  afterEach(() => source.restore());

  test("opens /events on the API origin", async () => {
    const { subscribe } = await load();
    subscribe(() => {});
    expect(source.made.map((s) => s.url)).toEqual(["http://localhost:8787/events"]);
  });

  test("delivers parsed events and drops a malformed frame without stopping", async () => {
    const { subscribe } = await load();
    const events: unknown[] = [];
    subscribe((e) => events.push(e));
    const [es] = source.made;
    es.onmessage?.({ data: "not json" });
    es.onmessage?.({ data: JSON.stringify({ kind: "status", payload: { n: 1 } }) });
    expect(events).toEqual([{ kind: "status", payload: { n: 1 } }]);
  });

  test("wires onOpen and onError only when given, and the returned function closes the source", async () => {
    const { subscribe } = await load();
    const onOpen = vi.fn();
    const onError = vi.fn();
    const close = subscribe(() => {}, onError, onOpen);
    const [es] = source.made;
    es.onopen?.();
    es.onerror?.(new Event("error"));
    expect(onOpen).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledOnce();
    close();
    expect(es.closed).toBe(true);

    subscribe(() => {});
    expect(source.made[1].onopen).toBeUndefined();
    expect(source.made[1].onerror).toBeUndefined();
  });

  test("a blank stored token still means a plain EventSource", async () => {
    const { subscribe } = await load();
    localStorage.setItem(TOKEN_KEY, "   ");
    subscribe(() => {});
    expect(source.made).toHaveLength(1);
  });
});

describe("subscribe with a token (fetch)", () => {
  beforeEach(() => localStorage.setItem(TOKEN_KEY, "secret"));

  test("reads /events with the token header instead of opening an EventSource", async () => {
    const { subscribe } = await load();
    const es = recordingEventSource();
    try {
      let token: string | null = null;
      mockApi({ "GET /events": (_, request) => ((token = request.headers.get("X-Groundwork-Token")), sse({ kind: "status", payload: 1 })) });
      const events: unknown[] = [];
      const onOpen = vi.fn();
      const close = subscribe((e) => events.push(e), undefined, onOpen);
      await vi.waitFor(() => expect(events).toEqual([{ kind: "status", payload: 1 }]));
      close();
      expect(token).toBe("secret");
      expect(onOpen).toHaveBeenCalled();
      expect(es.made).toHaveLength(0);
    } finally {
      es.restore();
    }
  });

  test("a stream that ends is reported through onError and reopened after the pause", async () => {
    const { subscribe } = await load();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const calls = mockApi({ "GET /events": () => sse({ kind: "a", payload: 1 }) });
    const onError = vi.fn();
    const events: string[] = [];
    const close = subscribe((e) => events.push(e.kind), onError);
    await until(() => onError.mock.calls.length === 1);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(2_999);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    await until(() => calls.length === 2);
    close();
    expect(events.length).toBeGreaterThanOrEqual(1);
  });

  test("a 5xx or network failure retries and never marks the token refused", async () => {
    const { subscribe, tokenRefused } = await load();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const calls = mockApi({ "GET /events": { status: 503, body: {} } });
    const onError = vi.fn();
    const close = subscribe(() => {}, onError);
    await until(() => onError.mock.calls.length === 1);
    await vi.advanceTimersByTimeAsync(3_000);
    await until(() => calls.length === 2);
    close();
    expect(tokenRefused()).toBe(false);
  });

  test("a 401 is reported once, raises one toast, and is not retried", async () => {
    const { subscribe, tokenRefused, toast } = await load();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const error = vi.spyOn(toast, "error").mockImplementation(() => "id");
    const calls = mockApi({ "GET /events": { status: 401, body: { detail: "bad token" } } });
    const onError = vi.fn();
    subscribe(() => {}, onError);
    await until(() => tokenRefused());
    await vi.advanceTimersByTimeAsync(10_000);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(calls).toHaveLength(1);
    expect(error).toHaveBeenCalledOnce();
    expect(error.mock.calls[0][0]).toMatch(/refused the stored token/);
    expect(error.mock.calls[0][1]).toMatchObject({ id: "token-refused" });
  });

  test("once refused, a later subscribe opens nothing and tells no one", async () => {
    const { subscribe, tokenRefused, toast } = await load();
    vi.spyOn(toast, "error").mockImplementation(() => "id");
    const calls = mockApi({ "GET /events": { status: 401, body: {} } });
    subscribe(() => {});
    await vi.waitFor(() => expect(tokenRefused()).toBe(true));
    const onError = vi.fn();
    const close = subscribe(() => {}, onError);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(calls).toHaveLength(1);
    expect(onError).not.toHaveBeenCalled();
    expect(() => close()).not.toThrow();
  });

  test("two streams refused together raise a single toast but both hear the error", async () => {
    const { subscribe, tokenRefused, toast } = await load();
    const error = vi.spyOn(toast, "error").mockImplementation(() => "id");
    mockApi({ "GET /events": { status: 401, body: {} } });
    const first = vi.fn();
    const second = vi.fn();
    subscribe(() => {}, first);
    subscribe(() => {}, second);
    await vi.waitFor(() => expect(first).toHaveBeenCalled());
    await vi.waitFor(() => expect(second).toHaveBeenCalled());
    expect(tokenRefused()).toBe(true);
    expect(error).toHaveBeenCalledOnce();
  });

  test("closing before a 401 arrives says nothing and does not mark the token refused", async () => {
    const { subscribe, tokenRefused, toast } = await load();
    const error = vi.spyOn(toast, "error").mockImplementation(() => "id");
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    const calls = mockApi({ "GET /events": async () => (await gate, json({ detail: "bad" }, 401)) });
    const onError = vi.fn();
    const close = subscribe(() => {}, onError);
    await vi.waitFor(() => expect(calls).toHaveLength(1));
    close();
    release();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(onError).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
    expect(tokenRefused()).toBe(false);
  });

  test("closing stops the reconnect loop", async () => {
    const { subscribe } = await load();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const calls = mockApi({ "GET /events": () => sse() });
    const onError = vi.fn();
    const close = subscribe(() => {}, onError);
    await until(() => onError.mock.calls.length === 1);
    close();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(calls).toHaveLength(1);
  });
});

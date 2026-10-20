import { describe, expect, test } from "vitest";
import {
  NO_FILTERS, RING, absoluteBase, alertDetail, bytes, curl, duration, filterChoices, isNoise, isSlow, matchesFilters, p95, pick, pushAlert, reqPerSecond,
  route, ruleFires, statusClass, upsert, type Rule, type RuleMemory, type TrafficRecord,
} from "./traffic";

let seq = 0;
function record(over: Partial<TrafficRecord> = {}): TrafficRecord {
  seq += 1;
  return {
    id: seq, ts: 1000 + seq, node: "mac", direction: "in", remote: null, method: "GET", path: "/queue/jobs", route: "/queue/jobs", handler: "jobs",
    model: null, request_model: null, status: 200, ms: 12, state: "closed", streaming: false, error: "",
    request: { headers: {}, query: {}, body: null }, response: { headers: {}, body: null }, ...over,
  };
}
const rule = (over: Partial<Rule>): Rule => ({ id: "r", label: "R", severity: "info", enabled: true, source: "traffic", new_min: false, ...over }) as Rule;
const waiting = { headers: {}, query: { wait: "30" }, body: null };

describe("statusClass", () => {
  test("groups by hundreds, failed for no answer and null while a stream is open", () => {
    expect(statusClass({ status: 204, state: "closed" })).toBe("2xx");
    expect(statusClass({ status: 308, state: "closed" })).toBe("3xx");
    expect(statusClass({ status: 409, state: "closed" })).toBe("4xx");
    expect(statusClass({ status: 503, state: "closed" })).toBe("5xx");
    expect(statusClass({ status: null, state: "closed" })).toBe("failed");
    expect(statusClass({ status: null, state: "open" })).toBeNull();
  });
});

describe("labels", () => {
  test("route reads from the caller to the callee", () => {
    expect(route({ direction: "in", node: "mac", remote: "BEAST" })).toBe("BEAST → mac");
    expect(route({ direction: "in", node: "mac", remote: null })).toBe("browser → mac");
    expect(route({ direction: "out", node: "mac", remote: "BEAST" })).toBe("mac → BEAST");
  });
  test("duration and bytes", () => {
    expect(duration(null)).toBe("open");
    expect(duration(212.4)).toBe("212 ms");
    expect(duration(1900)).toBe("1.9 s");
    expect(duration(11_200)).toBe("11 s");
    expect(bytes(900)).toBe("900 B");
    expect(bytes(206 * 1024 * 1024)).toBe("206 MB");
  });
});

describe("upsert", () => {
  test("replaces a record by id (a stream closing) and keeps order", () => {
    const open = record({ state: "open", status: null, ms: null });
    const other = record();
    const merged = upsert([open, other], [{ ...open, state: "closed", status: 200, ms: 5 }]);
    expect(merged.map((r) => r.id)).toEqual([open.id, other.id]);
    expect(merged[0].state).toBe("closed");
  });
  test("keeps the newest RING records", () => {
    const many = Array.from({ length: RING + 5 }, (_, i) => record({ id: 10_000 + i }));
    const merged = upsert([], many);
    expect(merged).toHaveLength(RING);
    expect(merged[0].id).toBe(10_005);
  });
});

describe("header numbers", () => {
  test("reqPerSecond counts the last ten seconds", () => {
    const rs = [record({ ts: 80 }), record({ ts: 95 }), record({ ts: 99 }), record({ ts: 100 })];
    expect(reqPerSecond(rs, 100)).toBeCloseTo(0.3);
  });
  test("p95 needs enough answered requests and skips streams and waits", () => {
    const fast = Array.from({ length: 19 }, (_, i) => record({ ms: i + 1 }));
    expect(p95(fast)).toBeNull();
    const rs = [...fast, record({ ms: 100 }), record({ ms: 30_000, request: waiting }), record({ streaming: true, ms: 60_000 })];
    expect(p95(rs)).toBe(19);
    expect(isSlow(record({ ms: 50 }), 19)).toBe(true);
    expect(isSlow(record({ ms: 30_000, request: waiting }), 19)).toBe(false);
    expect(isSlow(record({ ms: 5 }), null)).toBe(false);
  });
});

describe("filters", () => {
  const joinBody = { type: "application/json", bytes: 20, json: { node: "BEAST", lr: 0.003 } };
  const rs = [
    record({ method: "POST", path: "/collective/run1/join", status: 200, remote: "BEAST", request: { headers: {}, query: {}, body: joinBody } }),
    record({ method: "GET", path: "/health", status: 200 }),
    record({ method: "GET", path: "/events", status: 200, streaming: true }),
    record({ method: "GET", path: "/collective/run1/rounds/4", status: 200, request: waiting }),
    record({ method: "POST", path: "/queue/jobs", status: 503 }),
  ];
  const paths = (f: Partial<typeof NO_FILTERS>) => rs.filter((r) => matchesFilters(r, { ...NO_FILTERS, ...f })).map((r) => r.path);
  test("noise (health, the event stream, long-poll waits) is hidden by default and shown on request", () => {
    expect(rs.filter(isNoise)).toHaveLength(3);
    expect(paths({})).toEqual(["/collective/run1/join", "/queue/jobs"]);
    expect(paths({ hideNoise: false })).toHaveLength(5);
  });
  test("method, status class, node and path prefix", () => {
    expect(paths({ method: "POST" })).toEqual(["/collective/run1/join", "/queue/jobs"]);
    expect(paths({ status: "5xx" })).toEqual(["/queue/jobs"]);
    expect(paths({ node: "BEAST" })).toEqual(["/collective/run1/join"]);
    expect(paths({ prefix: "/collective" })).toEqual(["/collective/run1/join"]);
  });
  test("search reads the path, status and the JSON of the bodies, case-insensitively", () => {
    expect(paths({ search: "beast" })).toEqual(["/collective/run1/join"]);
    expect(paths({ search: "0.003" })).toEqual(["/collective/run1/join"]);
    expect(paths({ search: "503" })).toEqual(["/queue/jobs"]);
    expect(paths({ search: "  nothing here " })).toEqual([]);
  });
  test("choices list what is in the buffer", () => {
    expect(filterChoices(rs)).toEqual({ methods: ["GET", "POST"], nodes: ["BEAST", "mac"], prefixes: ["/collective", "/events", "/health", "/queue"] });
  });
});

describe("curl", () => {
  test("repeats the method, query and JSON body", () => {
    const body = { type: "application/json", bytes: 9, json: { name: "it's" } };
    const r = record({ method: "POST", path: "/queue/jobs", request: { headers: {}, query: { a: "b c" }, body } });
    const text = curl(r, "http://127.0.0.1:8787");
    expect(text).toContain("curl -X POST 'http://127.0.0.1:8787/queue/jobs?a=b%20c'");
    expect(text).toContain(`--data '{"name":"it'\\''s"}'`);
    expect(text).not.toContain("X-Groundwork-Token");
  });
  test("sends this browser's token when it has one", () => {
    expect(curl(record({ path: "/queue" }), "https://n.ts.net/api", "t0k")).toContain("-H 'X-Groundwork-Token: t0k'");
  });
  test("leaves a binary body out with a note", () => {
    const body = { type: "application/octet-stream", bytes: 4096, binary: true };
    const r = record({ method: "POST", request: { headers: {}, query: {}, body } });
    expect(curl(r, "")).toContain("# the binary application/octet-stream (4.0 KB) is not kept");
  });
});

describe("absoluteBase", () => {
  test("puts a relative API path on the page's origin", () => {
    expect(absoluteBase("/api", "https://n.ts.net/console?x=1")).toBe("https://n.ts.net/api");
    expect(absoluteBase("", "http://localhost:3000/console")).toBe("http://localhost:3000");
    expect(absoluteBase("http://127.0.0.1:8787", "http://localhost:3000/")).toBe("http://127.0.0.1:8787");
  });
});

describe("pick", () => {
  test("reads a dotted path and gives undefined for a missing one", () => {
    expect(pick({ a: { b: 3 } }, "a.b")).toBe(3);
    expect(pick({ a: 1 }, "a.b")).toBeUndefined();
    expect(pick(null, "a")).toBeUndefined();
  });
});

describe("ruleFires", () => {
  const memory = (): RuleMemory => new Map();
  const signal = (r: TrafficRecord) => ({ source: "traffic", record: r }) as const;
  test("a traffic rule matches method, route template and status class", () => {
    const join = rule({ method: "POST", path: "/collective/*/join", status: "2xx" });
    expect(ruleFires(join, signal(record({ method: "POST", path: "/collective/r1/join", route: "/collective/{run}/join" })), memory())).toBe(true);
    expect(ruleFires(join, signal(record({ method: "GET", path: "/collective/r1/join" })), memory())).toBe(false);
    expect(ruleFires(join, signal(record({ method: "POST", path: "/collective/r1/join", status: 409 })), memory())).toBe(false);
    expect(ruleFires(join, signal(record({ method: "POST", path: "/collective/r1/x/join", route: "/collective/{run}/x/join" })), memory())).toBe(false);
  });
  test("a stream still open does not fire, and a disabled rule never does", () => {
    expect(ruleFires(rule({}), signal(record({ state: "open", status: null })), memory())).toBe(false);
    expect(ruleFires(rule({ enabled: false }), signal(record()), memory())).toBe(false);
  });
  test("where/equals tests a field of the record", () => {
    const refused = rule({ status: "4xx", where: "response.body.json.detail", equals: "stale epoch" });
    const reply = (detail: string) => ({ headers: {}, body: { type: "application/json", bytes: 1, json: { detail } } });
    expect(ruleFires(refused, signal(record({ status: 409, response: reply("stale epoch") })), memory())).toBe(true);
    expect(ruleFires(refused, signal(record({ status: 409, response: reply("other") })), memory())).toBe(false);
  });
  test("an event rule matches the kind and a payload field, including a status class", () => {
    const done = rule({ source: "event", kind: "job", where: "status", equals: "done" });
    expect(ruleFires(done, { source: "event", kind: "job", payload: { status: "done" } }, memory())).toBe(true);
    expect(ruleFires(done, { source: "event", kind: "job", payload: { status: "running" } }, memory())).toBe(false);
    expect(ruleFires(done, { source: "event", kind: "train", payload: { status: "done" } }, memory())).toBe(false);
    expect(ruleFires(done, signal(record()), memory())).toBe(false);
    const err = rule({ source: "event", kind: "api_error", where: "status", equals: "5xx" });
    expect(ruleFires(err, { source: "event", kind: "api_error", payload: { status: 503 } }, memory())).toBe(true);
    expect(ruleFires(err, { source: "event", kind: "api_error", payload: { status: 404 } }, memory())).toBe(false);
  });
  test("Run finished ignores background reads but still announces completed queued runs", () => {
    const done = rule({ id: "run.finished", source: "event", kind: "job", where: "status", equals: "done" });
    const fire = (payload: unknown) => ruleFires(done, { source: "event", kind: "job", payload }, memory());
    expect(fire({ id: "read-1", kind: "design.notebook", status: "done", progress: "", error: "" })).toBe(false);
    expect(fire({ id: "read-2", kind: "design.guess-check", status: "done" })).toBe(false);
    expect(fire({ job: "run-1", status: "done", error: "" })).toBe(true);
    expect(fire({ job: "run-1", status: "failed", error: "Training failed" })).toBe(false);
    expect(fire({ job: "run-1", status: "running" })).toBe(false);
    expect(fire({ job: "", status: "done" })).toBe(false);
    // User-defined job rules retain their ordinary payload matching.
    expect(ruleFires({ ...done, id: "custom" }, { source: "event", kind: "job", payload: { id: "read-1", status: "done" } }, memory())).toBe(true);
  });
  test("new_min fires only when a value goes below the lowest seen, the first value just sets it", () => {
    const best = rule({ source: "event", kind: "train", where: "val_loss", new_min: true, id: "best" });
    const mem = memory();
    const fire = (v: unknown) => ruleFires(best, { source: "event", kind: "train", payload: { val_loss: v } }, mem);
    expect([fire(4.2), fire(4.5), fire(3.9), fire(3.9), fire("x"), fire(3.1)]).toEqual([false, false, true, false, false, true]);
  });
});

describe("alerts", () => {
  test("alertDetail names the call or the event", () => {
    const query = { round: "4" };
    const r = record({ method: "POST", path: "/collective/r/join", status: 409, ms: 30, remote: "BEAST", request: { headers: {}, query, body: null } });
    expect(alertDetail({ source: "traffic", record: r })).toBe("POST /collective/r/join · 409 · 30 ms · from BEAST · round 4");
    expect(alertDetail({ source: "event", kind: "job", payload: { status: "done", job: "j1", nested: {} } })).toBe("job · status done · job j1");
  });
  test("empty progress and error fields do not look like errors in a successful alert", () => {
    expect(alertDetail({ source: "event", kind: "job", payload: { job: "j1", status: "done", progress: " ", error: "", step: 0, stopped: false } }))
      .toBe("job · job j1 · status done · step 0 · stopped false");
    expect(alertDetail({ source: "event", kind: "job", payload: { job: "j1", status: "failed", error: "Out of memory" } }))
      .toContain("error Out of memory");
  });
  test("the same alert again bumps the earlier one instead of stacking", () => {
    const a = { rule: "x", label: "X", severity: "info" as const, detail: "d", at: 1 };
    let alerts = pushAlert([], a);
    alerts = pushAlert(alerts, { ...a, detail: "e", at: 2 });
    alerts = pushAlert(alerts, { ...a, at: 3 });
    expect(alerts.map((x) => [x.detail, x.count])).toEqual([["d", 2], ["e", 1]]);
    expect(pushAlert(alerts, { ...a, detail: "f" }, 2)).toHaveLength(2);
  });
});

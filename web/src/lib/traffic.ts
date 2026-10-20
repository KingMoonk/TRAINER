// The logic of the API traffic console (components/traffic): the record the server's GET /traffic sends, the filters and the
// browser-side search over the records in the buffer, the numbers in the header, "copy as curl", and the alert rule matcher.
// No React here: traffic-console.tsx holds the state, these are the pure parts. The record is studio_api/traffic.py's.

import type { components } from "@/lib/api-schema";

export type Rule = components["schemas"]["Rule"];
export type Severity = Rule["severity"];

/** What the server shows of a body (studio_api.traffic.body_view). `json` is redacted; a binary body is only its size and type. */
export type BodyView = {
  type: string;
  bytes: number;
  json?: unknown;
  form?: Record<string, unknown>;
  truncated?: boolean;
  binary?: boolean;
  invalid?: boolean;
};

export type TrafficRecord = {
  id: number;
  ts: number;
  node: string;
  direction: "in" | "out";
  remote: string | null;
  method: string;
  path: string;
  route: string | null;
  handler: string | null;
  model: string | null;
  request_model: string | null;
  status: number | null;
  ms: number | null;
  state: "open" | "closed";
  streaming: boolean;
  error: string;
  synthetic?: boolean;
  request: { headers: Record<string, string>; query: Record<string, unknown>; body: BodyView | null };
  response: { headers: Record<string, string>; body: BodyView | null };
};

/** The buffer the server keeps and the console mirrors (studio_api.traffic.RING). */
export const RING = 2000;
/** Bodies are searched up to this many characters, as the server keeps them (studio_api.traffic.BODY_LIMIT). */
export const SEARCH_BODY_LIMIT = 16 * 1024;
/** The window `reqPerSecond` counts over. */
export const RATE_WINDOW_S = 10;

/** Merge `incoming` into `records` by id (a stream's record arrives again when it closes), oldest first, at most RING. */
export function upsert(records: TrafficRecord[], incoming: TrafficRecord[]): TrafficRecord[] {
  if (!incoming.length) return records;
  const byId = new Map(records.map((r) => [r.id, r]));
  for (const record of incoming) byId.set(record.id, record);
  const merged = [...byId.values()].sort((a, b) => a.id - b.id);
  return merged.length > RING ? merged.slice(merged.length - RING) : merged;
}

// ---- status -------------------------------------------------------------------------------------------------------------------

export type StatusClass = "2xx" | "3xx" | "4xx" | "5xx" | "failed";

/** The class of a record's status: 2xx, 3xx, 4xx, 5xx, or "failed" for a call that got no answer. null while a stream is open. */
export function statusClass(record: Pick<TrafficRecord, "status" | "state">): StatusClass | null {
  if (record.status === null) return record.state === "open" ? null : "failed";
  return record.status >= 500 ? "5xx" : record.status >= 400 ? "4xx" : record.status >= 300 ? "3xx" : "2xx";
}

// ---- labels -------------------------------------------------------------------------------------------------------------------

/** "BEAST → Michaels-Mac" for a call to this node and "Michaels-Mac → BEAST" for one it made; "browser" when nobody names the caller. */
export function route(record: Pick<TrafficRecord, "direction" | "node" | "remote">): string {
  return record.direction === "out" ? `${record.node} → ${record.remote ?? "?"}` : `${record.remote ?? "browser"} → ${record.node}`;
}

/** "212 ms", "1.9 s", "11.2 s"; "open" for a stream still running. */
export function duration(ms: number | null): string {
  if (ms === null) return "open";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
}

export function clock(ts: number): string {
  const d = new Date(ts * 1000);
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

/** "206 MB" style size of a body. */
export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let value = n / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

// ---- header numbers -----------------------------------------------------------------------------------------------------------

/** Records per second over the RATE_WINDOW_S seconds before `now` (epoch seconds), counted by when they started. */
export function reqPerSecond(records: TrafficRecord[], now: number): number {
  const from = now - RATE_WINDOW_S;
  let n = 0;
  for (let i = records.length - 1; i >= 0 && records[i].ts >= from; i--) n += 1;
  return n / RATE_WINDOW_S;
}

/** Records that say nothing about speed: a stream's open record and a long-poll wait. */
function isWait(record: TrafficRecord): boolean {
  return record.streaming || "wait" in record.request.query;
}

/** The 95th-percentile duration (ms) of the answered requests in the buffer, waits left out; null with fewer than `min` of them. */
export function p95(records: TrafficRecord[], min = 20): number | null {
  const times = records.filter((r) => r.ms !== null && !isWait(r)).map((r) => r.ms as number).sort((a, b) => a - b);
  if (times.length < min) return null;
  return times[Math.min(times.length - 1, Math.ceil(times.length * 0.95) - 1)];
}

/** Whether a record is slower than the p95 (never a wait). */
export function isSlow(record: TrafficRecord, limit: number | null): boolean {
  return limit !== null && record.ms !== null && !isWait(record) && record.ms > limit;
}

// ---- filters and search -------------------------------------------------------------------------------------------------------

export type Filters = {
  method: string; // "all" or an HTTP method
  status: "all" | StatusClass;
  node: string; // "all", or a node name that is the record's node or its remote
  prefix: string; // "" or a path prefix such as /collective
  search: string;
  hideNoise: boolean;
};

export const NO_FILTERS: Filters = { method: "all", status: "all", node: "all", prefix: "", search: "", hideNoise: true };

/** The polls and waits that bury the rest: /health, the /events stream and any long-poll wait. */
export function isNoise(record: TrafficRecord): boolean {
  return record.path.startsWith("/health") || record.path === "/events" || "wait" in record.request.query;
}

const searchCache = new WeakMap<TrafficRecord, string>();

/**
 * What search looks at, lowercased: method, path, route, handler, models, node, remote, status and the JSON of the query and bodies,
 * each body cut at SEARCH_BODY_LIMIT. A binary body is its summary ("fp16 ... bytes"), which is all the server sends of it.
 */
export function searchText(record: TrafficRecord): string {
  const cached = searchCache.get(record);
  if (cached !== undefined) return cached;
  const part = (value: unknown) => (value === undefined || value === null ? "" : JSON.stringify(value).slice(0, SEARCH_BODY_LIMIT));
  const body = (view: BodyView | null) => (view ? [view.type, view.bytes, part(view.json), part(view.form)] : []);
  const text = [
    record.method, record.path, record.route, record.handler, record.model, record.request_model, record.node, record.remote,
    record.status, record.error, part(record.request.query), ...body(record.request.body), ...body(record.response.body),
  ].filter((x) => x !== null && x !== "").join(" ").toLowerCase();
  searchCache.set(record, text);
  return text;
}

export function matchesFilters(record: TrafficRecord, filters: Filters): boolean {
  if (filters.hideNoise && isNoise(record)) return false;
  if (filters.method !== "all" && record.method !== filters.method) return false;
  if (filters.status !== "all" && statusClass(record) !== filters.status) return false;
  if (filters.node !== "all" && record.node !== filters.node && record.remote !== filters.node) return false;
  if (filters.prefix && !record.path.startsWith(filters.prefix)) return false;
  const needle = filters.search.trim().toLowerCase();
  return !needle || searchText(record).includes(needle);
}

/** The distinct choices for the filters, from what is in the buffer: methods, nodes and first path segments. */
export function filterChoices(records: TrafficRecord[]): { methods: string[]; nodes: string[]; prefixes: string[] } {
  const methods = new Set<string>();
  const nodes = new Set<string>();
  const prefixes = new Set<string>();
  for (const record of records) {
    methods.add(record.method);
    nodes.add(record.node);
    if (record.remote) nodes.add(record.remote);
    const first = record.path.split("/")[1];
    if (first) prefixes.add(`/${first}`);
  }
  const sorted = (set: Set<string>) => [...set].sort();
  return { methods: sorted(methods), nodes: sorted(nodes), prefixes: sorted(prefixes) };
}

// ---- copy as curl -------------------------------------------------------------------------------------------------------------

const shell = (text: string) => `'${text.replace(/'/g, `'\\''`)}'`;

/**
 * A curl command for a record, against `base` (the API's absolute URL). Secrets in the record were masked by the server, so
 * the token header carries this browser's own token, and is left out when it has none (a node reached through
 * `tailscale serve` needs none). A JSON body is repeated as the server kept it; a binary or oversize body is left out, with
 * a comment.
 */
export function curl(record: TrafficRecord, base: string, token = ""): string {
  const query = Object.entries(record.request.query).flatMap(([key, value]) =>
    (Array.isArray(value) ? value : [value]).map((v) => `${encodeURIComponent(key)}=${encodeURIComponent(String(v))}`));
  const url = `${base}${record.path}${query.length ? `?${query.join("&")}` : ""}`;
  const lines = [`curl -X ${record.method} ${shell(url)}`];
  if (token) lines.push(`  -H ${shell(`X-Groundwork-Token: ${token}`)}`);
  const body = record.request.body;
  if (body?.json !== undefined) lines.push(`  -H 'Content-Type: application/json'`, `  --data ${shell(JSON.stringify(body.json))}`);
  else if (body) lines.push(`  # the ${body.binary ? "binary" : "oversize"} ${body.type || "body"} (${bytes(body.bytes)}) is not kept`);
  return lines.join(" \\\n");
}

// ---- alert rules ----------------------------------------------------------------------------------------------------------------

/** What a rule is tested against: a traffic record, or an event from /events. */
export type Signal = { source: "traffic"; record: TrafficRecord } | { source: "event"; kind: string; payload: unknown };

/** The value at a dotted path (`response.body.json.state`, `status`), or undefined. */
export function pick(value: unknown, path: string): unknown {
  let at = value;
  for (const key of path.split(".")) {
    if (at === null || typeof at !== "object") return undefined;
    at = (at as Record<string, unknown>)[key];
  }
  return at;
}

function globMatches(pattern: string, text: string): boolean {
  return new RegExp(`^${pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^/]+")}$`).test(text);
}

function sameValue(actual: unknown, expected: string | number | boolean): boolean {
  if (typeof expected === "string" && /^\dxx$/.test(expected) && typeof actual === "number") return Math.floor(actual / 100) === Number(expected[0]);
  return actual === expected || (actual !== undefined && actual !== null && typeof actual !== "object" && String(actual) === String(expected));
}

function statusMatches(pattern: string, status: number | null): boolean {
  if (status === null) return false;
  return /^\dxx$/.test(pattern) ? Math.floor(status / 100) === Number(pattern[0]) : String(status) === pattern;
}

/** The lowest number each `new_min` rule has seen; a rule fires when a value is below it (the first value sets it without firing). */
export type RuleMemory = Map<string, number>;

/** Whether `rule` fires on `signal`. An alert needs an enabled rule of the signal's source; every field the rule gives must match. */
export function ruleFires(rule: Rule, signal: Signal, memory: RuleMemory): boolean {
  if (!rule.enabled || rule.source !== signal.source) return false;
  let subject: unknown;
  if (signal.source === "traffic") {
    const { record } = signal;
    if (record.state !== "closed") return false;
    if (rule.method && rule.method !== record.method) return false;
    if (rule.path && !globMatches(rule.path, record.route ?? record.path) && !globMatches(rule.path, record.path)) return false;
    if (rule.status && !statusMatches(rule.status, record.status)) return false;
    subject = record;
  } else {
    if (rule.kind !== signal.kind) return false;
    subject = signal.payload;
    // Both queued runs and background API reads emit "job". The built-in run
    // alert (including saved copies of it) applies only to the queue's job id.
    if (rule.id === "run.finished" && signal.kind === "job") {
      const job = pick(subject, "job");
      if (typeof job !== "string" || !job.trim()) return false;
    }
  }
  if (rule.where === null || rule.where === undefined) return true;
  const actual = pick(subject, rule.where);
  if (rule.new_min) {
    if (typeof actual !== "number" || !Number.isFinite(actual)) return false;
    const lowest = memory.get(rule.id);
    memory.set(rule.id, lowest === undefined ? actual : Math.min(lowest, actual));
    return lowest !== undefined && actual < lowest;
  }
  return rule.equals === null || rule.equals === undefined ? actual !== undefined : sameValue(actual, rule.equals);
}

/** The second line of an alert: what happened, in a few words. */
export function alertDetail(signal: Signal): string {
  if (signal.source === "traffic") {
    const { record } = signal;
    const query = record.request.query;
    return [`${record.method} ${record.path}`, record.status ?? "no answer", duration(record.ms), record.direction === "out" ? `to ${record.remote}` : record.remote ? `from ${record.remote}` : "",
      query.round !== undefined ? `round ${String(query.round)}` : ""].filter((x) => x !== "").join(" · ");
  }
  const payload = signal.payload;
  if (payload && typeof payload === "object") {
    const fields = Object.entries(payload as Record<string, unknown>).filter(([, v]) =>
      typeof v === "string" ? v.trim().length > 0 : typeof v === "number" || typeof v === "boolean",
    ).slice(0, 5);
    return [signal.kind, ...fields.map(([k, v]) => `${k} ${String(v)}`)].join(" · ");
  }
  return signal.kind;
}

export type Alert = { key: string; rule: Rule["id"]; label: string; severity: Severity; detail: string; at: number; count: number };

/** Add an alert to the stack (newest first, at most `limit`); the same rule and detail again bumps the earlier one instead of stacking a copy. */
export function pushAlert(alerts: Alert[], next: Omit<Alert, "key" | "count">, limit = 50): Alert[] {
  const key = `${next.rule}|${next.detail}`;
  const earlier = alerts.find((a) => a.key === key);
  const added: Alert = { ...next, key, count: (earlier?.count ?? 0) + 1 };
  return [added, ...alerts.filter((a) => a.key !== key)].slice(0, limit);
}

/** The API's absolute base URL: `NEXT_PUBLIC_API_URL` when it is absolute, otherwise that path (or none) on the page's origin. */
export function absoluteBase(base: string, href: string): string {
  return new URL(base || "/", href).href.replace(/\/$/, "");
}

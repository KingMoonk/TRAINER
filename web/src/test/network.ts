// The fake network for component tests: `mockApi` answers fetches, `FakeEventSource` is the /events stream a test drives with `emit`.
// This is the Vitest counterpart of tests/qp-mock.ts (Playwright).

export type Call = { method: string; path: string; search: string; body: unknown };
export type Reply = { status?: number; body?: unknown; text?: string; headers?: Record<string, string> } | Response | undefined;
export type Handler = (call: Call, request: Request) => Reply | Promise<Reply>;

let current: (request: Request) => Promise<Response> = async () => new Response("no mock", { status: 404 });

export function stableFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const request = input instanceof Request && !init ? input : new Request(new URL(String(input instanceof Request ? input.url : input), "http://localhost"), init ?? (input instanceof Request ? input : undefined));
  return current(request);
}

/** A "METHOD /path" pattern; `*` matches one path segment (`DELETE /queue/jobs/*`). */
function matches(pattern: string, key: string): boolean {
  if (pattern === key) return true;
  if (!pattern.includes("*")) return false;
  return new RegExp(`^${pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^/]+")}$`).test(key);
}

/**
 * Answer the page's API calls. `handlers` maps "METHOD /path" (exact first, then `*` patterns) to a reply; a call with no
 * handler gets a 404 like the Playwright mock. A reply is `{ status?, body? }` (JSON) or a ready `Response`. Returns the calls made.
 */
export function mockApi(handlers: Record<string, Handler | Reply>): Call[] {
  const calls: Call[] = [];
  current = async (request) => {
    const url = new URL(request.url);
    const key = `${request.method} ${url.pathname.replace(/\/$/, "") || "/"}`;
    const raw = request.method === "GET" || request.method === "HEAD" ? "" : await request.clone().text();
    let body: unknown = raw || undefined;
    try {
      body = raw ? JSON.parse(raw) : undefined;
    } catch {
      // a body that is not JSON (an upload) is kept as text
    }
    const call: Call = { method: request.method, path: url.pathname, search: url.search, body };
    calls.push(call);
    const pattern = key in handlers ? key : Object.keys(handlers).find((p) => matches(p, key));
    const entry = pattern ? handlers[pattern] : undefined;
    const reply = typeof entry === "function" ? await entry(call, request) : entry;
    if (reply instanceof Response) return reply;
    if (!reply) return json({ detail: `no mock for ${key}` }, 404);
    if (reply.text !== undefined) return new Response(reply.text, { status: reply.status ?? 200, headers: reply.headers });
    if (reply.status === 204) return new Response(null, { status: 204 });
    return json(reply.body ?? {}, reply.status ?? 200);
  };
  return calls;
}

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const sources = new Set<FakeEventSource>();
let eventsDown = false;

class FakeEventSource {
  onmessage: ((m: MessageEvent<string>) => void) | null = null;
  onopen: (() => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  constructor() {
    sources.add(this);
    setTimeout(() => (eventsDown ? this.onerror?.(new Event("error")) : this.onopen?.()), 0);
  }
  close() {
    sources.delete(this);
  }
}
export const stableEventSource = FakeEventSource as unknown as typeof EventSource;

/** Deliver one /events message to every open stream. */
export function emit(kind: string, payload: unknown) {
  for (const source of [...sources]) source.onmessage?.({ data: JSON.stringify({ kind, payload }) } as MessageEvent<string>);
}

/** Drop every open /events stream, as a restarted server or a sleeping laptop does. */
export function dropEvents() {
  for (const source of [...sources]) source.onerror?.(new Event("error"));
}

/** Make new /events streams fail (the shell then reads Offline), as the Playwright static server does with its 404. */
export function eventsUnreachable(down = true) {
  eventsDown = down;
}

export function resetNetwork() {
  current = async () => new Response("no mock", { status: 404 });
  sources.clear();
  eventsDown = false;
}

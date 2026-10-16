import type { Page, Route } from "@playwright/test";

type Reply = { status?: number; body?: unknown } | undefined;
type Handler = (route: Route) => Reply | Promise<Reply>;

/**
 * Answer the page's API calls in a test (the static server has no API). `handlers` maps "METHOD /path" to a reply; a call
 * with no handler gets a 404. Only fetches are answered, so a navigation to a page of the same path still gets the page.
 */
export async function mockApi(page: Page, handlers: Record<string, Handler>) {
  const calls: string[] = [];
  await page.route(
    (url) => !/^\/(_next|events)/.test(url.pathname),
    async (route) => {
      const request = route.request();
      if (request.resourceType() !== "fetch") return route.fallback();
      const url = new URL(request.url());
      const key = `${request.method()} ${url.pathname}`;
      calls.push(key);
      const reply = await handlers[key]?.(route);
      if (!reply) return route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: `no mock for ${key}` }) });
      if (reply.status === 204) return route.fulfill({ status: 204 });
      return route.fulfill({ status: reply.status ?? 200, contentType: "application/json", body: JSON.stringify(reply.body ?? {}) });
    },
  );
  return calls;
}

/** Replace EventSource with one the test drives: `emit(page, kind, payload)` delivers a /events message to the page. */
export async function fakeEvents(page: Page) {
  await page.addInitScript(() => {
    const sources: { onmessage: ((m: MessageEvent<string>) => void) | null }[] = [];
    class FakeEventSource {
      onmessage: ((m: MessageEvent<string>) => void) | null = null;
      onopen: (() => void) | null = null;
      onerror: ((e: Event) => void) | null = null;
      constructor() {
        sources.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }
      close() {
        sources.splice(sources.indexOf(this), 1);
      }
    }
    (window as unknown as { EventSource: unknown }).EventSource = FakeEventSource;
    (window as unknown as { __emit: (e: unknown) => void }).__emit = (event) =>
      sources.forEach((s) => s.onmessage?.({ data: JSON.stringify(event) } as MessageEvent<string>));
  });
}

export function emit(page: Page, kind: string, payload: unknown) {
  return page.evaluate((event) => (window as unknown as { __emit: (e: unknown) => void }).__emit(event), { kind, payload });
}

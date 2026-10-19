import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { resetNav } from "./navigation";
import { resetNetwork, stableEventSource, stableFetch } from "./network";

// openapi-fetch captures `fetch` when `api` is created (at import), so the global is replaced once here by a function that
// calls whatever the current test installed with mockApi().
vi.stubGlobal("fetch", stableFetch);
vi.stubGlobal("EventSource", stableEventSource);

vi.mock("next/navigation", async () => {
  const { nav } = await import("./navigation");
  return {
    useSearchParams: () => nav.params,
    usePathname: () => nav.path,
    useRouter: () => ({ push: nav.push, replace: nav.replace, prefetch: () => {}, back: () => {}, refresh: () => {} }),
  };
});

// jsdom has no layout engine: the observers and media queries the UI libraries ask for are inert stand-ins.
class InertObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}
vi.stubGlobal("ResizeObserver", InertObserver);
vi.stubGlobal("IntersectionObserver", InertObserver);
window.matchMedia ??= ((query: string) => ({
  matches: false,
  media: query,
  onchange: null,
  addEventListener() {},
  removeEventListener() {},
  addListener() {},
  removeListener() {},
  dispatchEvent: () => false,
})) as typeof window.matchMedia;
Element.prototype.scrollIntoView ??= () => {};

beforeEach(() => {
  resetNav();
  resetNetwork();
});

afterEach(() => {
  cleanup();
  try {
    localStorage.clear();
  } catch {
    // a test that replaced localStorage owns its cleanup
  }
});

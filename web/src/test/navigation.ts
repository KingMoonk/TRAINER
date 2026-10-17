// The state behind the `next/navigation` mock in setup.ts. A test sets `nav.path` and `nav.params` before rendering and reads `nav.push` / `nav.replace` after.
import { vi } from "vitest";

export const nav = {
  path: "/",
  params: new URLSearchParams(),
  push: vi.fn<(href: string) => void>(),
  replace: vi.fn<(href: string) => void>(),
};

export function resetNav(path = "/", search = "") {
  nav.path = path;
  nav.params = new URLSearchParams(search);
  nav.push.mockReset();
  nav.replace.mockReset();
}

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderResult } from "@testing-library/react";
import type { ReactElement } from "react";
import { Sidebar } from "@/components/sidebar";
import { StatusStrip } from "@/components/status-strip";
import { Toaster } from "@/components/ui/sonner";
import { ShellEvents } from "@/lib/events";
import { nav } from "./navigation";

/**
 * Render a page the way app/layout.tsx does: query client, the /events stream (network.ts), toasts and, with `shell`, the
 * sidebar and status strip. `route` sets the pathname and `?query` that `next/navigation` reports. Queries do not retry
 * (a failing read shows its error at once) and nothing is cached between tests.
 */
export function renderPage(ui: ReactElement, { route = "/", shell = false }: { route?: string; shell?: boolean } = {}): RenderResult & { client: QueryClient } {
  const [path, search = ""] = route.split("?");
  nav.path = path;
  nav.params = new URLSearchParams(search);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 5_000 }, mutations: { retry: false } } });
  const result = render(
    <QueryClientProvider client={client}>
      <ShellEvents>
        {shell && <Sidebar />}
        {shell && <StatusStrip />}
        {ui}
        <Toaster />
      </ShellEvents>
    </QueryClientProvider>,
  );
  return Object.assign(result, { client });
}

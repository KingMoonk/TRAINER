import type { Route } from "@/lib/routes";

/** A page that exists but has no content yet (Phases 2 and 3 of docs/rewrite.md fill them in). */
export function PagePlaceholder({ route }: { route: Route }) {
  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-3 p-8">
      <h1 className="text-2xl font-semibold tracking-tight">{route.title}</h1>
      <p className="max-w-prose text-muted-foreground">{route.lead}</p>
      <p className="mt-6 rounded-md border border-dashed p-4 text-sm text-muted-foreground">
        This page is not built yet. The desktop app still has it.
      </p>
    </main>
  );
}

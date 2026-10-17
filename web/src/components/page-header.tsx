import type { ReactNode } from "react";

/** The one frame every page's <main> uses, so the width, padding and rhythm are the same on every page. */
export const PAGE_MAIN = "mx-auto flex w-full min-w-0 max-w-5xl flex-col gap-5 p-4 md:p-8";

/** The one page header: an optional breadcrumb, the page's only h1, an optional version line, the lead, and optional links. */
export function PageHeader({ title, lead, breadcrumb, meta, children }: { title: ReactNode; lead?: ReactNode; breadcrumb?: ReactNode; meta?: ReactNode; children?: ReactNode }) {
  return (
    <header className="flex flex-col gap-1.5">
      {breadcrumb && <div className="text-xs text-muted-foreground">{breadcrumb}</div>}
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {meta && <p className="text-sm text-muted-foreground">{meta}</p>}
      {lead && <p className="max-w-prose text-muted-foreground">{lead}</p>}
      {children}
    </header>
  );
}

"use client";

import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { AREAS, ROUTES } from "@/lib/routes";
import { SectionTabs } from "./section-tabs";

/** The one frame every page's <main> uses, so the width, padding and rhythm are the same on every page. */
export const PAGE_MAIN = "mx-auto flex w-full min-w-0 max-w-5xl flex-col gap-5 p-4 md:p-8";

/** The one page header: an optional eyebrow/breadcrumb, the page's only h1, an optional version line, the lead, optional actions, and section tabs. */
export function PageHeader({
  title,
  lead,
  breadcrumb,
  meta,
  actions,
  hideTabs,
  children,
}: {
  title: ReactNode;
  lead?: ReactNode;
  breadcrumb?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  hideTabs?: boolean;
  children?: ReactNode;
}) {
  const pathname = (usePathname() || "").replace(/\/$/, "");
  const current = ROUTES.find((route) => route.href === pathname);
  const area = AREAS.find((a) => a.id === current?.area);
  const eyebrow = breadcrumb ?? (area ? area.title : undefined);

  return (
    <header className="flex flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          {eyebrow && <div className="text-xs font-medium uppercase tracking-widest text-muted-foreground">{eyebrow}</div>}
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          {meta && <p className="text-sm text-muted-foreground">{meta}</p>}
          {lead && <p className="max-w-prose text-muted-foreground">{lead}</p>}
          {children}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>
      {!hideTabs && <SectionTabs />}
    </header>
  );
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Boxes, ChevronRight, Database, Layers3, ListOrdered, MessageSquare, Network, Settings, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { APPEARANCE } from "@/lib/copy/appearance";
import { AREAS, ROUTES, areaHome } from "@/lib/routes";
const gearClass = (active: boolean) => cn("flex size-9 shrink-0 items-center justify-center rounded-lg border border-transparent text-muted-foreground transition-colors hover:bg-surface-overlay hover:text-foreground", active && "border-surface-border bg-surface-raised text-foreground");
const icons = { designer: Network, pretraining: Layers3, post: Sparkles, chat: MessageSquare, queue: ListOrdered, resources: Database };

export function Sidebar() {
  const pathname = usePathname().replace(/\/$/, "");
  const current = ROUTES.find((route) => route.href === pathname);
  const pages = ROUTES.filter((route) => route.area === current?.area);
  return (
    <aside className="studio-glass z-20 flex w-full shrink-0 flex-col border-b border-sidebar-border md:sticky md:top-0 md:h-dvh md:w-60 md:border-r md:border-b-0">
      <Link href="/builder" className="flex items-center gap-3 px-5 py-5 md:py-7" aria-label="Groundwork Studio home">
        <span className="flex size-9 items-center justify-center rounded-xl border border-surface-border-strong bg-surface-overlay shadow-inner"><Boxes className="size-5" aria-hidden="true" /></span>
        <span className="text-sm font-semibold tracking-tight">Groundwork <span className="font-normal text-muted-foreground">Studio</span></span>
      </Link>
      <nav aria-label="Studio" className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-visible">
        {AREAS.map((area) => {
          const Icon = icons[area.id as keyof typeof icons];
          const selected = current?.area === area.id;
          return (
            <Link key={area.id} href={areaHome(area.id).href} aria-current={selected ? "page" : undefined}
              className={cn("flex shrink-0 items-center gap-3 rounded-lg border border-transparent px-3 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-surface-overlay hover:text-foreground", area.id === "queue" && "md:mt-5", selected && "border-surface-border bg-surface-raised font-medium text-foreground shadow-sm")}>
              <Icon className="size-4" aria-hidden="true" />{area.title}
              {selected && <ChevronRight className="ml-auto hidden size-3.5 text-muted-foreground md:block" aria-hidden="true" />}
            </Link>
          );
        })}
        {/* On a phone the sidebar is a bar across the top, so the gear joins the row; on a desktop it sits at the foot. */}
        <Link href="/settings" aria-current={pathname === "/settings" ? "page" : undefined} title={APPEARANCE.settingsLabel} aria-label={APPEARANCE.settingsLabel} className={cn(gearClass(pathname === "/settings"), "md:hidden")}>
          <Settings className="size-4" aria-hidden="true" />
        </Link>
      </nav>
      {pages.length > 1 && (
        <div role="group" aria-label="Pages in this area" className="flex gap-1 overflow-x-auto border-t border-surface-border px-3 py-2 md:mx-3 md:mt-3 md:flex-col md:gap-0.5 md:overflow-y-auto md:px-0 md:pt-4">
          <p className="hidden px-3 pb-2 text-[11px] font-medium tracking-widest text-muted-foreground uppercase md:block">Explore</p>
          {pages.map((route) => (
            <Link key={route.route} href={route.href} aria-current={pathname === route.href ? "location" : undefined}
              className={cn("shrink-0 rounded-md px-3 py-2 text-xs text-muted-foreground transition-colors hover:bg-surface-overlay hover:text-foreground", pathname === route.href && "bg-surface-overlay text-foreground")}>{route.title}</Link>
          ))}
        </div>
      )}
      <Link href="/settings" aria-current={pathname === "/settings" ? "page" : undefined}
        title={APPEARANCE.settingsLabel} aria-label={APPEARANCE.settingsLabel}
        className={cn(gearClass(pathname === "/settings"), "mx-4 mb-4 hidden md:mt-auto md:flex")}>
        <Settings className="size-4" aria-hidden="true" />
      </Link>
    </aside>
  );
}

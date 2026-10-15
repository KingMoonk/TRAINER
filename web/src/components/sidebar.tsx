"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { AREAS, ROUTES, areaHome } from "@/lib/routes";
import { Separator } from "@/components/ui/separator";

// The Tk shell's sidebar: four pipeline stages, a rule, then the two utility areas. The area that lights up is the
// one the current page belongs to.
export function Sidebar() {
  const pathname = usePathname();
  const current = ROUTES.find((r) => r.href === pathname);
  const stages = AREAS.filter((a) => !a.utility);
  const utilities = AREAS.filter((a) => a.utility);
  const row = (area: (typeof AREAS)[number]) => (
    <Link
      key={area.id}
      href={areaHome(area.id).href}
      aria-current={current?.area === area.id ? "page" : undefined}
      className={cn(
        "block rounded-md px-3 py-2 text-sm text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-foreground",
        current?.area === area.id && "bg-sidebar-accent font-medium text-sidebar-foreground",
      )}
    >
      {area.title}
    </Link>
  );
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-sidebar-border bg-sidebar p-3">
      <div className="px-3 pb-4 pt-2 text-base font-semibold tracking-tight">Groundwork Studio</div>
      <nav className="flex flex-col gap-1">
        {stages.map(row)}
        <Separator className="my-2" />
        {utilities.map(row)}
      </nav>
      <div className="mt-auto px-3 pt-4 text-xs text-muted-foreground">Nothing is training</div>
    </aside>
  );
}

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
  // The export uses trailing slashes (next.config.ts), so the pathname is "/queue/" and the route href is "/queue".
  const current = ROUTES.find((r) => r.href === pathname.replace(/\/$/, ""));
  const stages = AREAS.filter((a) => !a.utility);
  const utilities = AREAS.filter((a) => a.utility);
  const row = (area: (typeof AREAS)[number]) => (
    <Link
      key={area.id}
      href={areaHome(area.id).href}
      aria-current={current?.area === area.id ? "page" : undefined}
      className={cn(
        "block shrink-0 whitespace-nowrap rounded-md px-3 py-2 text-sm text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-foreground",
        current?.area === area.id && "bg-sidebar-accent font-medium text-sidebar-foreground",
      )}
    >
      {area.title}
    </Link>
  );
  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-sidebar-border bg-sidebar p-3 md:w-56 md:border-b-0 md:border-r">
      <div className="px-3 pb-2 pt-1 text-base md:pb-4 md:pt-2 font-semibold tracking-tight">Groundwork Studio</div>
      <nav className="flex gap-1 overflow-x-auto md:flex-col md:overflow-visible">
        {stages.map(row)}
        <Separator orientation="vertical" className="mx-1 md:hidden" />
        <Separator className="my-2 hidden md:block" />
        {utilities.map(row)}
      </nav>
    </aside>
  );
}

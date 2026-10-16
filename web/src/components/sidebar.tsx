"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Boxes, ChevronRight, Database, Layers3, ListOrdered, MessageSquare, Network, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { AREAS, ROUTES, areaHome } from "@/lib/routes";
const icons = { designer: Network, pretraining: Layers3, post: Sparkles, chat: MessageSquare, queue: ListOrdered, resources: Database };

export function Sidebar() {
  const pathname = usePathname().replace(/\/$/, "");
  const current = ROUTES.find((route) => route.href === pathname);
  const pages = ROUTES.filter((route) => route.area === current?.area);
  return (
    <aside className="studio-glass z-20 flex w-full shrink-0 flex-col border-b border-sidebar-border md:sticky md:top-0 md:h-dvh md:w-60 md:border-r md:border-b-0">
      <Link href="/builder" className="flex items-center gap-3 px-5 py-5 md:py-7" aria-label="Groundwork Studio home">
        <span className="flex size-9 items-center justify-center rounded-xl border border-white/15 bg-white/5 shadow-inner"><Boxes className="size-5" aria-hidden="true" /></span>
        <span className="text-sm font-semibold tracking-tight">Groundwork <span className="font-normal text-muted-foreground">Studio</span></span>
      </Link>
      <nav aria-label="Studio" className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-visible">
        {AREAS.map((area) => {
          const Icon = icons[area.id as keyof typeof icons];
          const selected = current?.area === area.id;
          return (
            <Link key={area.id} href={areaHome(area.id).href} aria-current={selected ? "page" : undefined}
              className={cn("flex shrink-0 items-center gap-3 rounded-lg border border-transparent px-3 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground", area.id === "queue" && "md:mt-5", selected && "border-white/10 bg-white/7 font-medium text-foreground shadow-sm")}>
              <Icon className="size-4" aria-hidden="true" />{area.title}
              {selected && <ChevronRight className="ml-auto hidden size-3.5 text-muted-foreground md:block" aria-hidden="true" />}
            </Link>
          );
        })}
      </nav>
      {pages.length > 1 && (
        <div role="group" aria-label="Pages in this area" className="flex gap-1 overflow-x-auto border-t border-white/8 px-3 py-2 md:mx-3 md:mt-3 md:flex-col md:gap-0.5 md:overflow-y-auto md:px-0 md:pt-4">
          <p className="hidden px-3 pb-2 text-[11px] font-medium tracking-widest text-muted-foreground uppercase md:block">Explore</p>
          {pages.map((route) => (
            <Link key={route.route} href={route.href} aria-current={pathname === route.href ? "location" : undefined}
              className={cn("shrink-0 rounded-md px-3 py-2 text-xs text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground", pathname === route.href && "bg-white/5 text-foreground")}>{route.title}</Link>
          ))}
        </div>
      )}
    </aside>
  );
}

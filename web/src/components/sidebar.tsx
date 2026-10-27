"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Boxes, Database, Layers3, MessageSquare, Network, Server, Settings, Terminal } from "lucide-react";
import { cn } from "@/lib/utils";
import { APPEARANCE } from "@/lib/copy/appearance";
import { AREAS, ROUTES, areaHome } from "@/lib/routes";
import { useTraffic } from "@/components/traffic/traffic-provider";
import { TRAFFIC } from "@/lib/copy/traffic";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const focusClass = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset";
const gearClass = (active: boolean) =>
  cn(
    "flex size-11 shrink-0 items-center justify-center rounded-lg border border-transparent text-muted-foreground transition-colors hover:bg-surface-overlay hover:text-foreground md:size-9",
    focusClass,
    active && "border-surface-border bg-surface-raised text-foreground",
  );
const icons = { models: Network, data: Database, train: Layers3, eval: MessageSquare, machines: Server };

export function Sidebar() {
  const pathname = (usePathname() || "").replace(/\/$/, "");
  const current = ROUTES.find((route) => route.href === pathname);
  const settingsActive = pathname === "/settings";

  const settingsGear = (className?: string) => (
    <Tooltip>
      <TooltipTrigger asChild>
        <Link
          href="/settings"
          aria-current={settingsActive ? "page" : undefined}
          aria-label={APPEARANCE.settingsLabel}
          className={cn(gearClass(settingsActive), className)}
        >
          <Settings className="size-4" aria-hidden="true" />
        </Link>
      </TooltipTrigger>
      <TooltipContent side="right">{APPEARANCE.settingsLabel}</TooltipContent>
    </Tooltip>
  );

  const traffic = useTraffic();
  const consoleButton = (className?: string) => (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={traffic.toggle}
          aria-pressed={traffic.open}
          aria-label={TRAFFIC.button}
          aria-keyshortcuts="Control+`"
          className={cn(gearClass(traffic.open), className)}
        >
          <Terminal className="size-4" aria-hidden="true" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="right">{TRAFFIC.button} ({TRAFFIC.shortcut})</TooltipContent>
    </Tooltip>
  );

  return (
    <aside className="studio-glass z-20 flex w-full shrink-0 flex-col border-b border-sidebar-border md:sticky md:top-0 md:h-dvh md:w-60 md:border-r md:border-b-0">
      {/* Top bar on phone: logo, settings gear and traffic console button */}
      <div className="flex items-center justify-between px-4 py-3 md:hidden">
        <Link href="/builder" className={cn("flex items-center gap-3", focusClass)} aria-label="Groundwork Studio home">
          <span className="flex size-9 items-center justify-center rounded-xl border border-surface-border-strong bg-surface-overlay shadow-inner">
            <Boxes className="size-5" aria-hidden="true" />
          </span>
          <span className="text-sm font-semibold tracking-tight">
            Groundwork <span className="font-normal text-muted-foreground">Studio</span>
          </span>
        </Link>
        <div className="flex items-center gap-1">
          {settingsGear()}
          {consoleButton()}
        </div>
      </div>

      {/* Desktop logo */}
      <Link href="/builder" className={cn("hidden items-center gap-3 px-5 py-5 md:flex md:py-7", focusClass)} aria-label="Groundwork Studio home">
        <span className="flex size-9 items-center justify-center rounded-xl border border-surface-border-strong bg-surface-overlay shadow-inner">
          <Boxes className="size-5" aria-hidden="true" />
        </span>
        <span className="text-sm font-semibold tracking-tight">
          Groundwork <span className="font-normal text-muted-foreground">Studio</span>
        </span>
      </Link>

      {/* Areas navigation: horizontally scrolling pills on phone, vertical list on desktop */}
      <nav aria-label="Studio" className="flex gap-1.5 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-visible md:px-3 md:pb-0">
        {AREAS.map((area) => {
          const Icon = icons[area.id as keyof typeof icons];
          const selected = current?.area === area.id;
          return (
            <Link
              key={area.id}
              href={areaHome(area.id).href}
              aria-current={selected ? "page" : undefined}
              className={cn(
                "flex min-h-[44px] shrink-0 items-center gap-3 rounded-lg border border-transparent px-3 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-surface-overlay hover:text-foreground",
                focusClass,
                area.id === "machines" && "md:mt-5",
                selected && "border-surface-border bg-surface-raised font-medium text-foreground shadow-sm",
              )}
            >
              <Icon className="size-4" aria-hidden="true" />
              {area.title}
            </Link>
          );
        })}
      </nav>

      {/* Pinned to the foot on desktop */}
      <div className="mx-4 mb-4 hidden items-center gap-1 pt-4 md:mt-auto md:flex">
        {settingsGear()}
        {consoleButton()}
      </div>
    </aside>
  );
}

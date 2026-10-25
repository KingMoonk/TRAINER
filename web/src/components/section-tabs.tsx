"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { AREAS, ROUTES } from "@/lib/routes";

export function SectionTabs({ className }: { className?: string }) {
  const pathname = (usePathname() || "").replace(/\/$/, "");
  const current = ROUTES.find((route) => route.href === pathname);
  if (!current) return null;

  const area = AREAS.find((a) => a.id === current.area);
  const pages = ROUTES.filter((route) => route.area === current.area && !route.hidden);
  if (pages.length <= 1) return null;

  const activeRoute = current.hidden && current.parent ? current.parent : current.route;

  return (
    <nav
      aria-label={`${area?.title ?? "Section"} pages`}
      data-testid="section-tabs"
      className={cn("flex gap-1 overflow-x-auto border-b border-border", className)}
    >
      {pages.map((route) => {
        const isSelected = route.route === activeRoute;
        return (
          <Link
            key={route.route}
            href={route.href}
            aria-current={isSelected ? "page" : undefined}
            className={cn(
              "-mb-px flex min-h-[44px] shrink-0 items-center border-b-2 border-transparent px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              isSelected && "border-primary font-medium text-foreground",
            )}
          >
            {route.tabTitle ?? route.title}
          </Link>
        );
      })}
    </nav>
  );
}

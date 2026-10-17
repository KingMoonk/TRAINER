"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

/** The tabs of the Cluster area, in order; the first is the default (`/cluster/` opens it). Add a tab by adding an entry here. */
export const CLUSTER_TABS = [{ id: "machines", title: "Machines", href: "/cluster/machines" }] as const;

/** The horizontal tab bar across the top of every page under /cluster. */
export function ClusterTabs() {
  const pathname = usePathname().replace(/\/$/, "");
  return (
    <nav aria-label="Cluster" className="flex gap-1 overflow-x-auto border-b border-border" data-testid="cluster-tabs">
      {CLUSTER_TABS.map((tab) => {
        const selected = pathname === tab.href;
        return (
          <Link
            key={tab.id}
            href={tab.href}
            aria-current={selected ? "page" : undefined}
            className={cn(
              "-mb-px shrink-0 border-b-2 border-transparent px-4 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground",
              selected && "border-primary font-medium text-foreground",
            )}
          >
            {tab.title}
          </Link>
        );
      })}
    </nav>
  );
}

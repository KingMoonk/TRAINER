"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { availableElsewhere, usePull, type PublishedVersion } from "@/lib/cluster";

/** The versions other machines published that this workspace does not have, each with the machine and a Pull action. */
export function AvailableVersions({ published }: { published: PublishedVersion[] | undefined }) {
  const pull = usePull();
  const rows = availableElsewhere(published);
  if (!rows.length) return null;
  return (
    <ul className="flex flex-col gap-2" aria-label="Available on other machines" data-testid="available-versions">
      {rows.map((entry) => {
        const pulling = pull.isPending && pull.variables?.key === entry.key;
        return (
          <li key={entry.key} className="flex flex-wrap items-center gap-3 rounded-lg border border-dashed p-3 text-sm">
            <span className="font-medium">Version {entry.number}</span>
            <Badge variant="outline">available, not local</Badge>
            <span className="text-muted-foreground">from {entry.origin}</span>
            <span className="min-w-0 flex-1 truncate text-muted-foreground">{entry.change || entry.name || ""}</span>
            <Button
              variant="outline"
              size="sm"
              disabled={pull.isPending}
              aria-label={`Pull version ${entry.number} from ${entry.origin}`}
              onClick={() => pull.mutate(entry)}
            >
              {pulling ? "Pulling…" : "Pull"}
            </Button>
          </li>
        );
      })}
    </ul>
  );
}

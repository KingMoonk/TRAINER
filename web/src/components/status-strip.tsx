"use client";

import { cn } from "@/lib/utils";
import { useShell } from "@/lib/events";
import { Skeleton } from "@/components/ui/skeleton";
import type { QueueState, StudioStatus } from "@/lib/types";

export function queueText(queue: StudioStatus["queue"]): string {
  const states: Record<QueueState, string> = {
    idle: queue.waiting ? `${queue.waiting} waiting` : "Nothing is training",
    running: queue.waiting ? `Training, ${queue.waiting} waiting` : "Training",
    paused: "Queue paused",
  };
  return states[queue.state];
}

function Item({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm">{children}</span>
    </div>
  );
}

// Current version, queue and resources, from the shell's event stream. While the first connection is pending the
// values are skeletons; once the stream errors the strip says "Offline" and keeps the last values, dimmed.
export function StatusStrip() {
  const { status, connection } = useShell();
  const loading = connection === "connecting";
  const { version, queue, resources } = status;
  const value = (text: string) => (loading ? <Skeleton className="inline-block h-4 w-20 align-middle" /> : text);
  return (
    <div
      role="status"
      aria-label="Status"
      data-connection={connection}
      className="studio-glass flex min-h-14 flex-wrap items-center gap-x-6 gap-y-2 border-b px-5 py-3 md:px-8"
    >
      <div className={cn("flex flex-wrap items-center gap-x-6 gap-y-1", connection === "offline" && "opacity-60")}>
        <Item label="Version">{value(version ?? "None yet")}</Item>
        <Item label="Queue">{value(queueText(queue))}</Item>
        <Item label="Resources">
          {value(
            [resources.device, resources.diskFreeGb === null ? null : `${resources.diskFreeGb.toFixed(0)} GB free`]
              .filter(Boolean)
              .join(", ") || "Not measured",
          )}
        </Item>
      </div>
      <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
        <span
          aria-hidden
          className={cn(
            "size-2 rounded-full",
            connection === "online" ? "bg-emerald-500" : connection === "offline" ? "bg-destructive" : "bg-muted-foreground",
          )}
        />
        {connection === "online" ? "Connected" : connection === "offline" ? "Offline" : "Connecting"}
      </div>
    </div>
  );
}

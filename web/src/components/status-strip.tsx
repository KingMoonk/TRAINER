"use client";

import { cn } from "@/lib/utils";
import { useShell } from "@/lib/events";
import { Skeleton } from "@/components/ui/skeleton";
import { availableElsewhere, isCluster, nodeName, useCluster, usePublished, type ClusterQueue, type PublishedVersion } from "@/lib/cluster";
import type { QueueState, StudioStatus } from "@/lib/types";

/** The machines in the cluster, this one marked and any that stopped answering. */
export function machinesText(cluster: ClusterQueue): string {
  return cluster.nodes.map((n) => `${nodeName(n)}${n.local ? " (this machine)" : n.online ? "" : " (offline)"}`).join(", ");
}

/** What is going on elsewhere: jobs on the other machines and versions to pull. "" when there is nothing. */
export function elsewhereText(cluster: ClusterQueue, published: PublishedVersion[] | undefined): string {
  const jobs = cluster.jobs.filter((row) => !row.queued_here || (row.status === "running" && row.node !== "local")).length;
  const versions = availableElsewhere(published).length;
  const noun = (n: number, one: string) => `${n} ${one}${n === 1 ? "" : "s"}`;
  return [jobs ? `${noun(jobs, "job")} on other machines` : "", versions ? `${noun(versions, "version")} to pull` : ""].filter(Boolean).join(", ");
}

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
    <div className="flex min-w-0 items-baseline gap-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="min-w-0 break-words text-sm">{children}</span>
    </div>
  );
}

// Current version, queue and resources, from the shell's event stream. While the first connection is pending the
// values are skeletons; once the stream errors the strip says "Offline" and keeps the last values, dimmed.
export function StatusStrip() {
  const { status, connection } = useShell();
  const loading = connection === "connecting";
  const { version, queue, resources } = status;
  const cluster = useCluster().data;
  const published = usePublished(cluster).data;
  const machines = isCluster(cluster) ? cluster : undefined; // a cluster of one shows nothing about machines
  const elsewhere = machines ? elsewhereText(machines, published) : "";
  const value = (text: string) => (loading ? <Skeleton className="inline-block h-4 w-20 align-middle" /> : text);
  return (
    <div
      role="status"
      aria-label="Status"
      data-connection={connection}
      className="studio-glass flex min-h-14 min-w-0 flex-wrap items-center gap-x-6 gap-y-2 border-b px-4 py-3 md:px-8"
    >
      <div className={cn("flex min-w-0 flex-wrap items-center gap-x-6 gap-y-1", connection === "offline" && "opacity-60")}>
        <Item label="Version">{value(version ? `${version}${machines ? " · on this machine" : ""}` : "None yet")}</Item>
        <Item label="Queue">{value(queueText(queue))}</Item>
        {machines && <Item label="Machines">{value(machinesText(machines))}</Item>}
        {elsewhere && <Item label="Elsewhere">{value(elsewhere)}</Item>}
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

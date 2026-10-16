"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { nodeName, seenText, type ClusterQueue, type QueueNode, type QueueRow } from "@/lib/cluster";
import { cn } from "@/lib/utils";
import { T, fill } from "./copy";
import { nodeLabel, pinChoice, placeText } from "./logic";
import type { QueueActions } from "./use-queue";

/** The line under a job that says where it runs, or where it will, and why it waits. Only for a cluster of several machines. */
export function PlaceLine({ row }: { row: QueueRow }) {
  return (
    <div className="text-xs text-muted-foreground" data-testid={`place-${row.id}`}>
      {placeText(row)}
      {row.waiting_reason && <div>{row.waiting_reason}</div>}
    </div>
  );
}

/** Pin a waiting job to one machine, or let it go anywhere again. */
export function PinSelect({ id, name, row, nodes, actions }: { id: string; name: string; row: QueueRow | undefined; nodes: QueueNode[]; actions: QueueActions }) {
  const pinned = row?.pinned ?? "";
  const value = pinChoice(nodes, pinned);
  return (
    <label className="flex items-center gap-1 text-xs text-muted-foreground">
      {T.runOn}
      <select
        aria-label={`${T.runOn}: ${name}`}
        data-testid={`pin-${id}`}
        value={value}
        disabled={actions.pin.isPending}
        onChange={(e) => actions.pin.mutate({ id, pinned: e.target.value || null })}
        className="h-8 rounded-md border border-input bg-background px-2 text-sm text-foreground"
      >
        <option value="">{T.anyMachine}</option>
        {pinned && !nodes.some((n) => n.pin === value) && <option value={value}>{value}</option>}
        {nodes.map((node) => (
          <option key={node.address} value={node.pin}>
            {nodeLabel(node, nodes)}
          </option>
        ))}
      </select>
    </label>
  );
}

/** What the other machines have: each machine, then the jobs that run or wait on it. A machine that stopped answering stays, greyed. */
export function OtherMachinesCard({ cluster }: { cluster: ClusterQueue }) {
  const rows = cluster.jobs.filter((row) => !row.queued_here || (row.status === "running" && row.node !== "local"));
  return (
    <Card data-testid="other-machines">
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.elsewhere}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <p className="text-xs text-muted-foreground">{T.elsewhereTip}</p>
        <ul className="flex flex-wrap gap-2" aria-label="Machines">
          {cluster.nodes.map((node) => (
            <li key={node.address} data-testid={`node-${node.address}`} className={cn(!node.online && "opacity-60")}>
              <Badge variant="outline" title={node.error || undefined}>
                {nodeLabel(node, cluster.nodes)}
              </Badge>
            </li>
          ))}
        </ul>
        {cluster.nodes
          .filter((node) => !node.online)
          .map((node) => (
            <p key={node.address} role="status" className="text-xs text-muted-foreground">
              {fill(T.notAnswering, { node: nodeName(node, cluster.nodes), seen: node.seen_at ? ` (last answered ${seenText(node.seen_at)})` : "" })}
            </p>
          ))}
        {rows.length === 0 ? (
          <p className="text-base text-muted-foreground">{T.nothingElsewhere}</p>
        ) : (
          <ol className="flex flex-col gap-2">
            {rows.map((row) => (
              <li
                key={`${row.queued_on}-${row.id}`}
                data-testid={`other-${row.id}`}
                data-stale={row.stale ? "true" : undefined}
                className={cn("flex flex-wrap items-center gap-3 rounded-lg border-2 border-border bg-muted/20 p-2", row.stale && "opacity-60")}
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium">{row.label || row.kind}</div>
                  <div className="text-xs text-muted-foreground">
                    {placeText(row)} · {fill(T.queuedOn, { node: row.queued_on })}
                  </div>
                  {row.note && <div className="text-sm text-muted-foreground">{row.note}</div>}
                </div>
                {row.stale && <Badge variant="outline">{T.lastKnown}</Badge>}
                {row.status === "held" && <Badge variant="outline">{T.onHold}</Badge>}
              </li>
            ))}
          </ol>
        )}
      </CardContent>
    </Card>
  );
}

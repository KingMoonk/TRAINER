"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { QueueRow } from "@/lib/cluster";
import { CLUSTER as T } from "@/lib/copy/cluster";
import { cn } from "@/lib/utils";
import { deviceText, fraction, lastSeen, machineJobs, progressText, refusedUs, runPage, waitingPlace, type ClusterNode } from "./logic";

type Props = {
  node?: ClusterNode; // undefined: added to the list but not saved yet
  address: string;
  rows: QueueRow[];
  seenAt: number;
  login: string;
  names: Record<string, string>;
  busy: boolean; // a save or a stop is in flight
  onRemove?: () => void;
  onStop: (row: QueueRow) => void;
};

/** One machine: its names, whether it answers (and why not), what it computes on, the job it runs and the jobs waiting on it. */
export function MachineCard({ node, address, rows, seenAt, login, names, busy, onRemove, onStop }: Props) {
  const local = node?.local ?? false;
  const title = local ? `${node?.hostname} · ${T.thisMachine}` : node?.hostname || address;
  const { running, waiting } = machineJobs(address, rows);
  const device = node ? deviceText(node) : "";
  const seen = node ? lastSeen(node, seenAt) : "";
  const stale = node && !node.online;
  return (
    <Card data-testid={`machine-${address}`} data-online={node ? String(node.online) : undefined} className={cn(stale && "opacity-90")}>
      <CardHeader className="flex flex-row flex-wrap items-center gap-2">
        <div className="min-w-0 flex-1">
          <CardTitle className="truncate">{title}</CardTitle>
          <div className="text-xs text-muted-foreground">
            {!local && node?.hostname ? T.listedAs.replace("{address}", address) : null}
            {node?.tailnet_name && <span data-testid={`tailnet-${address}`}>{!local && node.hostname ? " · " : ""}{node.tailnet_name}</span>}
          </div>
        </div>
        {node && <Badge variant={node.online ? "secondary" : "outline"}>{node.online ? T.online : T.offline}</Badge>}
        {node?.online && <Badge variant="outline">{node.busy ? T.busy : T.idle}</Badge>}
        {onRemove && (
          <Button variant="ghost" size="sm" aria-label={`${T.remove} ${address}`} disabled={busy} onClick={onRemove}>{T.remove}</Button>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        {!node && <p className="text-muted-foreground">{T.unsaved}</p>}
        {node && (device || node.version) && (
          <p className="text-muted-foreground" data-testid={`device-${address}`}>{[device, node.version && `version ${node.version}`].filter(Boolean).join(" · ")}</p>
        )}
        {node && !node.online && node.error && <p className="text-muted-foreground" data-testid={`machine-error-${address}`}>{node.error}</p>}
        {node && refusedUs(node) && <Refused name={node.hostname || address} login={login} />}
        {seen && <p className="text-xs text-muted-foreground" data-testid={`seen-${address}`}>{seen}</p>}
        {node && (
          <>
            <JobList title={T.running} empty={T.nothingRunning} rows={running} names={names} stale={!!stale} onStop={onStop} busy={busy} />
            <JobList title={T.waiting} empty={T.nothingWaiting} rows={waiting} names={names} stale={!!stale} onStop={onStop} busy={busy} />
          </>
        )}
      </CardContent>
    </Card>
  );
}

/** Why a machine answered 401, in a sentence, and what to add. The login is this computer's, as the Access card reads it. */
function Refused({ name, login }: { name: string; login: string }) {
  return (
    <div role="note" className="rounded-lg border border-warning/40 bg-warning-soft p-3" data-testid="machine-refused">
      <div className="font-medium">{T.refusedTitle.replace("{node}", name)}</div>
      <p className="text-muted-foreground">{(login ? T.refusedWithLogin : T.refusedNoLogin).replace("{login}", login)}</p>
    </div>
  );
}

function JobList({ title, empty, rows, names, stale, onStop, busy }: { title: string; empty: string; rows: QueueRow[]; names: Record<string, string>; stale: boolean; onStop: (row: QueueRow) => void; busy: boolean }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {title}
        {stale && rows.length > 0 && <span className="ml-1 normal-case">({T.lastKnown})</span>}
      </div>
      {rows.length === 0 ? (
        <p className="text-muted-foreground">{empty}</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((row) => (
            <JobItem key={row.id} row={row} names={names} onStop={onStop} busy={busy} stale={stale} />
          ))}
        </ul>
      )}
    </div>
  );
}

function JobItem({ row, names, onStop, busy, stale }: { row: QueueRow; names: Record<string, string>; onStop: (row: QueueRow) => void; busy: boolean; stale: boolean }) {
  const running = row.status === "running";
  const page = runPage(row);
  return (
    <li data-testid={`job-${row.id}`} className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-faint p-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="min-w-0 flex-1 truncate font-medium">{row.label}</div>
        {row.status === "held" && <Badge variant="outline">{T.onHold}</Badge>}
        {!row.queued_here && <Badge variant="outline">{T.queuedOn.replace("{node}", row.queued_on)}</Badge>}
        {page && (
          <Link href={page} className="text-sm underline underline-offset-2" data-testid={`run-link-${row.id}`}>{T.openRun}</Link>
        )}
        <Button variant={running ? "destructive" : "ghost"} size="sm" disabled={busy || stale} aria-label={`${running ? T.stop : T.removeJob} ${row.label}`} onClick={() => onStop(row)}>
          {running ? T.stop : T.removeJob}
        </Button>
      </div>
      {running ? (
        <>
          <div className="text-xs text-muted-foreground" data-testid={`progress-${row.id}`}>{progressText(row)}</div>
          <div className="h-1.5 rounded-full bg-muted" role="progressbar" aria-label={row.label} aria-valuenow={Math.round(fraction(row) * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-1.5 rounded-full bg-highlight-strong" style={{ width: `${Math.round(fraction(row) * 100)}%` }} />
          </div>
        </>
      ) : (
        <div className="text-xs text-muted-foreground">{waitingPlace(row, names)}</div>
      )}
      {row.waiting_reason && <div className="text-xs text-muted-foreground">{row.waiting_reason}</div>}
    </li>
  );
}

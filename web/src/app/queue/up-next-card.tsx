"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { ClusterQueue } from "@/lib/cluster";
import { cn } from "@/lib/utils";
import { T, fill } from "./copy";
import { allDoneLine, dropIndex, minutes, rowFor, type Job, type Queue } from "./logic";
import { MembersLine, PinSelect, PlaceLine } from "./nodes";
import { useJobOutlook, type QueueActions } from "./use-queue";

/** `cluster` is set only when other machines are listed: a cluster of one shows no machine on any row. */
export function UpNextCard({ queue, actions, cluster }: { queue: Queue | undefined; actions: QueueActions; cluster?: ClusterQueue }) {
  const [dragging, setDragging] = useState<string | null>(null);
  const [over, setOver] = useState<number | null>(null);
  const jobs = queue?.up_next ?? [];
  const end = () => {
    setDragging(null);
    setOver(null);
  };
  const drop = (target: number, id: string) => {
    const from = jobs.findIndex((j) => j.id === id);
    const index = dropIndex(target, jobs.length);
    if (from >= 0 && index !== from) actions.move.mutate({ id, index });
    end();
  };
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.upNext}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3" data-testid="up-next">
        {!queue ? (
          <div className="flex flex-col gap-2" aria-label="Reading the queue">
            <Skeleton className="h-14 w-full" />
            <Skeleton className="h-14 w-full" />
          </div>
        ) : (
          <>
            {allDoneLine(queue) && <p className="text-sm text-muted-foreground">{allDoneLine(queue)}</p>}
            {jobs.length === 0 ? (
              <div>
                <p className="text-base text-muted-foreground">{T.nothingWaiting}</p>
                <p className="text-xs text-muted-foreground">{T.nothingWaitingTip}</p>
              </div>
            ) : (
              <>
                <p className="text-xs text-muted-foreground">{T.reorderTip}</p>
                <ol className="flex flex-col gap-2">
                  {jobs.map((job, n) => (
                    <Row
                      key={job.id}
                      job={job}
                      n={n}
                      last={jobs.length - 1}
                      active={dragging === job.id || (over === n && dragging !== null)}
                      actions={actions}
                      cluster={cluster}
                      onDragStart={() => setDragging(job.id)}
                      onDragOver={() => setOver(n)}
                      onDrop={(id) => drop(n, id)}
                      onDragEnd={end}
                    />
                  ))}
                </ol>
              </>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function Row({
  job,
  n,
  last,
  active,
  actions,
  cluster,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
}: {
  job: Job;
  n: number;
  last: number;
  active: boolean;
  actions: QueueActions;
  cluster?: ClusterQueue;
  onDragStart: () => void;
  onDragOver: () => void;
  onDrop: (id: string) => void;
  onDragEnd: () => void;
}) {
  const held = job.status === "held";
  const name = job.label || job.kind;
  const outlook = useJobOutlook(job.id).data?.outlook;
  const move = (index: number) => actions.move.mutate({ id: job.id, index });
  const row = rowFor(cluster, job.id);
  return (
    <li
      data-testid={`job-${job.id}`}
      onDragOver={(e) => {
        e.preventDefault();
        onDragOver();
      }}
      onDrop={(e) => {
        e.preventDefault();
        onDrop(e.dataTransfer.getData("text/plain"));
      }}
      onDragEnd={onDragEnd}
      className={cn("flex flex-wrap items-center gap-3 rounded-lg border-2 p-2 transition-colors", active ? "selected-box" : "border-border bg-muted/20")}
    >
      <button
        type="button"
        draggable
        aria-label={`${T.dragHandle}: ${name}`}
        title={T.dragHandle}
        onDragStart={(e) => {
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", job.id);
          onDragStart();
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            const target = dropIndex(n + (e.key === "ArrowUp" ? -1 : 1), last + 1);
            if (target !== n) move(target);
          }
        }}
        className="cursor-grab select-none rounded px-1 text-lg text-muted-foreground focus-visible:outline-2 focus-visible:outline-ring active:cursor-grabbing"
      >
        {"⠇"}
      </button>
      <span className={cn("w-5 text-center font-semibold", held ? "text-muted-foreground" : "text-highlight")}>{n + 1}</span>
      <div className="min-w-0 flex-1">
        <div className={cn("truncate font-medium", held && "text-muted-foreground")}>
          {name}
          {held ? ` · ${T.onHold}` : ""}
        </div>
        {job.note && <div className="text-xs text-muted-foreground">{job.note}</div>}
        {outlook && <div className="text-xs text-muted-foreground">{fill(T.starts, { outlook })}</div>}
        {row && job.kind !== "pretrain_distributed" && <PlaceLine row={row} />}
        <MembersLine job={job} />
      </div>
      {job.eta_s != null && <span className="text-xs text-muted-foreground">{minutes(job.eta_s)}</span>}
      {cluster && job.kind !== "pretrain_distributed" && <PinSelect job={job} name={name} row={row} nodes={cluster.nodes} actions={actions} />}
      <div className="flex flex-wrap gap-1">
        <Button variant="outline" size="sm" aria-label={`${T.moveUp}: ${name}`} disabled={n === 0} onClick={() => move(n - 1)}>
          {"▲"}
        </Button>
        <Button variant="outline" size="sm" aria-label={`${T.moveDown}: ${name}`} disabled={n === last} onClick={() => move(n + 1)}>
          {"▼"}
        </Button>
        <Button variant="outline" size="sm" aria-label={`${T.runNext}: ${name}`} disabled={n === 0} onClick={() => move(0)}>
          {T.runNext}
        </Button>
        <Button variant="outline" size="sm" aria-label={`${held ? T.release : T.hold}: ${name}`} onClick={() => actions.hold.mutate({ id: job.id, held: !held })}>
          {held ? T.release : T.hold}
        </Button>
        <Button variant="destructive" size="sm" aria-label={`${T.remove}: ${name}`} onClick={() => actions.remove.mutate(job.id)}>
          {T.remove}
        </Button>
      </div>
    </li>
  );
}

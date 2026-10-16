"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { T, fill } from "./copy";
import { allDoneLine, dropIndex, minutes, type Job, type Queue } from "./logic";
import { useJobOutlook, type QueueActions } from "./use-queue";

export function UpNextCard({ queue, actions }: { queue: Queue | undefined; actions: QueueActions }) {
  const [dragging, setDragging] = useState<string | null>(null);
  const [over, setOver] = useState<number | null>(null);
  const jobs = queue?.up_next ?? [];
  const end = () => {
    setDragging(null);
    setOver(null);
  };
  const drop = (target: number) => {
    const from = jobs.findIndex((j) => j.id === dragging);
    const index = dropIndex(target, jobs.length);
    if (dragging && from >= 0 && index !== from) actions.move.mutate({ id: dragging, index });
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
                      onDragStart={() => setDragging(job.id)}
                      onDragOver={() => setOver(n)}
                      onDrop={() => drop(n)}
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
  onDragStart: () => void;
  onDragOver: () => void;
  onDrop: () => void;
  onDragEnd: () => void;
}) {
  const held = job.status === "held";
  const name = job.label || job.kind;
  const outlook = useJobOutlook(job.id).data?.outlook;
  const move = (index: number) => actions.move.mutate({ id: job.id, index });
  return (
    <li
      data-testid={`job-${job.id}`}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/plain", job.id);
        onDragStart();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        onDragOver();
      }}
      onDrop={(e) => {
        e.preventDefault();
        onDrop();
      }}
      onDragEnd={onDragEnd}
      className={cn("flex flex-wrap items-center gap-3 rounded-lg border-2 bg-muted/20 p-2 transition-colors", active ? "border-violet-500" : "border-border")}
    >
      <span aria-hidden title={T.dragHandle} className="cursor-grab select-none px-1 text-lg text-muted-foreground">
        {"⠇"}
      </span>
      <span className={cn("w-5 text-center font-semibold", held ? "text-muted-foreground" : "text-violet-400")}>{n + 1}</span>
      <div className="min-w-0 flex-1">
        <div className={cn("truncate font-medium", held && "text-muted-foreground")}>
          {name}
          {held ? ` · ${T.onHold}` : ""}
        </div>
        {job.note && <div className="text-xs text-muted-foreground">{job.note}</div>}
        {outlook && <div className="text-xs text-muted-foreground">{fill(T.starts, { outlook })}</div>}
      </div>
      {job.eta_s != null && <span className="text-xs text-muted-foreground">{minutes(job.eta_s)}</span>}
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

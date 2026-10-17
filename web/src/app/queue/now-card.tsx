"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { ClusterQueue } from "@/lib/cluster";
import { T } from "./copy";
import { placeText, progressLine, rowFor, stateLine, type Queue } from "./logic";
import type { QueueActions } from "./use-queue";

/** `cluster` is set only when other machines are listed: a cluster of one shows no machine anywhere. */
export function NowCard({ queue, actions, cluster }: { queue: Queue | undefined; actions: QueueActions; cluster?: ClusterQueue }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.now}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3" data-testid="now-card">
        {!queue ? (
          <div className="flex flex-col gap-2" aria-label="Reading the queue">
            <Skeleton className="h-5 w-64" />
            <Skeleton className="h-4 w-96 max-w-full" />
            <Skeleton className="h-2 w-full" />
          </div>
        ) : queue.job ? (
          <Running queue={queue} actions={actions} cluster={cluster} />
        ) : (
          <>
            <p className="text-base text-muted-foreground">{queue.paused ? T.pausedByYou : T.nothingTraining}</p>
            <div className="flex flex-wrap gap-2">
              {queue.paused ? (
                <Button onClick={() => actions.resume.mutate()} disabled={actions.resume.isPending}>
                  {T.resume}
                </Button>
              ) : (
                <>
                  <Button onClick={() => actions.pause.mutate()} disabled={actions.pause.isPending}>
                    {T.pause}
                  </Button>
                  {queue.up_next.length > 0 && (
                    <Button variant="outline" title={T.startDesc} onClick={() => actions.start.mutate()} disabled={actions.start.isPending}>
                      {T.start}
                    </Button>
                  )}
                </>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function Running({ queue, actions, cluster }: { queue: Queue; actions: QueueActions; cluster?: ClusterQueue }) {
  const job = queue.job!;
  const row = rowFor(cluster, job.id);
  const fraction = queue.total ? Math.min(1, queue.step / queue.total) : 0;
  return (
    <>
      <p className="text-base" data-testid="state-line">
        {stateLine(queue)}
      </p>
      {row && (
        <p className="text-sm text-muted-foreground" data-testid="now-place">
          {placeText(row)}
        </p>
      )}
      <p className="text-sm text-muted-foreground" data-testid="progress-line">
        {progressLine(queue)}
      </p>
      <div
        role="progressbar"
        aria-label="Job progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(fraction * 100)}
        className="h-2 w-full rounded-full bg-muted"
      >
        <div className="h-2 rounded-full bg-highlight-strong transition-all" style={{ width: `${Math.round(fraction * 100)}%` }} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {queue.paused ? (
          <Button onClick={() => actions.resume.mutate()} disabled={actions.resume.isPending}>
            {T.resume}
          </Button>
        ) : (
          <Button onClick={() => actions.pause.mutate()} disabled={actions.pause.isPending}>
            {T.pause}
          </Button>
        )}
        <Button variant="outline" title={T.checkpointDesc} onClick={() => actions.checkpointNow.mutate()} disabled={actions.checkpointNow.isPending}>
          {T.checkpointNow}
        </Button>
        <Button variant="outline" onClick={() => actions.stopAfterCooldown.mutate()} disabled={queue.stopping || actions.stopAfterCooldown.isPending}>
          {T.stopAfterCooldown}
        </Button>
        <Button variant="destructive" onClick={() => actions.remove.mutate(job.id)} disabled={actions.remove.isPending}>
          {T.stop}
        </Button>
      </div>
      {queue.stopping && <p className="text-sm text-highlight">{T.stopping}</p>}
      <p className="text-xs text-muted-foreground">{T.checkpointDesc}</p>
    </>
  );
}

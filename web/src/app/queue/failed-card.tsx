"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { T } from "./copy";
import type { Job, Queue } from "./logic";
import type { QueueActions } from "./use-queue";

/** Jobs that ended failed: the sentence says what happened and what to do next; the raw text sits behind a toggle. Dismissing removes the job. */
export function FailedCard({ queue, actions }: { queue: Queue | undefined; actions: QueueActions }) {
  const failed = queue?.failed ?? [];
  if (failed.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.failed}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2" data-testid="failed">
        <ul className="flex flex-col gap-2">
          {failed.map((job) => (
            <FailedRow key={job.id} job={job} actions={actions} />
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function FailedRow({ job, actions }: { job: Job; actions: QueueActions }) {
  const [open, setOpen] = useState(false);
  const name = job.label || job.kind;
  return (
    <li data-testid={`failed-${job.id}`} className="flex flex-col gap-1 rounded-lg border-2 border-destructive/40 p-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 flex-1 truncate font-medium">{name}</span>
        {job.details && (
          <Button variant="outline" size="sm" aria-expanded={open} aria-label={`${open ? T.hideDetails : T.showDetails}: ${name}`} onClick={() => setOpen(!open)}>
            {open ? T.hideDetails : T.showDetails}
          </Button>
        )}
        <Button variant="outline" size="sm" aria-label={`${T.dismiss}: ${name}`} onClick={() => actions.remove.mutate(job.id)}>
          {T.dismiss}
        </Button>
      </div>
      <p className="text-sm" data-testid={`failed-why-${job.id}`}>
        {job.error || T.failedNoReason}
      </p>
      {open && job.details && (
        <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded bg-muted p-2 text-xs" data-testid={`failed-details-${job.id}`}>
          {job.details}
        </pre>
      )}
    </li>
  );
}

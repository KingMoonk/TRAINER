"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { T, fill } from "./copy";
import type { Job } from "./logic";

/** The jobs running on another machine, with the job's note: where a peer's refused checkpoint is said (for example, that the bucket is full). */
export function AwayCard({ jobs }: { jobs: Job[] }) {
  if (jobs.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.away}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3" data-testid="away">
        <ul className="flex flex-col gap-2">
          {jobs.map((job) => (
            <li key={job.id} className="flex flex-col gap-1">
              <p className="text-base">
                {job.label || job.kind} · {fill(T.awayOn, { node: job.node })}
              </p>
              {job.note && <p className="text-sm text-muted-foreground">{job.note}</p>}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

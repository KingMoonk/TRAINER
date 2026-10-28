"use client";

import { useMutation } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api, unwrap } from "@/lib/api";
import { EmptyState } from "@/components/empty-state";
import { ROW_KINDS, T } from "./copy";
import { errorText, rowText, type VersionRow } from "./logic";
import { useJob } from "./use-events";

/** The versions, keyed by number so a row updates in place when its checkpoint arrives or a new one is saved. */
export function VersionsCard({ versions, startsFrom, loading }: { versions: VersionRow[] | undefined; startsFrom: number | undefined; loading: boolean }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Versions</CardTitle>
        <CardDescription>Each round starts from the best trained one.</CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex flex-col gap-2">
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-2/3" />
          </div>
        ) : !versions?.length ? (
          <EmptyState>{T.noVersions}</EmptyState>
        ) : (
          <ul className="flex flex-col gap-2" aria-label="Versions">
            {[...versions].reverse().map((v) => (
              <li key={v.number} data-testid={`version-${v.number}`} className="flex flex-wrap items-center gap-2 text-sm">
                <span className="font-medium">Version {v.number}</span>
                <span className="text-muted-foreground">{v.change || v.name}</span>
                {v.best_checkpoint ? <Badge variant="secondary">trained</Badge> : <Badge variant="outline">no checkpoint yet</Badge>}
                {v.number === startsFrom && <Badge>starts here</Badge>}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

/** The rows behind a round (or the held-back ones): a job, so a skeleton shows until its result event arrives. */
export function RowsCard() {
  const [kind, setKind] = useState<(typeof ROW_KINDS)[number]["value"]>("conversations");
  const [asked, setAsked] = useState<(typeof ROW_KINDS)[number]["value"] | null>(null);
  const request = useCallback(() => unwrap(api.POST("/post-training/rows", { body: { kind } })), [kind]);
  const { job, start, pending } = useJob(request);
  const ask = useMutation({
    mutationFn: start,
    onMutate: () => setAsked(kind),
    onError: (error) => toast.error(errorText(error)),
  });
  const rows = job?.status === "done" && Array.isArray(job.result) ? (job.result as unknown[]) : null;
  const waiting = ask.isPending || pending;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Look at the data</CardTitle>
        <CardDescription>The rows a round uses, or the held-back ones it is checked on.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <ToggleGroup type="single" variant="outline" aria-label="Which rows" value={kind} onValueChange={(v) => v && setKind(v as typeof kind)} className="flex-wrap">
            {ROW_KINDS.map((k) => (
              <ToggleGroupItem key={k.value} value={k.value}>
                {k.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <Button variant="outline" onClick={() => ask.mutate()} disabled={waiting}>
            Show rows
          </Button>
        </div>
        {waiting && (
          <div className="flex flex-col gap-2" aria-label="Reading rows">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-5/6" />
            <Skeleton className="h-5 w-2/3" />
          </div>
        )}
        {!waiting && job?.status === "failed" && <p className="text-sm text-destructive">{job.error}</p>}
        {!waiting && rows && (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted-foreground">
              {rows.length.toLocaleString("en-US")} {ROW_KINDS.find((k) => k.value === asked)?.label.toLowerCase()}
              {rows.length > 20 ? ", the first 20 shown" : ""}
            </p>
            {rows.slice(0, 20).map((row, i) => (
              <pre key={i} className="whitespace-pre-wrap rounded-lg bg-muted/50 p-2 text-xs">
                {rowText(row)}
              </pre>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

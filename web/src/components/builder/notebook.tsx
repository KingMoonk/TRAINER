"use client";

// The lab notebook (builder.py _notebook, design.notebook): one line per version, oldest first, saying what it changed, what
// that did to held-back error, step time and memory, and whether the guess was right. The server reads every version's
// run files (a job); the rows are keyed by version so a training run's result updates its row in place.

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { design, detailOf, type NotebookRow, type Version } from "@/lib/builder";
import { BUILDER, fill } from "@/lib/copy/builder";
import { cn } from "@/lib/utils";

/** What the notebook reads from a version: whether it has trained and its score, not every status change. */
const stamp = (v: Version) => {
  const card = v.scorecard as { bpc?: unknown; bpc_method?: unknown } | undefined;
  return [v.number, v.change ?? "", Boolean(v.best_checkpoint), card?.bpc ?? null, card?.bpc_method ?? null, JSON.stringify(v.prediction ?? null)];
};

const STATE = { attention: "text-warning", live: "text-success", neutral: "text-muted-foreground" } as const;

function Line({ row }: { row: NotebookRow }) {
  const question = row.question;
  return (
    <li className="flex flex-col gap-0.5 py-1" data-testid={`notebook-${row.number}`} data-state={row.state}>
      <p className="text-sm">
        <span className="font-medium">{fill(BUILDER.notebook_version, { n: row.number })}</span>
        {row.change ? ` · ${row.change}` : ""}
        {row.effects ? <span data-testid="notebook-effects"> → {row.effects}</span> : null}
      </p>
      {(question || row.guesses || row.note) && (
        <p className="flex flex-wrap gap-x-2 text-xs text-muted-foreground">
          {question && <span data-testid="notebook-question">{fill(BUILDER.notebook_asked, { question })}</span>}
          {row.guesses && (
            <span className={cn(STATE[row.state])} data-testid="notebook-guess">
              {row.guesses}
            </span>
          )}
          {row.note && <span data-testid="notebook-note">{row.note}</span>}
        </p>
      )}
    </li>
  );
}

export function LabNotebook({ versions }: { versions: Version[] | undefined }) {
  const notebookQ = useQuery({
    queryKey: ["design", "notebook", (versions ?? []).map(stamp)],
    queryFn: ({ signal }) => design.notebook(signal),
    enabled: versions !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
  });
  const rows = notebookQ.data?.rows;
  return (
    <Card size="sm" data-testid="notebook">
      <CardHeader>
        <CardTitle title={BUILDER.notebook_tip}>{BUILDER.notebook}</CardTitle>
      </CardHeader>
      <CardContent>
        {notebookQ.isError ? (
          <p role="alert" className="text-sm text-destructive">
            {BUILDER.notebook_failed} {detailOf(notebookQ.error)}
          </p>
        ) : !rows ? (
          <div role="status" aria-label={BUILDER.notebook_reading} aria-busy="true" className="flex flex-col gap-2" data-testid="notebook-loading">
            <Skeleton className="h-5 w-3/4" />
            <Skeleton className="h-5 w-2/3" />
          </div>
        ) : rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">{BUILDER.notebook_empty}</p>
        ) : (
          <ol className="flex flex-col divide-y" aria-label={BUILDER.notebook}>
            {rows.map((row) => (
              <Line key={row.number} row={row} />
            ))}
          </ol>
        )}
      </CardContent>
    </Card>
  );
}

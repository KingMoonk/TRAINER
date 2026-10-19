"use client";

// The starting questions (builder.py _experiments): each sets up a one-change draft from the current version, and the
// question goes with the version it is saved as (shown in the lab notebook). What a question can and cannot do is the server's.

import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { design, detailOf, type Choices, type Try } from "@/lib/builder";
import { BUILDER, fill } from "@/lib/copy/builder";

const AMBER = "text-warning";

type Props = {
  cur: Choices;
  /** The question the draft is exactly the one change of, or null. */
  asking: string | null;
  busy: boolean;
  onTry: (key: Try["key"]) => void;
};

export function Experiments({ cur, asking, busy, onTry }: Props) {
  const rowsQ = useQuery({ queryKey: ["design", "experiments", cur], queryFn: () => design.experiments(cur), staleTime: Infinity, retry: false });
  return (
    <Card size="sm" data-testid="experiments">
      <CardHeader>
        <CardTitle title={BUILDER.experiments_tip}>{BUILDER.experiments}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {rowsQ.isError ? (
          <p role="alert" className="text-sm text-destructive">
            {BUILDER.experiments_failed} {detailOf(rowsQ.error)}
          </p>
        ) : !rowsQ.data ? (
          <div role="status" aria-label={BUILDER.explain_loading} aria-busy="true" className="flex flex-col gap-2">
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
          </div>
        ) : (
          rowsQ.data.map((row) => (
            <div key={row.question} className="flex flex-col gap-1" data-testid="experiment">
              <p className="text-sm">{row.question}</p>
              <div className="flex flex-wrap gap-2">
                {row.tries.map((t) => (
                  <Button
                    key={t.key}
                    size="xs"
                    variant="outline"
                    disabled={busy || !!t.why}
                    aria-describedby={t.why ? `why-${t.key}` : undefined}
                    onClick={() => onTry(t.key)}
                    data-testid={`try-${t.key}`}
                  >
                    {t.label} ›
                  </Button>
                ))}
              </div>
              {row.tries.map(
                (t) =>
                  t.why && (
                    <p key={t.key} id={`why-${t.key}`} className="text-xs text-muted-foreground">
                      {t.why}
                    </p>
                  ),
              )}
            </div>
          ))
        )}
        {asking && (
          <p className={`text-xs ${AMBER}`} data-testid="experiment-asking">
            {fill(BUILDER.exp_asking, { question: asking })}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

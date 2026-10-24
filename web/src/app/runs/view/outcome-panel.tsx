"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SeriesChart } from "@/components/series-chart";
import type { Run } from "@/app/cluster/training/logic";
import { outcomeSeries } from "./outcome";

/** One outcome-and-participation view, used only on the canonical run monitor. */
export function OutcomePanel({ run }: { run: Run }) {
  const { axis, loss, participation, marks } = outcomeSeries(run);
  return (
    <Card data-testid="run-outcome">
      <CardHeader>
        <CardTitle role="heading" aria-level={2}>Learning progress</CardTitle>
        <CardDescription>Held-out quality and the workers behind each outer update, aligned by accepted work.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {run.history.length === 0 ? <p className="text-sm text-muted-foreground">No round has finished yet.</p> : (
          <div className="grid gap-6 lg:grid-cols-2">
            <figure className="flex min-w-0 flex-col gap-2" data-testid="outcome-loss">
              <figcaption className="text-sm font-medium">Validation loss vs. work</figcaption>
              {loss.length ? <SeriesChart series={loss} marks={marks} xLabel={axis.label} yLabel="Loss" className="h-56 w-full" /> :
                <p className="text-sm text-muted-foreground">No held-out validation result yet.</p>}
            </figure>
            <figure className="flex min-w-0 flex-col gap-2" data-testid="outcome-participation">
              <figcaption className="text-sm font-medium">Members and accepted contributions</figcaption>
              <SeriesChart series={participation} xLabel={axis.label} yLabel="Members" className="h-56 w-full" />
            </figure>
          </div>
        )}
        {run.history.length > 0 && <p className="text-xs text-muted-foreground" data-testid="outcome-axis-note">{axis.note}</p>}
      </CardContent>
    </Card>
  );
}

"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SeriesChart } from "@/components/series-chart";
import type { Run } from "@/app/cluster/training/logic";
import { liveLossSeries, tokensSeries } from "./live";

/** Trends within the current round; the topology above shows the latest value for each worker. */
export function OpenRoundPanel({ run }: { run: Run }) {
  const loss = liveLossSeries(run);
  const speed = tokensSeries(run);
  if (loss.length === 0 && speed.length === 0) return null;
  return (
    <Card data-testid="open-round-trends">
      <CardHeader>
        <CardTitle>Current round</CardTitle>
        <CardDescription>Each worker’s training loss and useful token rate as local steps progress.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 lg:grid-cols-2">
        {loss.length > 0 && (
          <figure className="flex min-w-0 flex-col gap-2" data-testid="open-round-loss">
            <figcaption className="text-sm font-medium">Training loss by worker</figcaption>
            <SeriesChart series={loss} xLabel="Local step" yLabel="Loss" className="h-48 w-full" />
          </figure>
        )}
        {speed.length > 0 && (
          <figure className="flex min-w-0 flex-col gap-2" data-testid="open-round-speed">
            <figcaption className="text-sm font-medium">Useful tokens/s by worker</figcaption>
            <SeriesChart series={speed} xLabel="Local step" yLabel="Tokens/s" className="h-48 w-full" />
          </figure>
        )}
      </CardContent>
    </Card>
  );
}

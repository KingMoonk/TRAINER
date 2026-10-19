"use client";

import { SeriesChart } from "@/components/series-chart";
import { TRAINING as T } from "@/lib/copy/training";
import { errorMarks, lossSeries, type Run } from "./logic";

/** The error curve: training and validation loss per round, with the best round and the rounds where the run changed course marked. */
export function ErrorCurve({ run }: { run: Run }) {
  const series = lossSeries(run.history);
  return (
    <section className="flex flex-col gap-2" aria-label={T.errorCurve} data-testid="error-curve">
      <h3 className="text-sm font-medium">{T.errorCurve}</h3>
      <p className="max-w-prose text-xs text-muted-foreground">{T.errorCurveLead}</p>
      {series.length === 0 ? (
        <p className="text-sm text-muted-foreground">{T.noRounds}</p>
      ) : (
        <SeriesChart series={series} marks={errorMarks(run)} xLabel={T.roundAxis} yLabel={T.lossAxis} className="h-48 w-full" />
      )}
    </section>
  );
}

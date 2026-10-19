"use client";

import { SeriesChart, type Mark, type Series } from "@/components/series-chart";
import { Hint } from "@/app/runs/hint";
import { cn } from "@/lib/utils";
import { T } from "./copy";
import { overfitBad, overfitText, type Overfit } from "./logic";

export type Readout = { id: string; label: string; value: string; tip: string; colour?: string };

/** A loss chart with its numbers above it (each with a Tooltip saying what it is) and, for a single job, the overfitting pill below. */
export function LossPanel({ id, title, series, xLabel, readouts, empty, overfit, marks }: {
  id: string; title: string; series: Series[]; xLabel: string; readouts: Readout[]; empty: string; overfit?: Overfit; marks?: Mark[];
}) {
  return (
    <section aria-labelledby={id} className="flex min-w-0 flex-1 basis-96 flex-col gap-3 rounded-xl border border-border bg-card p-5" data-testid="loss-panel">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 id={id} className="text-lg font-semibold">{title}</h2>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs" aria-label={title}>
          {readouts.map((readout) => (
            <li key={readout.id} data-testid={`readout-${readout.id}`}>
              {readout.colour && <span aria-hidden="true" className="mr-1" style={{ color: readout.colour }}>━</span>}
              <Hint tip={readout.tip}>{readout.label} {readout.value}</Hint>
            </li>
          ))}
        </ul>
      </div>
      {series.length === 0 ? (
        <p className="py-10 text-center text-sm text-muted-foreground">{empty}</p>
      ) : (
        <SeriesChart series={series} xLabel={xLabel} yLabel={T.lossAxis} marks={marks} className="h-48 w-full" />
      )}
      {overfit && (
        <Hint tip={T.overfitTip} plain className="self-start">
          <span
            data-testid="overfit"
            data-state={overfit.state}
            className={cn(
              "inline-block rounded-full border px-2.5 py-1 font-mono text-xs",
              overfitBad(overfit) ? "border-destructive text-destructive" : overfit.state === "early" ? "border-border text-muted-foreground" : "border-success text-success",
            )}
          >
            {overfitText(overfit)}
          </span>
        </Hint>
      )}
    </section>
  );
}

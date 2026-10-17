"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SeriesChart, type Series } from "@/components/series-chart";
import { TRAINING as T } from "@/lib/copy/training";
import { EventLog } from "@/app/runs/view/event-log";
import { liveLossSeries, tokensSeries } from "@/app/runs/view/live";
import { bytesText, lossSeries, trafficSeries, type Run } from "./logic";
import { RoundBanner } from "@/app/runs/view/round-banner";
import { Timeline } from "@/app/runs/view/timeline";
import { Topology } from "@/app/runs/view/topology";

const CELL = "px-2 py-1.5 text-left align-top";

function Chart({ title, lead, series, xLabel, yLabel, empty }: { title: string; lead: string; series: Series[]; xLabel: string; yLabel: string; empty: string }) {
  return (
    <section className="flex flex-col gap-2" aria-label={title}>
      <h3 className="text-sm font-medium">{title}</h3>
      <p className="max-w-prose text-xs text-muted-foreground">{lead}</p>
      {series.length === 0 ? <p className="text-sm text-muted-foreground">{empty}</p> : <SeriesChart series={series} xLabel={xLabel} yLabel={yLabel} className="h-48 w-full" />}
    </section>
  );
}

/** The four charts: loss per round, the open round's inner loss and speed per machine, and the bytes moved per round. */
function Charts({ run }: { run: Run }) {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Chart title={T.progress} lead={T.progressLead} series={lossSeries(run.history)} xLabel={T.roundAxis} yLabel={T.lossAxis} empty={T.noRounds} />
      <Chart title={T.liveLossTitle} lead={T.liveLossLead} series={liveLossSeries(run)} xLabel={T.stepAxis} yLabel={T.lossAxis} empty={T.noSamples} />
      <Chart title={T.tokensTitle} lead={T.tokensLead} series={tokensSeries(run)} xLabel={T.stepAxis} yLabel={T.tokensAxis} empty={T.noSamples} />
      <Chart title={T.traffic} lead={T.trafficLead} series={run.history.length === 0 ? [] : trafficSeries(run.history)} xLabel={T.roundAxis} yLabel="MB" empty={T.noRounds} />
    </div>
  );
}

function RoundsTable({ run }: { run: Run }) {
  if (run.history.length === 0) return null;
  return (
    <details>
      <summary className="cursor-pointer text-sm font-medium">{T.roundsTable}</summary>
      <div className="overflow-x-auto">
        <table className="mt-2 w-full min-w-[32rem] text-sm" aria-label={`${T.roundsTable}: ${run.label}`}>
          <thead>
            <tr className="text-xs text-muted-foreground">
              <th scope="col" className={CELL}>{T.roundCol}</th>
              <th scope="col" className={CELL}>{T.trainLoss}</th>
              <th scope="col" className={CELL}>{T.validationLoss}</th>
              <th scope="col" className={CELL}>{T.up}</th>
              <th scope="col" className={CELL}>{T.down}</th>
              <th scope="col" className={CELL}>{T.contributedCol}</th>
              <th scope="col" className={CELL}>{T.droppedCol}</th>
            </tr>
          </thead>
          <tbody>
            {run.history.map((row) => (
              <tr key={row.round} className="border-t border-border tabular-nums" data-testid={`round-${row.round + 1}`}>
                <th scope="row" className={CELL}>{row.round + 1}</th>
                <td className={CELL}>{row.loss == null ? "" : row.loss.toFixed(3)}</td>
                <td className={CELL}>{row.val_loss == null ? "" : row.val_loss.toFixed(3)}</td>
                <td className={CELL}>{bytesText(row.bytes_up)}</td>
                <td className={CELL}>{bytesText(row.bytes_down)}</td>
                <td className={CELL}>{row.contributed} of {row.members}</td>
                <td className={CELL}>{row.dropped.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/**
 * The live view of one run: its state and settings, the round and its phase, the machines with their edges to the coordinator, the round
 * timeline, the charts, the event log, and the table of finished rounds.
 */
export function LiveView({ run, stoppingRun, stoppingNode, onStopRun, onStopMember }: {
  run: Run; stoppingRun: boolean; stoppingNode: string; onStopRun: () => void; onStopMember: (node: string) => void;
}) {
  return (
    <Card data-testid={`run-${run.id}`} aria-label={T.liveRunLabel.replace("{label}", run.label)}>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2">
          {run.label}
          <Badge variant={run.state === "failed" ? "destructive" : "secondary"} data-testid={`run-state-${run.id}`}>{T.states[run.state]}</Badge>
        </CardTitle>
        <CardDescription className="flex flex-col gap-0.5">
          <span>{T.coordinates.replace("{owner}", run.owner)}</span>
          <span>{T.roundOf.replace("{round}", String(run.round)).replace("{rounds}", String(run.rounds))}</span>
          <span>
            {T.settingsLine.replace("{steps}", run.inner_steps.toLocaleString("en-US")).replace("{lr}", String(run.outer_lr)).replace("{momentum}", String(run.outer_momentum))}
          </span>
        </CardDescription>
        {run.note && <p className="text-sm text-muted-foreground" data-testid={`run-note-${run.id}`}>{run.note}</p>}
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        <RoundBanner run={run} />
        <Topology run={run} stoppingRun={stoppingRun} stoppingNode={stoppingNode} onStopRun={onStopRun} onStopMember={onStopMember} />
        <Timeline run={run} />
        <Charts run={run} />
        <EventLog run={run} />
        <RoundsTable run={run} />
      </CardContent>
    </Card>
  );
}

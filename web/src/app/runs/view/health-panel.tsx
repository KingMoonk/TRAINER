"use client";

import { SeriesChart } from "@/components/series-chart";
import { Hint } from "@/app/runs/hint";
import { cn } from "@/lib/utils";
import type { Run } from "@/app/cluster/training/logic";
import { T, fill } from "./copy";
import { bytesSeries, compressionText, linkSeries, lossSeries, percentText, rateText, speedupText, splitRows, throughputSeries, verdictTone } from "./health";

const PARTS = [
  { key: "compute", label: T.hCompute, colour: "var(--chart-1)" },
  { key: "wait", label: T.hWaitLabel, colour: "var(--chart-4)" },
  { key: "sync", label: T.hSync, colour: "var(--chart-3)" },
] as const;

/** The verdict on whether the cluster helps, the numbers behind it, and the charts of each round's measurements. */
export function HealthPanel({ run }: { run: Run }) {
  const health = run.health;
  if (!health) return null;
  const rows = splitRows(health);
  const longest = Math.max(1e-9, ...rows.map((row) => row.compute + row.wait + row.sync));
  const compression = compressionText(health);
  const link = linkSeries(health);
  const readouts = [
    { id: "speedup", label: T.hSpeedup, value: speedupText(health), tip: T.hSpeedupTip },
    { id: "together", label: T.hAggregate, value: rateText(health.aggregate_tokens_per_s), tip: T.hAggregateTip },
    { id: "baseline", label: fill(T.hBaseline, { node: health.baseline_node }), value: rateText(health.baseline_tokens_per_s), tip: T.hBaselineTip },
    { id: "comm", label: T.hComm, value: percentText(health.comm_share), tip: T.hCommTip },
    { id: "wait", label: fill(T.hWait, { node: health.wait_node }), value: percentText(health.wait_share), tip: T.hWaitTip },
    { id: "idle", label: fill(T.hIdle, { node: health.idle_node }), value: percentText(health.idle_share), tip: T.hIdleTip },
  ].filter((readout) => readout.value !== "");
  return (
    <section aria-labelledby="run-health" className="flex flex-col gap-4 rounded-xl border border-border bg-card p-5" data-testid="health-panel">
      <h2 id="run-health" className="text-lg font-semibold">{T.health}</h2>
      <Hint tip={health.basis} plain className="self-start">
        <p data-testid="verdict" data-state={health.verdict.state} className={cn("rounded-full border px-3 py-1.5 text-sm", verdictTone(health.verdict.state))}>
          {health.verdict.text}
        </p>
      </Hint>
      {readouts.length > 0 && (
        <ul className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs" aria-label={T.health}>
          {readouts.map((readout) => (
            <li key={readout.id} data-testid={`health-${readout.id}`}><Hint tip={readout.tip}>{readout.label} {readout.value}</Hint></li>
          ))}
        </ul>
      )}
      {health.window > 0 && <p className="text-xs text-muted-foreground">{fill(T.hOver, { n: health.window })}</p>}
      {health.rounds.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">{T.hNoRounds}</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          <figure className="flex min-w-0 flex-col gap-2" data-testid="health-loss">
            <figcaption className="text-sm font-medium">{T.hLossTitle}</figcaption>
            <SeriesChart series={lossSeries(health)} xLabel={T.hAxisRound} yLabel="loss" className="h-48 w-full" />
          </figure>
          <figure className="flex min-w-0 flex-col gap-2" data-testid="health-throughput">
            <figcaption className="text-sm font-medium">{T.hThroughputTitle}</figcaption>
            <SeriesChart series={throughputSeries(health)} xLabel={T.hAxisRound} yLabel="tokens/s" className="h-48 w-full" />
          </figure>
          <figure className="flex min-w-0 flex-col gap-2" data-testid="health-split">
            <figcaption className="text-sm font-medium"><Hint tip={T.hSplitTip}>{T.hSplitTitle}</Hint></figcaption>
            <ul className="flex flex-col gap-1" aria-label={T.hSplitTitle}>
              {rows.map((row) => (
                <li key={row.round} className="flex items-center gap-2 font-mono text-xs" data-testid={`split-${row.round}`}>
                  <span className="w-8 shrink-0 text-right text-muted-foreground">{row.round}</span>
                  <span className="flex h-4 min-w-0 flex-1 overflow-hidden rounded-sm" style={{ width: "100%" }}>
                    {PARTS.map((part) => (
                      <span key={part.key} title={`${part.label} ${fill(T.hSeconds, { n: row[part.key].toFixed(1) })}`} style={{ width: `${(row[part.key] / longest) * 100}%`, background: part.colour }} />
                    ))}
                  </span>
                </li>
              ))}
            </ul>
            <ul className="flex flex-wrap gap-3 text-xs text-muted-foreground" aria-label={T.hSplitTitle}>
              {PARTS.map((part) => (
                <li key={part.key}><span aria-hidden="true" className="mr-1" style={{ color: part.colour }}>{"■"}</span>{part.label}</li>
              ))}
            </ul>
          </figure>
          {link.length > 0 && (
            <figure className="flex min-w-0 flex-col gap-2" data-testid="health-link">
              <figcaption className="text-sm font-medium"><Hint tip={T.hLinkTip}>{T.hLinkTitle}</Hint></figcaption>
              <SeriesChart series={link} xLabel={T.hAxisRound} yLabel="MB/s" className="h-48 w-full" />
            </figure>
          )}
          <figure className="flex min-w-0 flex-col gap-2" data-testid="health-bytes">
            <figcaption className="text-sm font-medium">{T.hBytesTitle}</figcaption>
            <SeriesChart series={bytesSeries(health)} xLabel={T.hAxisRound} yLabel="MB" className="h-48 w-full" />
            {compression && <p className="text-xs text-muted-foreground">{fill(T.hCompression, { ratio: compression })}</p>}
          </figure>
        </div>
      )}
    </section>
  );
}

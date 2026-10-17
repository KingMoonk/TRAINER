"use client";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TRAINING as T } from "@/lib/copy/training";
import type { Run } from "@/app/cluster/training/logic";
import { Hint } from "@/app/runs/hint";
import { T as V, fill } from "./copy";
import { laneSegments, percent, recentTimeline, timelineEnd, timelineExtent, type Segment } from "./live";
import { clock } from "./logic";

const COLOURS: Record<Segment["kind"], string> = {
  download: "var(--chart-3)",
  train: "var(--chart-1)",
  wait: "var(--border)",
  upload: "var(--chart-2)",
};

const ROW = "h-6";

/**
 * One row per machine, one column per round (the open one and the last few): θ downloaded, trained, waited, Δ uploaded, with the deadline marked.
 * The columns share one time axis so the rounds compare. A column holds its machines in the same order, so the rows line up.
 */
export function Timeline({ run }: { run: Run }) {
  const lines = recentTimeline(run);
  if (lines.length === 0) return null;
  const extent = timelineExtent(lines, run);
  const nodes = [...new Set(lines.flatMap((line) => line.lanes.map((lane) => lane.node)))];
  const first = lines[0].round + 1;
  const last = lines[lines.length - 1].round + 1;
  return (
    <section aria-label={T.timeline} className="flex flex-col gap-2" data-testid="timeline">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-medium">
          <Hint tip={V.timelineTip} plain>{T.timeline} · {fill(V.roundsRange, { from: first, to: last })}</Hint>
        </h3>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label={T.timeline}>
          {(Object.keys(COLOURS) as Segment["kind"][]).map((kind) => (
            <li key={kind} className="flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-sm" style={{ background: COLOURS[kind] }} aria-hidden="true" />
              {T.segments[kind]}
            </li>
          ))}
        </ul>
      </div>
      <div className="flex gap-3 overflow-x-auto">
        <div className="flex shrink-0 flex-col gap-1 pt-5">
          {nodes.map((node) => (
            <span key={node} className={`${ROW} max-w-28 truncate text-xs leading-6`}>{node}</span>
          ))}
        </div>
        {lines.map((line) => {
          const end = timelineEnd(line, run);
          return (
            <div key={line.round} className="flex min-w-44 flex-1 flex-col gap-1" data-testid={`timeline-round-${line.round + 1}`} data-open={line.open}>
              <p className="h-4 text-xs font-medium tabular-nums">
                {line.open ? fill(V.roundOpen, { n: line.round + 1 }) : fill(V.roundClosed, { n: line.round + 1, time: clock(end) })}
              </p>
              {nodes.map((node) => {
                const lane = line.lanes.find((l) => l.node === node);
                if (!lane) return <div key={node} className={`${ROW} rounded bg-muted/20`} aria-hidden="true" />;
                return (
                  <div key={node} className={`relative ${ROW} rounded bg-muted/40`} data-testid={`lane-${line.round + 1}-${lane.node}`}>
                    {laneSegments(lane, end).map((segment) => (
                      <Tooltip key={segment.kind}>
                        <TooltipTrigger asChild>
                          <span
                            data-segment={segment.kind}
                            aria-label={`${T.segments[segment.kind]}: ${T.secondsShort.replace("{n}", (segment.to - segment.from).toFixed(1))}`}
                            className="absolute top-0 h-full"
                            style={{ left: `${percent(segment.from, extent)}%`, width: `${percent(segment.to - segment.from, extent)}%`, background: COLOURS[segment.kind] }}
                          />
                        </TooltipTrigger>
                        <TooltipContent>{T.segments[segment.kind]}: {T.secondsShort.replace("{n}", (segment.to - segment.from).toFixed(1))}</TooltipContent>
                      </Tooltip>
                    ))}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span
                          data-testid={`deadline-${line.round + 1}-${lane.node}`}
                          aria-label={`${T.deadlineMark}: ${T.secondsShort.replace("{n}", line.deadline.toFixed(1))}`}
                          className="absolute top-[-2px] h-[calc(100%+4px)] w-0.5 bg-foreground"
                          style={{ left: `${percent(line.deadline, extent)}%` }}
                        />
                      </TooltipTrigger>
                      <TooltipContent>{T.deadlineMark}: {T.secondsShort.replace("{n}", line.deadline.toFixed(1))}</TooltipContent>
                    </Tooltip>
                    {lane.counted === false && (
                      <span className="absolute inset-y-0 right-1 flex items-center text-[0.7rem] text-foreground" data-testid={`nothing-${line.round + 1}-${lane.node}`}>
                        {T.countedForNothing}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </section>
  );
}

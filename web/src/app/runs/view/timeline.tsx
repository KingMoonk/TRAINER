"use client";

import { TRAINING as T } from "@/lib/copy/training";
import { laneSegments, percent, recentTimeline, timelineEnd, timelineExtent, type Segment } from "./live";
import type { Run } from "@/app/cluster/training/logic";

const COLOURS: Record<Segment["kind"], string> = {
  download: "var(--chart-3)",
  train: "var(--chart-1)",
  wait: "var(--border)",
  upload: "var(--chart-2)",
};

/** One lane per machine over the open round and the last few: θ downloaded, trained, waited, Δ uploaded, with the deadline marked. */
export function Timeline({ run }: { run: Run }) {
  const lines = recentTimeline(run);
  if (lines.length === 0) return null;
  const extent = timelineExtent(lines, run);
  return (
    <section aria-label={T.timeline} className="flex flex-col gap-2" data-testid="timeline">
      <h3 className="text-sm font-medium">{T.timeline}</h3>
      <p className="max-w-prose text-xs text-muted-foreground">{T.timelineLead}</p>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label={T.timeline}>
        {(Object.keys(COLOURS) as Segment["kind"][]).map((kind) => (
          <li key={kind} className="flex items-center gap-1.5">
            <span className="inline-block size-2.5 rounded-sm" style={{ background: COLOURS[kind] }} aria-hidden="true" />
            {T.segments[kind]}
          </li>
        ))}
      </ul>
      {lines.map((line) => {
        const end = timelineEnd(line, run);
        return (
          <div key={line.round} className="flex flex-col gap-1" data-testid={`timeline-round-${line.round + 1}`} data-open={line.open}>
            <p className="text-xs font-medium">{(line.open ? T.timelineOpen : T.timelineRound).replace("{n}", String(line.round + 1))}</p>
            {line.lanes.map((lane) => (
              <div key={lane.node} className="grid grid-cols-[6rem_1fr] items-center gap-2" data-testid={`lane-${line.round + 1}-${lane.node}`}>
                <span className="truncate text-xs" title={lane.node}>{lane.node}</span>
                <div className="relative h-5 rounded bg-muted/40">
                  {laneSegments(lane, end).map((segment) => (
                    <span
                      key={segment.kind}
                      data-segment={segment.kind}
                      title={`${T.segments[segment.kind]}: ${T.secondsShort.replace("{n}", (segment.to - segment.from).toFixed(1))}`}
                      className="absolute top-0 h-full"
                      style={{ left: `${percent(segment.from, extent)}%`, width: `${percent(segment.to - segment.from, extent)}%`, background: COLOURS[segment.kind] }}
                    />
                  ))}
                  <span
                    data-testid={`deadline-${line.round + 1}-${lane.node}`}
                    title={`${T.deadlineMark}: ${T.secondsShort.replace("{n}", line.deadline.toFixed(1))}`}
                    className="absolute top-[-2px] h-[calc(100%+4px)] w-0.5 bg-foreground"
                    style={{ left: `${percent(line.deadline, extent)}%` }}
                  />
                  {lane.counted === false && (
                    <span className="absolute inset-y-0 right-1 flex items-center text-[0.7rem] text-foreground" data-testid={`nothing-${line.round + 1}-${lane.node}`}>
                      {T.countedForNothing}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </section>
  );
}

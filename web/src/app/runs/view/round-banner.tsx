"use client";

import { TRAINING as T } from "@/lib/copy/training";
import type { Run } from "@/app/cluster/training/logic";
import { Hint } from "@/app/runs/hint";
import { T as V, fill } from "./copy";
import { collectingText, countdownText, roundNumber, STAGES, stageIndex } from "./live";
import { percentOf } from "./logic";

/**
 * "Round 3 of 10" with the phase it is in (members training, collecting changes, outer step, checking held-back text, next round), a bar of the
 * rounds done, and the time to the round's deadline with the grace after it.
 */
export function RoundBanner({ run }: { run: Run }) {
  const stage = stageIndex(run);
  const countdown = countdownText(run);
  return (
    <section aria-label={T.banner} className="flex flex-col gap-2 rounded-lg border border-border p-3" data-testid="round-banner">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-base font-semibold" data-testid="round-title">
          {T.roundBanner.replace("{round}", String(roundNumber(run))).replace("{rounds}", String(run.rounds))}
          {stage < 0 && <span className="ml-2 text-sm font-normal text-muted-foreground" data-testid="round-phase">{T.phases[run.phase]}</span>}
        </p>
        {countdown && (
          <Hint tip={V.deadlineTip} plain className="text-sm tabular-nums text-muted-foreground">
            <span data-testid="countdown">{countdown}</span>
            {run.grace_s > 0 && <span data-testid="grace"> · {fill(V.grace, { n: Math.round(run.grace_s) })}</span>}
          </Hint>
        )}
      </div>
      <div
        role="progressbar"
        aria-label={V.roundsBar}
        aria-valuemin={0}
        aria-valuemax={run.rounds}
        aria-valuenow={run.round}
        className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
      >
        <div className="h-full bg-primary" style={{ width: `${percentOf(run.round, run.rounds)}%` }} />
      </div>
      <ol className="flex flex-wrap gap-2" aria-label={T.stagesLabel}>
        {STAGES.map((name, index) => {
          const active = index === stage;
          const extra = active ? collectingText(run) : "";
          return (
            <li
              key={name}
              aria-current={active ? "step" : undefined}
              data-active={active}
              data-testid={`stage-${name}`}
              className={`rounded-full px-3 py-1 text-xs ${active ? "bg-primary text-primary-foreground" : index < stage ? "bg-muted text-foreground" : "bg-muted text-muted-foreground"}`}
            >
              <Hint tip={V.stageTips[name]} plain>
                {index < stage && <span aria-hidden="true">✓ </span>}
                {T.stages[name]}
                {extra && <> ({extra})</>}
              </Hint>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

"use client";

import { TRAINING as T } from "@/lib/copy/training";
import { collectingText, countdownText, roundNumber, STAGES, stageIndex } from "./live";
import type { Run } from "./logic";

/** "Round 3 of 10" with the phase it is in (members training, collecting changes, outer step, checking held-back text, next round) and the time to its deadline. */
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
        {countdown && <p className="text-sm tabular-nums text-muted-foreground" data-testid="countdown">{countdown}</p>}
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
              className={`rounded-full px-3 py-1 text-xs ${active ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}
            >
              {T.stages[name]}
              {extra && <> ({extra})</>}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

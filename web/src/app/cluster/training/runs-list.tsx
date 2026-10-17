"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TRAINING as T } from "@/lib/copy/training";
import { otherRuns } from "./live";
import type { Run } from "./logic";

/** The other runs, a short list under the live one. Choosing one makes it the live run. */
export function RunsList({ runs, live, onSelect }: { runs: Run[]; live: Run | undefined; onSelect: (id: string) => void }) {
  const { shown, hidden } = otherRuns(runs, live);
  if (shown.length === 0) return null;
  return (
    <section aria-labelledby="training-other-runs" className="flex flex-col gap-2" data-testid="other-runs">
      <h2 id="training-other-runs" className="text-lg font-semibold tracking-tight">{T.otherRuns}</h2>
      <p className="text-xs text-muted-foreground">{T.otherRunsLead}</p>
      <ul className="flex flex-col gap-2">
        {shown.map((run) => (
          <li key={run.id} className="flex flex-wrap items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm" data-testid={`other-${run.id}`}>
            <span className="font-medium">{run.label}</span>
            <Badge variant={run.state === "failed" ? "destructive" : "secondary"} data-testid={`other-state-${run.id}`}>{T.states[run.state]}</Badge>
            <span className="text-xs text-muted-foreground">{T.roundOf.replace("{round}", String(run.round)).replace("{rounds}", String(run.rounds))}</span>
            <Button className="ml-auto" variant="outline" size="sm" aria-label={T.showRunLabel.replace("{label}", run.label)} onClick={() => onSelect(run.id)}>
              {T.showRun}
            </Button>
          </li>
        ))}
      </ul>
      {hidden > 0 && <p className="text-xs text-muted-foreground">{T.moreRuns.replace("{n}", String(hidden))}</p>}
    </section>
  );
}

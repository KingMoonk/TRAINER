"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ROUNDS, T, fill } from "./copy";
import { stepLine, type Gate, type Queue, type Status } from "./logic";

function Progress({ fraction }: { fraction: number }) {
  return (
    <div role="progressbar" aria-label="Round progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)} className="h-2 w-full max-w-md rounded-full bg-muted">
      <div className="h-2 rounded-full bg-violet-500 transition-all" style={{ width: `${Math.round(fraction * 100)}%` }} />
    </div>
  );
}

/** The verdict of the open round: the gate that decides whether Chat moves, with the step count while it trains. */
export function GateCard({
  round,
  status,
  gate,
  queue,
  checkpoint,
  overlap,
}: {
  round: number;
  status: string;
  gate: Gate | undefined;
  queue: Queue | undefined;
  checkpoint: string;
  overlap: Status["overlap"];
}) {
  const spec = ROUNDS[round - 1];
  const running = status === "running";
  const lead = spec.verdict;
  let verdict: string;
  if (gate) verdict = gate.verdict || (spec.counted ? "" : lead);
  else if (spec.counted) verdict = fill(running ? T.soFarCount : T.willCount, { what: lead });
  else if (running) verdict = fill(T.soFar, { verdict: lead[0].toLowerCase() + lead.slice(1) });
  else verdict = fill(T.willRead, { verdict: lead });
  const checks = gate?.checks ?? [
    { name: T.heldBackCheck, detail: T.measuredAtEnd, ok: null },
    { name: T.problemsCheck, detail: T.measuredAtEnd, ok: null },
  ];
  const result = gate ? (gate.ok ? T.resultLive : T.resultKept) : running ? T.chatKeeps : T.untilThen;
  const total = queue?.total ?? 0;
  const shown = Object.entries(overlap ?? {}).filter(([, o]) => o.left_out);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-violet-400">
          {gate ? `Result · Round ${round}` : running ? `Training · Round ${round}` : "When this round ends"}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {running && (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted-foreground" data-testid="step-line">
              {stepLine(queue)}
            </p>
            {total > 0 && <Progress fraction={Math.min(1, (queue?.step ?? 0) / total)} />}
          </div>
        )}
        <div>
          <div className="text-xs uppercase text-muted-foreground">Verdict</div>
          <p className={cn("mt-1 text-lg", gate ? "text-emerald-400" : "text-muted-foreground")}>{verdict}</p>
        </div>
        <p className="text-xs text-muted-foreground">{T.chatMoves}</p>
        <ul className="flex flex-col gap-2">
          {checks.map((c) => (
            <li key={c.name} className="flex items-start gap-3">
              <span aria-hidden className={cn("w-4 text-center font-bold", c.ok == null ? "text-muted-foreground" : c.ok ? "text-emerald-400" : "text-rose-400")}>
                {c.ok == null ? "…" : c.ok ? "✓" : "✕"}
              </span>
              <span>
                <span className="block text-sm">{c.name}</span>
                <span className="block text-xs text-muted-foreground">{c.detail}</span>
              </span>
            </li>
          ))}
        </ul>
        {gate?.judge ? (
          <p className="text-sm">
            <span className="font-mono">{gate.judge[0]}</span> of <span className="font-mono">{gate.judge[1]}</span> prompts preferred by the judge{" "}
            <span className="text-xs text-muted-foreground">({T.judgeNote})</span>
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">{T.judgeNone}</p>
        )}
        {shown.map(([kind, o]) => (
          <p key={kind} className="text-xs text-muted-foreground">
            {fill(T.overlapLeftOut, { leftOut: o.left_out, checked: o.checked, what: kind === "problems" ? "problems" : "documents" })}
          </p>
        ))}
        <p className="text-sm">{fill(result, { checkpoint })}</p>
      </CardContent>
    </Card>
  );
}

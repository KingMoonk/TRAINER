"use client";

import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { TRAINING as T } from "@/lib/copy/training";
import { cn } from "@/lib/utils";
import { isActive, isProblem, memberDevice, needsReason, reasonText, type Member, type Run } from "@/app/cluster/training/logic";
import { T as V } from "./copy";
import {
  deviceBadge, edges, liveLossText, liveSpeedText, memberPhaseText, progressFraction, reportedText, shardsText, stateColor, stepsText, type Edge,
} from "./live";

/** One edge between the coordinator and a member: θ goes out at the start of a round, Δ comes back at its end. It moves while the transfer runs. */
function EdgeView({ edge, node }: { edge: Edge; node: string }) {
  const down = edge.direction === "down";
  const colour = edge.active ? (down ? "var(--warning)" : "var(--info)") : "var(--border)";
  const head = (
    <svg viewBox="0 0 8 12" width="8" height="12" className="shrink-0" aria-hidden="true">
      <polygon points={down ? "0,0 8,6 0,12" : "8,0 0,6 8,12"} fill={colour} />
    </svg>
  );
  return (
    <div className="flex flex-col gap-0.5 text-xs" data-testid={`edge-${edge.direction}-${node}`} data-active={edge.active}>
      <span className="flex items-baseline justify-between gap-2">
        <span className="text-muted-foreground">{down ? T.thetaDown : T.deltaUp}</span>
        <span className="tabular-nums">{edge.label}</span>
      </span>
      <span className="flex items-center" aria-hidden="true">
        {!down && head}
        <svg height="12" className="min-w-8 flex-1">
          <line x1="0" x2="100%" y1="6" y2="6" stroke={colour} strokeWidth="2" className={edge.active ? "edge-flow" : undefined} />
        </svg>
        {down && head}
      </span>
    </div>
  );
}

function MemberNode({ run, member, action }: { run: Run; member: Member; action: ReactNode }) {
  const bad = isProblem(member);
  const why = needsReason(member);
  const fraction = progressFraction(member, run);
  const facts = [liveSpeedText(member), liveLossText(member), shardsText(member), reportedText(member)].filter(Boolean);
  const round = run.phase === "training" || run.phase === "collecting";
  return (
    <div
      className={cn("flex flex-col gap-2 rounded-lg border-2 bg-info/5 p-3", bad ? "border-destructive" : "border-info")}
      data-testid={`node-${member.node}`}
      data-state={member.state}
      data-phase={member.phase ?? ""}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{member.node}</span>
        {member.device && <Badge variant="outline" data-testid={`device-${member.node}`}>{deviceBadge(member)}</Badge>}
        <Badge variant={bad ? "destructive" : "secondary"} data-testid={`state-${member.node}`}>{T.memberStates[member.state]}</Badge>
        {member.local && <span className="text-xs text-muted-foreground">{T.thisComputer}</span>}
        {action && <span className="ml-auto">{action}</span>}
      </div>
      {memberDevice(member) && <p className="text-xs text-muted-foreground">{memberDevice(member)}</p>}
      {member.phase && member.state === "training" && <p className="text-sm" data-testid={`phase-${member.node}`}>{memberPhaseText(member, run)}</p>}
      <div className="flex flex-col gap-1">
        <div
          role="progressbar"
          aria-label={T.progressOf.replace("{node}", member.node)}
          aria-valuemin={0}
          aria-valuemax={member.round_steps || run.inner_steps}
          aria-valuenow={member.step_in_round}
          className="h-2 w-full overflow-hidden rounded-full bg-muted"
        >
          <div className="h-full" style={{ width: `${Math.round(fraction * 100)}%`, background: stateColor(member) }} />
        </div>
        <p className="text-xs tabular-nums text-muted-foreground">{stepsText(member, run)}</p>
      </div>
      <p className="flex flex-wrap gap-x-3 text-xs tabular-nums text-muted-foreground" data-testid={`facts-${member.node}`}>
        {facts.map((fact) => <span key={fact}>{fact}</span>)}
      </p>
      {round && member.state === "training" && (
        <p className="text-xs text-muted-foreground" data-testid={`sent-${member.node}`}>{member.sent ? V.deltaIn : V.deltaPending}</p>
      )}
      {why && <p className="text-sm text-muted-foreground" data-testid={`reason-${member.node}`}>{reasonText(member)}</p>}
    </div>
  );
}

/**
 * The run as a picture: the coordinator on the left (the shared weights θ, the outer optimiser and the held-back text) and a node for each
 * machine to its right, each joined to it by a θ edge and a Δ edge. The coordinator is the warning colour, a worker the info colour.
 * The page decides what controls go on the coordinator and on each machine (`coordinatorAction`, `memberAction`).
 */
export function Topology({ run, coordinatorAction, memberAction }: {
  run: Run; coordinatorAction?: ReactNode; memberAction?: (member: Member) => ReactNode;
}) {
  return (
    <section aria-label={`${T.topology}: ${run.label}`} className="flex flex-col gap-3" data-testid="topology">
      <ul
        className="grid gap-x-4 gap-y-4 md:grid-cols-[minmax(0,15rem)_minmax(6rem,1fr)_minmax(0,24rem)]"
        style={{ "--rows": Math.max(1, run.members.length) } as React.CSSProperties}
      >
        <li
          className="flex flex-col items-center gap-1 rounded-lg border-2 border-warning bg-warning/5 p-3 text-center md:col-start-1 md:row-start-1 md:self-center md:[grid-row:1/span_var(--rows)]"
          data-testid="coordinator"
        >
          <p className="font-mono text-xs text-warning">{V.coordinatorRole}</p>
          <p className="font-medium">{T.coordinatorTitle}: {run.owner}</p>
          <p className="text-xs text-muted-foreground">{T.coordinatorHolds}</p>
          {coordinatorAction && isActive(run) && coordinatorAction}
        </li>
        {run.members.map((member) => {
          const [down, up] = edges(run, member);
          return (
            <li key={member.node} className="contents" data-testid={`member-${member.node}`}>
              <div className="flex flex-col justify-center gap-2 md:col-start-2">
                <EdgeView edge={down} node={member.node} />
                <EdgeView edge={up} node={member.node} />
              </div>
              <div className="md:col-start-3">
                <MemberNode run={run} member={member} action={memberAction?.(member)} />
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TRAINING as T } from "@/lib/copy/training";
import {
  deviceBadge, edges, liveLossText, liveSpeedText, progressFraction, reportedText, shardsText, stateColor, stepsText, type Edge,
} from "./live";
import { isActive, memberDevice, needsReason, reasonText, type Member, type Run } from "@/app/cluster/training/logic";

/** One edge between the coordinator and a member: θ goes down at the start of a round, Δ comes up at its end. It moves while the transfer runs. */
function EdgeView({ edge, node }: { edge: Edge; node: string }) {
  const down = edge.direction === "down";
  const colour = edge.active ? (down ? "var(--chart-1)" : "var(--chart-2)") : "var(--border)";
  return (
    <div className="flex items-stretch gap-2" data-testid={`edge-${edge.direction}-${node}`} data-active={edge.active}>
      <svg viewBox="0 0 12 48" width="12" className="h-12 shrink-0" aria-hidden="true">
        <line x1="6" x2="6" y1={down ? 2 : 46} y2={down ? 38 : 10} stroke={colour} strokeWidth="2" className={edge.active ? "edge-flow" : undefined} />
        <polygon points={down ? "1,38 11,38 6,46" : "1,10 11,10 6,2"} fill={colour} />
      </svg>
      <span className="flex flex-col justify-center text-xs">
        <span className="text-muted-foreground">{down ? T.thetaDown : T.deltaUp}</span>
        <span className="tabular-nums">{edge.label}</span>
      </span>
    </div>
  );
}

function MemberNode({ run, member, stopping, onStop }: { run: Run; member: Member; stopping: boolean; onStop: () => void }) {
  const bad = needsReason(member);
  const fraction = progressFraction(member, run);
  const facts = [liveSpeedText(member), liveLossText(member), shardsText(member), reportedText(member)].filter(Boolean);
  return (
    <div
      className="flex flex-col gap-2 rounded-lg border-2 p-3"
      style={{ borderColor: stateColor(member) }}
      data-testid={`node-${member.node}`}
      data-state={member.state}
      data-phase={member.phase ?? ""}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{member.node}</span>
        {member.device && <Badge variant="outline" data-testid={`device-${member.node}`}>{deviceBadge(member)}</Badge>}
        <Badge variant={bad ? "destructive" : "secondary"} data-testid={`state-${member.node}`}>{T.memberStates[member.state]}</Badge>
        {member.local && <span className="text-xs text-muted-foreground">{T.thisComputer}</span>}
        {member.can_stop && (
          <Button className="ml-auto" variant="outline" size="sm" disabled={stopping} aria-label={T.stopMemberLabel.replace("{node}", member.node)} onClick={onStop}>
            {T.stopMember}
          </Button>
        )}
      </div>
      {memberDevice(member) && <p className="text-xs text-muted-foreground">{memberDevice(member)}</p>}
      {member.phase && member.state === "training" && <p className="text-sm" data-testid={`phase-${member.node}`}>{T.memberPhases[member.phase]}</p>}
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
      {bad && <p className="text-sm text-muted-foreground" data-testid={`reason-${member.node}`}>{reasonText(member)}</p>}
    </div>
  );
}

/**
 * The run as a picture: the coordinator on top (the shared weights θ, the outer optimizer and the held-back text) and a node for each machine below,
 * each joined to it by a θ edge and a Δ edge. Stop run sits on the coordinator, Stop on each machine.
 */
export function Topology({ run, stoppingRun, stoppingNode, onStopRun, onStopMember }: {
  run: Run; stoppingRun: boolean; stoppingNode: string; onStopRun: () => void; onStopMember: (node: string) => void;
}) {
  return (
    <section aria-label={`${T.topology}: ${run.label}`} className="flex flex-col gap-3" data-testid="topology">
      <div className="mx-auto flex w-full max-w-lg flex-col items-center gap-1 rounded-lg border-2 border-border p-3 text-center" data-testid="coordinator">
        <p className="font-medium">{T.coordinatorTitle}: {run.owner}</p>
        <p className="text-xs text-muted-foreground">{T.coordinatorHolds}</p>
        {run.can_stop && isActive(run) && (
          <Button variant="outline" size="sm" disabled={stoppingRun} onClick={onStopRun}>{stoppingRun ? T.stoppingRun : T.stopRun}</Button>
        )}
      </div>
      <ul className="grid gap-x-4 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
        {run.members.map((member) => {
          const [down, up] = edges(run, member);
          return (
            <li key={member.node} className="flex flex-col gap-1" data-testid={`member-${member.node}`}>
              <div className="grid grid-cols-2 gap-2">
                <EdgeView edge={down} node={member.node} />
                <EdgeView edge={up} node={member.node} />
              </div>
              <MemberNode run={run} member={member} stopping={stoppingNode === member.node} onStop={() => onStopMember(member.node)} />
            </li>
          );
        })}
      </ul>
    </section>
  );
}

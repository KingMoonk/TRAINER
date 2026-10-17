"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SeriesChart } from "@/components/series-chart";
import { TRAINING as T } from "@/lib/copy/training";
import { bytesText, deltaText, isActive, lossSeries, memberDevice, needsReason, reasonText, speedText, trafficSeries, type Member, type Run } from "./logic";

const CELL = "px-2 py-1.5 text-left align-top";

function MemberRow({ member, stopping, onStop }: { member: Member; stopping: boolean; onStop: () => void }) {
  const bad = needsReason(member);
  return (
    <>
      <tr className="border-t border-border" data-testid={`member-${member.node}`}>
        <th scope="row" className={`${CELL} font-medium`}>
          {member.node}
          {member.local && <span className="ml-2 text-xs font-normal text-muted-foreground">{T.thisComputer}</span>}
        </th>
        <td className={CELL}>{memberDevice(member)}</td>
        <td className={CELL}>
          <Badge variant={bad ? "destructive" : "secondary"}>{T.memberStates[member.state]}</Badge>
        </td>
        <td className={`${CELL} tabular-nums`}>{member.rounds_done}</td>
        <td className={`${CELL} tabular-nums`}>{member.steps.toLocaleString("en-US")}</td>
        <td className={`${CELL} tabular-nums`}>{speedText(member)}</td>
        <td className={`${CELL} tabular-nums`}>{deltaText(member)}</td>
        <td className={`${CELL} text-right`}>
          {member.can_stop && (
            <Button variant="outline" size="sm" disabled={stopping} aria-label={T.stopMemberLabel.replace("{node}", member.node)} onClick={onStop}>
              {T.stopMember}
            </Button>
          )}
        </td>
      </tr>
      {bad && (
        <tr data-testid={`reason-${member.node}`}>
          <td colSpan={8} className="px-2 pb-2 text-sm text-muted-foreground">{reasonText(member)}</td>
        </tr>
      )}
    </>
  );
}

function Members({ run, stoppingNode, onStopMember }: { run: Run; stoppingNode: string; onStopMember: (node: string) => void }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[40rem] text-sm" aria-label={`${T.members}: ${run.label}`}>
        <thead>
          <tr className="text-xs text-muted-foreground">
            <th scope="col" className={CELL}>{T.machine}</th>
            <th scope="col" className={CELL}>{T.device}</th>
            <th scope="col" className={CELL}>{T.state}</th>
            <th scope="col" className={CELL}>{T.round}</th>
            <th scope="col" className={CELL}>{T.steps}</th>
            <th scope="col" className={CELL}>{T.speed}</th>
            <th scope="col" className={CELL}>{T.lastDelta}</th>
            <th scope="col" className={CELL}><span className="sr-only">{T.stopMember}</span></th>
          </tr>
        </thead>
        <tbody>
          {run.members.map((member) => (
            <MemberRow key={member.node} member={member} stopping={stoppingNode === member.node} onStop={() => onStopMember(member.node)} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Progress({ run }: { run: Run }) {
  const loss = lossSeries(run.history);
  if (run.history.length === 0) return <p className="text-sm text-muted-foreground">{T.noRounds}</p>;
  return (
    <div className="flex flex-col gap-6">
      <section className="flex flex-col gap-2" aria-label={T.progress}>
        <h3 className="text-sm font-medium">{T.progress}</h3>
        <p className="max-w-prose text-xs text-muted-foreground">{T.progressLead}</p>
        <SeriesChart series={loss} xLabel={T.roundAxis} yLabel={T.lossAxis} className="h-56 w-full" />
      </section>
      <section className="flex flex-col gap-2" aria-label={T.traffic}>
        <h3 className="text-sm font-medium">{T.traffic}</h3>
        <p className="max-w-prose text-xs text-muted-foreground">{T.trafficLead}</p>
        <SeriesChart series={trafficSeries(run.history)} xLabel={T.roundAxis} yLabel="MB" className="h-48 w-full" />
      </section>
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
    </div>
  );
}

/** One run: its state, who coordinates, every machine in it with the reason it is out (if it is), and how the loss and traffic went. */
export function RunCard({ run, stoppingRun, stoppingNode, onStopRun, onStopMember }: {
  run: Run; stoppingRun: boolean; stoppingNode: string; onStopRun: () => void; onStopMember: (node: string) => void;
}) {
  return (
    <Card data-testid={`run-${run.id}`}>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex flex-wrap items-center gap-2">
            {run.label}
            <Badge variant={run.state === "failed" ? "destructive" : "secondary"} data-testid={`run-state-${run.id}`}>{T.states[run.state]}</Badge>
          </CardTitle>
          {run.can_stop && isActive(run) && (
            <Button variant="outline" size="sm" disabled={stoppingRun} onClick={onStopRun}>{stoppingRun ? T.stoppingRun : T.stopRun}</Button>
          )}
        </div>
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
        <Members run={run} stoppingNode={stoppingNode} onStopMember={onStopMember} />
        <Progress run={run} />
      </CardContent>
    </Card>
  );
}

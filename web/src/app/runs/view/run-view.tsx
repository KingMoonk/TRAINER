"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { isActive, lessonsText, stopReasonText, type Run } from "@/app/cluster/training/logic";
import { Lessons } from "@/app/cluster/training/lessons-list";
import { MemberAlerts } from "@/app/cluster/training/member-alerts";
import { RunControls } from "@/app/cluster/training/run-controls";
import { useStopMember, useStopRun } from "@/app/cluster/training/use-training";
import { useQueueActions } from "@/app/queue/use-queue";
import { etaText, runEta } from "@/app/runs/logic";
import { Hint } from "@/app/runs/hint";
import { TRAINING as TR } from "@/lib/copy/training";
import { stateLabel } from "@/lib/members";
import { RemoveMemberButton, SoloControls, StopRunButton } from "./controls";
import { T, fill } from "./copy";
import { EventLog } from "./event-log";
import { ExtendRunButton } from "./extend-run";
import { ProgressCard, RunHeader } from "./header";
import { HealthPanel } from "./health-panel";
import { factAge, memberPhaseText, roundNumber } from "./live";
import { LossPanel } from "./loss-panel";
import { dilocoShape, runViewHref, gapNow, gapText, lastOf, lossText, overfit, percentOf, soloShape, stepSeries, trainLine } from "./logic";
import { OutcomePanel } from "./outcome-panel";
import { OpenRoundPanel } from "./open-round-panel";
import { RoundBanner } from "./round-banner";
import { Timeline } from "./timeline";
import { Topology } from "./topology";
import type { RunView } from "./use-run-view";

const PANEL = "flex min-w-0 flex-1 basis-96 flex-col gap-3 rounded-xl border border-border bg-card p-5";

/** A run spread over several machines: the round, who is doing what, the round timeline, the loss by round and the events. */
export function DilocoView({ run }: { run: Run }) {
  const router = useRouter();
  const stopRun = useStopRun();
  const stopMember = useStopMember();
  const eta = runEta(run);
  const status = [TR.phases[run.phase], eta == null ? "" : fill(T.eta, { eta: etaText(eta) })].filter(Boolean).join(" · ");
  return (
    <div className="flex flex-col gap-6" data-testid={`run-${run.id}`}>
      <RunHeader
        title={run.label || T.untitled}
        shape={dilocoShape(run)}
        freshness={factAge(run)}
        controls={
          <div className="flex flex-wrap items-center gap-2">
            <RunControls run={run} onRestarted={(made) => router.push(runViewHref(made.id))} />
            {run.can_stop && isActive(run) && (
              <div role="group" aria-label={T.controls} className="flex items-center gap-2">
                <span className="font-mono text-xs text-warning">{T.coordinatorLabel}</span>
                <ExtendRunButton run={run} />
                <StopRunButton pending={stopRun.isPending && stopRun.variables === run.id} onConfirm={() => stopRun.mutate(run.id)} />
              </div>
            )}
          </div>
        }
      />
      {run.paused && <p className="text-sm text-warning" data-testid="run-paused">{TR.pausedNote}</p>}
      {stopReasonText(run) && <p className="text-sm text-muted-foreground" data-testid="run-stop-reason">{stopReasonText(run)}</p>}
      {lessonsText(run) && <p className="text-sm text-muted-foreground" data-testid={`run-lessons-${run.id}`}>{lessonsText(run)}</p>}
      {run.note && <p className="text-sm text-muted-foreground" data-testid="run-note">{run.note}</p>}
      <ProgressCard
        now={fill(T.roundOf, { round: roundNumber(run) })}
        of={fill(T.roundTotal, { rounds: run.rounds })}
        percent={percentOf(run.round, run.rounds)}
        status={status}
        statusTip={T.etaTip}
      />
      <OutcomePanel run={run} />
      {isActive(run) && run.lessons + run.lessons_left_out > 0 && <Lessons run={run} />}
      <MemberAlerts run={run} />
      <RoundBanner run={run} />
      <section aria-labelledby="run-who" className="flex flex-col gap-3 rounded-xl border border-border bg-card p-5">
        <h2 id="run-who" className="text-lg font-semibold"><Hint tip={T.whoTip} plain>{T.who}</Hint></h2>
        <Topology
          run={run}
          memberAction={(member) =>
            member.can_stop && isActive(run) ? (
              <RemoveMemberButton
                node={member.node}
                pending={stopMember.isPending && stopMember.variables.id === run.id && stopMember.variables.node === member.node}
                onConfirm={() => stopMember.mutate({ id: run.id, node: member.node })}
              />
            ) : null
          }
        />
      </section>
      <OpenRoundPanel run={run} />
      <HealthPanel run={run} />
      <div className="rounded-xl border border-border bg-card p-5">
        <Timeline run={run} />
      </div>
      <div className={PANEL}><EventLog run={run} /></div>
    </div>
  );
}

/** A job on this computer alone: |W| = 1, so no round, timeline or transfer panels. */
export function SoloView({ view }: { view: Extract<RunView, { kind: "solo" }> }) {
  const { job, queue, running, machine, curve } = view;
  const actions = useQueueActions();
  const busy = actions.pause.isPending || actions.resume.isPending || actions.stopAfterCooldown.isPending || actions.remove.isPending;
  const step = running ? queue.step : 0;
  const total = running ? queue.total : 0;
  const eta = running ? queue.eta_s : job.eta_s;
  const status = [
    running ? queue.phase || T.phaseWord : job.status,
    running && queue.lr != null ? fill(T.lr, { lr: queue.lr.toExponential(1) }) : "",
    eta != null ? fill(T.eta, { eta: etaText(eta) }) : "",
  ].filter(Boolean).join(" · ");
  const train = curve ? lastOf(trainLine(curve)) : null;
  const val = curve ? lastOf(curve.heldout) : null;
  const readouts = [
    { id: "train", label: T.lTrain, value: lossText(train), tip: T.lTrainTip, colour: "var(--info)" },
    { id: "val", label: T.lVal, value: lossText(val), tip: T.lValTip, colour: "var(--warning)" },
    { id: "gap", label: T.gap, value: gapText(curve ? gapNow(curve) : null), tip: T.gapTip },
  ].filter((readout) => readout.value !== "");
  return (
    <div className="flex flex-col gap-6" data-testid={`run-${job.id}`}>
      <RunHeader
        title={job.label || job.kind}
        shape={soloShape(machine)}
        controls={
          running ? (
            <SoloControls
              paused={queue.paused}
              stopping={queue.stopping}
              busy={busy}
              onPause={() => actions.pause.mutate()}
              onResume={() => actions.resume.mutate()}
              onCoolDown={() => actions.stopAfterCooldown.mutate()}
              onStopNow={() => actions.remove.mutate(job.id)}
            />
          ) : undefined
        }
      />
      {job.note && <p className="text-sm text-muted-foreground">{job.note}</p>}
      <ProgressCard
        now={fill(T.stepOf, { step: step.toLocaleString("en-US") })}
        of={total > 0 ? fill(T.stepTotal, { total: total.toLocaleString("en-US") }) : ""}
        percent={percentOf(step, total)}
        status={status}
        statusTip={T.soloStatusTip}
      />
      <section aria-labelledby="run-who" className="flex flex-col gap-3 rounded-xl border border-border bg-card p-5">
        <h2 id="run-who" className="text-lg font-semibold">{T.who}</h2>
        <Hint tip={T.soloNodeTip} plain className="w-fit max-w-md">
          <span className="flex flex-col gap-1.5 rounded-xl border-2 border-warning bg-info/10 p-5" data-testid="solo-node">
            <span><b className="text-lg">{machine}</b></span>
            <span className="font-mono text-xs text-warning">{T.coordinatorWorker}</span>
            <span className="text-sm">{T.localData}</span>
            {train != null && <span className="font-mono text-xs text-muted-foreground">{fill(T.soloLoss, { value: lossText(train) })}</span>}
            {!running && <Badge variant="secondary" className="w-fit">{job.status}</Badge>}
          </span>
        </Hint>
      </section>
      <div className="flex flex-wrap gap-4">
        <LossPanel
          id="run-loss"
          title={T.lossVsStep}
          series={stepSeries(curve)}
          xLabel={T.lossAxisStep}
          readouts={readouts}
          empty={T.noCurveSolo}
          overfit={curve ? overfit(curve) : undefined}
        />
        <section aria-labelledby="run-events" className={PANEL}>
          <h2 id="run-events" className="text-lg font-semibold">{T.events}</h2>
          <p className="text-sm text-muted-foreground">{T.soloNoEvents}</p>
          <Link href="/runs/new/" className="w-fit text-sm text-highlight underline-offset-4 hover:underline">{T.relaunch}</Link>
        </section>
      </div>
    </div>
  );
}

/**
 * A member job on this computer: a worker of a run another computer coordinates. It says whose run it is and links to the run's own page
 * (the coordinator's account, from the replica), and shows what this machine is doing now: downloading θ, loading shards, warming up, training.
 */
export function MemberView({ view }: { view: Extract<RunView, { kind: "member" }> }) {
  const { job, invite, run, machine } = view;
  const owner = run?.owner || invite?.owner || "";
  const label = run?.label || invite?.run_label || job.label || job.kind;
  const mine = run?.members.find((m) => m.node === (invite?.node ?? machine));
  const doing = (run && mine ? memberPhaseText(mine, run) : "") || invite?.reason || "";
  const rounds = run?.rounds ?? invite?.rounds ?? 0;
  const done = run?.round ?? invite?.round ?? 0;
  const roundText = rounds > 0 ? fill(T.memberRound, { round: Math.min(done + 1, rounds), rounds }) : "";
  return (
    <div className="flex flex-col gap-6" data-testid={`run-${job.id}`}>
      <RunHeader title={label} shape={owner ? fill(T.memberOf, { run: label, owner }) : fill(T.memberOfUnknown, { run: label })} />
      <section aria-labelledby="run-who" className="flex flex-col gap-3 rounded-xl border border-border bg-card p-5">
        <h2 id="run-who" className="text-lg font-semibold">{T.who}</h2>
        <span className="flex w-fit max-w-md flex-col gap-1.5 rounded-xl border-2 border-info bg-info/10 p-5" data-testid="member-node">
          <span><b className="text-lg">{invite?.node || machine}</b></span>
          <span className="font-mono text-xs text-info">{T.memberRole}</span>
          {owner && <span className="text-sm">{fill(T.memberData, { owner })}</span>}
          {invite && <Badge variant="secondary" className="w-fit" data-testid="member-state">{stateLabel(invite.state)}</Badge>}
          {doing && <span className="text-sm" data-testid="member-doing">{doing}</span>}
          {roundText && <span className="text-xs text-muted-foreground" data-testid="member-round">{roundText}</span>}
        </span>
        {invite?.run_id && (
          <Link href={runViewHref(invite.run_id)} className="w-fit text-sm text-highlight underline-offset-4 hover:underline" data-testid="member-run-link">
            {fill(T.openRun, { owner: owner || T.coordinatorLabel })} →
          </Link>
        )}
      </section>
    </div>
  );
}

"use client";

import Link from "next/link";
import { Disclosure } from "@/components/disclosure";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { KeptCard } from "../post-training/kept-card";
import { PHASE_STATUS_TEXT, STATUS_TEXT, T, fill } from "./copy";
import { PlanEditor } from "./plan-editor";
import {
  PHASES,
  checkText,
  choiceLabel,
  controls,
  loopRounds,
  phaseTitle,
  progressLine,
  resultChecks,
  resultVerdict,
  roundText,
  runChoices,
  runDecisions,
  runLabel,
  timelinePhases,
  valueText,
  when,
  type Progress,
  type RunView,
} from "./logic";
import { useRunActions } from "./use-auto-run";

function Bar({ fraction }: { fraction: number }) {
  const percent = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  return (
    <div role="progressbar" aria-label="Job progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} className="h-2 w-full max-w-md rounded-full bg-muted">
      <div className="h-2 rounded-full bg-highlight-strong transition-all" style={{ width: `${percent}%` }} />
    </div>
  );
}

/** The job the current phase waits for or runs: its step, the time left, and where to watch it. */
function ProgressCard({ progress }: { progress: Progress }) {
  return (
    <Card data-testid="auto-run-progress">
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.progressTitle}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <p className="text-sm font-medium">{progress.label || progress.kind}</p>
        {progress.fraction != null && <Bar fraction={progress.fraction} />}
        <p className="text-sm text-muted-foreground">{progressLine(progress)}</p>
        <p className="flex flex-wrap gap-4 text-sm">
          <Link href="/runs" className="underline underline-offset-2">{T.openRuns}</Link>
          <Link href="/queue" className="underline underline-offset-2">{T.openQueue}</Link>
        </p>
      </CardContent>
    </Card>
  );
}

function Timeline({ run }: { run: RunView }) {
  return (
    <section className="flex flex-col gap-2" aria-label={T.timeline}>
      <h2 className="text-sm font-medium">{T.timeline}</h2>
      <ol className="flex flex-col gap-2">
        {timelinePhases(run).map(({ state }) => {
          const checks = resultChecks(state.result);
          const verdict = resultVerdict(state.result);
          const rounds = loopRounds(state.result);
          const current = run.state.current === state.kind;
          const spec = PHASES[state.kind];
          return (
            <li key={state.kind} aria-current={current ? "step" : undefined} data-testid={`phase-${state.kind}`} className={cn("rounded-lg border p-3", current && "border-highlight")}>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-sm font-medium">{phaseTitle(state.kind)}</span>
                <span className="text-xs uppercase tracking-wide text-muted-foreground">
                  {PHASE_STATUS_TEXT[state.status] ?? state.status}
                  {state.attempt > 1 ? ` · ${fill(T.attempt, { n: state.attempt })}` : ""}
                </span>
              </div>
              {state.error && <p className="mt-1 text-sm text-destructive">{state.error}</p>}
              {verdict && <p className="mt-1 text-sm text-muted-foreground">{verdict}</p>}
              {checks.length > 0 && (
                <ul aria-label={`${T.checks}, ${phaseTitle(state.kind)}`} className="mt-1 flex flex-col gap-0.5 text-sm">
                  {checks.map((check) => (
                    <li key={check.name}>
                      {check.name}: <span className="text-muted-foreground">{checkText(check.ok)}{check.detail ? `, ${check.detail}` : ""}</span>
                    </li>
                  ))}
                </ul>
              )}
              {rounds.length > 0 && (
                <ol aria-label={`${T.rounds}, ${phaseTitle(state.kind)}`} className="mt-1 flex flex-col gap-1 text-sm">
                  {rounds.map((round) => (
                    <li key={round.round}>
                      {roundText(round)}
                      {round.reason && <span className="block text-xs text-muted-foreground">{round.reason}</span>}
                    </li>
                  ))}
                </ol>
              )}
              {state.checkpoint && (
                <p className="mt-1 break-all text-xs text-muted-foreground">
                  {T.checkpoint}: {state.checkpoint}
                </p>
              )}
              {spec && state.kind !== "data" && (state.status === "running" || state.status === "queued" || state.status === "passed" || state.status === "failed") && (
                <Link href={spec.href} className="mt-1 inline-block text-sm underline underline-offset-2">
                  {spec.link}
                </Link>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/** What the parameter controller chose between phases, each with the evidence-based reason (and a bigger design as a proposal only). */
function Choices({ run }: { run: RunView }) {
  const choices = runChoices(run);
  return (
    <section className="flex flex-col gap-2" aria-label={T.controllerTitle} data-testid="auto-run-choices">
      <h2 className="text-sm font-medium">{T.controllerTitle}</h2>
      {run.plan.controller && !run.plan.controller.enabled ? (
        <p className="text-sm text-muted-foreground">{T.controllerOff}</p>
      ) : choices.length === 0 ? (
        <p className="text-sm text-muted-foreground">{T.noChoices}</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {choices.map((choice) => (
            <li key={`${choice.phase}-${choice.attempt}-${choice.key}`} className="rounded-lg border p-3 text-sm">
              {choice.applied ? (
                <p className="font-medium">
                  {fill(T.chosenFrom, { label: choiceLabel(choice), phase: phaseTitle(choice.phase), n: choice.attempt, was: valueText(choice.was), value: valueText(choice.value) })}
                  {choice.clamped ? ` (${T.heldToLimit})` : ""}
                </p>
              ) : (
                <p className="font-medium">
                  {choiceLabel(choice)}: {T.proposal}
                </p>
              )}
              <p className="mt-1 text-muted-foreground">{choice.why}</p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** The controller's decision after each cycle: start another or finish, with the evidence it read (the RLHF loop's stop reason and kept rounds). */
function Decisions({ run }: { run: RunView }) {
  const decisions = runDecisions(run);
  const max = run.plan.controller?.max_cycles ?? 1;
  return (
    <section className="flex flex-col gap-2" aria-label={T.cycleTitle} data-testid="auto-run-cycles">
      <h2 className="text-sm font-medium">{T.cycleTitle}</h2>
      <p className="text-sm text-muted-foreground">{fill(T.cycleNow, { n: run.state.cycle ?? 1, max })}</p>
      <ul className="flex flex-col gap-2">
        {decisions.map((d) => {
          const evidence = d.evidence as { kept?: number; rounds?: number; stop_reason?: string };
          return (
            <li key={d.cycle} className="rounded-lg border p-3 text-sm">
              <p className="font-medium">{fill(d.decision === "another" ? T.cycleAnother : T.cycleFinish, { n: d.cycle })}</p>
              <p className="mt-1 text-muted-foreground">{d.why}</p>
              {evidence.stop_reason ? <p className="mt-1 text-muted-foreground">{evidence.stop_reason} {fill(T.cycleKept, { kept: evidence.kept ?? 0, rounds: evidence.rounds ?? 0 })}</p> : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

/** One run: where it is, what it needs from the person, its phases, and (while it is a draft or paused) its plan. */
export function RunPanel({ run }: { run: RunView }) {
  const actions = useRunActions(run.id);
  const can = controls(run);
  const status = run.state.status;
  const busy = actions.control.isPending || actions.choose.isPending || actions.savePlan.isPending;
  const journal = run.journal.filter((line) => line.event);
  const finished = ["done", "stopped", "failed"].includes(status);

  return (
    <div className="flex flex-col gap-4" data-testid="auto-run-view">
      <Card>
        <CardHeader>
          <CardTitle className="flex flex-wrap items-baseline justify-between gap-2 text-base">
            <span>{runLabel({ name: run.plan.name, version: run.plan.version })}</span>
            <span data-testid="auto-run-status" className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              {STATUS_TEXT[status] ?? status}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {(run.state.reason || status === "pausing") && <p className="text-sm">{status === "pausing" && !run.state.reason ? T.pausing : run.state.reason}</p>}
          {!finished && status !== "draft" && (
            <div className="flex flex-wrap gap-2">
              {can.pause && (
                <Button variant="outline" disabled={busy} onClick={() => actions.control.mutate("pause")}>
                  {T.pause}
                </Button>
              )}
              {can.resume && (
                <Button disabled={busy} onClick={() => actions.control.mutate("resume")}>
                  {T.resume}
                </Button>
              )}
              {can.decide && (
                <>
                  <Button disabled={busy} onClick={() => actions.choose.mutate("retry")}>
                    {T.retry}
                  </Button>
                  <Button variant="outline" disabled={busy} onClick={() => actions.choose.mutate("next")}>
                    {T.skip}
                  </Button>
                </>
              )}
              {can.stop && (
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="outline" disabled={busy}>
                      {T.stop}
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>{T.stopTitle}</AlertDialogTitle>
                      <AlertDialogDescription>{T.stopBody}</AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>{T.keepGoing}</AlertDialogCancel>
                      <AlertDialogAction onClick={() => actions.control.mutate("stop")}>{T.stop}</AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {can.kept && (
        <div className="flex flex-col gap-2">
          <p className="text-sm text-muted-foreground">{T.keptCard}</p>
          <KeptCard
            round={can.kept.round}
            version={run.plan.version}
            onAdopted={() => {}}
            adoptWith={async () => {
              await actions.choose.mutateAsync("adopt");
              return T.adopted;
            }}
          />
        </div>
      )}

      {run.progress && !finished && <ProgressCard progress={run.progress} />}
      {status === "draft" ? (
        <PlanEditor
          key={JSON.stringify(run.plan)}
          initial={run.plan}
          pending={busy}
          actions={[
            { label: T.start, primary: true, run: (plan) => actions.savePlan.mutate(plan, { onSuccess: () => actions.control.mutate("start") }) },
            { label: T.savePlan, run: (plan) => actions.savePlan.mutate(plan) },
          ]}
        />
      ) : (
        <Timeline run={run} />
      )}
      {status !== "draft" && run.plan.controller && <Choices run={run} />}
      {status !== "draft" && run.plan.controller?.enabled && run.plan.phases.some((p) => p.kind === "rlhf" && p.enabled) && <Decisions run={run} />}
      {status === "paused" && (
        <Disclosure title={T.changePlan} hint={T.changePlanHint}>
          <PlanEditor key={JSON.stringify(run.plan)} initial={run.plan} pending={busy} actions={[{ label: T.saveChanges, primary: true, run: (plan) => actions.savePlan.mutate(plan) }]} />
        </Disclosure>
      )}
      {status !== "draft" && (
        <Disclosure title={T.journal} hint={T.journalHint}>
          {journal.length === 0 ? (
            <p className="text-sm text-muted-foreground">{T.noJournal}</p>
          ) : (
            <ul className="flex flex-col gap-1 text-sm" data-testid="auto-run-journal">
              {journal.map((line, i) => (
                <li key={`${line.t}-${i}`}>
                  <span className="mr-2 text-xs text-muted-foreground">{when(line.t)}</span>
                  {line.event.replaceAll("_", " ")}
                  {line.reason ? `: ${line.reason}` : ""}
                </li>
              ))}
            </ul>
          )}
        </Disclosure>
      )}
    </div>
  );
}

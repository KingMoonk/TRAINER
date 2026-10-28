"use client";

import { TipButton } from "@/app/runs/view/controls";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { TRAINING as T } from "@/lib/copy/training";
import { holdButtons, type Run } from "./logic";
import { useAutoJoin, useHoldRun, useRestartRun } from "./use-training";

/** The run's switch for inviting machines that come online while it goes (on by default); the server owns the setting. */
export function AutoJoinSwitch({ run }: { run: Run }) {
  const change = useAutoJoin();
  const id = `auto-join-${run.id}`;
  return (
    <div className="flex items-center gap-2" data-testid="auto-join">
      <Switch id={id} checked={run.auto_join} disabled={change.isPending} onCheckedChange={(enabled) => change.mutate({ id: run.id, enabled })} />
      <Label htmlFor={id} title={T.autoJoinTip}>{T.autoJoin}</Label>
    </div>
  );
}

/**
 * Pause, Resume and Restart for one run, each only while the server says the run can do it. Stop is its own button where each page
 * already had it. `onRestarted` gets the new run so the page can show it.
 */
export function RunControls({ run, onRestarted }: { run: Run; onRestarted: (run: Run) => void }) {
  const pause = useHoldRun("pause");
  const resume = useHoldRun("resume");
  const restart = useRestartRun(onRestarted);
  const show = holdButtons(run);
  if (!show.pause && !show.resume && !show.restart) return null;
  return (
    <div role="group" aria-label={T.controls} className="flex flex-wrap items-center gap-2">
      {show.pause && (
        <TipButton variant="outline" size="sm" tip={T.pauseTip} disabled={pause.isPending} onClick={() => pause.mutate(run.id)} data-testid="pause-run">
          {pause.isPending ? T.working : T.pause}
        </TipButton>
      )}
      {show.resume && (
        <TipButton variant="outline" size="sm" tip={T.resumeTip} disabled={resume.isPending} onClick={() => resume.mutate(run.id)} data-testid="resume-run">
          {resume.isPending ? T.working : T.resume}
        </TipButton>
      )}
      {show.restart && (
        <TipButton variant="outline" size="sm" tip={T.restartTip} disabled={restart.isPending} onClick={() => restart.mutate(run.id)} data-testid="restart-run">
          {restart.isPending ? T.working : T.restart}
        </TipButton>
      )}
    </div>
  );
}

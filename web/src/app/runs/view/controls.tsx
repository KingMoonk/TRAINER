"use client";

import type { ComponentProps } from "react";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader,
  AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { T, fill } from "./copy";

/** A button that says what it does in a Tooltip. */
export function TipButton({ tip, ...props }: { tip: string } & ComponentProps<typeof Button>) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button {...props} />
      </TooltipTrigger>
      <TooltipContent>{tip}</TooltipContent>
    </Tooltip>
  );
}

/** A button that asks before it acts: its Tooltip says what it does, the dialog asks. */
export function ConfirmButton({ label, ariaLabel, tip, title, body, confirm, onConfirm, disabled, danger, size = "default", testId }: {
  label: string; ariaLabel?: string; tip: string; title: string; body: string; confirm: string; onConfirm: () => void;
  disabled?: boolean; danger?: boolean; size?: "default" | "sm"; testId?: string;
}) {
  return (
    <AlertDialog>
      <Tooltip>
        <TooltipTrigger asChild>
          <AlertDialogTrigger asChild>
            <Button
              variant="outline"
              size={size}
              disabled={disabled}
              aria-label={ariaLabel}
              data-testid={testId}
              className={cn(danger && "border-destructive/50 text-destructive hover:text-destructive")}
            >
              {label}
            </Button>
          </AlertDialogTrigger>
        </TooltipTrigger>
        <TooltipContent>{tip}</TooltipContent>
      </Tooltip>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{body}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{T.cancel}</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm} variant={danger ? "destructive" : "default"}>{confirm}</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/** Stop the whole DiLoCo run (shown only where the run can be stopped). */
export function StopRunButton({ pending, onConfirm }: { pending: boolean; onConfirm: () => void }) {
  return (
    <ConfirmButton
      label={pending ? T.working : T.stopRun}
      tip={T.stopRunTip}
      title={T.stopRunTitle}
      body={T.stopRunBody}
      confirm={T.stopRunConfirm}
      onConfirm={onConfirm}
      disabled={pending}
      danger
      testId="stop-run"
    />
  );
}

/** Take one machine out of the run; it stays out (there is no rejoin). */
export function RemoveMemberButton({ node, pending, onConfirm }: { node: string; pending: boolean; onConfirm: () => void }) {
  return (
    <ConfirmButton
      label={pending ? T.working : T.removeMember}
      ariaLabel={fill(T.removeMemberLabel, { node })}
      tip={fill(T.removeMemberTip, { node })}
      title={fill(T.removeMemberTitle, { node })}
      body={T.removeMemberBody}
      confirm={T.removeMemberConfirm}
      onConfirm={onConfirm}
      disabled={pending}
      size="sm"
      danger
    />
  );
}

/** Pause or resume the queue, cool down and stop, stop now: a single job on this computer. */
export function SoloControls({ paused, stopping, busy, onPause, onResume, onCoolDown, onStopNow }: {
  paused: boolean; stopping: boolean; busy: boolean; onPause: () => void; onResume: () => void; onCoolDown: () => void; onStopNow: () => void;
}) {
  return (
    <div role="group" aria-label={T.controls} className="flex flex-wrap gap-2">
      <TipButton variant="outline" tip={T.pauseTip} disabled={busy} onClick={paused ? onResume : onPause}>
        {paused ? T.resume : T.pause}
      </TipButton>
      <ConfirmButton
        label={T.coolDown}
        tip={T.coolDownTip}
        title={T.coolDownTitle}
        body={T.coolDownBody}
        confirm={T.coolDownConfirm}
        onConfirm={onCoolDown}
        disabled={busy || stopping}
      />
      <ConfirmButton label={T.stopNow} tip={T.stopNowTip} title={T.stopNowTitle} body={T.stopNowBody} confirm={T.stopNowConfirm} onConfirm={onStopNow} disabled={busy} danger />
    </div>
  );
}

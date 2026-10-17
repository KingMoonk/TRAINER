"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { nodeName } from "@/lib/node-name";
import { cn } from "@/lib/utils";
import { deviceText } from "../../cluster/machines/logic";
import { fill, NEW_RUN as T } from "./copy";
import type { Node } from "./logic";

export const FIELD = "min-h-11 rounded-lg border border-input bg-background px-3 text-base text-foreground";

/** A label (or any text) with its explanation in a tooltip, marked by a dotted underline. */
export function Hint({ tip, children, className }: { tip: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className={cn("cursor-help underline decoration-dotted decoration-muted-foreground underline-offset-4", className)}>
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent>{tip}</TooltipContent>
    </Tooltip>
  );
}

/** The workers: every machine on the Machines list, with its device and whether it answers. Several can be ticked for Pretrain. */
export function Workers({
  nodes,
  workers,
  forced,
  multiple,
  onToggle,
}: {
  nodes: Node[];
  workers: string[];
  forced: boolean;
  multiple: boolean;
  onToggle: (address: string) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-3">
      <legend className="mb-3 flex items-center gap-2.5 text-xl font-semibold">
        <span aria-hidden="true" className="size-3 rounded-[3px] bg-info" />
        <Hint tip={multiple ? T.workersTip : T.workersTipOne}>{T.workers}</Hint>
      </legend>
      {nodes.length === 0 && <p className="text-sm text-muted-foreground">{T.noMachines}</p>}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {nodes.map((node) => {
          const on = workers.includes(node.address);
          const required = on && node.local && forced;
          const status = !node.online ? T.notAnswering : node.busy ? T.busy : T.ready;
          const name = node.local ? T.thisComputer : nodeName(node, nodes);
          const button = (
            <button
              key={node.address}
              type="button"
              aria-pressed={on}
              disabled={!node.online || required}
              data-testid={`worker-${node.address}`}
              onClick={() => onToggle(node.address)}
              className={cn(
                "flex min-h-20 w-full flex-col gap-1.5 rounded-xl border-2 p-4 text-left outline-none transition-colors focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed",
                on ? "border-info bg-info/15" : "border-border bg-card hover:bg-muted/30",
                !node.online && "opacity-60",
              )}
            >
              <span className="flex items-center justify-between gap-3">
                <b className="text-lg font-semibold">{name}</b>
                <span className="font-mono text-sm text-muted-foreground">{status}</span>
              </span>
              <span className="text-sm text-muted-foreground">{deviceText(node) || node.device_name || node.hostname}</span>
            </button>
          );
          const tip = required ? T.localRequired : !node.online ? node.error : "";
          return tip ? (
            <Tooltip key={node.address}>
              <TooltipTrigger asChild>
                <div>{button}</div>
              </TooltipTrigger>
              <TooltipContent>{tip}</TooltipContent>
            </Tooltip>
          ) : (
            button
          );
        })}
      </div>
      <Link href="/cluster/machines" className="w-fit text-sm text-highlight hover:underline">
        {T.manageNodes}
      </Link>
    </fieldset>
  );
}

/** The coordinator is always this computer; `StartRun` has no field to choose another. */
export function Coordinator() {
  return (
    <div className="flex items-center gap-2.5" data-testid="coordinator">
      <span aria-hidden="true" className="size-3 rounded-full bg-warning" />
      <span className="text-xl font-semibold">
        <Hint tip={T.coordinatorTip}>{T.coordinator}</Hint>
      </span>
      <span className="text-base text-muted-foreground">{T.coordinatorValue}</span>
    </div>
  );
}

function Setting({ id, label, tip, value, onChange, step, readOnly }: { id: string; label: string; tip: string; value: string; onChange?: (value: string) => void; step?: string; readOnly?: boolean }) {
  return (
    <label htmlFor={id} className="flex flex-col gap-1.5 text-sm text-muted-foreground">
      <span>
        <Hint tip={tip}>{label}</Hint> <span aria-hidden="true" className="text-xs">ⓘ</span>
      </span>
      {readOnly ? (
        <input id={id} readOnly value={value} className={cn(FIELD, "bg-muted/40 font-mono")} />
      ) : (
        <input id={id} type="number" inputMode="decimal" min="0" step={step} value={value} onChange={(e) => onChange?.(e.target.value)} className={cn(FIELD, "font-mono")} />
      )}
    </label>
  );
}

export type Outer = { rounds: string; innerSteps: string; outerLr: string; outerMomentum: string };

/** H, R, η_outer and μ, which `StartRun` takes, and Δ dtype, which the server fixes. Shown only when W has several machines. */
export function OuterLoop({ values, deltaDtype, onChange }: { values: Outer; deltaDtype: string; onChange: (patch: Partial<Outer>) => void }) {
  return (
    <fieldset className="flex flex-col gap-3" data-testid="outer-loop">
      <legend className="mb-3 text-xl font-semibold">
        <Hint tip={T.outerLoopTip}>{T.outerLoop}</Hint>
      </legend>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(9.5rem,1fr))] gap-3">
        <Setting id="outer-h" label={T.innerSteps} tip={T.innerStepsTip} value={values.innerSteps} onChange={(innerSteps) => onChange({ innerSteps })} step="1" />
        <Setting id="outer-r" label={T.rounds} tip={T.roundsTip} value={values.rounds} onChange={(rounds) => onChange({ rounds })} step="1" />
        <Setting id="outer-lr" label={T.outerLr} tip={T.outerLrTip} value={values.outerLr} onChange={(outerLr) => onChange({ outerLr })} step="0.1" />
        <Setting id="outer-mu" label={T.outerMomentum} tip={T.outerMomentumTip} value={values.outerMomentum} onChange={(outerMomentum) => onChange({ outerMomentum })} step="0.1" />
        <Setting id="outer-dtype" label={T.deltaDtype} tip={T.deltaDtypeTip} value={deltaDtype} readOnly />
      </div>
    </fieldset>
  );
}

/** "{count} selected" for the Text picker. */
export function selectedText(count: number): string {
  return fill(T.textSelected, { count });
}

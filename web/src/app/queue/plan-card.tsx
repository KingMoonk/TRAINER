"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { T, TOGGLES } from "./copy";
import { parseHour, resumesText, toggleState, type Plan, type Queue } from "./logic";
import type { QueueActions } from "./use-queue";

const HOURS = Object.entries(T.hoursOptions) as [Plan["hours"], string][];
const MEMORY = Object.entries(T.memoryOptions) as [Plan["memory"], string][];

export function PlanCard({ queue, actions }: { queue: Queue | undefined; actions: QueueActions }) {
  const plan = queue?.plan;
  const save = (patch: Partial<Plan>) => plan && actions.savePlan.mutate(patch);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.whenTrain}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4" data-testid="plan-card">
        <p className="text-xs text-muted-foreground">{T.whenTrainTip}</p>
        {!queue || !plan ? (
          <div className="flex flex-col gap-3" aria-label="Reading the plan">
            <Skeleton className="h-8 w-72" />
            <Skeleton className="h-8 w-56" />
            <Skeleton className="h-6 w-80" />
            <Skeleton className="h-6 w-72" />
          </div>
        ) : (
          <>
            <Gate queue={queue} />
            <Segmented label={T.hours} value={plan.hours} options={HOURS} onChange={(hours) => save({ hours })} />
            {plan.hours === "set" && (
              <div className="flex flex-wrap items-center gap-3">
                <HourBox label={T.fromHour} value={plan.start_hour} onCommit={(start_hour) => save({ start_hour })} />
                <HourBox label={T.toHour} value={plan.end_hour} onCommit={(end_hour) => save({ end_hour })} />
              </div>
            )}
            <Segmented label={T.memory} value={plan.memory} options={MEMORY} onChange={(memory) => save({ memory })} />
            <div className="flex flex-col gap-3">
              {TOGGLES.map(({ key, name, note }) => {
                const state = toggleState(key, note, queue.gate.supported);
                return (
                  <div key={key} className="flex items-start gap-3">
                    <Switch
                      id={`plan-${key}`}
                      checked={state.enabled && plan[key]}
                      disabled={!state.enabled}
                      onCheckedChange={(on) => save({ [key]: on })}
                      className="mt-0.5"
                    />
                    <Label htmlFor={`plan-${key}`} className="flex flex-col items-start gap-0.5">
                      <span className={state.enabled ? "" : "text-muted-foreground"}>{name}</span>
                      <span className="text-xs font-normal text-muted-foreground">{state.note}</span>
                    </Label>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

/** Why training is not running right now: the plan's gate, or "paused by you". Nothing when it is free to go. */
function Gate({ queue }: { queue: Queue }) {
  const reason = queue.gate.reason ?? queue.blocked;
  if (!reason) return null;
  return (
    <div role="status" data-testid="gate" className="rounded-lg border border-highlight-strong/50 bg-part-own/10 p-3 text-sm">
      <span className="font-medium">{T.blockedHead}: </span>
      {reason}. {resumesText(queue.gate.resumes_at)}
    </div>
  );
}

function Segmented<V extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: V;
  options: [V, string][];
  onChange: (value: V) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="text-sm text-muted-foreground">{label}</span>
      <ToggleGroup
        type="single"
        variant="outline"
        aria-label={label}
        value={value}
        onValueChange={(next) => next && onChange(next as V)}
      >
        {options.map(([v, text]) => (
          <ToggleGroupItem key={v} value={v} aria-label={text}>
            {text}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

/** A whole hour 0-23, saved when the box loses focus or Enter is pressed; anything else goes back to the saved hour. */
function HourBox({ label, value, onCommit }: { label: string; value: number; onCommit: (hour: number) => void }) {
  const [text, setText] = useState<string | null>(null);
  const commit = () => {
    const hour = parseHour(text ?? "");
    if (text !== null && hour !== null && hour !== value) onCommit(hour);
    setText(null);
  };
  return (
    <label className="flex items-center gap-2 text-sm text-muted-foreground">
      {label}
      <input
        inputMode="numeric"
        aria-label={label}
        value={text ?? String(value)}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
        className="h-8 w-14 rounded-lg border border-input bg-transparent px-2 text-center text-foreground outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
      />
    </label>
  );
}

"use client";

import { useId, useState } from "react";
import { useExtendRun } from "@/app/cluster/training/use-training";
import type { Run } from "@/app/cluster/training/logic";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TRAINING as TR } from "@/lib/copy/training";
import { T, fill } from "./copy";
import { extendRequest } from "./logic";

/** "Add rounds…" next to Stop: how many rounds to add and, if wanted, another H; the run is extended without its machines leaving. */
export function ExtendRunButton({ run }: { run: Run }) {
  const [open, setOpen] = useState(false);
  const [add, setAdd] = useState("10");
  const [steps, setSteps] = useState("");
  const ids = { add: useId(), steps: useId() };
  const extend = useExtendRun(run.id, () => setOpen(false));
  const asked = extendRequest(run, add, steps);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <Button variant="outline" data-testid="extend-run">{T.extendRun}</Button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent>{T.extendTip}</TooltipContent>
      </Tooltip>
      <PopoverContent align="end" aria-label={T.extendTitle} className="flex flex-col gap-3">
        <p className="font-medium">{T.extendTitle}</p>
        <div className="flex flex-col gap-1">
          <Label htmlFor={ids.add}>{T.extendRounds}</Label>
          <Input id={ids.add} type="number" min={0} step={1} inputMode="numeric" value={add} onChange={(e) => setAdd(e.target.value)} />
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor={ids.steps}>{T.extendSteps}</Label>
          <Input id={ids.steps} type="number" min={1} step={1} inputMode="numeric" placeholder={String(run.inner_steps)} value={steps} onChange={(e) => setSteps(e.target.value)} />
          <p className="text-xs text-muted-foreground">{fill(T.extendStepsHint, { steps: run.inner_steps })}</p>
        </div>
        <p className="text-xs text-muted-foreground" data-testid="extend-summary">
          {asked ? fill(T.extendNow, { rounds: run.rounds, steps: asked.inner_steps ?? run.inner_steps, total: asked.rounds ?? run.rounds }) : T.extendInvalid}
        </p>
        {asked?.inner_steps !== undefined && (
          <p className="text-xs text-warning" data-testid="extend-tuned">
            {fill(T.extendTuned, { lr: run.outer_lr, momentum: run.outer_momentum, steps: run.inner_steps })}
          </p>
        )}
        <Button disabled={!asked || extend.isPending} onClick={() => asked && extend.mutate(asked)} data-testid="extend-confirm">
          {extend.isPending ? TR.working : T.extendConfirm}
        </Button>
      </PopoverContent>
    </Popover>
  );
}

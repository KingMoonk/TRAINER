"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { toast } from "sonner";
import { runViewHref } from "@/app/runs/view/logic";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { api, unwrap } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Choice } from "./cards";
import { T, fill } from "./copy";
import { LENGTHS, STOP_WHEN_FLAT, errorText, lrLine, modeTexts, poolLines, queueNote, stepsLine, type Length, type Plan, type PlanBody, type PreparedStatus, type StopWhenFlat } from "./logic";
import { optionBox } from "@/components/option-box";

/** Continue or start over, how long, and Add to the queue: the job is the one the server's plan built. While the size of the text is
 * still being worked out, adding puts the run in the queue at once (POST /pretraining/start); it gets its steps once the size is known. */
export function TrainCard({
  plan,
  body,
  estimating,
  mode,
  onMode,
  length,
  onLength,
  stopWhenFlat,
  onStopWhenFlat,
  prepared,
}: {
  plan: Plan | undefined;
  body: PlanBody;
  estimating: boolean;
  mode: "catch_up" | "from_scratch";
  onMode: (m: "catch_up" | "from_scratch") => void;
  length: Length;
  onLength: (l: Length) => void;
  stopWhenFlat: StopWhenFlat;
  onStopWhenFlat: (s: StopWhenFlat) => void;
  prepared?: PreparedStatus | null;
}) {
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [solo, setSolo] = useState(false);
  // The machines a start would train on, asked of the server (it surveys the Machines list): shown before Start is pressed.
  const pool = useQuery({
    queryKey: ["pretraining", "pool", body, plan?.version, solo],
    queryFn: () => unwrap(api.POST("/pretraining/pool", { body: { ...body, version: plan!.version, solo } })),
    enabled: !!plan && plan.version !== null && body.dataset_ids.length > 0,
    placeholderData: keepPreviousData,
    staleTime: 15_000,
  });
  const add = useMutation({
    mutationFn: () => unwrap(api.POST("/queue/jobs", { body: { ...plan!.job!, note: "" } })),
    onSuccess: (job) => {
      setJobId(job.id);
      setRunId(null);
      toast.success(fill(T.added, { label: job.label }));
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  const start = useMutation({
    mutationFn: () => unwrap(api.POST("/pretraining/start", { body: { ...body, version: plan!.version, solo } })),
    onSuccess: (job) => {
      setJobId(job.id);
      setRunId(job.run_id ?? null);
      toast.success(fill(job.run_id ? T.startedPooled : T.added, { label: job.label }));
      if (job.note) toast.info(job.note);
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  // The only thing missing is the size of the text: the run may be added now, and waits in the queue until that is known.
  const sizeFirst = !!plan && !plan.can_add && plan.estimating && plan.version !== null;
  const busy = add.isPending || start.isPending;
  const pooled = plan?.can_add && pool.data?.pooled === true;
  const lines = pool.data ? poolLines(pool.data) : null;
  // When the job this page added starts; before one is added (or once it has left the queue), what adding one now would do.
  const outlook = useQuery({
    queryKey: ["queue", "outlook", jobId],
    queryFn: () => unwrap(api.GET("/queue/jobs/{job_id}/outlook", { params: { path: { job_id: jobId! } } })),
    enabled: jobId !== null,
  });
  const when = jobId ? outlook.data?.outlook : "";
  const queueLine = when ? `Queued. ${when}.` : plan ? `${plan.outlook}.` : "";
  const texts = plan ? modeTexts(plan) : null;
  const shown = plan?.mode ?? mode;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{T.train}</CardTitle>
        <CardDescription>{T.continueStart}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {!plan || !texts ? (
          <div className="flex flex-col gap-2" aria-label="Reading the version">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : (
          <>
            {plan.version && <p className="text-sm text-muted-foreground">{fill(T.versionNeeds, { n: plan.version })}</p>}
            <div role="radiogroup" aria-label="How to train" className="flex flex-col gap-2">
              {(["catch_up", "from_scratch"] as const).map((key) => {
                const on = shown === key;
                const blocked = key === "catch_up" && !plan.can_continue;
                return (
                  <button
                    key={key}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    data-testid={`mode-${key}`}
                    onClick={() => onMode(key)}
                    className={optionBox(on, "flex flex-col gap-0.5 px-3 py-2")}
                  >
                    <span className="font-medium">{texts[key].title}</span>
                    <span className={cn("text-sm", blocked ? "text-warning" : "text-muted-foreground")}>{texts[key].note}</span>
                  </button>
                );
              })}
            </div>
          </>
        )}
        <Choice label={T.howLong} value={length} items={Object.keys(LENGTHS) as Length[]} onChange={onLength} />
        <p className="text-sm text-muted-foreground" data-testid="steps-line">
          {stepsLine(plan, estimating)}
        </p>
        <Choice label={T.stopFlat} hint={T.stopFlatHint} value={stopWhenFlat} items={Object.keys(STOP_WHEN_FLAT) as StopWhenFlat[]} onChange={onStopWhenFlat} />
        {plan?.plateau_line && (
          <p className="text-sm text-muted-foreground" data-testid="plateau-line">
            {plan.plateau_line}
          </p>
        )}
        <p className="text-sm">{lrLine(plan?.lr)}</p>
        <p className="text-sm" title={T.scheduleDesc}>
          {T.scheduleLabel} {T.scheduleValue} {T.scheduleNote}
        </p>
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2" title={T.soloHint}>
            <Switch id="pretrain-solo" checked={solo} onCheckedChange={setSolo} />
            <Label htmlFor="pretrain-solo">{T.soloSwitch}</Label>
          </div>
          {lines && (
            <div className="flex flex-col gap-0.5 text-sm" data-testid="pool">
              <p data-testid="pool-line">{lines.lead}</p>
              {lines.why && (
                <p className="text-muted-foreground" data-testid="pool-why">
                  {lines.why}
                </p>
              )}
              {lines.leftOut.map((text) => (
                <p key={text} className="text-muted-foreground" data-testid="pool-left-out">
                  {text}
                </p>
              ))}
            </div>
          )}
          {prepared?.prepared && (
            <p className="text-xs text-success flex items-center gap-1.5" data-testid="train-prepared-note">
              <span>✓</span> {T.readyToTrain}: {fill(T.shardedOnDisk, { windows: (prepared.windows ?? 0).toLocaleString("en-US"), tokens: (prepared.tokens ?? 0).toLocaleString("en-US") })}
            </p>
          )}
          <div className="flex items-center gap-3">
            <Button disabled={!(plan?.can_add || sizeFirst) || busy || (pool.isLoading && !sizeFirst)} onClick={() => (plan?.can_add && !pooled ? add.mutate() : start.mutate())}>
              {T.addQueue}
            </Button>
            {runId && (
              <Link href={runViewHref(runId)} className="text-sm text-highlight hover:underline" data-testid="open-run">
                {T.openRun} ›
              </Link>
            )}
          </div>
          {sizeFirst ? (
            <p className="text-sm text-muted-foreground" data-testid="size-first">
              {T.sizeFirst}
            </p>
          ) : plan && !plan.can_add && plan.reason && (
            <p className="text-sm text-muted-foreground" data-testid="reason">
              {plan.reason}
            </p>
          )}
          {plan && <p className="text-xs text-muted-foreground">{queueNote(plan)}</p>}
          <p className="text-sm text-muted-foreground" data-testid="queue-line">
            {queueLine}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

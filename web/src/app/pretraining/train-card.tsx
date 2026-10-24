"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { runJob, type JobLike } from "../your-data/jobs";
import { cn } from "@/lib/utils";
import { Choice } from "./cards";
import { T, fill } from "./copy";
import { LENGTHS, STOP_WHEN_FLAT, errorText, lrLine, modeTexts, queueNote, stepsLine, type Length, type Plan, type PlanBody, type StopWhenFlat } from "./logic";
import { optionBox } from "@/components/option-box";

/** Continue or start over, how long, and Add to the queue: the job is the one the server's plan built. While the size of the text is
 * still being worked out, adding asks the server to finish that first and then add the job (POST /pretraining/start). */
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
}) {
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const add = useMutation({
    mutationFn: () => unwrap(api.POST("/queue/jobs", { body: { ...plan!.job!, note: "" } })),
    onSuccess: (job) => {
      setJobId(job.id);
      toast.success(fill(T.added, { label: job.label }));
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  const start = useMutation({
    mutationFn: () =>
      runJob<{ id: string; label: string }>(() => unwrap(api.POST("/pretraining/start", { body: { ...body, version: plan!.version } })) as Promise<JobLike>),
    onSuccess: (job) => {
      setJobId(job.id);
      toast.success(fill(T.added, { label: job.label }));
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  // The only thing missing is the size of the text: the run may be asked for now, and is added once that is known.
  const sizeFirst = !!plan && !plan.can_add && plan.estimating && plan.version !== null;
  const busy = add.isPending || start.isPending;
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
          <div>
            <Button disabled={!(plan?.can_add || sizeFirst) || busy} onClick={() => (plan?.can_add ? add.mutate() : start.mutate())}>
              {T.addQueue}
            </Button>
          </div>
          {start.isPending ? (
            <p className="text-sm text-muted-foreground" data-testid="sizing-then-adding">
              {T.sizingThenAdding}
            </p>
          ) : sizeFirst ? (
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

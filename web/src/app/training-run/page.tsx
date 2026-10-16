"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useEffect, useMemo, useRef } from "react";
import { toast } from "sonner";
import { SeriesChart } from "@/components/series-chart";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError, subscribe, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { cn } from "@/lib/utils";
import { fill, STATUS, T } from "./copy";
import { bits, chartSeries, last, phaseBands, progressLine, sampleRows, switchText, type JobParams } from "./logic";

const REFRESH_MS = 1500; // live events redraw at most this often, as the Tk page does

function describe(error: unknown): string {
  if (error instanceof ApiError) return `The server answered ${error.status}${error.body ? `: ${error.body}` : ""}`;
  return error instanceof Error ? error.message : "Something went wrong.";
}

/** Refetch the run and the queue when the server says they moved, at most once per REFRESH_MS. */
function useLiveRefresh() {
  const queryClient = useQueryClient();
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const refresh = () => {
      timer = undefined;
      queryClient.invalidateQueries({ queryKey: ["training-run"] });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    };
    const close = subscribe((event) => {
      if (["queue", "train", "job", "versions"].includes(event.kind) && !timer) timer = setTimeout(refresh, REFRESH_MS);
    });
    return () => {
      clearTimeout(timer);
      close();
    };
  }, [queryClient]);
}

export default function Page() {
  const route = routeFor("training_run");
  const queryClient = useQueryClient();
  useLiveRefresh();

  // Both are read on load, so a page opened mid-run or after it ended shows the right state, not only what events brought since.
  const runQuery = useQuery({
    queryKey: ["training-run"],
    queryFn: () => unwrap(api.GET("/training-run")),
    placeholderData: keepPreviousData,
  });
  const queueQuery = useQuery({ queryKey: ["queue"], queryFn: () => unwrap(api.GET("/queue")), placeholderData: keepPreviousData });
  const run = runQuery.data;
  const queue = queueQuery.data;

  const control = useMutation({
    mutationFn: (action: "pause" | "resume" | "stop_after_cooldown") => unwrap(api.POST(`/queue/${action}`)),
    onSuccess: (state) => queryClient.setQueryData(["queue"], state),
    onError: (error) => toast.error(describe(error)),
  });

  // The chart's inputs only change when the curve does, so a poll that brings the same points leaves it alone.
  const curve = run?.curve;
  const refs = run?.references;
  const parent = run?.parent ?? null;
  const parentName = parent ? `version ${parent}` : T.earlierVersion;
  const series = useMemo(
    () =>
      curve && refs
        ? chartSeries(curve, refs, {
            train: T.trainingText,
            raw: T.perStep,
            lr: T.learningRate,
            held: T.heldBack,
            uniform: T.uniformGuess,
            unigram: T.wordFrequencies,
            best: parent ? fill(T.versionsBest, { parent: `version ${parent}` }) : T.earlierBest,
          })
        : [],
    [curve, refs, parent],
  );
  const bands = useMemo(() => phaseBands(curve?.phases ?? []), [curve?.phases]);
  const hasPoints = !!curve && (curve.steps.length > 0 || curve.smooth.length > 0 || curve.heldout.length > 0 || curve.lr.length > 0);

  const nowStep = run?.running && run.total ? run.step : null;
  const samples = curve?.samples;
  const rows = useMemo(() => (samples ? sampleRows(samples, nowStep, parent) : []), [samples, nowStep, parent]);
  const everyLabel = curve?.sample_every ? fill(T.whatWritesNow, { every: curve.sample_every.toLocaleString("en-US") }) : T.whatWritesNowPlain;

  const loading = runQuery.isPending || queueQuery.isPending;
  const failed = runQuery.isError ? describe(runQuery.error) : null;
  const shownError = useRef<string | null>(null);
  useEffect(() => {
    if (failed && failed !== shownError.current) toast.error(failed);
    shownError.current = failed;
  }, [failed]);

  const title = run?.version == null ? route.title : `${route.title} · version ${run.version}${run.running ? " catching up" : ""}`;
  const change = !run
    ? ""
    : run.version == null
      ? T.noVersion
      : parent == null
        ? `Version ${run.version} starts from random weights: there is no earlier version to continue from.`
        : `Version ${run.version} = version ${parent}${run.change ? ` with ${run.change}` : ""}. The new parts start at zero, so it began exactly at version ${parent}’s error.`;

  const job = queue?.job ?? null;
  const progress = job && queue ? progressLine({ ...queue, params: job.params as JobParams }) : T.nothingTraining;
  const status = run?.status ? STATUS[run.status] : null;
  const chatName = run?.chat_version ? `version ${run.chat_version}` : T.noTrainedVersion;
  const numbers: [string, number | null][] = [
    [T.errorTraining, curve ? last(curve.smooth) : null],
    [T.errorHeldBack, curve ? last(curve.heldout) : null],
    [fill(T.versionBest, { parent: parentName }), refs?.best_so_far_bpc ?? null],
  ];

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-5 p-8">
      <div className="flex flex-col gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="max-w-prose text-muted-foreground">{route.lead}</p>
      </div>

      {loading ? (
        <div className="flex flex-col gap-3" aria-label="Reading the run">
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-9 w-64" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-72 w-full" />
        </div>
      ) : (
        <>
          <div className="flex flex-col gap-1">
            <p className="max-w-prose text-base text-muted-foreground">{change}</p>
            {run?.parent != null && (
              <Link href={routeFor("builder").href} className="w-fit text-sm text-violet-400 underline-offset-4 hover:underline">
                {T.seeChange}
              </Link>
            )}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button disabled={!job && !queue?.paused} onClick={() => control.mutate(queue?.paused ? "resume" : "pause")}>
              {queue?.paused ? T.resume : T.pause}
            </Button>
            <Button variant="outline" disabled={!job || queue?.stopping} onClick={() => control.mutate("stop_after_cooldown")}>
              {T.stopCooldown}
            </Button>
          </div>
          <p className="text-sm text-muted-foreground" data-testid="progress">
            {progress}
          </p>

          <Card>
            <CardContent className="flex flex-col gap-3">
              <div className="grid gap-4 sm:grid-cols-3" title={T.bitsPerCharNote}>
                {numbers.map(([label, value]) => (
                  <div key={label}>
                    <div className="font-mono text-3xl tabular-nums">{bits(value)}</div>
                    <div className="text-xs text-muted-foreground">{label}</div>
                  </div>
                ))}
              </div>
              {status && (
                <Badge variant="outline" className={cn("w-fit", status.tone)} data-testid="status">
                  {status.label}
                </Badge>
              )}
              {hasPoints ? (
                <div title={T.coolDownDrop}>
              <SeriesChart series={series} bands={bands} xLabel="step" yLabel="bits per character" rightLabel={T.learningRate} />
                </div>
              ) : (
                <p className="py-10 text-center text-sm text-muted-foreground">{T.noPointsYet}</p>
              )}
              <p className="text-xs text-muted-foreground">
                {T.trainingText} · {T.heldBack}
              </p>
            </CardContent>
          </Card>

          {run && (
            <div className="rounded-lg border border-sky-500/50 bg-sky-500/10 p-4" data-testid="switch">
              <div className="font-semibold">{fill(T.chatStillUses, { chat: chatName })}</div>
              <p className="mt-1 text-sm text-muted-foreground">{switchText(run.switch, run.version ?? null, parent, run.gate_vs ?? null, run.running)}</p>
              <Link href={routeFor("chat").href} className={cn(buttonVariants({ variant: "link" }), "h-auto px-0")}>
                {T.compareChat}
              </Link>
            </div>
          )}

          <details className="rounded-lg border p-4 text-sm text-muted-foreground">
            <summary className="cursor-pointer font-medium text-foreground">{T.howStatusWork}</summary>
            <p className="mt-2 max-w-prose">
              {T.statusOneOf} {T.statusLearning} {T.learningDesc} {T.statusFlat} {T.flatDesc} {T.statusMemorising} {T.memorisingDesc}
            </p>
            <p className="mt-2 max-w-prose">{T.scheduleNote}</p>
            <p className="mt-2 max-w-prose">{T.lrAutoNote}</p>
            <p className="mt-2 max-w-prose">{T.gateHow}</p>
          </details>

          <Card>
            <CardContent className="flex flex-col gap-2">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground" title={T.alwaysPicks}>
                {everyLabel}
              </div>
              {curve?.prompt && <p className="font-serif text-base text-muted-foreground">{curve.prompt}</p>}
              <ul className="flex flex-col gap-1">
                {rows.map((row, i) => (
                  <li key={i} className="flex gap-3" data-testid="sample">
                    <span className="w-28 shrink-0 text-xs text-muted-foreground">{row.label}</span>
                    <span className="font-serif text-base">{row.text}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </>
      )}
    </main>
  );
}

"use client";

import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useEffect, useState } from "react";
import { SeriesChart, type Series } from "@/components/series-chart";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AvailableVersions } from "@/components/available-versions";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError, subscribe, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { cameFrom, isCluster, useCluster, usePublished } from "@/lib/cluster";
import { routeFor } from "@/lib/routes";
import { cn } from "@/lib/utils";
import { T } from "./copy";

type Version = components["schemas"]["Version"];
type Queue = components["schemas"]["QueueStateModel"];
type Curve = { steps: number[][]; smooth: number[][]; heldout: number[][]; total: number };

const INSPECTORS = ["trace", "head_lab", "head_zoom", "loss", "value_map", "inside3d"] as const;

export function versionState(version: Version, queue?: Queue, editingVersion?: number) {
  if (queue?.job?.params.version === version.number) return queue.paused ? T.stateTrainingPaused : T.stateTraining;
  const waiting = queue?.up_next.find((job) => job.params.version === version.number);
  if (waiting) return waiting.status === "held" ? T.stateHeld : T.stateQueued;
  if (editingVersion === version.number) return T.stateEditing;
  return ({ draft: T.stateDraft, saved: T.stateSaved, queued: T.stateQueued, training: T.stateTraining, trained: T.stateTrained } as Record<string, string>)[version.status] ?? T.stateSaved;
}

export function comparisonSeries(current: { number: number; curve: Curve | null }, previous?: { number: number; curve: Curve | null }): Series[] {
  const out: Series[] = [];
  const offset = previous?.curve?.total || previous?.curve?.steps.at(-1)?.[0] || 0;
  for (const [entry, shift] of [[previous, 0], [current, offset]] as const) {
    if (!entry?.curve) continue;
    const curve = entry.curve;
    const training = curve.smooth.length ? curve.smooth : curve.steps;
    for (const [kind, points] of [[T.stateTraining, training], [T.heldBack, curve.heldout]] as const) {
      if (points.length) out.push({ id: `${entry.number}-${kind}`, label: `${T.version} ${entry.number} · ${kind}`,
        points: points.map(([x, y]) => ({ x: x + shift, y })) });
    }
  }
  return out;
}

function useVersionEvents() {
  const cache = useQueryClient();
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const keys = new Set<string>();
    const close = subscribe((event) => {
      if (event.kind === "train") keys.add("builder");
      if (["versions", "checkpoint"].includes(event.kind)) { keys.add("versions"); keys.add("builder"); }
      if (event.kind === "queue") keys.add("queue");
      if (event.kind === "job" && event.payload && typeof event.payload === "object" && "job" in event.payload) {
        keys.add("queue"); keys.add("versions"); keys.add("builder");
      }
      if (keys.size && timer === undefined) timer = setTimeout(() => {
        timer = undefined;
        for (const key of keys) void cache.invalidateQueries({ queryKey: [key] });
        keys.clear();
      }, 500);
    });
    return () => { clearTimeout(timer); close(); };
  }, [cache]);
}

/** The model editor can pass editingVersion while its unsaved draft is open. */
export function VersionTray({ editingVersion }: { editingVersion?: number }) {
  useVersionEvents();
  const versions = useQuery({ queryKey: ["versions"], queryFn: ({ signal }) => unwrap(api.GET("/versions", { signal })), refetchInterval: 15_000 });
  const queue = useQuery({ queryKey: ["queue"], queryFn: ({ signal }) => unwrap(api.GET("/queue", { signal })), refetchInterval: 15_000 });
  const cluster = useCluster().data;
  const published = usePublished(cluster).data;
  const multi = isCluster(cluster); // a cluster of one shows nothing about machines
  const [picked, setPicked] = useState<number>();
  const rows = versions.data ?? [];
  const selected = rows.find((version) => version.number === picked) ?? rows.at(-1);
  const parent = selected && (rows.find((version) => version.number === selected.parent) ?? rows[rows.indexOf(selected) - 1]);
  const shown = [parent, selected].filter((version): version is Version => !!version);
  const curves = useQueries({ queries: shown.map((version) => ({
    queryKey: ["builder", "curve", version.number],
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      try { return await unwrap(api.GET("/pretraining/versions/{number}/curve", { params: { path: { number: version.number } }, signal })); }
      catch (error) { if (error instanceof ApiError && error.status === 404) return null; throw error; }
    },
    retry: false,
  })) });
  const series = selected ? comparisonSeries({ number: selected.number, curve: curves.at(-1)?.data ?? null },
    parent ? { number: parent.number, curve: curves[0]?.data ?? null } : undefined) : [];

  return <section className="flex min-w-0 flex-col gap-5" aria-label={T.versionHistory}>
    <Card>
      <CardHeader><CardTitle>{T.versions}</CardTitle><CardDescription>{T.versionsDescription}</CardDescription></CardHeader>
      <CardContent>
        {versions.isPending ? <div aria-label={T.readingVersions} className="space-y-2"><Skeleton className="h-20 w-full" /><Skeleton className="h-20 w-full" /></div>
          : versions.isError ? <p role="alert">{T.couldNotReadVersions} {versions.error.message}</p>
          : !rows.length ? <p className="text-sm text-muted-foreground">{T.empty}</p>
          : <ol className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3" aria-label={T.savedVersions}>
            {rows.map((version) => <li key={version.number}>
              <button type="button" aria-pressed={selected?.number === version.number} onClick={() => setPicked(version.number)}
                className={cn("flex h-full w-full flex-col gap-3 rounded-xl border p-4 text-left transition-colors hover:bg-surface-overlay",
                  selected?.number === version.number ? "selected-box" : "border-border bg-surface-faint")}>
                <span className="flex w-full flex-wrap items-center justify-between gap-2"><span className="font-medium">{T.version} {version.number}</span>
                  <Badge variant="outline">{versionState(version, queue.data, editingVersion)}</Badge></span>
                <span className="text-sm text-muted-foreground">{version.change || version.name || T.savedDesign}</span>
                {multi && <span className="text-xs text-muted-foreground" data-testid={`where-${version.number}`}>
                  {T.onThisMachine}{cameFrom(published, version.number) ? ` · ${T.pulledFrom} ${cameFrom(published, version.number)}` : ""}</span>}
              </button>
            </li>)}
          </ol>}
        {multi && !!published?.some((entry) => !entry.local) && <div className="mt-4 flex flex-col gap-2">
          <h3 className="text-sm font-medium">{T.availableElsewhere}</h3>
          <AvailableVersions published={published} />
        </div>}
        {queue.isError && <p role="alert" className="mt-3 text-sm">{T.queueUnavailable}</p>}
      </CardContent>
    </Card>
    {selected && <>
      <Card>
        <CardHeader><CardTitle>{T.version} {selected.number}{parent ? ` ${T.afterVersion} ${parent.number}` : ` · ${T.trainingHistory}`}</CardTitle>
          <CardDescription>{selected.change || T.modelDesign}</CardDescription></CardHeader>
        <CardContent className="flex min-w-0 flex-col gap-4">
          {curves.some((curve) => curve.isPending) ? <Skeleton className="h-64 w-full" aria-label={T.readingCurves} />
            : curves.some((curve) => curve.isError) ? <p role="alert">{T.curvesFailed}</p>
            : series.length ? <SeriesChart series={series} xLabel={T.trainingStep} yLabel={T.errorAxis} className="h-72 w-full" />
            : <p className="text-sm text-muted-foreground">{T.noCurve}</p>}
          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
            {shown.map((version, index) => { const curve = curves[index]?.data; const held = curve?.heldout.at(-1)?.[1];
              return <p key={version.number}>{T.version} {version.number} · {T.heldBackError} {held == null ? T.notMeasured : `${held.toFixed(3)} ${T.bitsPerCharacter}`}</p>; })}
          </div>
          <Link href={routeFor("queue").href} className="text-sm underline underline-offset-4">{T.openQueue}</Link>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>{T.lookInside} {selected.number}</CardTitle><CardDescription>{T.inspectDescription}</CardDescription></CardHeader>
        <CardContent className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {INSPECTORS.map((key) => { const route = routeFor(key); return <Link key={key} href={`${route.href}/?version=${selected.number}`}
            className="rounded-xl border border-border bg-surface-faint px-4 py-3 text-sm transition-colors hover:bg-surface-overlay">{route.title}<span aria-hidden className="float-right text-muted-foreground">↗</span></Link>; })}
        </CardContent>
      </Card>
    </>}
  </section>;
}

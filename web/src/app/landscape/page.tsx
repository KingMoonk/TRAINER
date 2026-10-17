"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import dynamic from "next/dynamic";
import { Suspense, useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { HeatMap } from "@/components/viz/heat-map";
import { useRamp } from "@/components/viz/use-ramp";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { api, unwrap } from "@/lib/api";
import { useEvent } from "@/lib/events";
import { axisUnit, costLine, fetchLandscape, GRIDS, LIVE_GRID_MAX, type LandscapeMode, type LandscapeResult } from "@/lib/landscape";
import { RAMP_TOKENS } from "@/lib/ramp";
import { useVersionParameter } from "@/lib/version-param";
import { LANDSCAPE as T } from "./copy";
import { isSnapshotEvent, requestedGrid, versionLabel } from "./logic";

// WebGL does not exist while the static export prerenders, so the surface only ever loads in the browser.
const HeatSurface = dynamic(() => import("@/components/viz/heat-surface"), {
  ssr: false,
  loading: () => <div className="grid h-full place-items-center text-sm text-muted-foreground">Loading the 3D view...</div>,
});

const NARROW = "(max-width: 640px)";
const REDUCED = "(prefers-reduced-motion: reduce)";

function useMedia(query: string): boolean {
  return useSyncExternalStore(
    (notify) => { const list = window.matchMedia(query); list.addEventListener("change", notify); return () => list.removeEventListener("change", notify); },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

export default function Page() {
  return <Suspense fallback={<Skeleton className="m-8 h-64" />}><VersionedLandscape /></Suspense>;
}

function VersionedLandscape() {
  const version = useVersionParameter();
  return <LandscapePage key={version ?? "latest"} initialVersion={version} />;
}

function LandscapePage({ initialVersion }: { initialVersion?: number }) {
  const versions = useQuery({ queryKey: ["versions"], queryFn: ({ signal }) => unwrap(api.GET("/versions", { signal })), refetchInterval: 15_000 });
  const usable = (versions.data ?? []).filter((item) => item.best_checkpoint || item.status === "training");
  const [selected, setSelected] = useState<number | undefined>(initialVersion);
  const version = selected !== undefined && usable.some((item) => item.number === selected) ? selected
    : (usable.find((item) => item.status === "training") ?? usable.at(-1))?.number;

  const [mode, setMode] = useState<LandscapeMode>("random");
  const [grid, setGrid] = useState<number>(21);
  const [span, setSpan] = useState(1);
  const [seed, setSeed] = useState(0);
  const [live, setLive] = useState(false);
  const [chosenView, setChosenView] = useState<"surface" | "map" | null>(null);
  const narrow = useMedia(NARROW);
  const reduceMotion = useMedia(REDUCED);
  const fallback = narrow || reduceMotion;
  const view = chosenView ?? (fallback ? "map" : "surface");
  const ramp = useRamp();

  const still = useQuery({
    queryKey: ["landscape", version, mode, grid, span, seed],
    enabled: version !== undefined && !live,
    queryFn: ({ signal }) => fetchLandscape({ version, mode, grid, span, seed }, signal),
    placeholderData: keepPreviousData, retry: false, staleTime: 60_000,
  });

  const frame = useLiveFrames(live, version, seed);
  const result: LandscapeResult | undefined = live ? frame.result ?? still.data : still.data;
  const error = live ? frame.error : still.isError ? still.error.message : "";
  const pending = live ? frame.busy : still.isFetching;

  return <main className="mx-auto flex w-full min-w-0 max-w-5xl flex-col gap-4 p-4 md:p-8">
    <p className="text-xs text-muted-foreground">{T.breadcrumb}</p>
    <h1 className="text-2xl font-semibold tracking-tight">{T.title}</h1>
    <p className="max-w-prose text-muted-foreground">{T.lead}</p>
    {versions.isPending ? <Skeleton className="h-40 w-full" aria-label="Loading versions" /> : versions.isError ?
      <Card><CardContent role="alert">{T.versionsFailed} <Button variant="outline" onClick={() => void versions.refetch()}>{T.retry}</Button></CardContent></Card> : !usable.length ?
      <Card><CardContent><p>{T.empty}</p><Link href="/pretraining/" className="text-sm underline">{T.openPretraining}</Link></CardContent></Card> : <>
      <Card><CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap items-end gap-4">
          <label className="flex flex-col gap-2 text-sm">{T.version}
            <select aria-label={T.version} value={version ?? ""} onChange={(event) => setSelected(Number(event.target.value))}
              className="h-10 max-w-full rounded-lg border border-input bg-background px-3">
              {version === undefined && <option value="" disabled>{T.chooseVersion}</option>}
              {usable.map((item) => <option key={item.number} value={item.number}>{versionLabel(item.number, item.name)}</option>)}
            </select>
          </label>
          <div className="flex flex-col gap-2 text-sm">{T.directions}
            <div className="flex gap-2" role="group" aria-label={T.directions}>
              {(["random", "pca"] as const).map((value) => <Button key={value} variant={(live ? "random" : mode) === value ? "default" : "outline"}
                aria-pressed={(live ? "random" : mode) === value} disabled={live && value === "pca"} onClick={() => setMode(value)}>
                {value === "random" ? T.random : T.pca}</Button>)}
            </div>
          </div>
          <label className="flex flex-col gap-2 text-sm">{T.grid}
            <select aria-label={T.grid} value={requestedGrid(grid, live)} disabled={live} onChange={(event) => setGrid(Number(event.target.value))}
              className="h-10 rounded-lg border border-input bg-background px-3">
              {GRIDS.map((value) => <option key={value} value={value}>{T.gridUnit(value)}</option>)}
              {live && <option value={LIVE_GRID_MAX}>{T.gridUnit(LIVE_GRID_MAX)}</option>}
            </select>
          </label>
          <label className="flex flex-col gap-2 text-sm">{T.range}
            <select aria-label={T.range} value={span} disabled={live} onChange={(event) => setSpan(Number(event.target.value))}
              className="h-10 rounded-lg border border-input bg-background px-3">
              {[0.5, 1, 2, 4].map((value) => <option key={value} value={value}>±{value} {T.rangeUnit(mode === "random")}</option>)}
            </select>
          </label>
          <Button variant="outline" disabled={mode !== "random"} onClick={() => setSeed((value) => value + 1)}>{T.newDirections}</Button>
        </div>
        <p className="text-sm text-muted-foreground">{mode === "random" || live ? T.randomHelp : T.pcaHelp}</p>
        <div className="flex flex-wrap items-center gap-3">
          <Switch id="landscape-live" checked={live} onCheckedChange={setLive} aria-label={T.live} />
          <label htmlFor="landscape-live" className="text-sm font-medium">{T.live}</label>
          <span className="min-w-0 text-sm text-muted-foreground" data-testid="landscape-cost">
            {live ? T.liveHelp(LIVE_GRID_MAX) : `${T.liveOff} ${T.estimate(grid)}.`}
          </span>
        </div>
      </CardContent></Card>
      {error ? <Card><CardContent className="flex flex-col items-start gap-3" role="alert">
        <p>{T.failed}: {error}</p>
        {!live && <Button variant="outline" onClick={() => void still.refetch()}>{T.retry}</Button>}
      </CardContent></Card> : !result ? <div aria-label={T.loading} aria-busy="true" className="space-y-3">
        {live && !frame.busy ? <p className="text-sm text-muted-foreground">{T.liveWaiting}</p> : <Skeleton className="h-96 w-full" />}
      </div> : <section className="flex min-w-0 flex-col gap-4" aria-label="Loss landscape" aria-busy={pending}>
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label={T.view}>
          {(["surface", "map"] as const).map((value) => <Button key={value} variant={view === value ? "default" : "outline"}
            aria-pressed={view === value} onClick={() => setChosenView(value)}>{value === "surface" ? T.surface : T.heatMap}</Button>)}
        </div>
        {fallback && chosenView === null && <p className="text-sm text-muted-foreground">{T.heatMapFallback}</p>}
        <div className="flex flex-col gap-4 lg:flex-row">
          <div className="min-w-0 flex-1">
            {view === "map" ? <HeatMap result={result} ramp={ramp} /> :
              <div className="h-[28rem] w-full overflow-hidden rounded-lg border border-border" data-testid="landscape-surface">
                {ramp && <HeatSurface result={result} ramp={ramp} reduceMotion={reduceMotion} />}
              </div>}
            <p className="mt-2 text-xs text-muted-foreground">{T.axisX(axisUnit(result))} across, {T.axisY(axisUnit(result))} {view === "map" ? "up" : "into the screen"}.</p>
            <Legend />
          </div>
          <Readout result={result} />
        </div>
      </section>}
    </>}
  </main>;
}

function Readout({ result }: { result: LandscapeResult }) {
  return <dl className="flex min-w-0 flex-col gap-3 text-sm lg:w-72" data-testid="landscape-readout">
    <div><dt className="flex items-center gap-2 font-medium"><i className="inline-block size-3 rounded-full bg-landscape-current" aria-hidden="true" />{T.here}</dt>
      <dd className="tabular-nums" data-testid="landscape-here">{T.nats(result.current.loss)}</dd></div>
    <div><dt className="flex items-center gap-2 font-medium"><i className="inline-block size-3 rotate-45 bg-landscape-minimum" aria-hidden="true" />{T.lowest}</dt>
      <dd className="tabular-nums" data-testid="landscape-lowest">{T.nats(result.minimum.loss)}</dd>
      <dd className="text-muted-foreground">{result.minimum.inside ? T.inside : T.atEdge}</dd></div>
    {result.trajectory.length > 0 && <div><dt className="flex items-center gap-2 font-medium"><i className="inline-block h-0.5 w-4 bg-landscape-path" aria-hidden="true" />{T.path}</dt>
      <dd className="text-muted-foreground">{result.trajectory.length} snapshots. {T.pathNote}</dd></div>}
    <p className="text-muted-foreground">{T.centre(result.center.name, result.center.step)}</p>
    <p className="text-muted-foreground" data-testid="landscape-measured">{costLine(result)}.</p>
    <p className="text-xs text-muted-foreground">{result.note}</p>
    <p className="text-xs text-muted-foreground">{T.method}</p>
  </dl>;
}

function Legend() {
  return <div className="mt-3 flex max-w-xl items-center gap-2 text-xs text-muted-foreground" aria-label={T.colourNote}>
    <span>{T.legendLow}</span>
    <div className="flex h-3 flex-1 overflow-hidden rounded-sm border border-border" aria-hidden="true">
      {RAMP_TOKENS.map((token) => <i key={token} className="flex-1" style={{ background: `var(${token})` }} />)}
    </div>
    <span>{T.legendHigh}</span>
  </div>;
}

/**
 * Live landscape frames: one in flight at a time, one more queued if a snapshot arrived meanwhile. The first frame (and each
 * "new directions") sets the filter norms every later frame reuses, so the surface keeps its shape and only the loss moves.
 */
function useLiveFrames(live: boolean, version: number | undefined, seed: number) {
  const [state, setState] = useState<{ result?: LandscapeResult; error: string; busy: boolean }>({ error: "", busy: false });
  const running = useRef(false);
  const queued = useRef(false);
  const fresh = useRef(true);
  const controller = useRef<AbortController | null>(null);
  const on = useRef(live);

  const request = useCallback(async () => {
    if (!on.current || version === undefined) return;
    if (running.current) { queued.current = true; return; }
    running.current = true;
    const abort = new AbortController();
    controller.current = abort;
    setState((was) => ({ ...was, busy: true }));
    try {
      const result = await fetchLandscape({ version, live: true, grid: LIVE_GRID_MAX, seed, new_anchor: fresh.current }, abort.signal);
      fresh.current = false;
      if (on.current) setState({ result, error: "", busy: false });
    } catch (caught) {
      if (!abort.signal.aborted && on.current) setState((was) => ({ ...was, error: caught instanceof Error ? caught.message : String(caught), busy: false }));
    } finally {
      running.current = false;
      if (queued.current && on.current && !abort.signal.aborted) { queued.current = false; void request(); }
      else setState((was) => (was.busy ? { ...was, busy: false } : was));
    }
  }, [version, seed]);

  useEffect(() => {
    on.current = live;
    if (!live) return;
    fresh.current = true;
    void request();
    return () => {
      on.current = false;
      queued.current = false;
      controller.current?.abort();
      setState({ error: "", busy: false });
    };
  }, [live, request]);

  useEvent("train", (payload) => { if (isSnapshotEvent(payload)) void request(); });
  return state;
}

"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { awaitJob } from "@/app/benchmark/job";
import { SeriesChart } from "@/components/series-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ApiError, api, subscribe, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { COMPARE } from "./copy";
import {
  axisFor,
  curveSeries,
  deltaCell,
  finalCell,
  guessLine,
  letter,
  matchingText,
  queueBlock,
  variantName,
  type AxisId,
  type CompareRun,
  type CurvePoint,
} from "./logic";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

type Setup = components["schemas"]["CompareSetup"];
type Version = components["schemas"]["Version"];

const RUNS_KEY = ["compare", "runs"] as const;
const CURRENT = "current";

/** A server message: the `detail` of an error reply, else the error's own text. */
function message(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const detail = (JSON.parse(error.body) as { detail?: unknown }).detail;
      if (typeof detail === "string") return detail;
    } catch {
      // not JSON: fall through to the generic text
    }
  }
  return error instanceof Error ? error.message : String(error);
}

const versionLabel = (v: Version) => v.name || `Version ${v.number}`;

export default function Page() {
  const client = useQueryClient();
  const versions = useQuery({ queryKey: ["compare", "versions"], queryFn: () => unwrap(api.GET("/versions")) });
  const runs = useQuery({
    queryKey: RUNS_KEY,
    queryFn: async () => awaitJob<CompareRun[]>(await unwrap(api.POST("/benchmark/compare-runs"))),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    retry: false,
  });

  const [base, setBase] = useState(CURRENT);
  const [variants, setVariants] = useState<number[]>([]);
  const [mode, setMode] = useState<Setup["mode"]>("params");
  const [seeds, setSeeds] = useState<Setup["seeds"]>(3);
  const [retune, setRetune] = useState(true);
  const [question, setQuestion] = useState("");
  const [guess, setGuess] = useState("");
  const [axis, setAxis] = useState<AxisId>("steps");
  const [picked, setPicked] = useState<number | null>(null); // index into the finished runs; null = the newest
  const [status, setStatus] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [live, setLive] = useState<Map<number, { seed: number; points: CurvePoint[] }>>(new Map());
  const [doneGuess, setDoneGuess] = useState("");
  const [queuedNames, setQueuedNames] = useState<string[]>([]);

  const all = versions.data ?? [];
  const baseNumber = base === CURRENT ? (all.at(-1)?.number ?? null) : Number(base);
  const others = all.filter((v) => v.number !== baseNumber && !variants.includes(v.number));
  const setup: Setup = { base: base === CURRENT ? null : Number(base), variants, mode, seeds, retune_lr: retune, question, guess };

  // Planning is a server call: it matches each variant to the base and counts the runs.
  const plan = useQuery({
    queryKey: ["compare", "plan", base, variants, mode, seeds, retune],
    queryFn: () => unwrap(api.POST("/benchmark/compare-plan", { body: { ...setup, question: "", guess: "" } })),
    enabled: variants.length > 0,
    retry: false,
  });
  const block = queueBlock(variants.length, guess);

  const queue = useMutation({
    mutationFn: () => unwrap(api.POST("/benchmark/compare-queue", { body: setup })),
    onSuccess: (queued) => {
      setQueuedNames(plan.data?.variants.map((r) => r.name) ?? []);
      setJobId(queued.id);
      setLive(new Map());
      setDoneGuess(guess);
      setStatus(COMPARE.queued(queued.label, queued.outlook));
      toast.success(COMPARE.queued(queued.label, queued.outlook));
    },
    onError: (error) => {
      setStatus(message(error));
      toast.error(message(error));
    },
  });

  // Curves stream in while the queued runs train; the finished comparison is read again when they end.
  useEffect(() => {
    if (!jobId) return;
    return subscribe((event) => {
      const p = event.payload as Record<string, unknown> | null;
      if (!p || typeof p !== "object") return;
      if (event.kind === "job" && (p.job ?? p.id) === jobId && (p.status === "failed" || p.status === "stopped" || p.status === "cancelled")) {
        setStatus(COMPARE.runsEnded(String(p.status), String(p.error ?? "")));
        setJobId(null);
        return;
      }
      if (event.kind !== "train" || p.job !== jobId) return;
      const variant = Number(p.variant ?? 0);
      if (p.type === "compare" && p.phase === "run") {
        const point: CurvePoint = [Number(p.step), Number(p.bpc), (p.trained_tokens as number | null) ?? null, (p.elapsed_seconds as number | null) ?? null];
        setLive((old) => {
          const kept = old.get(variant);
          const next = new Map(old);
          next.set(variant, { seed: Number(p.seed), points: kept && kept.seed === Number(p.seed) ? [...kept.points, point].slice(-400) : [point] });
          return next;
        });
        setStatus(COMPARE.runningLive(letter(variant), Number(p.seed), Number(p.step), p.total as number | undefined));
      } else if (p.type === "compare" && p.phase === "sweep") {
        setStatus(COMPARE.sweepLive(letter(variant), Number(p.step), p.total as number | undefined));
      } else if (p.type === "compare_done") {
        setJobId(null);
        setLive(new Map());
        setPicked(null);
        void client.invalidateQueries({ queryKey: RUNS_KEY });
      }
    });
  }, [jobId, client]);

  const finished = runs.data ?? [];
  const index = picked !== null && picked < finished.length ? picked : finished.length - 1;
  const run = finished[index];
  const streaming = jobId !== null && live.size > 0;
  const names = streaming ? queuedNames : run?.names;

  return (
    <main className={PAGE_MAIN}>
      <PageHeader breadcrumb="Network designer" title={COMPARE.h1} lead={COMPARE.lead} />

      <Card>
        <CardHeader>
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{COMPARE.setupTitle}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="compare-question">{COMPARE.question}</Label>
            <input
              id="compare-question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={COMPARE.questionPlaceholder}
              className="h-9 w-full rounded-lg border bg-transparent px-3 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{COMPARE.base}</Label>
            {versions.isPending ? (
              <Skeleton aria-label="Loading the versions" className="h-9 w-56" />
            ) : (
              <Select value={base} onValueChange={(v) => { setBase(v); setVariants([]); }}>
                <SelectTrigger aria-label={COMPARE.base} className="w-56">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={CURRENT}>{COMPARE.baseCurrent}</SelectItem>
                  {all.map((v) => (
                    <SelectItem key={v.number} value={String(v.number)}>
                      {versionLabel(v)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{COMPARE.holdEqual}</Label>
            <ToggleGroup type="single" variant="outline" value={mode} onValueChange={(v) => v && setMode(v as Setup["mode"])} aria-label={COMPARE.holdEqual}>
              {COMPARE.modes.map((m) => (
                <ToggleGroupItem key={m.id} value={m.id}>
                  {m.label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <p className="text-xs text-muted-foreground">{COMPARE.modes.find((m) => m.id === mode)?.note}</p>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{COMPARE.seeds}</Label>
            <ToggleGroup type="single" variant="outline" value={String(seeds)} onValueChange={(v) => v && setSeeds(Number(v) as Setup["seeds"])} aria-label={COMPARE.seeds}>
              {["1", "3", "5"].map((n) => (
                <ToggleGroupItem key={n} value={n}>
                  {n}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <div className="mt-2 flex items-center gap-2">
              <Switch id="compare-retune" checked={retune} onCheckedChange={setRetune} />
              <Label htmlFor="compare-retune">{COMPARE.retuneLr}</Label>
            </div>
            <p className="text-xs text-muted-foreground">{COMPARE.seedsNote(retune ? COMPARE.lrNoteOn : COMPARE.lrNoteOff)}</p>
          </div>

          <div className="flex flex-col gap-2">
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-muted-foreground">
                <tr>
                  {[COMPARE.run, "", COMPARE.changeFromBase, COMPARE.matching, ""].map((h, i) => (
                    <th key={i} className="pb-1 pr-4 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="py-1 pr-4 font-medium">A</td>
                  <td className="py-1 pr-4">{COMPARE.baseTitle(baseNumber)}</td>
                  <td className="py-1 pr-4 text-muted-foreground">{COMPARE.baseDiff}</td>
                  <td className="py-1 pr-4">—</td>
                  <td />
                </tr>
                {variants.map((n, i) => {
                  const v = all.find((x) => x.number === n);
                  const planned = plan.data?.variants[i + 1];
                  return (
                    <tr key={n}>
                      <td className="py-1 pr-4 font-medium">{letter(i + 1)}</td>
                      <td className="py-1 pr-4">{v ? versionLabel(v) : `Version ${n}`}</td>
                      <td className="py-1 pr-4 font-mono text-xs text-muted-foreground">{planned?.changes?.join(" · ") || (planned ? "no change" : "")}</td>
                      <td className="py-1 pr-4 text-muted-foreground">
                        {planned ? matchingText(planned, plan.data?.variants[0]?.steps, mode, plan.data?.wall_seconds ?? 0, retune) : plan.isPending ? "…" : ""}
                      </td>
                      <td>
                        <Button variant="ghost" size="xs" aria-label={`${COMPARE.remove} ${v ? versionLabel(v) : n}`} onClick={() => setVariants((old) => old.filter((x) => x !== n))}>
                          {COMPARE.remove}
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {others.length > 0 ? (
              <Select value="" onValueChange={(v) => setVariants((old) => [...old, Number(v)])}>
                <SelectTrigger aria-label={COMPARE.addVariant} className="w-72">
                  <SelectValue placeholder={COMPARE.addVariantPlaceholder} />
                </SelectTrigger>
                <SelectContent>
                  {others.map((v) => (
                    <SelectItem key={v.number} value={String(v.number)}>
                      {versionLabel(v)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              !versions.isPending && variants.length === 0 && <p className="text-sm text-muted-foreground">{COMPARE.noVariantVersions}</p>
            )}
            {plan.isError && (
              <p role="alert" className="text-sm text-warning">
                {message(plan.error)}
              </p>
            )}
            {plan.data?.variants.filter((r) => !r.matched).map((r) => (
              <p key={r.variant} className="text-sm text-warning">{`Variant ${r.variant}: ${r.reason}`}</p>
            ))}
          </div>

          <div className="flex flex-col gap-2">
            <p className="text-sm">{COMPARE.guess}</p>
            <ToggleGroup type="single" variant="outline" value={guess} onValueChange={setGuess} aria-label={COMPARE.guess}>
              {COMPARE.guesses.map((g) => (
                <ToggleGroupItem key={g} value={g}>
                  {g}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
          <div className="flex flex-col items-start gap-2">
            <Button disabled={block !== "" || queue.isPending || plan.isError || !plan.data} onClick={() => queue.mutate()}>
              {plan.data ? COMPARE.queueRuns(plan.data.n_runs, plan.data.n_sweeps) : COMPARE.queueRuns(0, 0)}
            </Button>
            {block && variants.length > 0 && <p className="text-xs text-muted-foreground">{block}</p>}
            {status && (
              <p role="status" className="text-sm">
                {status}
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{COMPARE.finishedTitle}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {runs.isPending && <Skeleton aria-label="Loading the finished comparisons" className="h-40 w-full" />}
          {runs.isError && (
            <div className="flex flex-col items-start gap-2">
              <p role="alert" className="text-sm text-warning">
                {COMPARE.readFailed} {message(runs.error)}
              </p>
              <Button variant="outline" onClick={() => void runs.refetch()}>
                {COMPARE.retry}
              </Button>
            </div>
          )}
          {runs.data && finished.length === 0 && !streaming && <p className="text-sm text-muted-foreground">{COMPARE.noFinished}</p>}
          {finished.length > 1 && !streaming && (
            <Select value={String(index)} onValueChange={(v) => setPicked(Number(v))}>
              <SelectTrigger aria-label={COMPARE.finishedPick} className="w-full max-w-md">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {finished.map((r, i) => (
                  <SelectItem key={i} value={String(i)}>
                    {runTitle(r, i)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {(run || streaming) && (
            <Curves
              run={streaming ? undefined : run}
              live={streaming ? live : undefined}
              names={names}
              axis={axis}
              onAxis={setAxis}
              guess={streaming ? doneGuess : (run?.guess ?? "")}
            />
          )}
        </CardContent>
      </Card>
    </main>
  );
}

const when = (r: CompareRun) => (typeof r.finished === "number" ? new Date(r.finished * 1000).toLocaleString() : "");
const runTitle = (r: CompareRun, i: number) => [r.question?.trim() || `Comparison ${i + 1}`, when(r)].filter(Boolean).join(" · ");

/** The curves of one finished comparison (or the runs streaming now) on one chart, and its table. */
function Curves({
  run,
  live,
  names,
  axis,
  onAxis,
  guess,
}: {
  run?: CompareRun;
  live?: Map<number, { seed: number; points: CurvePoint[] }>;
  names?: string[];
  axis: AxisId;
  onAxis: (a: AxisId) => void;
  guess: string;
}) {
  const stats = new Map((run?.results?.variants ?? []).map((v) => [v.variant, v]));
  const curves = new Map<number, CurvePoint[]>(live ? [...live].map(([n, l]) => [n, l.points]) : [...stats].map(([n, v]) => [n, v.curve]));
  const used = axisFor([...curves.values()], axis);
  const title = COMPARE.axes.find((a) => a.id === used.axis)!.title;
  const series = curveSeries(curves, names, used.axis, run?.results?.references);
  const missing = used.missing && COMPARE.axes.find((a) => a.id === used.missing)!.title;
  return (
    <div className="flex flex-col gap-3">
      {run?.question?.trim() && <p className="text-sm">{run.question.trim()}</p>}
      <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{COMPARE.curvesTitle(title)}</p>
      <p className="text-xs text-muted-foreground">{COMPARE.curvesDesc}</p>
      <ToggleGroup type="single" variant="outline" size="sm" value={axis} onValueChange={(v) => v && onAxis(v as AxisId)} aria-label={COMPARE.axisLabel}>
        {COMPARE.axes.map((a) => (
          <ToggleGroupItem key={a.id} value={a.id}>
            {a.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <p className="text-xs text-muted-foreground">{COMPARE.axisFair}</p>
      {missing && <p className="text-sm text-warning">{COMPARE.axisFallback(missing)}</p>}
      {series.length > 0 ? <SeriesChart series={series} xLabel={COMPARE.axes.find((a) => a.id === used.axis)!.label} yLabel={COMPARE.yLabel} /> : <p className="text-sm text-muted-foreground">{COMPARE.curvesStream}</p>}
      {run && stats.size > 0 && (
        <>
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted-foreground">
              <tr>
                {[COMPARE.run, "", COMPARE.finalError, COMPARE.vsBase].map((h, i) => (
                  <th key={i} className="pb-1 pr-4 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...stats.values()].map((s) => (
                <tr key={s.variant} className="align-top">
                  <td className="py-1 pr-4 font-medium">{letter(s.variant)}</td>
                  <td className="py-1 pr-4">{variantName(run.names, s.variant)}</td>
                  <td className="whitespace-pre-line py-1 pr-4 text-muted-foreground">{finalCell(s)}</td>
                  <td className={`py-1 pr-4 ${s.inside_spread ? "text-muted-foreground" : ""}`}>{s.variant ? deltaCell(s) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {guess && <p className="text-sm">{guessLine(guess, "B", stats.get(1))}</p>}
          {run.results?.cancelled && <p className="text-sm text-warning">{COMPARE.cancelled}</p>}
        </>
      )}
    </div>
  );
}

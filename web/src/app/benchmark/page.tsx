"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { SeriesChart } from "@/components/series-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { api, subscribe, unwrap } from "@/lib/api";
import { BENCH } from "./copy";
import { awaitJob } from "./job";
import {
  CHARTS,
  chartFor,
  fewShotNote,
  fewShotRows,
  hasItems,
  pending,
  shotsText,
  statusText,
  trendTables,
  valueText,
  versionName,
  type BenchmarkOut,
  type BenchmarkRow,
  type FewShotOut,
} from "./logic";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { TrainingSpeed } from "./training-speed";

const READ_KEY = ["benchmark", "read"] as const;
const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

export default function Page() {
  const client = useQueryClient();
  const read = useQuery({
    queryKey: READ_KEY,
    queryFn: async () => awaitJob<BenchmarkOut>(await unwrap(api.POST("/benchmark/read"))),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    retry: false,
  });

  // A version finished training: show it, waiting to be checked.
  useEffect(
    () =>
      subscribe((event) => {
        if (event.kind === "versions") void client.invalidateQueries({ queryKey: READ_KEY });
      }),
    [client],
  );

  const state = read.data;
  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={BENCH.h1} lead={BENCH.lead} />
      {read.isPending && <BenchmarkSkeleton />}
      {read.isError && (
        <Card>
          <CardContent className="flex flex-col items-start gap-3">
            <p role="alert" className="text-sm text-warning">
              {BENCH.readFailed} {message(read.error)}
            </p>
            <Button variant="outline" onClick={() => void read.refetch()}>
              {BENCH.retry}
            </Button>
          </CardContent>
        </Card>
      )}
      {state && <Loaded state={state} />}
      <TrainingSpeed />
    </main>
  );
}

function BenchmarkSkeleton() {
  return (
    <div aria-label="Loading the benchmark" className="flex flex-col gap-4">
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-64 w-full" />
      <Skeleton className="h-64 w-full" />
    </div>
  );
}

function Loaded({ state }: { state: BenchmarkOut }) {
  const client = useQueryClient();
  const rows = state.rows ?? [];
  const todo = pending(state);
  const [failed, setFailed] = useState<Record<number, string>>({});
  const [progress, setProgress] = useState("");

  // Run the suite on every unchecked version, one after another; a version that fails does not stop the others.
  const check = useMutation({
    mutationFn: async (numbers: number[]) => {
      const errors: Record<number, string> = {};
      for (const [done, number] of numbers.entries()) {
        setProgress(BENCH.running(number, done + 1, numbers.length));
        try {
          await awaitJob(await unwrap(api.POST("/benchmark/run", { body: { version: number } })));
        } catch (error) {
          errors[number] = message(error);
        }
      }
      return { numbers, errors };
    },
    onSuccess: ({ numbers, errors }) => {
      setFailed((old) => ({ ...Object.fromEntries(Object.entries(old).filter(([n]) => !numbers.includes(Number(n)))), ...errors }));
      const bad = Object.keys(errors).length;
      if (bad) toast.warning(`${numbers.length - bad} of ${numbers.length} versions checked`);
      else toast.success(numbers.length === 1 ? "Version checked" : `${numbers.length} versions checked`);
      void client.invalidateQueries({ queryKey: READ_KEY });
    },
    onError: (error) => toast.error(message(error)),
    onSettled: () => setProgress(""),
  });

  const dropped = Object.values(state.dropped ?? {}).reduce((a, b) => a + b, 0);
  const sizes = state.sizes ?? {};
  const anyData = Object.values(sizes).some(Boolean);
  const shots = shotsText(state);
  return (
    <>
      <Card>
        <CardContent className="flex flex-col gap-2">
          {anyData && <p className="text-sm text-muted-foreground">{BENCH.fixed(sizes.docs ?? 0, sizes.choice ?? 0, sizes.problems ?? 0)}</p>}
          {(state.sources ?? []).length > 0 && <p className="text-sm text-muted-foreground">{BENCH.sources(state.sources ?? [])}</p>}
          <p className="text-sm">{statusText(state)}</p>
          {Object.entries(failed).map(([number, error]) => (
            <p key={number} className="text-sm text-warning">
              {BENCH.failed(Number(number), error)}
            </p>
          ))}
          {dropped > 0 && <p className="text-sm text-muted-foreground">{BENCH.dropped(dropped)}</p>}
          {check.isPending ? (
            <div className="flex flex-col gap-1">
              <p role="status" className="text-sm">
                {progress}
              </p>
              <p className="text-xs text-muted-foreground">{BENCH.runningNow}</p>
            </div>
          ) : (
            todo.length > 0 && (
              <Button className="mt-2 self-start" onClick={() => check.mutate(todo)}>
                {BENCH.run}
              </Button>
            )
          )}
        </CardContent>
      </Card>
      {rows.length > 0 && anyData && CHARTS.map((c) => <MetricCard key={c.metric} state={state} chart={c} shots={shots} />)}
      {rows.length > 0 && anyData && <TrendCard state={state} />}
      {rows.length > 0 && <FewShotCard rows={rows} />}
      {rows.length > 1 && <VerdictCard rows={rows} />}
    </>
  );
}

function MetricCard({ state, chart, shots }: { state: BenchmarkOut; chart: (typeof CHARTS)[number]; shots: string }) {
  const { series, chance, unplaced } = hasItems(state, chart.metric) ? chartFor(state, chart.metric) : { series: [], chance: null, unplaced: [] };
  const left = !hasItems(state, chart.metric) ? BENCH.noMetricData(chart.what) : series.length === 0 ? BENCH.emptyMetric : "";
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{chart.title}</CardTitle>
        <CardDescription>{chart.desc(shots)}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {left ? (
          <p className="text-sm text-muted-foreground">{left}</p>
        ) : (
          <>
            <SeriesChart series={series} xLabel={BENCH.xLabel} yLabel={chart.axis} />
            {chance !== null && <p className="text-xs text-muted-foreground">{BENCH.chance(valueText(chart.metric, chance))}</p>}
            <p className="text-xs text-muted-foreground">
              {BENCH.axis} {BENCH.linked}
            </p>
          </>
        )}
        {unplaced.length > 0 && <p className="text-xs text-warning">{BENCH.unplaced(unplaced)}</p>}
      </CardContent>
    </Card>
  );
}

/** The suite across the versions of each lineage, each against the one before with the server's paired test, so a gain is not just a lower number. */
function TrendCard({ state }: { state: BenchmarkOut }) {
  const tables = trendTables(state);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{BENCH.trendTitle}</CardTitle>
        <CardDescription>{BENCH.trendHint}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {tables.length === 0 && <p className="text-sm text-muted-foreground">{BENCH.trendEmpty}</p>}
        {tables.map((table) => (
          <div key={table.lineage} className="flex flex-col gap-1 overflow-x-auto">
            <h3 className="text-sm font-medium">{table.title}</h3>
            <table aria-label={table.title} className="w-full text-left text-sm">
              <thead className="text-xs text-muted-foreground">
                <tr>
                  {BENCH.trendHead.map((h, i) => (
                    <th key={i} className="pb-1 pr-4 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((cells) => (
                  <tr key={cells[0]}>
                    {cells.map((c, i) => (
                      <td key={i} className={`py-1 pr-4 ${i >= 1 ? "font-mono" : ""}`}>
                        {c}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function VersionSelect({ rows, value, onChange, label }: { rows: BenchmarkRow[]; value: string; onChange: (v: string) => void; label: string }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger aria-label={label} className="w-56">
        <SelectValue placeholder={label} />
      </SelectTrigger>
      <SelectContent>
        {rows.map((r) => (
          <SelectItem key={r.number} value={String(r.number)}>
            {versionName(r)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/** The few-shot check for one version: asked on request (it takes minutes), answered as a job. */
function FewShotCard({ rows }: { rows: BenchmarkRow[] }) {
  const [version, setVersion] = useState(String(rows.at(-1)?.number ?? ""));
  const [tables, setTables] = useState<Record<string, FewShotOut | { error: string }>>({});
  const ask = useMutation({
    mutationFn: async (number: string) => ({
      number,
      table: await awaitJob<FewShotOut>(await unwrap(api.POST("/benchmark/few-shot", { body: { version: Number(number) } }))),
    }),
    onSuccess: ({ number, table }) => setTables((old) => ({ ...old, [number]: table })),
    onError: (error, number) => {
      setTables((old) => ({ ...old, [number]: { error: message(error) } }));
      toast.error(message(error));
    },
  });
  const table = tables[version];
  const failed = table && "error" in table ? table.error : "";
  const result = table && !("error" in table) ? table : undefined;
  const shown = fewShotRows(result);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{BENCH.shotsTitle}</CardTitle>
        <CardDescription>{BENCH.shotsHint}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <VersionSelect rows={rows} value={version} onChange={setVersion} label={BENCH.shotsVersion} />
          <Button disabled={ask.isPending || !version} onClick={() => ask.mutate(version)}>
            {result ? BENCH.shotsAgain : BENCH.shotsRun}
          </Button>
        </div>
        {ask.isPending ? (
          <div className="flex flex-col gap-2">
            <p role="status" className="text-sm text-muted-foreground">
              {BENCH.shotsRunning}
            </p>
            <Skeleton aria-label="Asking the questions" className="h-24 w-full" />
          </div>
        ) : shown.length > 0 && result ? (
          <>
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-muted-foreground">
                <tr>
                  {BENCH.shotsHead.map((h) => (
                    <th key={h} className="pb-1 pr-4 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {shown.map((cells) => (
                  <tr key={cells[0]}>
                    {cells.map((c, i) => (
                      <td key={i} className={`py-1 pr-4 ${i >= 1 && i <= 3 ? "font-mono" : ""}`}>
                        {c}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-muted-foreground">{fewShotNote(result)}</p>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">{failed || (result ? BENCH.shotsNone : BENCH.shotsNotRun)}</p>
        )}
      </CardContent>
    </Card>
  );
}

/** How much training the newer of two versions saved over the older; the server says why when there is no verdict. */
function VerdictCard({ rows }: { rows: BenchmarkRow[] }) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const ready = a !== "" && b !== "" && a !== b;
  const verdict = useQuery({
    queryKey: ["benchmark", "verdict", a, b],
    queryFn: () => unwrap(api.GET("/benchmark/verdict", { params: { query: { a: Number(a), b: Number(b) } } })),
    enabled: ready,
    retry: false,
  });
  const [older, newer] = [Number(a), Number(b)].sort((x, y) => x - y);
  const name = (n: number) => versionName(rows.find((r) => r.number === n) ?? { number: n });
  const v = verdict.data;
  const ok = Boolean(v?.comparable && v.percent != null);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{BENCH.verdictTitle}</CardTitle>
        <CardDescription>{BENCH.verdictText}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-2">
          <VersionSelect rows={rows} value={a} onChange={setA} label={BENCH.verdictOlder} />
          <VersionSelect rows={rows} value={b} onChange={setB} label={BENCH.verdictNewer} />
        </div>
        {!ready && <p className="text-sm text-muted-foreground">{BENCH.verdictPick}</p>}
        {ready && verdict.isPending && <Skeleton aria-label="Reading the verdict" className="h-10 w-full" />}
        {ready && verdict.isError && <p className="text-sm text-warning">{message(verdict.error)}</p>}
        {ready && v && (
          <div role="status" className="flex flex-col gap-1">
            <p className={`text-lg font-semibold ${ok ? "" : "text-warning"}`}>
              {ok ? BENCH.verdictLine(name(newer), name(older), Math.round(100 - (v.percent ?? 0))) : BENCH.verdictNone(v.reason ?? "")}
            </p>
            <p className="text-xs text-muted-foreground">{ok ? BENCH.verdictSub : BENCH.verdictNoneSub}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { SeriesChart, type Series } from "@/components/series-chart";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ApiError, api, subscribe, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { cn } from "@/lib/utils";
import { QUICKTEST } from "./copy";
import { NO_RUN, applyEvent, checkMark, secondsLeft, signalText, type Result, type RunState, type Which } from "./logic";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

const route = routeFor("quicktest");
const STORE = "groundwork.quicktest"; // this tab's last run: when it was queued (server clock), its job and the guess made
const NEVER = 1e15; // `since` for "no run started from this tab": nothing finished after it
const REFRESH_MS = 5_000;

type Started = { since: number; jobId: string; guess: string };

function loadStarted(): Started | null {
  try {
    const raw = window.sessionStorage.getItem(STORE);
    return raw ? (JSON.parse(raw) as Started) : null;
  } catch {
    return null;
  }
}

function saveStarted(value: Started | null) {
  try {
    if (value) window.sessionStorage.setItem(STORE, JSON.stringify(value));
    else window.sessionStorage.removeItem(STORE);
  } catch {
    // storage can be blocked; the page then only forgets its run when it is reloaded
  }
}

function detail(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const body = JSON.parse(error.body) as { detail?: unknown };
      if (typeof body.detail === "string") return body.detail;
    } catch {
      // not JSON: fall through to the status line
    }
  }
  return error instanceof Error ? error.message : String(error);
}

function Kicker({ children }: { children: React.ReactNode }) {
  return <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{children}</span>;
}

function Choice<T extends string>({
  label,
  hint,
  value,
  onChange,
  options,
}: {
  label: string;
  hint: string;
  value: T;
  onChange: (value: string) => void;
  options: { id: T; label: string }[];
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-sm font-medium">{label}</span>
      <ToggleGroup
        type="single"
        variant="outline"
        value={value}
        onValueChange={(next) => next && onChange(next)}
        aria-label={label}
        className="flex-wrap justify-start"
      >
        {options.map((o) => (
          <ToggleGroupItem key={o.id} value={o.id}>
            {o.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <span className="text-xs text-muted-foreground">{hint}</span>
    </div>
  );
}

function Design({ kicker, title, description, tone, children }: { kicker: string; title: string; description: string; tone: string; children?: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-1 rounded-md border bg-muted/30 p-4">
      <span className={cn("text-xs font-medium tracking-wide uppercase", tone)}>{kicker}</span>
      <span className="text-lg font-semibold">{title}</span>
      {children}
      <span className="text-sm text-muted-foreground">{description}</span>
    </div>
  );
}

export default function Page() {
  const client = useQueryClient();
  const router = useRouter();

  const [started, setStarted] = useState<Started | null>(loadStarted);
  const [run, setRun] = useState<RunState>(NO_RUN);
  const [dataset, setDataset] = useState<"stories" | "slice" | "problems">("stories");
  const [minutes, setMinutes] = useState<"1" | "3" | "10">("3");
  const [size, setSize] = useState<"full" | "shrunk">("full");
  const [pickedGuess, setGuess] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [now, setNow] = useState(() => Date.now());

  const state = useQuery({
    queryKey: ["quicktest", started?.since ?? NEVER],
    queryFn: () => unwrap(api.GET("/quicktest", { params: { query: { since: started?.since ?? NEVER } } })),
    refetchInterval: REFRESH_MS,
  });
  const queue = useQuery({ queryKey: ["queue"], queryFn: () => unwrap(api.GET("/queue")), refetchInterval: REFRESH_MS });

  // A quick test already queued (from an earlier visit or another tab) is watched like one started here.
  const queued = useMemo(
    () => [queue.data?.job, ...(queue.data?.up_next ?? [])].find((j) => j?.kind === "quicktest") ?? null,
    [queue.data],
  );
  const jobId = started?.jobId ?? queued?.id ?? null;
  const guess = pickedGuess ?? started?.guess ?? (queued?.params as { guess?: string } | undefined)?.guess ?? "";

  useEffect(
    () =>
      subscribe((event) => {
        const at = Date.now();
        setRun((r) => applyEvent(r, jobId, event.kind, event.payload, at));
        if (event.kind === "queue") void client.invalidateQueries({ queryKey: ["queue"] });
        if (event.kind === "train" && (event.payload as { type?: string } | null)?.type === "quicktest_done") {
          void client.invalidateQueries({ queryKey: ["queue"] });
          void client.invalidateQueries({ queryKey: ["quicktest"] });
        }
      }),
    [client, jobId],
  );

  const result: Result | null = run.result ?? state.data?.last ?? null;
  const inQueue = queue.data ? [queue.data.job, ...queue.data.up_next].some((j) => j?.id === jobId) : true;
  const running = !!jobId && !result && !run.ended && inQueue;
  useEffect(() => {
    if (!running) return;
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, [running]);

  const start = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/quicktest/run", {
          body: { dataset, minutes: Number(minutes) as 1 | 3 | 10, shrink: size === "shrunk", guess: guess as "Lower" | "Same" | "Higher" },
        }),
      ),
    onSuccess: async (made) => {
      const next = { since: made.started_at, jobId: made.job_id, guess };
      saveStarted(next);
      setStarted(next);
      setRun(NO_RUN);
      setGuess(guess);
      setStatus(QUICKTEST.queued(made.outlook));
      await client.invalidateQueries({ queryKey: ["queue"] });
    },
    onError: (error) => setStatus(detail(error)),
  });
  const save = useMutation({
    mutationFn: () => unwrap(api.POST("/quicktest/save")),
    onSuccess: () => {
      saveStarted(null);
      void client.invalidateQueries({ queryKey: ["quicktest"] });
      router.push("/builder");
    },
    onError: (error) => toast.error(`${QUICKTEST.saveFailed}: ${detail(error)}`),
  });
  const discard = useMutation({
    mutationFn: () => unwrap(api.DELETE("/quicktest/draft")),
    onSuccess: () => {
      saveStarted(null);
      void client.invalidateQueries({ queryKey: ["quicktest"] });
      router.push("/builder");
    },
    onError: (error) => toast.error(`${QUICKTEST.discardFailed}: ${detail(error)}`),
  });

  const onStart = () => {
    if (!state.data?.draft) setStatus(QUICKTEST.needDraft);
    else if (!guess) setStatus(QUICKTEST.needGuess);
    else start.mutate();
  };

  const name = state.data?.current.title ?? "Current version";
  const shownStatus = run.ended || status;
  const busy = running || start.isPending;

  return (
    <main className={PAGE_MAIN}>
      <PageHeader
        breadcrumb={<Link href="/builder" className="hover:underline">‹ Network designer</Link>}
        title={route.title}
        lead={QUICKTEST.lead}
      />

      <Card>
        <CardContent className="flex flex-wrap items-center gap-3">
          <Badge variant="outline" className="border-warning/50 text-warning-soft">
            {QUICKTEST.throwaway}
          </Badge>
          <span className="text-sm text-muted-foreground">{QUICKTEST.throwawayDesc}</span>
        </CardContent>
      </Card>

      {state.isError ? (
        <p role="alert" className="text-sm text-destructive">
          {QUICKTEST.loadFailed} {detail(state.error)}
        </p>
      ) : !state.data ? (
        <div className="flex flex-col gap-4" aria-label="Loading">
          <Skeleton className="h-36 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>
                <Kicker>{QUICKTEST.testing}</Kicker>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-col gap-4 md:flex-row">
                <Design kicker={QUICKTEST.currentDesign} title={name} description={state.data.current.description} tone="text-side-left" />
                <Design
                  kicker={QUICKTEST.yourIdea}
                  title={QUICKTEST.withChange(name)}
                  description={state.data.draft ? state.data.draft.cost : QUICKTEST.noChange}
                  tone="text-side-draft"
                >
                  {state.data.draft && state.data.draft.changes.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 py-1">
                      {state.data.draft.changes.map((change) => (
                        <Badge key={change} variant="secondary">
                          {change}
                        </Badge>
                      ))}
                    </div>
                  )}
                </Design>
              </div>
              <Link href="/builder" className={cn(buttonVariants({ variant: "outline" }), "self-start")}>
                {QUICKTEST.changeDesigner}
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="flex flex-col gap-5">
              <Choice
                label={QUICKTEST.tinyDataset}
                hint={QUICKTEST.datasets.find((d) => d.id === dataset)!.note}
                value={dataset}
                onChange={(v) => setDataset(v as typeof dataset)}
                options={QUICKTEST.datasets}
              />
              <Choice label={QUICKTEST.timeEach} hint={QUICKTEST.bothGetTime} value={minutes} onChange={(v) => setMinutes(v as typeof minutes)} options={QUICKTEST.minutes} />
              <Choice
                label={QUICKTEST.sizeLabel}
                hint={QUICKTEST.sizes.find((s) => s.id === size)!.note}
                value={size}
                onChange={(v) => setSize(v as typeof size)}
                options={QUICKTEST.sizes}
              />
            </CardContent>
          </Card>

          <div className="flex flex-col gap-2">
            <span className="text-base font-medium">
              {QUICKTEST.yourGuess} <span className="text-sm font-normal text-muted-foreground">{QUICKTEST.resultShows}</span>
            </span>
            <ToggleGroup type="single" variant="outline" value={guess} onValueChange={(v) => v && setGuess(v)} aria-label={QUICKTEST.yourGuess} className="justify-start">
              {QUICKTEST.guesses.map((g) => (
                <ToggleGroupItem key={g} value={g}>
                  {g}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <Button className="mt-2 self-start" disabled={busy} onClick={onStart}>
              {busy ? `${QUICKTEST.running}…` : result ? QUICKTEST.runAgain : QUICKTEST.run}
            </Button>
            {shownStatus && (
              <p role="status" data-testid="quicktest-status" className="text-sm text-notice">
                {shownStatus}
              </p>
            )}
          </div>

          {(running || result) && (
            <ResultCard
              result={result}
              run={run}
              now={now}
              name={name}
              guess={guess}
              nextNumber={state.data.next_number}
              canSave={!!state.data.draft}
              saving={save.isPending}
              onSave={() => save.mutate()}
              onDiscard={() => discard.mutate()}
            />
          )}
        </>
      )}

      <details className="rounded-md border p-4">
        <summary className="cursor-pointer text-sm font-medium">{QUICKTEST.howKicker}</summary>
        <div className="mt-3 flex flex-col gap-3">
          {QUICKTEST.how.map((item) => (
            <div key={item.head}>
              <p className="text-sm font-semibold">{item.head}</p>
              <p className="max-w-prose text-sm text-muted-foreground">{item.text}</p>
            </div>
          ))}
        </div>
      </details>
    </main>
  );
}

function ResultCard({
  result,
  run,
  now,
  name,
  guess,
  nextNumber,
  canSave,
  saving,
  onSave,
  onDiscard,
}: {
  result: Result | null;
  run: RunState;
  now: number;
  name: string;
  guess: string;
  nextNumber: number;
  canSave: boolean;
  saving: boolean;
  onSave: () => void;
  onDiscard: () => void;
}) {
  if (result?.stopped) {
    return (
      <Card>
        <CardContent className="flex flex-col gap-1">
          <Kicker>{QUICKTEST.stoppedKicker}</Kicker>
          <p className="text-sm">{QUICKTEST.stoppedText}</p>
        </CardContent>
      </Card>
    );
  }
  const rows: { which: Which; label: string; tone: string }[] = [
    { which: "current", label: name, tone: "bg-side-left" },
    { which: "draft", label: QUICKTEST.yourIdeaName, tone: "bg-side-draft" },
  ];
  const checks = Object.fromEntries((result?.checks ?? []).map((c) => [c.id, c]));
  const curves: Series[] =
    result?.current && result.draft
      ? [
          { id: "current", label: name, points: result.current.curve.map(([x, y]) => ({ x, y })) },
          { id: "draft", label: QUICKTEST.yourIdeaName, points: result.draft.curve.map(([x, y]) => ({ x, y })) },
        ]
      : [];
  const xs = curves.flatMap((s) => s.points.map((p) => p.x));
  if (curves.length && result?.references && xs.length) {
    const [lo, hi] = [Math.min(...xs), Math.max(...xs)];
    curves.push(
      { id: "uniform", label: QUICKTEST.uniformGuess, points: [{ x: lo, y: result.references.uniform }, { x: hi, y: result.references.uniform }] },
      { id: "unigram", label: QUICKTEST.wordFrequencies, points: [{ x: lo, y: result.references.unigram }, { x: hi, y: result.references.unigram }] },
    );
  }

  return (
    <Card data-testid="quicktest-result">
      <CardContent className="flex flex-col gap-4">
        <Kicker>{result ? QUICKTEST.resultKicker : QUICKTEST.runningKicker}</Kicker>
        <div className="flex flex-col gap-2">
          {rows.map(({ which, label, tone }) => {
            const prog = run.progress[which];
            const first = run.firstSeen[which];
            let fraction = 0;
            let text: string = QUICKTEST.waiting;
            const finished = result?.[which];
            if (finished) {
              fraction = 1;
              text = QUICKTEST.doneSteps(finished.steps);
            } else if (result) {
              text = QUICKTEST.notRun;
            } else if (prog?.total && prog.step) {
              fraction = prog.step / prog.total;
              const left = first ? secondsLeft(first.at, prog.step - first.step, prog.total - first.step, now) : null;
              text = QUICKTEST.stepLeft(prog.step, left);
            } else if (which === "current" && run.progress.draft) {
              text = QUICKTEST.done;
            }
            return (
              <div key={which} className="flex items-center gap-3">
                <span className="w-40 shrink-0 truncate text-sm">{label}</span>
                <div
                  role="progressbar"
                  aria-label={label}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(fraction * 100)}
                  className="h-2 flex-1 overflow-hidden rounded-full bg-muted"
                >
                  <div className={cn("h-full", tone)} style={{ width: `${fraction * 100}%` }} />
                </div>
                <span className="w-36 shrink-0 text-sm text-muted-foreground">{text}</span>
              </div>
            );
          })}
        </div>

        {result && (
          <>
            {curves.length > 0 && <SeriesChart series={curves} xLabel={QUICKTEST.chartX} yLabel={QUICKTEST.chartY} />}

            <div className="flex flex-col gap-1">
              <Kicker>{QUICKTEST.earlySignal}</Kicker>
              <p className="max-w-prose text-lg">{signalText(result, name, guess || null)}</p>
              <p className="max-w-prose text-xs text-muted-foreground">{QUICKTEST.onePrefix}</p>
              {result.references && !result.learned_context && <p className="max-w-prose text-sm text-warning-soft">{QUICKTEST.guard}</p>}
            </div>

            <div className="flex flex-col gap-1">
              <Kicker>{QUICKTEST.doesWork}</Kicker>
              <ul className="flex flex-col gap-1">
                {QUICKTEST.checks.map((c) => {
                  const check = checks[c.id];
                  const ok = check?.ok;
                  return (
                    <li key={c.id} className="flex gap-3 text-sm" data-check={c.id}>
                      <span className={cn("w-4 shrink-0", ok ? "text-highlight" : ok === false ? "text-destructive" : "text-muted-foreground")}>
                        {checkMark(ok)}
                      </span>
                      <span>
                        {c.name} — {c.what} ({check?.detail ?? "not run"})
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>

            {result.current && result.draft && (
              <div className="flex flex-col gap-2">
                <Kicker>{QUICKTEST.whatEachWrites}</Kicker>
                {(
                  [
                    [name, result.current.sample],
                    [QUICKTEST.yourIdeaName, result.draft.sample],
                  ] as const
                ).map(([who, sample]) => (
                  <div key={who}>
                    <p className="text-xs text-muted-foreground">{who}</p>
                    <p className="font-mono text-sm break-words whitespace-pre-wrap">{sample}</p>
                  </div>
                ))}
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              <Button disabled={!canSave || saving} onClick={onSave}>
                {QUICKTEST.saveAsVersion(nextNumber)}
              </Button>
              <Link href="/compare" className={buttonVariants({ variant: "outline" })}>
                {QUICKTEST.runRealComparison}
              </Link>
              <Button variant="outline" onClick={onDiscard}>
                {QUICKTEST.discard}
              </Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

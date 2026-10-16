"use client";

import { useQuery } from "@tanstack/react-query";
import { Suspense, useState } from "react";
import { useVersionParameter } from "@/lib/version-param";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { cn } from "@/lib/utils";
import { TRACE } from "./copy";
import { reason, runJob } from "./use-job";

type Row = {
  stage: string;
  kind: "embedding" | "mixer" | "channel" | "output";
  norm: number | null;
  rel_change: number | null;
  cosine: number | null;
  top_token: string;
  top_p: number;
  approximate: boolean;
};
type Trace = { tokens: string[]; pos: number; rows: Row[] };

const DEFAULT_TEXT = `${TRACE.example_text} ${TRACE.last_position}`;
const WHAT = { embedding: TRACE.embedding_what, mixer: TRACE.mixer_what, channel: TRACE.channel_what, output: TRACE.output_what };
const GRID = "grid grid-cols-2 gap-x-4 gap-y-1 md:grid-cols-[minmax(0,1.4fr)_repeat(3,minmax(0,1fr))_minmax(0,1.4fr)]";

const num = (value: number | null, digits = 2) => (value === null || value === undefined ? "—" : value.toFixed(digits));
const stageName = (row: Row) => (row.kind === "embedding" ? TRACE.embedding : row.kind === "output" ? TRACE.output : row.stage);

export default function Page() {
  return <Suspense fallback={<Skeleton className="m-8 h-64" />}><VersionedTrace /></Suspense>;
}

function VersionedTrace() {
  const version = useVersionParameter();
  return <TracePage key={version ?? "best"} version={version} />;
}

function TracePage({ version }: { version?: number }) {
  const route = routeFor("trace");
  const [text, setText] = useState(DEFAULT_TEXT);
  const [submitted, setSubmitted] = useState(DEFAULT_TEXT);

  // pos -1 follows the last position, as the Tk page does. version null is the best trained version.
  const trace = useQuery({
    queryKey: ["inspect", "trace", version, submitted],
    queryFn: ({ signal }) =>
      runJob<Trace>(() => unwrap(api.POST("/inspect/trace", { body: { version, text: submitted, pos: -1 } })), signal),
    retry: false,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
  const data = trace.data;

  return (
    <main className="mx-auto flex w-full max-w-5xl min-w-0 flex-col gap-4 p-4 md:p-8">
      <p className="text-xs text-muted-foreground">Network designer › Look inside</p>
      <h1 className="text-2xl font-semibold tracking-tight">{route.title}</h1>
    {version !== undefined && <p className="text-sm text-muted-foreground">Version {version}</p>}
      <p className="max-w-prose text-muted-foreground">
        {route.lead} {TRACE.no_numbers}
      </p>

      <Card>
        <CardContent className="flex flex-col gap-3">
          <p className="text-xs font-medium tracking-wider text-muted-foreground">{TRACE.example_prompt}</p>
          <form
            className="flex flex-col gap-2 sm:flex-row"
            onSubmit={(event) => {
              event.preventDefault();
              if (text.trim()) setSubmitted(text);
            }}
          >
            <input
              aria-label={TRACE.prompt_label}
              value={text}
              onChange={(event) => setText(event.target.value)}
              className="h-9 min-w-0 flex-1 rounded-lg border border-input bg-transparent px-3 text-base outline-none focus-visible:ring-3 focus-visible:ring-ring/50 md:text-sm"
            />
            <Button type="submit" disabled={!text.trim() || trace.isFetching}>
              {trace.isFetching ? TRACE.running : TRACE.run}
            </Button>
          </form>
          {data ? (
            <p aria-label="Tokens" className="text-lg break-words whitespace-pre-wrap">
              {data.tokens.map((token, i) => (
                <span key={i} className={i === data.pos ? "font-bold text-amber-500" : undefined}>
                  {token}
                </span>
              ))}
            </p>
          ) : trace.isError ? null : (
            <Skeleton aria-label="Loading the prompt" className="h-6 w-3/4" />
          )}
        </CardContent>
      </Card>

      {trace.isError && (
        <p role="alert" className="text-muted-foreground">
          {reason(trace.error, TRACE.no_numbers, TRACE.server_error)}
        </p>
      )}

      {trace.isPending && !trace.isError && (
        <div aria-label="Loading the trace" role="status" className="flex flex-col gap-2">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      )}

      {data && (
        <div role="table" aria-label="Stages" aria-busy={trace.isFetching} className={cn("flex flex-col gap-2", trace.isFetching && "opacity-60")}>
          <div role="row" className={cn(GRID, "hidden px-3 text-xs text-muted-foreground md:grid")}>
            {[TRACE.stage, TRACE.size_norm, TRACE.relative_change, TRACE.direction_kept, TRACE.top_guess].map((head) => (
              <div role="columnheader" key={head}>
                {head}
              </div>
            ))}
          </div>
          {data.rows.map((row) => (
            <div role="row" key={row.stage} className={cn(GRID, "rounded-lg border bg-card px-3 py-2 text-sm")}>
              <div role="cell" className="col-span-2 min-w-0 md:col-span-1">
                <div className="font-medium break-words">{stageName(row)}</div>
                <div className="text-xs text-muted-foreground">{WHAT[row.kind]}</div>
              </div>
              <Cell label={TRACE.size_norm} value={num(row.norm)} />
              <Cell label={TRACE.relative_change} value={num(row.rel_change)} />
              <Cell label={TRACE.direction_kept} value={num(row.cosine)} />
              <Cell
                label={TRACE.top_guess}
                value={`“${row.top_token}”  ${row.top_p.toFixed(2)}${row.approximate ? " ≈" : ""}`}
                muted={row.approximate}
              />
            </div>
          ))}
        </div>
      )}

      {data && (
        <div className="flex max-w-prose flex-col gap-2 text-sm text-muted-foreground">
          <p>{TRACE.relative_note}</p>
          <p>{TRACE.approx_legend}</p>
        </div>
      )}
    </main>
  );
}

/** One number of a row. Below md the column heads are hidden, so each cell carries its own label. */
function Cell({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div role="cell" className="min-w-0">
      <div aria-hidden className="text-xs text-muted-foreground md:hidden">
        {label}
      </div>
      <div className={cn("font-mono break-words", muted && "text-muted-foreground")}>{value}</div>
    </div>
  );
}

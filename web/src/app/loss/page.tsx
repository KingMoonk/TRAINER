"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useEffect, useState } from "react";
import { InspectControls } from "@/components/inspect-controls";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, subscribe, unwrap } from "@/lib/api";
import { awaitJob } from "@/lib/job-result";
import { LOSS as T } from "./copy";
import { lossView, probability, showPiece, type LossResult } from "./logic";

export default function Page() {
  const cache = useQueryClient();
  const versions = useQuery({ queryKey: ["versions"], queryFn: ({ signal }) => unwrap(api.GET("/versions", { signal })), refetchInterval: 15_000 });
  const trained = (versions.data ?? []).filter((version) => version.best_checkpoint);
  const [selected, setSelected] = useState<number>();
  const best = useQuery({ queryKey: ["inspect", "best-version", trained.map((item) => [item.number, item.best_checkpoint, item.status])], enabled: trained.length > 0,
    queryFn: async ({ signal }) => {
      const value = await awaitJob<unknown>(await unwrap(api.POST("/versions/best", { signal })), signal);
      if (value === null) return null;
      if (!value || typeof value !== "object" || !("number" in value) ||
          typeof value.number !== "number" || !Number.isInteger(value.number) || value.number < 1) {
        throw new Error("The server returned an invalid best version.");
      }
      if (!trained.some((item) => item.number === value.number)) throw new Error("The best version changed. Refresh the version list and try again.");
      return value.number;
    },
    staleTime: 30_000, retry: false,
  });
  const version = trained.some((item) => item.number === selected) ? selected :
    trained.some((item) => item.number === best.data) ? best.data ?? undefined : best.isSuccess ? trained.at(-1)?.number : undefined;
  const checkpoint = trained.find((item) => item.number === version)?.best_checkpoint;
  const [text, setText] = useState<string>(T.defaultText);
  const [submitted, setSubmitted] = useState<string>(T.defaultText);
  const [afterTraining, setAfterTraining] = useState(true);
  const result = useQuery({
    queryKey: ["inspect", "loss", version, checkpoint, submitted, afterTraining],
    enabled: version !== undefined,
    queryFn: async ({ signal }) => {
      const data = await awaitJob<LossResult>(await unwrap(api.POST("/inspect/loss", {
        body: { text: submitted, version, pos: -1, trained: afterTraining }, signal,
      })), signal);
      return { data, view: lossView(data) };
    },
    retry: false, staleTime: 30_000,
  });

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const close = subscribe((event) => {
      const payload = event.payload as { job?: string; status?: string } | null;
      if (event.kind !== "checkpoint" && event.kind !== "versions" &&
          !(event.kind === "job" && payload?.job && payload.status !== "running")) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        void cache.invalidateQueries({ queryKey: ["inspect"] }, { cancelRefetch: false });
        void cache.invalidateQueries({ queryKey: ["versions"] }, { cancelRefetch: false });
      }, 500);
    });
    return () => { close(); if (timer) clearTimeout(timer); };
  }, [cache]);

  return <main className="mx-auto flex w-full min-w-0 max-w-5xl flex-col gap-4 p-4 md:p-8">
    <p className="text-xs text-muted-foreground">{T.breadcrumb}</p>
    <h1 className="text-2xl font-semibold tracking-tight">{T.title}</h1>
    <p className="max-w-prose text-muted-foreground">{T.lead}</p>
    {versions.isPending ? <Skeleton className="h-40 w-full" aria-label="Loading versions" /> : versions.isError ?
      <Card><CardContent role="alert">{T.versionsFailed} <Button variant="outline" onClick={() => void versions.refetch()}>{T.retry}</Button></CardContent></Card> : !trained.length ?
      <Card><CardContent><p>{T.empty}</p><Link href="/pretraining/" className="text-sm underline">{T.openPretraining}</Link></CardContent></Card> : <>
      <Card><CardContent className="flex flex-col gap-4">
        <InspectControls text={text} onText={setText} version={version} onVersion={setSelected} versions={trained}
          pending={result.isFetching} labels={T} onSubmit={() => {
            if (!text.trim()) return;
            if (text === submitted) void result.refetch(); else setSubmitted(text);
          }} />
        <p className="text-sm text-muted-foreground">{T.context}</p>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Model weights">
          {[false, true].map((value) => <Button key={String(value)} variant={afterTraining === value ? "default" : "outline"}
            aria-pressed={afterTraining === value} onClick={() => setAfterTraining(value)}>{value ? T.trained : T.random}</Button>)}
        </div>
      </CardContent></Card>
      {best.isError && selected === undefined ? <Card><CardContent className="flex flex-col items-start gap-3" role="alert">
        <p>Could not find the best version: {best.error.message}. Choose a version above or try again.</p>
        <Button variant="outline" onClick={() => void best.refetch()}>{T.retry}</Button>
      </CardContent></Card> : result.isError ? <Card><CardContent className="flex flex-col items-start gap-3" role="alert">
        <p>{T.failed}: {result.error.message}</p><Button variant="outline" onClick={() => void result.refetch()}>{T.retry}</Button>
      </CardContent></Card> : !result.data ? <div aria-label={T.loading} aria-busy="true" className="space-y-3">
        <Skeleton className="h-72 w-full" /><Skeleton className="h-32 w-full" />
      </div> : <section className="flex min-w-0 flex-col gap-4" aria-label="Prediction results" aria-busy={result.isFetching}>
        <p className="text-sm text-muted-foreground">Version {version} · {afterTraining ? T.trained : T.random}</p>
        {result.isFetching && <Skeleton className="h-1 w-full" aria-label={T.loading} />}
        <Prediction data={result.data.data} view={result.data.view} trained={afterTraining} />
      </section>}
    </>}
  </main>;
}

function Prediction({ data, view, trained }: { data: LossResult; view: ReturnType<typeof lossView>; trained: boolean }) {
  const context = data.pos >= 0 ? data.tokens.slice(0, data.pos).join("") : "";
  return <>
    <Card><CardHeader>
      <CardTitle className="break-words whitespace-pre-wrap">…{Array.from(context).slice(-24).join("")} ?</CardTitle>
      <p className="text-sm text-muted-foreground">{T.softmax(data.pieces.length, showPiece(view.piece))}</p>
      {data.grad_error != null && <p className="text-xs text-muted-foreground">{T.gradientCheck(data.grad_error)}</p>}
    </CardHeader><CardContent>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs sm:text-sm">
          <caption className="sr-only">{T.tableCaption}</caption>
          <thead><tr>{[T.token, T.probability, T.target, T.gradient].map((label) => <th key={label} scope="col" className="px-2 pb-3 font-medium text-muted-foreground">{label}</th>)}</tr></thead>
          <tbody>{view.rows.map((row) => <tr key={row.index} className={row.target ? "bg-amber-400/10 text-amber-500" : "border-t border-white/5"}>
            <th scope="row" className="max-w-36 break-all px-2 py-3 font-mono font-normal">{showPiece(row.piece)}</th>
            <td className="px-2 py-3 font-mono">{probability(row.probability)}<div className="mt-1 h-1 w-full rounded bg-muted" aria-hidden="true"><div className="h-1 rounded bg-current" style={{ width: `${row.probability * 100}%` }} /></div></td>
            <td className="px-2 py-3 font-mono">{row.target}</td>
            <td className="px-2 py-3 font-mono">{row.gradient < 0 ? "−" : "+"}{probability(Math.abs(row.gradient))}</td>
          </tr>)}
          <tr className="border-t text-muted-foreground"><th scope="row" className="px-2 py-3 font-normal">all {view.others.toLocaleString()} others</th><td className="px-2 font-mono">{probability(view.remaining)}</td><td className="px-2">0 each</td><td className="px-2">+p each</td></tr>
          </tbody>
        </table>
      </div>
    </CardContent></Card>
    <Card><CardHeader><CardTitle>{T.entropy}</CardTitle></CardHeader><CardContent className="space-y-2">
      <p className="break-words font-mono text-sm">loss = −ln p({showPiece(view.piece)}) = −ln {probability(view.probability)}</p>
      <p className="text-3xl font-semibold tabular-nums" data-testid="loss-nats">{view.loss.toFixed(2)} nats</p>
      <p className="text-sm text-muted-foreground">{trained ? T.trainedNote : T.randomNote(data.pieces.length)}</p>
    </CardContent></Card>
    <Card><CardHeader><CardTitle>{T.whyBits}</CardTitle></CardHeader><CardContent className="space-y-2 text-sm">
      <p className="text-muted-foreground">{T.natsDepend}</p><p className="break-words font-mono">{T.formula}</p>
      <p className="break-words">This piece, “{showPiece(view.piece)}”, {view.characters} character{view.characters === 1 ? "" : "s"}: {view.loss.toFixed(2)} ÷ ({view.characters} × {Math.LN2.toFixed(3)}) = <strong data-testid="loss-bpc">{view.bits.toFixed(2)} bits per character</strong>.</p>
      <p className="text-muted-foreground">{T.total}</p>
    </CardContent></Card>
  </>;
}

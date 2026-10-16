"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { awaitJob } from "@/lib/job-result";
import { CHAT as T } from "./copy";
import { bestNumber, checkpointStep, fewShotNote, fewShotResult, fewShotRows, type Version } from "./logic";

/** A new checkpoint gets its own cache entry even when it belongs to the same version. */
export const checkKey = (kind: "checkpoint" | "few-shot", version: Version) => ["chat", kind, version.number, version.best_checkpoint] as const;

export function useBestVersion(versions: Version[]) {
  return useQuery({
    queryKey: ["chat", "best", versions.map((v) => [v.number, v.best_checkpoint, v.status])],
    enabled: versions.length > 0,
    queryFn: async ({ signal }) => {
      const initial = await unwrap(api.POST("/versions/best"));
      const number = bestNumber(await awaitJob<unknown>(initial, signal));
      if (number !== null && !versions.some((version) => version.number === number)) throw new Error(T.bestChanged);
      return number;
    },
    staleTime: 30_000,
    retry: false,
  });
}

export function CheckpointStep({ version }: { version: Version }) {
  const query = useQuery({
    queryKey: checkKey("checkpoint", version),
    queryFn: async ({ signal }) => {
      const initial = await unwrap(api.POST("/versions/{number}/checkpoint-step", { params: { path: { number: version.number } } }));
      return checkpointStep(await awaitJob<unknown>(initial, signal));
    },
    staleTime: Infinity,
    refetchInterval: version.status === "training" ? 30_000 : false,
    retry: false,
  });
  return <div aria-label={`Checkpoint for Version ${version.number}`} className="text-xs text-muted-foreground">
    {query.isFetching && <span role="status" className="flex items-center gap-2"><Skeleton className="h-3 w-20" />{T.checkpointLoading}</span>}
    {query.isError && <div><span role="alert">{T.checkpointFailed}: {query.error.message}</span> <Button variant="link" size="sm" onClick={() => void query.refetch()}>{T.retryStep}</Button></div>}
    {query.data === null ? T.checkpointUnknown : query.data !== undefined ? T.checkpoint(query.data) : null}
  </div>;
}

/** Mount with the version/checkpoint key: pending, error and cancellation state must never cross versions. */
export function FewShotCheck({ version }: { version: Version }) {
  const client = useQueryClient();
  const [cancelled, setCancelled] = useState(false);
  const queryKey = checkKey("few-shot", version);
  const query = useQuery({
    queryKey,
    queryFn: async ({ signal }) => {
      const initial = await unwrap(api.POST("/benchmark/few-shot", { body: { version: version.number } }));
      return fewShotResult(await awaitJob<unknown>(initial, signal));
    },
    enabled: false,
    staleTime: Infinity,
    gcTime: 30 * 60_000,
    retry: false,
  });
  const table = query.data;
  const rows = table ? fewShotRows(table) : [];
  return <Card aria-label={`${T.fewShot}: Version ${version.number}`}>
    <CardHeader><CardTitle>{T.fewShot}</CardTitle><p className="text-sm text-muted-foreground">Version {version.number} · {T.fewShotHint}</p></CardHeader>
    <CardContent className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" disabled={query.isFetching} onClick={() => { setCancelled(false); void query.refetch(); }}>{table || query.isError || cancelled ? T.fewShotAgain : T.fewShotRun}</Button>
        {query.isFetching && <Button variant="outline" onClick={() => { setCancelled(true); void client.cancelQueries({ queryKey, exact: true }); }}>{T.cancelCheck}</Button>}
      </div>
      {query.isFetching && <div role="status" aria-label={T.fewShotRunning}><p className="mb-2 text-sm">{T.fewShotRunning}…</p><Skeleton className={table ? "h-3 w-32" : "h-28 w-full"} /></div>}
        {cancelled && <p role="status">{T.fewShotCancelled}</p>}
        {query.isError && <p role="alert">{query.error.message}</p>}
        {!table && !query.isError && !cancelled && !query.isFetching && <p className="text-sm text-muted-foreground">{T.fewShotNotRun}</p>}
        {table && (rows.length ? <>
          <div className="overflow-x-auto"><table className="w-full text-left text-sm"><caption className="sr-only">Example check for Version {version.number}</caption><thead><tr>{T.fewShotHead.map((label) => <th key={label} scope="col" className="p-2 font-medium">{label}</th>)}</tr></thead><tbody>{rows.map(({ key, cells }) => <tr key={key} className="border-t">{cells.map((cell, column) => column === 0 ? <th key={T.fewShotHead[column]} scope="row" className="p-2 font-normal">{cell}</th> : <td key={T.fewShotHead[column]} className="p-2">{cell}</td>)}</tr>)}</tbody></table></div>
          <p className="text-xs text-muted-foreground">{fewShotNote(table)}</p>
        </> : <p className="text-sm text-muted-foreground">{T.fewShotNone}</p>)}
    </CardContent>
  </Card>;
}

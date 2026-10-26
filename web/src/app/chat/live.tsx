"use client";

import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, unwrap } from "@/lib/api";
import { awaitJob } from "@/lib/job-result";
import { CHAT as T } from "./copy";
import { liveLabel, liveScore, versionName, type LiveCheckpoint } from "./logic";

/** Seconds since the epoch, refreshed every `ms`, so "saved 3 min ago" keeps counting between reads. */
export function useNowSeconds(ms = 20_000): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now() / 1000), ms);
    return () => clearInterval(timer);
  }, [ms]);
  return now;
}

/** Versions whose first run is still training, with the newest checkpoint each has saved: chat with it, or score it now. */
export function LiveCheckpoints({ live, answering, busy, onPick }: { live: LiveCheckpoint[]; answering: number | null; busy: boolean; onPick: (version: number) => void }) {
  const now = useNowSeconds();
  return (
    <Card aria-label={T.liveTitle}>
      <CardHeader>
        <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">{T.liveTitle}</CardTitle>
        <p className="text-sm text-muted-foreground">{T.liveHint}</p>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {live.map((l) => <LiveRow key={l.version} live={l} now={now} chatting={answering === l.version} busy={busy} onPick={onPick} />)}
      </CardContent>
    </Card>
  );
}

function LiveRow({ live, now, chatting, busy, onPick }: { live: LiveCheckpoint; now: number; chatting: boolean; busy: boolean; onPick: (version: number) => void }) {
  const score = useMutation({
    mutationFn: async (signal?: AbortSignal) => {
      const initial = await unwrap(api.POST("/versions/{number}/score-live", { params: { path: { number: live.version } } }));
      return liveScore(await awaitJob<unknown>(initial, signal ?? new AbortController().signal));
    },
  });
  const result = score.data;
  return (
    <div className="flex flex-col gap-2 rounded-lg border p-3" aria-label={`Live checkpoint for ${versionName({ number: live.version })}`}>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className="w-24 font-semibold">{versionName({ number: live.version })}</span>
        <Badge variant="outline">still training</Badge>
        <span className="flex-1 text-muted-foreground">{liveLabel(live, now)}</span>
        <Button variant="outline" size="sm" aria-pressed={chatting} disabled={busy || chatting} onClick={() => onPick(live.version)}>{chatting ? "answering now" : T.liveChat}</Button>
        <Button variant="outline" size="sm" disabled={score.isPending} onClick={() => score.mutate(undefined)}>{T.liveScoreNow}</Button>
      </div>
      {score.isPending && <p role="status" className="text-sm text-muted-foreground">{T.liveScoring}</p>}
      {score.isError && <p role="alert" className="text-sm">{T.liveScoreFailed}: {score.error.message}</p>}
      {result && !score.isPending && (
        <p className="text-sm" aria-label={`Score for ${versionName({ number: live.version })}`}>
          {T.liveScored(result.bpc.toFixed(2), result.step === null ? T.liveStepUnknown : `step ${result.step.toLocaleString()}`, result.documents)}
        </p>
      )}
    </div>
  );
}

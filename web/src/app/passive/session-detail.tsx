"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { SeriesChart } from "@/components/series-chart";
import { api, unwrap } from "@/lib/api";
import { problem } from "@/lib/cluster";
import { PASSIVE } from "./copy";
import { Lessons } from "./lessons-list";
import { errorSeries, eventLine, facts, isTrouble, mergeEvents, STOP_REASONS, type JournalEvent } from "./detail";
import { ACTIVE, when } from "./session";

const PAGE = 50;
const REFRESH_MS = 5_000;

/** One session opened: phase, round, best error, the test error of each checkpoint, and the trainer's journal (newest first). */
export function SessionDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const detail = useQuery({
    queryKey: ["passive", "detail", id],
    queryFn: () => unwrap(api.GET("/passive/{session_id}", { params: { path: { session_id: id }, query: { limit: PAGE } } })),
    refetchInterval: (query) => (query.state.data && !ACTIVE.includes(query.state.data.session.status) ? false : REFRESH_MS),
  });
  const [older, setOlder] = useState<JournalEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [failure, setFailure] = useState("");

  const events = mergeEvents(older, detail.data?.journal.events ?? []);
  const first = events[0]?.index ?? 0;
  const loadEarlier = async () => {
    setLoading(true);
    setFailure("");
    try {
      const page = await unwrap(api.GET("/passive/{session_id}", { params: { path: { session_id: id }, query: { limit: PAGE, before: first } } }));
      setOlder((now) => mergeEvents(now, page.journal.events));
    } catch (error) {
      setFailure(problem(error));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card data-testid="passive-detail">
      <CardHeader>
        <CardTitle>{PASSIVE.detailTitle}</CardTitle>
        <CardDescription>{detail.data ? PASSIVE.started(PASSIVE.status[detail.data.session.status] ?? detail.data.session.status, when(detail.data.session.created)) : id}</CardDescription>
        <div className="pt-2">
          <Button variant="outline" size="sm" onClick={onClose}>
            {PASSIVE.detailClose}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        {detail.isError ? (
          <p role="alert" className="text-sm text-destructive">
            {PASSIVE.detailFailed} {problem(detail.error)}
          </p>
        ) : !detail.data ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <>
            <dl className="grid grid-cols-1 gap-x-8 gap-y-2 text-sm sm:grid-cols-2" data-testid="passive-facts">
              {facts(detail.data.progress).map((fact) => (
                <div key={fact.label} className="flex justify-between gap-4 border-b py-1">
                  <dt className="text-muted-foreground">{fact.label}</dt>
                  <dd className="font-medium">{fact.value}</dd>
                </div>
              ))}
            </dl>
            {detail.data.progress.stop_reason && (
              <p className="text-sm">{PASSIVE.endedBecause(STOP_REASONS[detail.data.progress.stop_reason] ?? detail.data.progress.stop_reason)}</p>
            )}
            {detail.data.session.error ? (
              <p role="alert" className="text-sm break-words text-destructive">
                {String(detail.data.session.error)}
              </p>
            ) : null}

            <section className="flex flex-col gap-2" aria-label={PASSIVE.curveTitle}>
              <h3 className="text-sm font-medium">{PASSIVE.curveTitle}</h3>
              {detail.data.progress.checkpoints.length === 0 ? (
                <p className="text-sm text-muted-foreground">{PASSIVE.curveNone}</p>
              ) : (
                <div data-testid="error-curve">
                  <SeriesChart series={errorSeries(detail.data.progress)} xLabel={PASSIVE.curveX} yLabel={PASSIVE.curveY} className="h-48 w-full" />
                  <p className="text-xs text-muted-foreground">{PASSIVE.curveHint}</p>
                </div>
              )}
            </section>

            <section className="flex flex-col gap-1" aria-label={PASSIVE.planTitle}>
              <h3 className="text-sm font-medium">{PASSIVE.planTitle}</h3>
              {Object.values(detail.data.plan).map((line) => (
                <p key={line} className="text-sm text-muted-foreground">
                  {line}
                </p>
              ))}
            </section>

            <Lessons id={id} />

            <section className="flex flex-col gap-2" aria-label={PASSIVE.journalTitle}>
              <h3 className="text-sm font-medium">{PASSIVE.journalTitle}</h3>
              {events.length === 0 ? (
                <p className="text-sm text-muted-foreground">{PASSIVE.journalNone}</p>
              ) : (
                <ol className="flex flex-col gap-1" data-testid="passive-journal">
                  {[...events].reverse().map((event) => (
                    <li key={event.index} data-type={event.type} className={`flex gap-3 text-sm ${isTrouble(event) ? "text-warning-soft" : ""}`}>
                      <span className="w-28 shrink-0 text-xs text-muted-foreground">{when(event.at)}</span>
                      <span className="min-w-0 break-words">{eventLine(event)}</span>
                    </li>
                  ))}
                </ol>
              )}
              {first > 0 && (
                <div className="flex items-center gap-3">
                  <Button variant="outline" size="sm" disabled={loading} onClick={() => void loadEarlier()}>
                    {PASSIVE.journalEarlier}
                  </Button>
                  {failure && (
                    <span role="alert" className="text-sm text-destructive">
                      {failure}
                    </span>
                  )}
                </div>
              )}
            </section>
          </>
        )}
      </CardContent>
    </Card>
  );
}

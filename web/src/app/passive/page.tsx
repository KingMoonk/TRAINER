"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, subscribe, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { PASSIVE } from "./copy";
import { competes, sessionView, type SessionRow } from "./session";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

const route = routeFor("passive");
const REFRESH_MS = 5_000;

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function Page() {
  const client = useQueryClient();
  const passive = useQuery({
    queryKey: ["passive"],
    queryFn: () => unwrap(api.GET("/passive")),
    refetchInterval: REFRESH_MS,
  });
  const queue = useQuery({
    queryKey: ["queue"],
    queryFn: () => unwrap(api.GET("/queue")),
    refetchInterval: REFRESH_MS,
  });

  // The passive methods emit nothing, so the page polls; a queue event also says the computer's load changed.
  useEffect(
    () =>
      subscribe((event) => {
        if (event.kind === "queue") {
          void client.invalidateQueries({ queryKey: ["queue"] });
          void client.invalidateQueries({ queryKey: ["passive"] });
        }
      }),
    [client],
  );

  const [confirming, setConfirming] = useState<string | null>(null);
  const [note, setNote] = useState("");

  const stop = useMutation({
    mutationFn: (id: string) => unwrap(api.POST("/passive/{session_id}/stop", { params: { path: { session_id: id } } })),
    onSuccess: ({ status }) => {
      setConfirming(null);
      setNote(status === "stopped" ? PASSIVE.stopped : PASSIVE.stopping);
    },
    onError: (error) => toast.error(`${PASSIVE.actionFailed}: ${message(error)}`),
    onSettled: () => client.invalidateQueries({ queryKey: ["passive"] }),
  });
  const command = useMutation({
    mutationFn: ({ id, action }: { id: string; action: "pause" | "resume" }) =>
      unwrap(api.POST(action === "pause" ? "/passive/{session_id}/pause" : "/passive/{session_id}/resume", { params: { path: { session_id: id } } })),
    onSuccess: (_, { action }) => setNote(action === "pause" ? PASSIVE.pausing : PASSIVE.resuming),
    onError: (error) => toast.error(`${PASSIVE.actionFailed}: ${message(error)}`),
    onSettled: () => client.invalidateQueries({ queryKey: ["passive"] }),
  });
  const keepGoing = useMutation({
    mutationFn: (id: string) => unwrap(api.POST("/passive/{session_id}/keep-going", { params: { path: { session_id: id } } })),
    onSuccess: () => setNote(PASSIVE.keptGoing),
    onError: (error) => toast.error(`${PASSIVE.actionFailed}: ${message(error)}`),
    onSettled: () => client.invalidateQueries({ queryKey: ["passive"] }),
  });
  const signOut = useMutation({
    mutationFn: () => unwrap(api.DELETE("/passive/signin")),
    onSuccess: () => setNote(PASSIVE.signinRemoved),
    onError: (error) => toast.error(`${PASSIVE.actionFailed}: ${message(error)}`),
    onSettled: () => client.invalidateQueries({ queryKey: ["passive"] }),
  });

  const sessions: SessionRow[] = passive.data?.sessions ?? [];
  const warning = competes(sessions, queue.data);

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={route.title} lead={PASSIVE.lead} />

      {warning && (
        <p role="alert" data-testid="passive-warning" className="rounded-md border border-warning/40 bg-warning/10 p-3 text-sm text-warning-soft">
          {warning}
        </p>
      )}

      {passive.isError ? (
        <p role="alert" data-testid="passive-error" className="text-sm text-destructive">
          {PASSIVE.loadFailed} {message(passive.error)}
        </p>
      ) : !passive.data ? (
        <div className="flex flex-col gap-4" aria-label="Loading">
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{PASSIVE.kicker}</CardTitle>
              <CardDescription>{PASSIVE.desc}</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-2">
              {sessions.length === 0 && <p className="text-sm text-muted-foreground">{PASSIVE.none}</p>}
              {sessions.map((row) => {
                const view = sessionView(row);
                return (
                  <div key={row.id} data-testid={`session-${row.id}`} className="flex items-start justify-between gap-4 rounded-md border p-3">
                    <div className="flex min-w-0 flex-col gap-1">
                      <span className="font-medium">{view.headline}</span>
                      {view.lines.map((line, i) => (
                        <span key={i} className="text-sm break-words text-muted-foreground">
                          {line}
                        </span>
                      ))}
                    </div>
                    {(view.canPause || view.canResume) && (
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={command.isPending}
                        onClick={() => command.mutate({ id: row.id, action: view.canPause ? "pause" : "resume" })}
                      >
                        {view.canPause ? PASSIVE.pause : PASSIVE.resume}
                      </Button>
                    )}
                    {view.canKeepGoing && (
                      <Button size="sm" title={PASSIVE.keepGoingHint} disabled={keepGoing.isPending} onClick={() => keepGoing.mutate(row.id)}>
                        {PASSIVE.keepGoing}
                      </Button>
                    )}
                    {view.canStop &&
                      (confirming === row.id ? (
                        <div className="flex max-w-xs shrink-0 flex-col items-end gap-2 text-right">
                          <p className="text-sm">{PASSIVE.stopConfirm}</p>
                          <div className="flex gap-2">
                            <Button variant="outline" size="sm" onClick={() => setConfirming(null)}>
                              {PASSIVE.cancel}
                            </Button>
                            <Button variant="destructive" size="sm" disabled={stop.isPending} onClick={() => stop.mutate(row.id)}>
                              {PASSIVE.stopConfirmYes}
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <Button variant="outline" size="sm" onClick={() => setConfirming(row.id)}>
                          {PASSIVE.stop}
                        </Button>
                      ))}
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{PASSIVE.signinKicker}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-start gap-3">
              <p className="text-sm text-muted-foreground">{passive.data.signin ? PASSIVE.signinOn : PASSIVE.signinOff}</p>
              {passive.data.signin && (
                <Button variant="outline" size="sm" disabled={signOut.isPending} onClick={() => signOut.mutate()}>
                  {PASSIVE.signinRemove}
                </Button>
              )}
            </CardContent>
          </Card>

          {note && (
            <p role="status" data-testid="passive-note" className="text-sm text-muted-foreground">
              {note}
            </p>
          )}
        </>
      )}
    </main>
  );
}

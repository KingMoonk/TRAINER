"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { cn } from "@/lib/utils";
import { errorText, type JobOut } from "./logic";
import {
  LOOP_DEFAULTS,
  NO_JUDGE,
  STATUS_WORDS,
  elapsed,
  headline,
  loopFor,
  pairsLine,
  roundEvidence,
  startRequest,
  type LoopForm,
  type LoopRound,
} from "./loop-logic";
import { useJob } from "./use-events";
import { useLive } from "@/lib/live";

const LOOPS_KEY = ["post-training", "loops"];

function Checks({ round }: { round: LoopRound }) {
  const audit = round.audit as { checks?: { name: string; detail: string; ok: boolean | null }[] } | null;
  const checks = [...(round.checks ?? []), ...(round.audit_flag ? [] : (audit?.checks ?? []).map((c) => ({ ...c, name: `Audit half: ${c.name.toLowerCase()}` })))];
  return (
    <ul className="flex flex-col gap-1">
      {checks.map((c) => (
        <li key={c.name} className="flex items-start gap-2 text-xs">
          <span aria-hidden className={cn("w-4 text-center font-bold", c.ok == null ? "text-muted-foreground" : c.ok ? "text-success" : "text-destructive")}>
            {c.ok == null ? "…" : c.ok ? "✓" : "✕"}
          </span>
          <span>
            {c.name}: <span className="text-muted-foreground">{c.detail}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

/** The automated loop: start it, stop it, carry it on, and read what every round did and why it was kept or rolled back. */
export function LoopCard({ version, hasCheckpoint }: { version: number | undefined; hasCheckpoint: boolean }) {
  const queryClient = useQueryClient();
  const loops = useQuery({
    queryKey: LOOPS_KEY,
    queryFn: ({ signal }) => unwrap(api.GET("/post-training/loops", { signal })),
  });
  // a loop's rounds arrive as `train` events of type loop_*, at most one re-read a second
  useLive(LOOPS_KEY, ["queue", "train"], { match: (kind, payload) => kind !== "train" || String((payload as { type?: unknown } | null)?.type ?? "").startsWith("loop_"), throttleMs: 1_000 });
  const judges = useQuery({ queryKey: ["post-training", "judges"], queryFn: ({ signal }) => unwrap(api.GET("/post-training/judges", { signal })), retry: false });
  const [form, setForm] = useState<LoopForm>(LOOP_DEFAULTS);
  const set = (patch: Partial<LoopForm>) => setForm((f) => ({ ...f, ...patch }));

  const loop = loopFor(loops.data, version);
  const active = !!loop?.active;
  const start = useMutation({
    mutationFn: (resume: string | null) => {
      const request = startRequest(form, version ?? null, resume);
      if ("error" in request) throw new Error(request.error);
      return unwrap(api.POST("/post-training/loops", { body: request.body }));
    },
    onSuccess: (queued) => {
      toast.success(`Added to the queue: ${queued.label}`);
      queryClient.invalidateQueries({ queryKey: LOOPS_KEY });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  const stop = useMutation({
    mutationFn: (id: string) => unwrap(api.POST("/post-training/loops/{loop_id}/stop", { params: { path: { loop_id: id } } })),
    onSuccess: () => {
      toast.success("Stopping. It keeps its last kept round; Continue picks up from there.");
      queryClient.invalidateQueries({ queryKey: LOOPS_KEY });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  const adoptRequest = useCallback(
    () => unwrap(api.POST("/post-training/loops/{loop_id}/adopt", { params: { path: { loop_id: loop?.id ?? "" } } })) as Promise<JobOut>,
    [loop?.id],
  );
  const adopt = useJob(adoptRequest, (job) => {
    if (job.status === "failed") return void toast.error(job.error);
    const result = job.result as { ok?: boolean; verdict?: string } | null;
    (result?.ok ? toast.success : toast.error)(result?.verdict ?? "Done.");
    queryClient.invalidateQueries({ queryKey: ["versions"] });
  });
  const adopting = useMutation({ mutationFn: adopt.start, onError: (error) => toast.error(errorText(error)) });

  const [opened, setOpened] = useState<{ loop: string; round: number } | null>(null);
  const open = opened && opened.loop === loop?.id ? opened.round : null;
  const setOpen = (round: number | null) => setOpened(round === null || !loop ? null : { loop: loop.id, round });
  const rounds = [...(loop?.rounds ?? [])].reverse();
  const canAdopt = !!loop && !active && loop.kept > 0 && loop.current !== loop.anchor;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Automated loop</CardTitle>
        <CardDescription>
          Round after round: the model answers, a rule or a judge picks the better answers, a preference round trains on them, and held-back data the judge never
          sees decides whether the result is kept or rolled back. It runs as one long job and carries on after a restart.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        {loops.isPending ? (
          <Skeleton className="h-20 w-full" aria-label="Reading the loops" />
        ) : (
          <>
            {loop && (
              <div className="flex flex-col gap-1" data-testid="loop-status">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={active ? "default" : "secondary"}>{active ? (loop.status === "waiting_judge" ? "waiting" : "running") : loop.status}</Badge>
                  <span className="text-sm">{headline(loop, active)}</span>
                </div>
                {loop.stop_reason && <p className="text-sm text-muted-foreground">{loop.stop_reason}</p>}
                {loop.note && <p className="text-xs text-muted-foreground">{loop.note}</p>}
                <p className="text-xs text-muted-foreground">Working time so far: {elapsed(loop.elapsed_s)}.</p>
              </div>
            )}
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <label className="flex flex-col gap-1 text-sm">
                {loop && !active ? "More rounds" : "Rounds at most"}
                <Input inputMode="numeric" value={form.rounds} onChange={(e) => set({ rounds: e.target.value })} disabled={active} />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                Stop after this many rounds in a row not kept
                <Input inputMode="numeric" value={form.patience} onChange={(e) => set({ patience: e.target.value })} disabled={active} />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                Hours at most (optional)
                <Input inputMode="decimal" value={form.hours} onChange={(e) => set({ hours: e.target.value })} disabled={active} placeholder="no limit" />
              </label>
              <div className="flex flex-col gap-1 text-sm">
                Judge
                <Select value={form.judge} onValueChange={(v) => set({ judge: v })} disabled={active}>
                  <SelectTrigger aria-label="Judge model" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={NO_JUDGE}>{judges.data?.selected ? `Workspace teacher (${judges.data.selected})` : "None: checked problems only"}</SelectItem>
                    {(judges.data?.models ?? []).map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            {judges.data?.error && <p className="text-xs text-muted-foreground">Ollama could not be read, so only the workspace&apos;s teacher can judge: {judges.data.error}</p>}
            <div className="flex flex-wrap gap-2">
              {active ? (
                <Button variant="outline" onClick={() => loop && stop.mutate(loop.id)} disabled={stop.isPending}>
                  Stop the loop
                </Button>
              ) : (
                <>
                  <Button onClick={() => start.mutate(null)} disabled={start.isPending || !hasCheckpoint}>
                    {loop ? "Start a new loop" : "Start the loop"}
                  </Button>
                  {loop && (
                    <Button variant="outline" onClick={() => start.mutate(loop.id)} disabled={start.isPending}>
                      Continue
                    </Button>
                  )}
                </>
              )}
              {canAdopt && (
                <Button variant="outline" onClick={() => adopting.mutate()} disabled={adopting.isPending || adopt.pending}>
                  {adopting.isPending || adopt.pending ? "Checking…" : "Use its best as this version"}
                </Button>
              )}
            </div>
            {!hasCheckpoint && <p className="text-sm text-muted-foreground">Train this version in Pretraining first.</p>}
            {loop && (
              <div className="flex flex-col gap-2">
                <h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Rounds</h3>
                {rounds.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No round has finished yet.</p>
                ) : (
                  <ul className="flex flex-col gap-2" aria-label="Loop rounds">
                    {rounds.map((r) => (
                      <li key={r.round} data-testid={`loop-round-${r.round}`} className="rounded-lg border p-3">
                        <button className="flex w-full flex-wrap items-center gap-2 text-left text-sm" onClick={() => setOpen(open === r.round ? null : r.round)} aria-expanded={open === r.round}>
                          <span className="font-medium">Round {r.round}</span>
                          <Badge variant={r.status === "kept" ? "default" : "outline"}>{STATUS_WORDS[r.status] ?? r.status}</Badge>
                          {r.audit_flag && <Badge variant="outline">audit flag</Badge>}
                          <span className="text-muted-foreground">{roundEvidence(r)}</span>
                        </button>
                        <p className="mt-1 text-xs text-muted-foreground">{r.reason}</p>
                        {open === r.round && (
                          <div className="mt-2 flex flex-col gap-2">
                            <p className="text-xs">{pairsLine(r)}</p>
                            <Checks round={r} />
                            <p className="text-xs text-muted-foreground">
                              {r.seconds ? `Took ${elapsed(r.seconds)}.` : ""} {r.node ? `Collected on ${r.node}.` : ""}
                            </p>
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            {!loop && <p className="text-sm text-muted-foreground">No loop has run for this version yet.</p>}
          </>
        )}
      </CardContent>
    </Card>
  );
}

"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { api, subscribe, unwrap, type ApiEvent } from "@/lib/api";
import { T } from "./copy";
import { awayJobs, errorText, type Job, type Plan, type Queue } from "./logic";

const QUEUE = ["queue"] as const;
const JOBS = ["queue-jobs"] as const;
const TICK_MS = 500;

/** Call `handler` for every event on the stream. The shell holds its own stream; this one is the page's, for `queue` and `train`. */
function useEventStream(handler: (event: ApiEvent) => void) {
  const latest = useRef(handler);
  useEffect(() => {
    latest.current = handler;
  });
  useEffect(() => subscribe((event) => latest.current(event)), []);
}

/**
 * The queue state, read once and then kept current from `/events`: a `queue` event (a job added, moved, held or removed, the
 * plan or pause changed) re-reads it at once, and `train` events (several a second) at most every half second, so the running
 * job's numbers move in place without the page redrawing for each step.
 */
export function useQueue() {
  const client = useQueryClient();
  const query = useQuery({ queryKey: QUEUE, queryFn: ({ signal }) => unwrap(api.GET("/queue", { signal })) });
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  useEventStream((event) => {
    if (event.kind === "queue") {
      client.invalidateQueries({ queryKey: QUEUE });
      client.invalidateQueries({ queryKey: ["queue-outlook"] });
    } else if (event.kind === "train" && timer.current === undefined) {
      timer.current = setTimeout(() => {
        timer.current = undefined;
        client.invalidateQueries({ queryKey: QUEUE });
      }, TICK_MS);
    }
  });
  return query;
}

/**
 * The jobs running on other machines, from GET /queue/jobs (the queue state has only this machine's own). A `cluster` event (a
 * peer took a job, or said its checkpoint or result could not be stored) and a `queue` event re-read them.
 */
export function useAwayJobs() {
  const client = useQueryClient();
  const query = useQuery({ queryKey: JOBS, queryFn: ({ signal }) => unwrap(api.GET("/queue/jobs", { signal })) });
  useEventStream((event) => {
    if (event.kind === "queue" || event.kind === "cluster") client.invalidateQueries({ queryKey: JOBS });
  });
  return { ...query, away: awayJobs((query.data ?? []) as Job[]) };
}

/** When a waiting job starts, from GET /queue/jobs/{id}/outlook. Empty once it has left the queue. */
export function useJobOutlook(jobId: string) {
  return useQuery({
    queryKey: ["queue-outlook", jobId],
    queryFn: () => unwrap(api.GET("/queue/jobs/{job_id}/outlook", { params: { path: { job_id: jobId } } })),
  });
}

/** Every control on the page. A reply that carries the new queue state replaces it at once; a problem is a toast. */
export function useQueueActions() {
  const client = useQueryClient();
  const keep = (state: Queue) => client.setQueryData(QUEUE, state);
  const failed = (error: unknown) => {
    toast.error(errorText(error));
    client.invalidateQueries({ queryKey: QUEUE });
  };
  const job = (job_id: string) => ({ params: { path: { job_id } } });

  const pause = useMutation({ mutationFn: () => unwrap(api.POST("/queue/pause")), onSuccess: keep, onError: failed });
  const resume = useMutation({ mutationFn: () => unwrap(api.POST("/queue/resume")), onSuccess: keep, onError: failed });
  const start = useMutation({ mutationFn: () => unwrap(api.POST("/queue/start")), onSuccess: keep, onError: failed });
  const stopAfterCooldown = useMutation({
    mutationFn: () => unwrap(api.POST("/queue/stop_after_cooldown")),
    onSuccess: keep,
    onError: failed,
  });
  const checkpointNow = useMutation({
    mutationFn: () => unwrap(api.POST("/queue/checkpoint_now")),
    onSuccess: (r) => (r.started ? toast.success(T.checkpointStarted) : toast.warning(T.checkpointNothing)),
    onError: failed,
  });
  const remove = useMutation({
    mutationFn: (id: string) => unwrap(api.DELETE("/queue/jobs/{job_id}", job(id))),
    onSuccess: keep,
    onError: failed,
  });
  const hold = useMutation({
    mutationFn: (v: { id: string; held: boolean }) => unwrap(api.POST("/queue/jobs/{job_id}/hold", { ...job(v.id), body: { held: v.held } })),
    onSuccess: keep,
    onError: failed,
  });
  const move = useMutation({
    mutationFn: (v: { id: string; index: number }) => unwrap(api.POST("/queue/jobs/{job_id}/move", { ...job(v.id), body: { index: v.index } })),
    onSuccess: keep,
    onError: failed,
  });
  const savePlan = useMutation({
    scope: { id: "queue-plan" },
    // Each edit carries only its changed fields. Read the last saved plan when
    // this edit reaches the front of the queue, not from its render's closure.
    mutationFn: async (patch: Partial<Plan>) => {
      await client.cancelQueries({ queryKey: QUEUE });
      const plan = await unwrap(api.GET("/queue/plan"));
      return unwrap(api.PUT("/queue/plan", { body: { ...plan, ...patch } }));
    },
    // Show the choice at once; the reply (or the next queue event) is what the server kept.
    onMutate: (patch) => {
      const before = client.getQueryData<Queue>(QUEUE);
      if (before) client.setQueryData(QUEUE, { ...before, plan: { ...before.plan, ...patch } });
      return { before };
    },
    onSuccess: (plan) => client.setQueryData<Queue>(QUEUE, (q) => (q ? { ...q, plan } : q)),
    onError: (error, _patch, context) => {
      if (context?.before) client.setQueryData(QUEUE, context.before);
      failed(error);
    },
  });
  return { pause, resume, start, stopAfterCooldown, checkpointNow, remove, hold, move, savePlan };
}

export type QueueActions = ReturnType<typeof useQueueActions>;

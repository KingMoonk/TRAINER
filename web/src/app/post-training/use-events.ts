"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, subscribe, unwrap, type ApiEvent } from "@/lib/api";
import type { JobOut } from "./logic";

/** Call `handler` for every event on the stream. The shell holds its own stream; this one is the page's, for the kinds
 *  the shell ignores (job, queue, versions, train). The browser reconnects an EventSource on its own. */
export function useEventStream(handler: (event: ApiEvent) => void) {
  const latest = useRef(handler);
  useEffect(() => {
    latest.current = handler;
  });
  useEffect(() => subscribe((event) => latest.current(event)), []);
}

export const FINISHED = ["done", "failed", "cancelled"];

/**
 * A job route (POST returns an id; progress and the result arrive as `job` events). `start` posts it; `job` is its latest
 * record. The record is also read once from GET /jobs/{id}, since the result may have arrived before the page listened.
 */
export function useJob(startRequest: () => Promise<JobOut>, onFinished?: (job: JobOut) => void) {
  const finished = useRef(onFinished);
  useEffect(() => {
    finished.current = onFinished;
  });
  const [job, setJob] = useState<JobOut | null>(null);
  const current = useRef<string | null>(null);
  const seen = useRef(new Map<string, JobOut>());
  const announced = useRef(new Set<string>());

  const take = useCallback((next: JobOut) => {
    seen.current.set(next.id, next);
    if (seen.current.size > 50) seen.current.delete(seen.current.keys().next().value as string);
    if (next.id !== current.current) return;
    setJob((prev) => (prev && prev.id === next.id && FINISHED.includes(prev.status) ? prev : next));
    if (FINISHED.includes(next.status) && !announced.current.has(next.id)) {
      announced.current.add(next.id);
      finished.current?.(next);
    }
  }, []);

  useEventStream((event) => {
    if (event.kind === "job") take(event.payload as JobOut);
  });

  const start = useCallback(async () => {
    const started = await startRequest();
    current.current = started.id;
    take(seen.current.get(started.id) ?? started);
    try {
      take((await unwrap(api.GET("/jobs/{job_id}", { params: { path: { job_id: started.id } } }))) as JobOut);
    } catch {
      // the events will still bring it
    }
    return started;
  }, [startRequest, take]);

  return { job, start, pending: job !== null && !FINISHED.includes(job.status) };
}

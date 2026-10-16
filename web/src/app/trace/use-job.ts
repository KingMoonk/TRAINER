"use client";

import { fetchJson, subscribe } from "@/lib/api";

/** A `job` event (studio_api/jobs.py JobRecord). */
export type Job = { id: string; kind: string; status: string; progress: string; fraction: number | null; result: unknown; error: string };

/** A job that ended failed or cancelled. `message` is the server's "ErrorType: text". */
export class JobError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "JobError";
  }
}

const POLL_MS = 1_000;

/**
 * Start a job with `start` (a POST that answers with its id) and resolve with its result. The result arrives as a `job` event
 * on GET /events. The stream is opened before the POST so no event is missed; GET /jobs/{id} is read once in case the job
 * finished first, and then every second only while the stream is down.
 */
export function runJob<T>(start: () => Promise<{ id: string }>, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const seen = new Map<string, Job>();
    let id: string | null = null;
    let settled = false;
    let poll: ReturnType<typeof setInterval> | undefined;
    let close = () => {};

    const end = (then: () => void) => {
      if (settled) return;
      settled = true;
      close();
      clearInterval(poll);
      signal?.removeEventListener("abort", onAbort);
      then();
    };
    const onAbort = () => end(() => reject(new DOMException("aborted", "AbortError")));
    const take = (job: Job | undefined) => {
      if (!job || job.id !== id) return;
      if (job.status === "done") end(() => resolve(job.result as T));
      else if (job.status === "failed") end(() => reject(new JobError(job.error)));
      else if (job.status === "cancelled") end(() => reject(new JobError("cancelled")));
    };
    const read = () => {
      if (id) fetchJson<Job>(`/jobs/${id}`).then(take, () => {});
    };

    if (signal?.aborted) return onAbort();
    signal?.addEventListener("abort", onAbort);
    close = subscribe(
      (event) => {
        if (event.kind !== "job") return;
        const job = event.payload as Job;
        seen.set(job.id, job);
        take(job);
      },
      () => {
        if (!poll && !settled) poll = setInterval(read, POLL_MS);
      },
    );
    start().then(
      (job) => {
        id = job.id;
        take(job as Job);
        take(seen.get(job.id));
        read();
      },
      (error) => end(() => reject(error)),
    );
  });
}

/** The sentence for a failed read: the engine's own message for a ValueError, else the page's "numbers appear" line. */
export function reason(error: unknown, noNumbers: string, serverError: string): string {
  if (error instanceof JobError) return /^ValueError: ([\s\S]+)$/.exec(error.message)?.[1] ?? noNumbers;
  const status = (error as { status?: number } | null)?.status;
  return status ? `${serverError} (${status}).` : noNumbers;
}

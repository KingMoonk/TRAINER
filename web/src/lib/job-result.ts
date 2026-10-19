import { api, ApiError, subscribe, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";

export type JobOut = components["schemas"]["studio_api__models_work__JobOut"];

const POLL_MS = 1_000;
const MAX_READ_FAILURES = 3;
const STATUSES = new Set(["queued", "running", "done", "failed", "cancelled"]);

export class JobFailedError extends Error {
  constructor(readonly jobId: string, message: string) {
    super(message);
    this.name = "JobFailedError";
  }
}

export class JobCancelledError extends Error {
  constructor(readonly jobId: string) {
    super("The job was cancelled.");
    this.name = "JobCancelledError";
  }
}

function abortError() {
  return new DOMException("The job wait was aborted.", "AbortError");
}

/** Wait one poll interval, or less when `onWake` is given a function that ends the wait early. */
function pause(signal: AbortSignal, onWake?: (wake: () => void) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(abortError()); return; }
    const abort = () => { clearTimeout(timer); reject(abortError()); };
    const done = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      resolve();
    };
    const timer = setTimeout(done, POLL_MS);
    signal.addEventListener("abort", abort, { once: true });
    onWake?.(done);
  });
}

/** GET /jobs has an untyped result in OpenAPI. Validate the envelope before following it. */
function jobRecord(value: unknown, id: string): JobOut {
  if (!value || typeof value !== "object" || !("id" in value) || value.id !== id ||
      !("status" in value) || typeof value.status !== "string" || !STATUSES.has(value.status)) {
    throw new Error("The server returned an invalid job record.");
  }
  return value as JobOut;
}

/**
 * Follow a submitted job, including jobs that finish before the first poll or whose SSE event was missed.
 * T describes the route's documented result; consumers should validate unknown result shapes when needed.
 * With `events`, the job's "job" events on /events end the wait as they arrive; polling stays as the fallback for a dropped stream.
 * Abort rejects promptly with AbortError and requests server cancellation (best effort if disconnected).
 * Transient read failures retry without resubmitting work; a missing job or repeated failure is reported.
 */
export async function awaitJob<T>(initial: JobOut, signal: AbortSignal, options: { events?: boolean } = {}): Promise<T> {
  let job = jobRecord(initial, initial.id);
  let failures = 0;
  const cancel = () => {
    if (job.status === "done" || job.status === "failed" || job.status === "cancelled") return;
    void api.DELETE("/jobs/{job_id}", {
      params: { path: { job_id: initial.id } }, signal: AbortSignal.timeout(5_000),
    }).catch(() => { /* An unavailable server cannot acknowledge cancellation. */ });
  };
  let pushed: JobOut | undefined;
  let wake = () => {};
  const closeEvents = options.events ? subscribe((event) => {
    if (event.kind !== "job") return;
    try { pushed = jobRecord(event.payload, initial.id); } catch { return; } // Another job's event, or one that is not a job record.
    wake();
  }) : () => {};
  signal.addEventListener("abort", cancel, { once: true });
  try {
    if (signal.aborted) { cancel(); throw abortError(); }
    for (;;) {
      if (signal.aborted) throw abortError();
      if (pushed) [job, pushed] = [pushed, undefined];
      if (job.status === "done") return job.result as T;
      if (job.status === "failed") throw new JobFailedError(job.id, job.error || "The job failed.");
      if (job.status === "cancelled") throw new JobCancelledError(job.id);
      let value: unknown;
      try {
        value = await unwrap(api.GET("/jobs/{job_id}", { params: { path: { job_id: initial.id } }, signal }));
        failures = 0;
      } catch (error) {
        if (signal.aborted) throw abortError();
        if ((error instanceof ApiError && error.status < 500 && error.status !== 429) || ++failures >= MAX_READ_FAILURES) throw error;
        await pause(signal);
        continue;
      }
      if (signal.aborted) throw abortError();
      job = jobRecord(value, initial.id);
      if (!pushed && (job.status === "queued" || job.status === "running")) {
        await pause(signal, (end) => { wake = end; });
        wake = () => {};
      }
    }
  } finally {
    closeEvents();
    signal.removeEventListener("abort", cancel);
  }
}

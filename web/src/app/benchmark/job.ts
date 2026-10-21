import { fetchJson } from "@/lib/api";
import { RESYNC, watchEvents } from "@/lib/events";
import type { components } from "@/lib/api-schema";

export type JobOut = components["schemas"]["studio_api__models_work__JobOut"];

const FINISHED = new Set(["done", "failed", "cancelled"]);
const BACKSTOP_MS = 15_000;

/** A job that ended without a result: failed (the server's message) or cancelled. */
export class JobFailed extends Error {}

/**
 * Wait for a job the server started to finish and return its result. The result arrives as a "job" event on /events; the
 * job is also read once from GET /jobs/{id} in case it finished before the stream was open, and again when the stream
 * reconnects, the tab is shown again and every few seconds (a lost event), so a dropped stream cannot leave the page waiting forever.
 */
export function awaitJob<T>(started: JobOut): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    let close = () => {};
    const settle = (job: JobOut) => {
      if (settled || !FINISHED.has(job.status)) return;
      settled = true;
      close();
      if (job.status === "done") resolve(job.result as T);
      else reject(new JobFailed(job.status === "cancelled" ? "The job was cancelled." : job.error || "The job failed."));
    };
    settle(started);
    if (settled) return;
    const read = () => fetchJson<JobOut>(`/jobs/${started.id}`).then(settle, () => {});
    close = watchEvents((event) => {
      if (event.kind === RESYNC) void read();
      const job = event.kind === "job" ? (event.payload as JobOut) : null;
      if (job?.id === started.id) settle(job);
    }, BACKSTOP_MS);
    void read();
  });
}

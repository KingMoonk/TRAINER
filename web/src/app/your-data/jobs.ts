import { api, unwrap } from "@/lib/api";
import { onApiEvent } from "./events";

/** The parts of a job record this code reads (studio_api.jobs.JobRecord). */
export type JobLike = { id: string; status: string; result?: unknown; error?: string };

const FINISHED = ["done", "failed", "cancelled"];
const POLL_MS = 2_000;

function settle<T>(job: JobLike): T {
  if (job.status === "done") return job.result as T;
  throw new Error(job.error || (job.status === "cancelled" ? "It was cancelled." : "It did not finish."));
}

/**
 * Start a long read (a POST that answers with a job) and resolve with the job's result, or reject with its error sentence.
 * The result arrives as a `job` event; GET /jobs/{id} backs that up for an event that came before the page listened, or never.
 * `signal` stops waiting (the job keeps running on the server).
 */
export async function runJob<T>(start: () => Promise<JobLike>, signal?: AbortSignal): Promise<T> {
  const started = await start();
  if (FINISHED.includes(started.status)) return settle<T>(started);
  return new Promise<T>((resolve, reject) => {
    let over = false;
    const timer: { id?: ReturnType<typeof setInterval> } = {};
    let off = () => {};
    const end = (job: JobLike | null, why?: Error) => {
      if (over) return;
      if (job && !FINISHED.includes(job.status)) return;
      over = true;
      off();
      clearInterval(timer.id);
      signal?.removeEventListener("abort", aborted);
      try {
        if (why) reject(why);
        else resolve(settle<T>(job as JobLike));
      } catch (error) {
        reject(error);
      }
    };
    const aborted = () => end(null, new DOMException("Aborted", "AbortError"));
    if (signal?.aborted) return aborted();
    signal?.addEventListener("abort", aborted);
    off = onApiEvent((event) => {
      if (event.kind === "job" && (event.payload as JobLike).id === started.id) end(event.payload as JobLike);
    });
    const poll = async () => {
      try {
        end((await unwrap(api.GET("/jobs/{job_id}", { params: { path: { job_id: started.id } } }))) as JobLike);
      } catch {
        // the events (or the next poll) will still bring it
      }
    };
    void poll();
    timer.id = setInterval(poll, POLL_MS);
  });
}

import { expect, test } from "@playwright/test";
import { api } from "../src/lib/api";
import { awaitJob, JobCancelledError, JobFailedError, type JobOut } from "../src/lib/job-result";

const record = (changes: Partial<JobOut> = {}): JobOut => ({
  id: "job-1", kind: "test", status: "queued", created_at: 0, progress: "", error: "", ...changes,
});
const originalGet = api.GET;
const originalDelete = api.DELETE;
const reply = (data: unknown) => ({ data, response: new Response(null, { status: 200 }) });
test.afterEach(() => { api.GET = originalGet; api.DELETE = originalDelete; });

test("returns an already completed result without polling", async () => {
  api.GET = (() => { throw new Error("Unexpected poll"); }) as typeof api.GET;
  await expect(awaitJob(record({ status: "done", result: { step: 0 } }), new AbortController().signal)).resolves.toEqual({ step: 0 });
});

test("reports failed and server-cancelled jobs distinctly", async () => {
  await expect(awaitJob(record({ status: "failed", error: "Checkpoint missing" }), new AbortController().signal)).rejects.toThrow(JobFailedError);
  await expect(awaitJob(record({ status: "failed", error: "Checkpoint missing" }), new AbortController().signal)).rejects.toThrow("Checkpoint missing");
  await expect(awaitJob(record({ status: "cancelled" }), new AbortController().signal)).rejects.toThrow(JobCancelledError);
});

test("polling recovers a completion without an SSE event", async () => {
  let reads = 0;
  api.GET = (async () => reply(record(++reads === 1 ? { status: "running" } : { status: "done", result: 42 }))) as typeof api.GET;
  await expect(awaitJob<number>(record(), new AbortController().signal)).resolves.toBe(42);
  expect(reads).toBe(2);
});

test("retries a temporary disconnect without resubmitting the job", async () => {
  let reads = 0;
  api.GET = (async () => {
    if (++reads === 1) throw new TypeError("Network unavailable");
    return reply(record({ status: "done", result: null }));
  }) as typeof api.GET;
  await expect(awaitJob(record(), new AbortController().signal)).resolves.toBeNull();
  expect(reads).toBe(2);
});

test("a missing job is reported without an endless retry", async () => {
  let reads = 0;
  api.GET = (async () => { reads++; return { error: { detail: "No job" }, response: new Response(null, { status: 404 }) }; }) as typeof api.GET;
  await expect(awaitJob(record(), new AbortController().signal)).rejects.toThrow("404");
  expect(reads).toBe(1);
});

test("rejects a polling result belonging to a different job", async () => {
  api.GET = (async () => reply(record({ id: "other", status: "done", result: "wrong" }))) as typeof api.GET;
  await expect(awaitJob(record(), new AbortController().signal)).rejects.toThrow("invalid job record");
});

test("aborting a wait requests cancellation once and ignores a late result", async () => {
  const controller = new AbortController();
  let deletes = 0;
  let finish!: (value: ReturnType<typeof reply>) => void;
  api.GET = (() => new Promise<ReturnType<typeof reply>>((resolve) => { finish = resolve; })) as typeof api.GET;
  api.DELETE = (async () => { deletes++; return reply(record({ status: "running" })); }) as typeof api.DELETE;
  const waiting = awaitJob(record(), controller.signal);
  const rejected = expect(waiting).rejects.toMatchObject({ name: "AbortError" });
  controller.abort();
  finish(reply(record({ status: "done", result: "late" })));
  await rejected;
  expect(deletes).toBe(1);
});

test("an already aborted caller cancels a job submitted before its id arrived", async () => {
  let deletes = 0;
  api.DELETE = (async () => { deletes++; return reply(record({ status: "cancelled" })); }) as typeof api.DELETE;
  const controller = new AbortController();
  controller.abort();
  await expect(awaitJob(record(), controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  expect(deletes).toBe(1);
});

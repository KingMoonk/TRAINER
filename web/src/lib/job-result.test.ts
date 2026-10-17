import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ApiError } from "./api-core";
import { awaitJob, JobCancelledError, JobFailedError, type JobOut } from "./job-result";
import { emit, json, mockApi, type Handler } from "@/test/network";

const record = (changes: Partial<JobOut> = {}): JobOut => ({
  id: "job-1", kind: "test", status: "queued", created_at: 0, progress: "", error: "", ...changes,
});

/** Answers GET /jobs/job-1 with each handler in turn (the last one repeats) and DELETE /jobs/job-1, counting both. */
function server(reads: (Handler | object)[], opts: { onDelete?: Handler } = {}) {
  const counts = { reads: 0, deletes: 0 };
  const calls = mockApi({
    "GET /jobs/job-1": async (call, request) => {
      const step = reads[Math.min(counts.reads++, reads.length - 1)];
      return typeof step === "function" ? step(call, request) : { body: step };
    },
    "DELETE /jobs/job-1": async (call, request) => {
      counts.deletes++;
      return opts.onDelete ? opts.onDelete(call, request) : { body: record({ status: "cancelled" }) };
    },
  });
  return { counts, calls };
}

/** Real timers cannot run a 1 s poll pause quickly: fake only the timeout and step it while the wait is unsettled. */
async function drive<T>(waiting: Promise<T>): Promise<T> {
  let settled = false;
  waiting.then(() => (settled = true), () => (settled = true));
  for (let turn = 0; turn < 100 && !settled; turn++) {
    await vi.advanceTimersByTimeAsync(1_000);
    await new Promise((resolve) => setImmediate(resolve));
  }
  return waiting;
}

/** An EventSource subclass that remembers its instances, so a test can see whether the wait closed its stream. */
function trackSources() {
  const original = globalThis.EventSource;
  const Base = original as unknown as new () => { close(): void };
  const made: { closed: boolean }[] = [];
  globalThis.EventSource = class extends Base {
    closed = false;
    constructor() {
      super();
      made.push(this);
    }
    close() {
      this.closed = true;
      super.close();
    }
  } as unknown as typeof EventSource;
  return { made, restore: () => (globalThis.EventSource = original) };
}

const never = new AbortController().signal;
const fresh = () => new AbortController().signal;

describe("a job that is already settled", () => {
  test("returns an already completed result without polling", async () => {
    const { calls } = server([]);
    await expect(awaitJob(record({ status: "done", result: { step: 0 } }), fresh())).resolves.toEqual({ step: 0 });
    expect(calls).toHaveLength(0);
  });

  test("a done job with no result gives undefined", async () => {
    server([]);
    await expect(awaitJob(record({ status: "done" }), fresh())).resolves.toBeUndefined();
  });

  test("reports failed and server-cancelled jobs distinctly", async () => {
    server([]);
    const failed = record({ status: "failed", error: "Checkpoint missing" });
    await expect(awaitJob(failed, fresh())).rejects.toThrow(JobFailedError);
    await expect(awaitJob(failed, fresh())).rejects.toThrow("Checkpoint missing");
    await expect(awaitJob(record({ status: "cancelled" }), fresh())).rejects.toThrow(JobCancelledError);
  });

  test("the errors say which job and have a readable name", async () => {
    server([]);
    const failed = await awaitJob(record({ status: "failed", error: "boom" }), fresh()).catch((e: unknown) => e);
    expect(failed).toMatchObject({ name: "JobFailedError", jobId: "job-1", message: "boom" });
    const cancelled = await awaitJob(record({ status: "cancelled" }), fresh()).catch((e: unknown) => e);
    expect(cancelled).toMatchObject({ name: "JobCancelledError", jobId: "job-1", message: "The job was cancelled." });
  });

  test("a failed job with an empty error still has a message", async () => {
    server([]);
    await expect(awaitJob(record({ status: "failed", error: "" }), fresh())).rejects.toThrow("The job failed.");
  });

  test("an initial record with an unknown status is rejected", async () => {
    server([]);
    await expect(awaitJob(record({ status: "weird" as JobOut["status"] }), fresh())).rejects.toThrow("invalid job record");
  });
});

describe("polling", () => {
  beforeEach(() => vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] }));
  afterEach(() => vi.useRealTimers());

  test("polling recovers a completion without an SSE event", async () => {
    const { counts } = server([record({ status: "running" }), record({ status: "done", result: 42 })]);
    await expect(drive(awaitJob<number>(record(), fresh()))).resolves.toBe(42);
    expect(counts.reads).toBe(2);
  });

  test("the first poll happens at once, not after a pause", async () => {
    const { counts } = server([record({ status: "done", result: 1 })]);
    await expect(awaitJob<number>(record(), fresh())).resolves.toBe(1);
    expect(counts.reads).toBe(1);
  });

  test("a polled failure rejects like an initial one", async () => {
    server([record({ status: "failed", error: "Out of memory" })]);
    await expect(drive(awaitJob(record(), fresh()))).rejects.toThrow(JobFailedError);
  });

  test("a polled cancellation rejects with JobCancelledError", async () => {
    server([record({ status: "running" }), record({ status: "cancelled" })]);
    await expect(drive(awaitJob(record(), fresh()))).rejects.toThrow(JobCancelledError);
  });

  test("retries a temporary disconnect without resubmitting the job", async () => {
    const { counts } = server([
      () => { throw new TypeError("Network unavailable"); },
      record({ status: "done", result: null }),
    ]);
    await expect(drive(awaitJob(record(), fresh()))).resolves.toBeNull();
    expect(counts.reads).toBe(2);
  });

  test("a server error or rate limit is retried too", async () => {
    const { counts } = server([() => ({ status: 503, body: {} }), () => ({ status: 429, body: {} }), record({ status: "done", result: "ok" })]);
    await expect(drive(awaitJob(record(), fresh()))).resolves.toBe("ok");
    expect(counts.reads).toBe(3);
  });

  test("gives up on the third failure in a row and reports the error", async () => {
    const { counts } = server([() => ({ status: 500, body: { detail: "down" } })]);
    const error = await drive(awaitJob(record(), fresh())).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 500 });
    expect(counts.reads).toBe(3);
  });

  test("a success resets the failure count", async () => {
    const fail = () => ({ status: 500, body: {} });
    const { counts } = server([fail, fail, record({ status: "running" }), fail, fail, record({ status: "done", result: "late" })]);
    await expect(drive(awaitJob(record(), fresh()))).resolves.toBe("late");
    expect(counts.reads).toBe(6);
  });

  test("a missing job is reported without an endless retry", async () => {
    const { counts } = server([() => ({ status: 404, body: { detail: "No job" } })]);
    await expect(awaitJob(record(), fresh())).rejects.toThrow("404");
    expect(counts.reads).toBe(1);
  });

  test("any other 4xx is reported at once", async () => {
    const { counts } = server([() => ({ status: 401, body: {} })]);
    await expect(awaitJob(record(), fresh())).rejects.toMatchObject({ status: 401 });
    expect(counts.reads).toBe(1);
  });

  test("rejects a polling result belonging to a different job", async () => {
    server([record({ id: "other", status: "done", result: "wrong" })]);
    await expect(awaitJob(record(), fresh())).rejects.toThrow("invalid job record");
  });

  test.each([
    ["not an object", "oops"],
    ["null", null],
    ["a status it does not know", { id: "job-1", status: "paused" }],
    ["no status", { id: "job-1" }],
  ])("rejects a polling result that is %s", async (_, body) => {
    server([() => json(body)]);
    await expect(awaitJob(record(), fresh())).rejects.toThrow("invalid job record");
  });
});

describe("aborting", () => {
  /** A GET that never answers until released, and fails like fetch does when the request is aborted. */
  function hangingRead() {
    let release!: (reply: object) => void;
    const handler: Handler = (_, request) =>
      new Promise((resolve, reject) => {
        release = (body) => resolve({ body });
        const abort = () => reject(new DOMException("aborted", "AbortError"));
        if (request.signal.aborted) abort();
        else request.signal.addEventListener("abort", abort);
      });
    return { handler, release: (body: object) => release(body) };
  }

  test("aborting a wait requests cancellation once and ignores a late result", async () => {
    const controller = new AbortController();
    const read = hangingRead();
    const { counts } = server([read.handler]);
    const waiting = awaitJob(record(), controller.signal);
    const rejected = expect(waiting).rejects.toMatchObject({ name: "AbortError" });
    await vi.waitFor(() => expect(counts.reads).toBe(1));
    controller.abort();
    read.release(record({ status: "done", result: "late" }));
    await rejected;
    expect(counts.deletes).toBe(1);
  });

  test("an already aborted caller cancels a job submitted before its id arrived", async () => {
    const { counts } = server([]);
    const controller = new AbortController();
    controller.abort();
    await expect(awaitJob(record(), controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(counts.deletes).toBe(1);
    expect(counts.reads).toBe(0);
  });

  test("an abort that lands during the pause between polls rejects promptly", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      const controller = new AbortController();
      const { counts } = server([record({ status: "running" })]);
      const waiting = awaitJob(record(), controller.signal);
      const rejected = expect(waiting).rejects.toMatchObject({ name: "AbortError" });
      while (counts.reads < 1) await new Promise((resolve) => setImmediate(resolve));
      await new Promise((resolve) => setImmediate(resolve));
      controller.abort();
      await rejected;
      expect(counts.reads).toBe(1);
      expect(counts.deletes).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  test("a cancellation request that cannot reach the server does not change the outcome", async () => {
    const controller = new AbortController();
    controller.abort();
    const { counts } = server([], { onDelete: () => { throw new TypeError("offline"); } });
    await expect(awaitJob(record(), controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    await vi.waitFor(() => expect(counts.deletes).toBe(1));
  });

  test("a failed read after an abort is an AbortError, not the read's own error", async () => {
    const controller = new AbortController();
    const { counts } = server([() => { controller.abort(); throw new TypeError("Failed to fetch"); }]);
    await expect(awaitJob(record(), controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(counts.reads).toBe(1);
  });

  test("a job that has finished is not cancelled by a later abort", async () => {
    const controller = new AbortController();
    const { counts } = server([]);
    await expect(awaitJob(record({ status: "done", result: 1 }), controller.signal)).resolves.toBe(1);
    controller.abort(); // the listener was removed when the wait ended
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(counts.deletes).toBe(0);
  });
});

describe("with events", () => {
  test("the job's /events message ends the wait before the next poll", async () => {
    const sources = trackSources();
    try {
      const { counts } = server([record({ status: "running" })]);
      const waiting = awaitJob<number>(record(), never, { events: true });
      await vi.waitFor(() => expect(counts.reads).toBe(1));
      emit("job", record({ id: "other", status: "done", result: "wrong" })); // another job
      emit("job", record({ status: "done", result: 7 }));
      await expect(waiting).resolves.toBe(7);
      expect(counts.reads).toBe(1);
      expect(sources.made.map((source) => source.closed)).toEqual([true]);
    } finally {
      sources.restore();
    }
  });

  test("a failure message rejects like a polled one", async () => {
    const sources = trackSources();
    try {
      server([record({ status: "running" })]);
      const waiting = awaitJob(record(), never, { events: true });
      const rejected = expect(waiting).rejects.toThrow("Checkpoint missing");
      await vi.waitFor(() => expect(sources.made).toHaveLength(1));
      emit("job", record({ status: "failed", error: "Checkpoint missing" }));
      await rejected;
      expect(sources.made.map((source) => source.closed)).toEqual([true]);
    } finally {
      sources.restore();
    }
  });

  test("a cancellation message rejects with JobCancelledError", async () => {
    server([record({ status: "running" })]);
    const waiting = awaitJob(record(), never, { events: true });
    const rejected = expect(waiting).rejects.toThrow(JobCancelledError);
    emit("job", record({ status: "cancelled" })); // the stream is opened synchronously, before the first poll
    await rejected;
  });

  test("other event kinds and non-job payloads are ignored", async () => {
    const { counts } = server([record({ status: "running" })]);
    const waiting = awaitJob<string>(record(), never, { events: true });
    await vi.waitFor(() => expect(counts.reads).toBe(1));
    emit("status", record({ status: "done", result: "wrong" })); // not a job event
    emit("job", "garbage");
    emit("job", { id: "job-1", status: "bogus" });
    emit("job", record({ status: "done", result: "right" }));
    await expect(waiting).resolves.toBe("right");
  });

  test("without the option no stream is opened, and with it the stream is closed when the wait fails", async () => {
    const sources = trackSources();
    try {
      server([record({ status: "done", result: 1 })]);
      await awaitJob(record(), fresh());
      expect(sources.made).toHaveLength(0);
      server([() => ({ status: 404, body: {} })]);
      await expect(awaitJob(record(), fresh(), { events: true })).rejects.toThrow("404");
      expect(sources.made.map((source) => source.closed)).toEqual([true]);
    } finally {
      sources.restore();
    }
  });

  test("an event that arrives while the first poll is in flight is used on the next turn", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    server([async () => (await gate, { body: record({ status: "running" }) })]);
    const waiting = awaitJob<number>(record(), never, { events: true });
    emit("job", record({ status: "done", result: 9 }));
    release();
    await expect(waiting).resolves.toBe(9);
  });
});

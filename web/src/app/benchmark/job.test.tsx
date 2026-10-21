import { render } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { ShellEvents } from "@/lib/events";
import { dropEvents, emit, mockApi } from "@/test/network";
import { awaitJob, JobFailed, type JobOut } from "./job";

const record = (changes: Partial<JobOut> = {}): JobOut => ({
  id: "job-1", kind: "benchmark", status: "queued", created_at: 0, progress: "", error: "", ...changes,
});

/** Answers GET /jobs/job-1 with each record in turn (the last repeats) and counts the reads. */
function server(reads: JobOut[]) {
  const counts = { reads: 0 };
  mockApi({ "GET /jobs/job-1": () => ({ body: reads[Math.min(counts.reads++, reads.length - 1)] }) });
  return counts;
}

beforeEach(() => {
  render(<ShellEvents>{null}</ShellEvents>);
});

describe("awaitJob", () => {
  test("an already finished job resolves without reading or listening", async () => {
    const counts = server([]);
    await expect(awaitJob(record({ status: "done", result: 5 }))).resolves.toBe(5);
    expect(counts.reads).toBe(0);
  });

  test("the job's event ends the wait, with no read beyond the first", async () => {
    const counts = server([record({ status: "running" })]);
    const waiting = awaitJob<string>(record());
    await vi.waitFor(() => expect(counts.reads).toBe(1));
    emit("job", record({ id: "other", status: "done", result: "wrong" }));
    emit("job", record({ status: "done", result: "right" }));
    await expect(waiting).resolves.toBe("right");
    expect(counts.reads).toBe(1);
  });

  test("a failed or cancelled event rejects with the server's message", async () => {
    server([record({ status: "running" })]);
    const failed = awaitJob(record());
    const rejected = expect(failed).rejects.toThrow(JobFailed);
    emit("job", record({ status: "failed", error: "Out of memory" }));
    await rejected;
    const cancelled = awaitJob(record());
    const cancelledRejected = expect(cancelled).rejects.toThrow("The job was cancelled.");
    emit("job", record({ status: "cancelled" }));
    await cancelledRejected;
  });

  test("a quiet stream is not read on a short interval", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    try {
      const counts = server([record({ status: "running" })]);
      const waiting = awaitJob(record());
      await vi.advanceTimersByTimeAsync(10_000);
      expect(counts.reads).toBe(1);
      emit("job", record({ status: "done", result: 1 })); // ends the wait, so it stops listening
      await waiting;
    } finally {
      vi.useRealTimers();
    }
  });

  test("the backstop reads the job when its event was lost", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    try {
      const counts = server([record({ status: "running" }), record({ status: "done", result: "late" })]);
      const waiting = awaitJob<string>(record());
      await vi.advanceTimersByTimeAsync(15_001);
      await expect(waiting).resolves.toBe("late");
      expect(counts.reads).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });

  test("the stream coming back after a drop reads the job again", async () => {
    await new Promise((resolve) => setTimeout(resolve, 5)); // the shell's first open is not a reconnect
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      const counts = server([record({ status: "running" }), record({ status: "done", result: "after" })]);
      const waiting = awaitJob<string>(record());
      await vi.advanceTimersByTimeAsync(1);
      expect(counts.reads).toBe(1);
      dropEvents();
      await vi.advanceTimersByTimeAsync(5_001);
      await expect(waiting).resolves.toBe("after");
      expect(counts.reads).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });
});

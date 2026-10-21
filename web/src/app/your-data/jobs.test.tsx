import { render } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { ShellEvents } from "@/lib/events";
import { dropEvents, emit, mockApi } from "@/test/network";
import { runJob, type JobLike } from "./jobs";

const record = (changes: Partial<JobLike> = {}): JobLike => ({ id: "job-1", status: "queued", ...changes });

/** Answers GET /jobs/job-1 with each record in turn (the last repeats) and counts the reads. */
function server(reads: JobLike[]) {
  const counts = { reads: 0 };
  mockApi({ "GET /jobs/job-1": () => ({ body: reads[Math.min(counts.reads++, reads.length - 1)] }) });
  return counts;
}

beforeEach(() => {
  render(<ShellEvents>{null}</ShellEvents>);
});

describe("runJob", () => {
  test("an already finished job resolves with its result without a read", async () => {
    const counts = server([]);
    await expect(runJob(async () => record({ status: "done", result: 4 }))).resolves.toBe(4);
    expect(counts.reads).toBe(0);
  });

  test("the job's event ends the wait", async () => {
    const counts = server([record({ status: "running" })]);
    const waiting = runJob<string>(async () => record());
    await vi.waitFor(() => expect(counts.reads).toBe(1));
    emit("job", record({ id: "other", status: "done", result: "wrong" }));
    emit("job", record({ status: "running" }));
    emit("job", record({ status: "done", result: "right" }));
    await expect(waiting).resolves.toBe("right");
    expect(counts.reads).toBe(1);
  });

  test("a failed event rejects with the server's sentence", async () => {
    const counts = server([record({ status: "running" })]);
    const waiting = runJob(async () => record());
    const rejected = expect(waiting).rejects.toThrow("No such file");
    await vi.waitFor(() => expect(counts.reads).toBe(1));
    emit("job", record({ status: "failed", error: "No such file" }));
    await rejected;
  });

  test("an abort stops the wait", async () => {
    server([record({ status: "running" })]);
    const controller = new AbortController();
    const waiting = runJob(async () => record(), controller.signal);
    const rejected = expect(waiting).rejects.toMatchObject({ name: "AbortError" });
    controller.abort();
    await rejected;
  });

  test("a quiet stream is not read on a short interval", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    try {
      const counts = server([record({ status: "running" })]);
      const controller = new AbortController();
      const waiting = runJob(async () => record(), controller.signal).catch(() => {});
      await vi.advanceTimersByTimeAsync(10_000);
      expect(counts.reads).toBe(1);
      controller.abort();
      await waiting;
    } finally {
      vi.useRealTimers();
    }
  });

  test("the backstop reads the job when its event was lost", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    try {
      const counts = server([record({ status: "running" }), record({ status: "done", result: "late" })]);
      const waiting = runJob<string>(async () => record());
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
      const waiting = runJob<string>(async () => record());
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

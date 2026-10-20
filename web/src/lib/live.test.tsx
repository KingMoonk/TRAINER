import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ShellEvents, useLiveRefresh } from "@/lib/events";
import { payloadIs } from "@/lib/live";
import { dropEvents, emit } from "@/test/network";

function Probe({ refetch, kinds = ["queue"], backstopMs = 60_000 as number | false, throttleMs, match }: {
  refetch: () => unknown; kinds?: string[]; backstopMs?: number | false; throttleMs?: number; match?: (kind: string, payload: unknown) => boolean;
}) {
  useLiveRefresh(kinds, refetch, backstopMs, { throttleMs, match });
  return null;
}

function mount(props: Parameters<typeof Probe>[0]) {
  return render(<ShellEvents><Probe {...props} /></ShellEvents>);
}

async function flush(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

function hide(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });
  document.dispatchEvent(new Event("visibilitychange"));
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  hide(false);
  vi.useRealTimers();
});

describe("useLiveRefresh", () => {
  test("an event of its kinds re-reads; other kinds do not", async () => {
    const refetch = vi.fn();
    mount({ refetch });
    await flush();
    emit("versions", {});
    await flush();
    expect(refetch).not.toHaveBeenCalled();
    emit("queue", {});
    await flush();
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  test("the stream coming back after a drop re-reads, the first open does not", async () => {
    const refetch = vi.fn();
    mount({ refetch });
    await flush();
    expect(refetch).not.toHaveBeenCalled();
    act(() => dropEvents());
    await flush(5_000); // the shell retries the stream after five seconds
    await flush(1); // and the new stream opens
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  test("showing the tab again re-reads", async () => {
    const refetch = vi.fn();
    mount({ refetch });
    await flush();
    act(() => hide(true));
    act(() => hide(false));
    await flush();
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  test("the backstop re-reads every minute, but not while the tab is hidden", async () => {
    const refetch = vi.fn();
    mount({ refetch });
    await flush(60_000);
    expect(refetch).toHaveBeenCalledTimes(1);
    act(() => hide(true));
    await flush(120_000);
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  test("a burst while a read is in flight costs one more read", async () => {
    let finish = () => {};
    const refetch = vi.fn(() => new Promise<void>((resolve) => { finish = resolve; }));
    mount({ refetch });
    await flush();
    emit("queue", {});
    await flush();
    emit("queue", {});
    emit("queue", {});
    emit("queue", {});
    await flush();
    expect(refetch).toHaveBeenCalledTimes(1);
    await act(async () => finish());
    await flush();
    expect(refetch).toHaveBeenCalledTimes(2);
  });

  test("throttleMs keeps reads apart and still makes the trailing one", async () => {
    const refetch = vi.fn();
    mount({ refetch, kinds: ["train"], throttleMs: 1_000 });
    await flush();
    emit("train", {});
    await flush();
    emit("train", {});
    await flush(500);
    expect(refetch).toHaveBeenCalledTimes(1);
    await flush(500);
    expect(refetch).toHaveBeenCalledTimes(2);
  });

  test("payloadIs keeps the events of one run", async () => {
    const refetch = vi.fn();
    mount({ refetch, kinds: ["run", "facts"], match: payloadIs("run", "r1", ["run"]) });
    await flush();
    emit("run", { run: "r2" });
    await flush();
    expect(refetch).not.toHaveBeenCalled();
    emit("run", { run: "r1" });
    await flush();
    emit("facts", { kinds: ["progress"] });
    await flush();
    expect(refetch).toHaveBeenCalledTimes(2);
  });
});

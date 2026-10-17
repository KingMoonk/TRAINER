import { act, screen, waitFor } from "@testing-library/react";
import { useEffect, useState } from "react";
import { describe, expect, test, vi } from "vitest";
import { emit, eventsUnreachable, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { TOKEN_KEY } from "./token";
import { PLACEHOLDER_STATUS, type ProblemReport } from "./types";
import { asStudioEvent, problemReportMessage, useEvent, useShell } from "./events";

const report = (over: Partial<ProblemReport> = {}): ProblemReport => ({ items: [], added: 0, duplicates: 0, checked: 0, passed: 0, ...over });

describe("problemReportMessage", () => {
  test("says how many were added, singular and plural", () => {
    expect(problemReportMessage(report({ added: 1 })).text).toBe("Added 1 problem");
    expect(problemReportMessage(report({ added: 12 })).text).toBe("Added 12 problems");
    expect(problemReportMessage(report({ added: 0 })).text).toBe("Added 0 problems");
  });

  test("mentions duplicates only when there are some", () => {
    expect(problemReportMessage(report({ added: 3, duplicates: 2 })).text).toBe("Added 3 problems; 2 already there");
  });

  test("nothing checked says nothing about checking, and is clean", () => {
    expect(problemReportMessage(report({ added: 2 }))).toEqual({ text: "Added 2 problems", clean: true });
  });

  test("all checked passed is clean", () => {
    expect(problemReportMessage(report({ added: 5, checked: 5, passed: 5 }))).toEqual({ text: "Added 5 problems; all 5 checked passed", clean: true });
  });

  test("some that did not pass their rule make it not clean", () => {
    expect(problemReportMessage(report({ added: 5, checked: 5, passed: 3 }))).toEqual({
      text: "Added 5 problems; 2 of 5 checked did not pass their rule",
      clean: false,
    });
  });

  test("puts every part in order", () => {
    expect(problemReportMessage(report({ added: 1, duplicates: 4, checked: 3, passed: 0 })).text).toBe(
      "Added 1 problem; 4 already there; 3 of 3 checked did not pass their rule",
    );
  });
});

describe("asStudioEvent", () => {
  test("passes status and problem_report events through", () => {
    const status = { kind: "status", payload: PLACEHOLDER_STATUS };
    const problems = { kind: "problem_report", payload: report() };
    expect(asStudioEvent(status)).toBe(status);
    expect(asStudioEvent(problems)).toBe(problems);
  });

  test("any other kind is for the page that owns it", () => {
    expect(asStudioEvent({ kind: "queue", payload: {} })).toBeNull();
    expect(asStudioEvent({ kind: "", payload: null })).toBeNull();
  });
});

/** Lets a test change the props of a mounted Listener (rerender() would drop the providers renderPage wraps). */
let setListener!: (props: { kind: string; onEvent: (payload: unknown) => void }) => void;
function Swappable(initial: { kind: string; onEvent: (payload: unknown) => void }) {
  const [props, set] = useState(initial);
  useEffect(() => {
    setListener = set;
  }, [set]);
  return <Listener {...props} />;
}

function Listener({ kind, onEvent }: { kind: string; onEvent: (payload: unknown) => void }) {
  useEvent(kind, onEvent);
  return null;
}

describe("useEvent", () => {
  test("calls the handler for its kind only, with the payload", () => {
    const handler = vi.fn();
    renderPage(<Listener kind="queue" onEvent={handler} />);
    emit("queue", { n: 1 });
    emit("cluster", { n: 2 });
    emit("queue", { n: 3 });
    expect(handler.mock.calls.map((c) => c[0])).toEqual([{ n: 1 }, { n: 3 }]);
  });

  test("calls the latest handler after a re-render, without resubscribing", () => {
    const first = vi.fn();
    const second = vi.fn();
    renderPage(<Swappable kind="queue" onEvent={first} />);
    act(() => setListener({ kind: "queue", onEvent: second }));
    emit("queue", { n: 1 });
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledWith({ n: 1 });
  });

  test("stops listening when the component unmounts", () => {
    const handler = vi.fn();
    const { unmount } = renderPage(<Listener kind="queue" onEvent={handler} />);
    emit("queue", { n: 1 });
    unmount();
    emit("queue", { n: 2 });
    expect(handler).toHaveBeenCalledTimes(1);
  });

  test("a changed kind moves the subscription", () => {
    const handler = vi.fn();
    renderPage(<Swappable kind="queue" onEvent={handler} />);
    act(() => setListener({ kind: "cluster", onEvent: handler }));
    emit("queue", 1);
    emit("cluster", 2);
    expect(handler.mock.calls.map((c) => c[0])).toEqual([2]);
  });

  test("every listener of a kind hears it", () => {
    const a = vi.fn();
    const b = vi.fn();
    renderPage(
      <>
        <Listener kind="queue" onEvent={a} />
        <Listener kind="queue" onEvent={b} />
      </>,
    );
    emit("queue", 1);
    expect(a).toHaveBeenCalledOnce();
    expect(b).toHaveBeenCalledOnce();
  });
});

function Shell() {
  const { status, connection } = useShell();
  return (
    <div>
      <p data-testid="connection">{connection}</p>
      <p data-testid="version">{status.version ?? "none"}</p>
      <p data-testid="queue">{status.queue.state}</p>
    </div>
  );
}

describe("useShell and ShellEvents", () => {
  test("starts connecting with the placeholder status, then reads online once the stream opens", async () => {
    renderPage(<Shell />);
    expect(screen.getByTestId("connection")).toHaveTextContent("connecting");
    expect(screen.getByTestId("version")).toHaveTextContent("none");
    await waitFor(() => expect(screen.getByTestId("connection")).toHaveTextContent("online"));
  });

  test("a status event replaces the status whole", async () => {
    renderPage(<Shell />);
    await waitFor(() => expect(screen.getByTestId("connection")).toHaveTextContent("online"));
    act(() => emit("status", { version: "version 3", queue: { state: "running", running: 1, waiting: 0 }, resources: { device: null, diskFreeGb: null } }));
    expect(screen.getByTestId("version")).toHaveTextContent("version 3");
    expect(screen.getByTestId("queue")).toHaveTextContent("running");
  });

  test("events of other kinds are forwarded to listeners and leave the status alone", async () => {
    const handler = vi.fn();
    renderPage(
      <>
        <Shell />
        <Listener kind="versions" onEvent={handler} />
      </>,
    );
    await waitFor(() => expect(screen.getByTestId("connection")).toHaveTextContent("online"));
    act(() => emit("versions", { number: 2 }));
    expect(handler).toHaveBeenCalledWith({ number: 2 });
    expect(screen.getByTestId("version")).toHaveTextContent("none");
  });

  test("a status event is also offered to listeners of status", () => {
    const handler = vi.fn();
    renderPage(<Listener kind="status" onEvent={handler} />);
    act(() => emit("status", PLACEHOLDER_STATUS));
    expect(handler).toHaveBeenCalledWith(PLACEHOLDER_STATUS);
  });

  test("an unreachable stream reads offline", async () => {
    eventsUnreachable();
    renderPage(<Shell />);
    await waitFor(() => expect(screen.getByTestId("connection")).toHaveTextContent("offline"));
  });

  test("an offline stream is retried every few seconds and comes back online", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      eventsUnreachable();
      const advance = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));
      renderPage(<Shell />);
      await advance(1);
      expect(screen.getByTestId("connection")).toHaveTextContent("offline");
      eventsUnreachable(false);
      await advance(4_999);
      expect(screen.getByTestId("connection")).toHaveTextContent("offline");
      await advance(2);
      expect(screen.getByTestId("connection")).toHaveTextContent("online");
    } finally {
      vi.useRealTimers();
    }
  });

  test("unmounting cancels the pending retry", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const Base = globalThis.EventSource as unknown as new () => object;
    let opened = 0;
    globalThis.EventSource = class extends Base {
      constructor() {
        super();
        opened++;
      }
    } as unknown as typeof EventSource;
    try {
      eventsUnreachable();
      const { unmount } = renderPage(<Shell />);
      await act(() => vi.advanceTimersByTimeAsync(1));
      expect(opened).toBe(1);
      unmount();
      await vi.advanceTimersByTimeAsync(20_000);
      expect(opened).toBe(1);
    } finally {
      globalThis.EventSource = Base as unknown as typeof EventSource;
      vi.useRealTimers();
    }
  });
});

describe("problem_report events", () => {
  test("a clean report raises a success toast", async () => {
    renderPage(<Shell />);
    await waitFor(() => expect(screen.getByTestId("connection")).toHaveTextContent("online"));
    act(() => emit("problem_report", report({ added: 4, checked: 4, passed: 4 })));
    const toast = await screen.findByText("Added 4 problems; all 4 checked passed");
    expect(toast.closest("[data-type]")).toHaveAttribute("data-type", "success");
  });

  test("a report with failures raises a warning toast", async () => {
    renderPage(<Shell />);
    await waitFor(() => expect(screen.getByTestId("connection")).toHaveTextContent("online"));
    act(() => emit("problem_report", report({ added: 2, checked: 2, passed: 1 })));
    const toast = await screen.findByText("Added 2 problems; 1 of 2 checked did not pass their rule");
    expect(toast.closest("[data-type]")).toHaveAttribute("data-type", "warning");
  });
});

// Last in the file: a refused token is remembered by the module until the page reloads.
describe("a refused token", () => {
  test("reads offline and the stream is not retried", async () => {
    localStorage.setItem(TOKEN_KEY, "wrong");
    const calls = mockApi({ "GET /events": { status: 401, body: { detail: "bad token" } } });
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      renderPage(<Shell />);
      for (let turn = 0; turn < 100 && !calls.length; turn++) await new Promise((resolve) => setImmediate(resolve));
      await vi.advanceTimersByTimeAsync(1);
      for (let turn = 0; turn < 20; turn++) await new Promise((resolve) => setImmediate(resolve));
      expect(screen.getByTestId("connection")).toHaveTextContent("offline");
      await vi.advanceTimersByTimeAsync(20_000);
      expect(calls).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });
});

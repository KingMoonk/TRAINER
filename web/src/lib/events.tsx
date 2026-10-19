"use client";

import { createContext, useContext, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { subscribe, tokenRefused, type ApiEvent } from "@/lib/api";
import { PLACEHOLDER_STATUS, type Connection, type ProblemReport, type StudioEvent, type StudioStatus } from "@/lib/types";

const RETRY_MS = 5_000;

type Shell = { status: StudioStatus; connection: Connection };

const ShellContext = createContext<Shell>({ status: PLACEHOLDER_STATUS, connection: "connecting" });

/** The shell's view of the server: the latest status and whether the event stream is up. */
export function useShell(): Shell {
  return useContext(ShellContext);
}

/** Every page listener the shell forwards raw events to (useEvent). A module-level set: the stream is opened once. */
const listeners = new Set<(event: ApiEvent) => void>();

/**
 * Call `handler` for every `kind` event on the shell's stream while the component is mounted. The handler may change
 * between renders; the latest one is called. Pages use this to update a keyed list in place instead of polling.
 */
export function useEvent(kind: string, handler: (payload: unknown) => void): void {
  const latest = useRef(handler);
  useEffect(() => {
    latest.current = handler;
  });
  useEffect(() => {
    const listener = (event: ApiEvent) => {
      if (event.kind === kind) latest.current(event.payload);
    };
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, [kind]);
}

/** Call `handler` for every event of every kind on the shell's stream while mounted (the traffic console's alert rules). */
export function useAnyEvent(handler: (event: ApiEvent) => void): void {
  const latest = useRef(handler);
  useEffect(() => {
    latest.current = handler;
  });
  useEffect(() => {
    const listener = (event: ApiEvent) => latest.current(event);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);
}

/** Narrow a raw stream event to one the shell handles; any other kind is for the page that owns it. */
export function asStudioEvent(event: ApiEvent): StudioEvent | null {
  return event.kind === "status" || event.kind === "problem_report" ? (event as StudioEvent) : null;
}

export function problemReportMessage(report: ProblemReport): { text: string; clean: boolean } {
  const noun = (n: number) => `${n} problem${n === 1 ? "" : "s"}`;
  const parts = [`Added ${noun(report.added)}`];
  if (report.duplicates) parts.push(`${report.duplicates} already there`);
  const failed = report.checked - report.passed;
  if (report.checked) parts.push(failed ? `${failed} of ${report.checked} checked did not pass their rule` : `all ${report.checked} checked passed`);
  return { text: parts.join("; "), clean: failed === 0 };
}

/**
 * Opens the event stream once for the whole app. `status` events feed the status strip; `problem_report` events raise a
 * toast. The stream is retried every few seconds while it is down, since the server may start after the page loads.
 */
export function ShellEvents({ children }: { children: React.ReactNode }) {
  const [shell, setShell] = useState<Shell>({ status: PLACEHOLDER_STATUS, connection: "connecting" });

  useEffect(() => {
    let close = () => {};
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      close = subscribe(
        (raw) => {
          for (const listener of listeners) listener(raw);
          const event = asStudioEvent(raw);
          if (event?.kind === "status") setShell((s) => ({ ...s, status: event.payload }));
          if (event?.kind === "problem_report") {
            const { text, clean } = problemReportMessage(event.payload);
            (clean ? toast.success : toast.warning)(text);
          }
        },
        () => {
          close();
          setShell((s) => ({ ...s, connection: "offline" }));
          if (!tokenRefused()) timer = setTimeout(connect, RETRY_MS);  // a refused token is not retried until the page reloads
        },
        () => setShell((s) => ({ ...s, connection: "online" })),
      );
    };
    connect();
    return () => {
      clearTimeout(timer);
      close();
    };
  }, []);

  return <ShellContext.Provider value={shell}>{children}</ShellContext.Provider>;
}

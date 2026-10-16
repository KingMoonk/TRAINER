"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { toast } from "sonner";
import { subscribe, type ApiEvent } from "@/lib/api";
import { PLACEHOLDER_STATUS, type Connection, type ProblemReport, type StudioEvent, type StudioStatus } from "@/lib/types";

const RETRY_MS = 5_000;

type Shell = { status: StudioStatus; connection: Connection };

const ShellContext = createContext<Shell>({ status: PLACEHOLDER_STATUS, connection: "connecting" });

/** The shell's view of the server: the latest status and whether the event stream is up. */
export function useShell(): Shell {
  return useContext(ShellContext);
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
          timer = setTimeout(connect, RETRY_MS);
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

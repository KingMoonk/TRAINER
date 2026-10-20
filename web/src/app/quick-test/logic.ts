import type { components } from "@/lib/api-schema";
import { QUICKTEST } from "./copy";

export type Result = components["schemas"]["QuickTestResult"];
export type Which = "current" | "draft";
export type Progress = { step: number; total: number | null };

/** Progress and the finished result of the run being watched, built from the `train` and `job` events of /events. */
export type RunState = {
  result: Result | null;
  progress: Partial<Record<Which, Progress>>;
  firstSeen: Partial<Record<Which, { at: number; step: number }>>;
  ended: string; // "failed" | "stopped" message once the job ended without a result
};

export const NO_RUN: RunState = { result: null, progress: {}, firstSeen: {}, ended: "" };

/** Time left from the pace so far, or null before the first step. */
export function secondsLeft(startedMs: number, step: number, total: number | null, nowMs: number): number | null {
  if (!step || !total) return null;
  return Math.max(0, ((nowMs - startedMs) / 1000 / step) * (total - step));
}

export function checkMark(ok: boolean | null | undefined): string {
  return ok ? "✓" : ok === null || ok === undefined ? "…" : "✗";
}

/** The early-signal sentence; a failed check replaces it with the part to look at. */
export function signalText(result: Result, name: string): string {
  if (result.broken) {
    const check = QUICKTEST.checks.find((c) => c.id === result.broken)?.name ?? result.broken;
    return `A check failed: ${check}. ${QUICKTEST.lookAt[result.broken] ?? ""}`.trim();
  }
  const rel = result.verdict === "about the same" ? "about the same as" : `${result.verdict} than`;
  return `After the same ${result.minutes} minutes, your idea's error is ${rel} ${name}'s.`;
}

/** What one `train` or `job` event does to the watched run; any other event, or another job's, leaves it as it was. */
export function applyEvent(run: RunState, jobId: string | null, kind: string, payload: unknown, nowMs: number): RunState {
  if (!payload || typeof payload !== "object") return run;
  const p = payload as Record<string, unknown>;
  if (jobId === null || p.job !== jobId) return run;
  if (kind === "job" && (p.status === "failed" || p.status === "stopped")) {
    return { ...run, ended: QUICKTEST.finished(String(p.status), typeof p.error === "string" && p.error ? p.error : undefined) };
  }
  if (kind !== "train") return run;
  if (p.type === "quicktest_done") return { ...run, result: p.results as Result, ended: "" };
  if (p.type === "quicktest" && p.step) {
    const which = p.which as Which;
    const step = Number(p.step);
    return {
      ...run,
      ended: "",
      progress: { ...run.progress, [which]: { step, total: typeof p.total === "number" ? p.total : null } },
      firstSeen: run.firstSeen[which] ? run.firstSeen : { ...run.firstSeen, [which]: { at: nowMs, step } },
    };
  }
  return run;
}

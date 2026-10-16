import { PASSIVE } from "./copy";

/** studio_passive.ACTIVE: the statuses a session can still be stopped in. */
export const ACTIVE = ["queued", "running", "paused"];

export type SessionRow = {
  id: string;
  status: string;
  created?: string | null;
  stop?: unknown;
  trainer?: string;
  watchdog?: string;
  restarts?: number;
  heard?: number | null;
  error?: unknown;
};

/** studio_ux.format_duration, without its "about " prefix. */
export function formatAgo(seconds: number): string {
  if (seconds < 45) return "under a minute";
  const minutes = seconds / 60;
  if (minutes < 90) return `${Math.max(1, Math.round(minutes))} min`;
  const hours = minutes / 60;
  return hours < 10 ? `${hours.toFixed(1)} hours` : `${Math.round(hours)} hours`;
}

/** A stored timestamp as local day and time, e.g. "15 Oct 09:30"; the stamp itself if it does not parse. */
export function when(stamp: string | null | undefined): string {
  if (!stamp) return "";
  const date = new Date(stamp);
  if (Number.isNaN(date.getTime())) return stamp;
  const day = date.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
  const time = date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false });
  return `${day} ${time}`;
}

const text = (value: unknown) => (value === null || value === undefined ? "" : String(value));

/** The headline, the lines under it (the stop plan, the facts, any error) and whether Stop applies for one session row. */
export function sessionView(row: SessionRow): { headline: string; lines: string[]; canStop: boolean } {
  const status = PASSIVE.status[row.status] ?? row.status;
  const active = ACTIVE.includes(row.status);
  let facts = PASSIVE.facts(row.trainer ?? "", row.watchdog ?? "", row.restarts ?? 0);
  if (row.heard !== null && row.heard !== undefined && active) facts += PASSIVE.heard(formatAgo(row.heard));
  return {
    headline: PASSIVE.started(status, when(row.created)),
    lines: [text(row.stop), facts, text(row.error)].filter(Boolean),
    canStop: active,
  };
}

/** The warning when a passive session and a queue job both want the computer, else "". */
export function competes(sessions: SessionRow[], queue: { job?: unknown; paused?: boolean } | undefined): string {
  const running = sessions.some((s) => s.status === "running");
  return running && queue?.job != null && !queue.paused ? PASSIVE.competes : "";
}

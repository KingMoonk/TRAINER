import type { components } from "@/lib/api-schema";
import { ApiError } from "@/lib/api";
import { nodeName, type ClusterQueue, type QueueNode, type QueueRow } from "@/lib/cluster";
import { T, fill } from "./copy";

export type Queue = components["schemas"]["QueueStateModel"];
export type Job = components["schemas"]["JobModel"];
export type Plan = components["schemas"]["PlanModel"];
/** The jobs this machine handed to a peer and that are running there: they are in the jobs list but not in the queue state. */
export const awayJobs = (jobs: Job[]): Job[] => jobs.filter((j) => j.node && j.status === "running" && !j.owner);
export type ToggleKey = "idle_only" | "pause_on_battery" | "keep_awake" | "chat_follows_best";

export function minutes(seconds: number | null | undefined): string {
  return seconds == null ? "" : `about ${Math.max(1, Math.round(seconds / 60))} min`;
}

/** Hours until everything running or waiting is done; null unless every job has an estimate. */
export function totalHours(state: Queue): number | null {
  const jobs = [...(state.job ? [state.job] : []), ...state.up_next];
  const etas = jobs.map((j) => (j.id === state.job?.id ? state.eta_s : j.eta_s));
  return jobs.length === 0 || etas.some((e) => e == null) ? null : (etas as number[]).reduce((a, b) => a + b, 0) / 3600;
}

export function allDoneLine(state: Queue): string {
  const hours = totalHours(state);
  if (hours != null) return fill(T.allDoneIn, { hours: hours.toFixed(1) });
  return state.up_next.length ? T.finishNeedsEstimates : "";
}

export function stateLine(s: Queue): string {
  const state = s.paused ? "paused" : s.blocked ? `waiting: ${s.blocked}` : "running";
  return `${s.job?.label || s.job?.kind || ""} · ${state}`;
}

export function progressLine(s: Queue): string {
  const lr = s.lr == null ? "auto" : `auto (${Number(s.lr.toPrecision(2))})`;
  const parts = [
    s.total ? `step ${s.step.toLocaleString("en-US")} of ${s.total.toLocaleString("en-US")}` : "starting",
    s.phase ? `schedule: ${s.phase}` : "",
    `learning rate: ${lr}`,
    s.eta_s != null ? `${minutes(s.eta_s)} left` : "",
  ];
  return parts.filter(Boolean).join(" · ");
}

/** (enabled, note) for a plan toggle. Idle detection the computer cannot do greys "Only when I'm not using the computer". */
export function toggleState(key: string, note: string, supported: boolean): { enabled: boolean; note: string } {
  return key === "idle_only" && !supported ? { enabled: false, note: T.idleUnsupported } : { enabled: true, note };
}

/** The place a dragged row lands: dropped on row `target`, it takes that row's position. */
export function dropIndex(target: number, length: number): number {
  return Math.max(0, Math.min(target, length - 1));
}

export function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const body = JSON.parse(error.body) as { detail?: unknown };
      if (typeof body.detail === "string") return body.detail;
    } catch {
      // not JSON: fall through to the status line
    }
    return `${error.message}${error.body ? `: ${error.body}` : ""}`;
  }
  return error instanceof Error ? error.message : String(error);
}

/** A whole hour 0-23 from what was typed, or null. */
export function parseHour(text: string): number | null {
  if (!/^\d{1,2}$/.test(text.trim())) return null;
  const n = Number(text);
  return n >= 0 && n <= 23 ? n : null;
}

export function resumesText(resumesAt: string | null | undefined): string {
  if (!resumesAt) return "";
  const when = new Date(resumesAt);
  return Number.isNaN(when.getTime()) ? "" : fill(T.blockedAt, { when: when.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) });
}

/** The machine-related row for job `id` in the merged queue, if the cluster has one. */
export function rowFor(cluster: ClusterQueue | undefined, id: string): QueueRow | undefined {
  return cluster?.jobs.find((row) => row.id === id);
}

/** Where a job runs, or where it will: its own words, for the line under its name. */
export function placeText(row: QueueRow): string {
  if (row.status === "running") return fill(T.runningOn, { node: row.hostname || row.node });
  return row.pinned ? fill(T.pinnedTo, { node: row.hostname || row.pinned }) : T.runsAnywhere;
}

/** What a machine is called in the "Run on" list: its name (with its address when another listed machine shares it), marked when it is this machine or not answering. */
export function nodeLabel(node: QueueNode, nodes: QueueNode[]): string {
  const name = nodeName(node, nodes);
  return node.local ? `${name} (${T.thisMachine})` : node.online ? name : `${name} (${T.offline})`;
}

/** The `pin` of the machine a job is pinned to, as the "Run on" list spells it ("" for any machine; the raw pin when no machine answers to it, or a name two machines share). */
export function pinChoice(nodes: QueueNode[], pinned: string): string {
  const lower = pinned.trim().toLowerCase();
  const byAddress = nodes.find((n) => n.pin.toLowerCase() === lower || n.address.toLowerCase() === lower);
  if (byAddress) return byAddress.pin;
  const byName = nodes.filter((n) => n.hostname.toLowerCase() === lower);
  return byName.length === 1 ? byName[0].pin : pinned;
}

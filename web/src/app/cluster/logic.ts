// The Cluster page's rules, apart from the page so a test can check them without a browser.
import type { components } from "@/lib/api-schema";
import { CLUSTER as T } from "@/lib/copy/cluster";
import { seenText, type ClusterQueue, type QueueRow } from "@/lib/cluster";

export type ClusterNode = components["schemas"]["ClusterNode"];
export type Queue = components["schemas"]["QueueStateModel"];

/** What a machine computes on: "NVIDIA RTX 4090 (cuda, 24 GB)", or "" for a machine that did not answer. */
export function deviceText(node: ClusterNode): string {
  if (!node.device) return "";
  const gb = node.memory_bytes > 0 ? `, ${Math.round(node.memory_bytes / 2 ** 30)} GB` : "";
  return node.device_name ? `${node.device_name} (${node.device}${gb})` : `${node.device}${gb}`;
}

/** Whether a machine refused this one (studio_cluster.failure words a peer's 401 as "<address> answered 401 to GET /node: ..."). */
export function refusedUs(node: ClusterNode): boolean {
  return !node.online && /\banswered 401\b/.test(node.error);
}

/** The list as typed: trimmed, blank and repeated entries dropped, in order (studio_cluster.clean_nodes does the same). */
export function cleanList(addresses: string[]): string[] {
  return addresses.map((address) => address.trim()).filter((address, i, all) => address && all.indexOf(address) === i);
}

/** Whether two lists differ in their entries or order. */
export function changed(a: string[], b: string[]): boolean {
  return a.length !== b.length || a.some((address, i) => address !== b[i]);
}

/**
 * Every job of every machine, as rows. The merged queue has them once other machines are listed (or a job still runs on a removed
 * one); for a cluster of one it is empty, so this machine's own queue becomes the rows.
 */
export function jobRows(merged: ClusterQueue | undefined, queue: Queue | undefined): QueueRow[] {
  if (merged && (merged.nodes.length > 0 || merged.jobs.length > 0)) return merged.jobs;
  if (!queue) return [];
  const mine = (job: NonNullable<Queue["job"]>, status: QueueRow["status"]): QueueRow => ({
    id: job.id, kind: job.kind, label: job.label, status, node: "local", hostname: "", queued_on: "", queued_here: true, queued_address: "local",
    pinned: "", waiting_reason: "", stale: false, note: job.note, step: 0, total: 0, eta_s: null,
  });
  return [
    ...(queue.job ? [{ ...mine(queue.job, "running"), step: queue.step, total: queue.total, eta_s: queue.eta_s ?? null }] : []),
    ...queue.up_next.map((job) => mine(job, job.status === "held" ? "held" : "waiting")),
  ];
}

/** What one machine is doing: the job it runs, and the jobs in its own queue that are waiting (including held ones). */
export function machineJobs(address: string, rows: QueueRow[]): { running: QueueRow[]; waiting: QueueRow[] } {
  return {
    running: rows.filter((row) => row.status === "running" && row.node === address),
    waiting: rows.filter((row) => row.status !== "running" && row.queued_address === address),
  };
}

/** A running job's progress in words: "Step 40 of 100 · about 2 min left", or "Starting" before its machine reports a total. */
export function progressText(row: QueueRow): string {
  const left = row.eta_s != null ? T.left.replace("{time}", `about ${Math.max(1, Math.round(row.eta_s / 60))} min`) : "";
  const steps = row.total > 0 ? T.stepOf.replace("{step}", row.step.toLocaleString("en-US")).replace("{total}", row.total.toLocaleString("en-US")) : T.starting;
  return [steps, left].filter(Boolean).join(" · ");
}

/** How far along a running job is, 0 to 1. */
export function fraction(row: QueueRow): number {
  return row.total > 0 ? Math.min(1, Math.max(0, row.step / row.total)) : 0;
}

/** Where a waiting job will run, in words. */
export function waitingPlace(row: QueueRow, nodeNames: Record<string, string>): string {
  return row.pinned ? T.pinnedTo.replace("{node}", row.hostname || nodeNames[row.node] || row.pinned) : T.runsAnywhere;
}

/** When a machine last answered: "Last answered 3:41 PM", or that it has not. This machine is always answering, so it says nothing. */
export function lastSeen(node: ClusterNode, seenAt: number): string {
  if (node.local) return "";
  const when = seenText(seenAt);
  return when ? T.lastAnswered.replace("{when}", when) : node.online ? "" : T.neverAnswered;
}

/** A job that has a page of its own on this computer: a training run that was queued here (its events arrive here whatever machine runs it). */
export function runPage(row: QueueRow): string | null {
  return row.status === "running" && row.queued_here && row.kind === "pretrain" ? "/training-run" : null;
}

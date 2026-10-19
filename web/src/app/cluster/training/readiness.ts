// Whether each machine chosen for a run is ready to join it now, apart from the page so a test can check the rules.
import type { ClusterQueue, QueueRow } from "@/lib/cluster";
import { JOIN } from "@/lib/copy/join";
import { etaText, fill, pasteAddress } from "@/lib/members";
import type { ClusterNode, StartRun } from "./logic";

/** One chosen machine: free (ready), busy with a job (and which, and for how long), or not answering (and why, and the address to fix). */
export type Readiness = {
  address: string;
  name: string;
  state: "ready" | "busy" | "unreachable";
  /** The job it is busy with, "" when it is busy but the queue does not say which. */
  label: string;
  eta_s: number | null;
  /** Why an unreachable machine did not answer. */
  why: string;
  /** The https:// address to paste into a Machines list. */
  paste: string;
};

function runningOn(node: ClusterNode, cluster: ClusterQueue | undefined): QueueRow | undefined {
  return cluster?.jobs.find((row) => row.status === "running" && (row.node === node.address || (node.hostname !== "" && row.hostname === node.hostname)));
}

/** The readiness of the chosen machines, in the order they were listed. */
export function readiness(nodes: ClusterNode[], chosen: string[], cluster: ClusterQueue | undefined): Readiness[] {
  return nodes
    .filter((node) => chosen.includes(node.address))
    .map((node) => {
      const name = node.hostname || node.address;
      const row = runningOn(node, cluster);
      const base = { address: node.address, name, label: "", eta_s: null, why: node.error, paste: pasteAddress(node.address) };
      if (!node.online) return { ...base, state: "unreachable" as const };
      if (node.busy || row) return { ...base, state: "busy" as const, label: row?.label ?? "", eta_s: row?.eta_s ?? null };
      return { ...base, state: "ready" as const };
    });
}

/** Every chosen machine is ready (or none is chosen): the Start button is then plain. */
export function allReady(items: Readiness[]): boolean {
  return items.every((item) => item.state === "ready");
}

/** "BEAST is training 'Version 1', about 12 min left." */
export function busyText(item: Readiness): string {
  return item.label ? fill(JOIN.busyText, { node: item.name, label: item.label, eta: etaText(item.eta_s) }) : fill(JOIN.busyNoLabel, { node: item.name });
}

export function unreachableText(item: Readiness): string {
  return item.why ? fill(JOIN.unreachableText, { node: item.name, why: item.why }) : fill(JOIN.unreachableNoWhy, { node: item.name });
}

/** "Wait for all": the run holds until this computer and every chosen machine that answers are free at once. An unreachable machine is not waited for. */
export function waitForAll(body: StartRun, items: Readiness[]): StartRun {
  const answering = items.filter((item) => item.state !== "unreachable").length;
  const most = body.max_members ?? 64;
  const wanted = Math.min(answering + 1, most);
  return { ...body, min_members: Math.max(body.min_members ?? 1, wanted) };
}

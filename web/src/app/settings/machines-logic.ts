// The Machines section's rules, apart from the page so a test can check them without a browser.
import type { components } from "@/lib/api-schema";

export type ClusterNode = components["schemas"]["ClusterNode"];

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

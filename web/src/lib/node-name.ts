import type { components } from "@/lib/api-schema";

type Machine = Pick<components["schemas"]["QueueNode"], "address" | "hostname" | "local">;

/**
 * What a machine is called on a page: its name, else the address it was listed by. When another listed machine has the same name
 * (any letter case), the address is added (a job is pinned to the address). This machine keeps its bare name: its address is only
 * "local". The same rule as `_names` in studio_cluster_queue.py; tests/fixtures/node-labels.json holds the cases both must meet.
 */
export function nodeName(node: Machine, nodes: Machine[]): string {
  const name = node.hostname || node.address;
  const shared = node.hostname && !node.local && nodes.some((n) => n !== node && n.hostname.toLowerCase() === node.hostname.toLowerCase());
  return shared ? `${name} (${node.address})` : name;
}

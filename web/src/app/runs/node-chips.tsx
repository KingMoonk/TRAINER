import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import type { ClusterNode } from "@/app/cluster/training/logic";
import { nodeName } from "@/lib/node-name";
import { cn } from "@/lib/utils";
import { T, fill } from "./copy";
import { Hint } from "./hint";
import { gb, rolesOn, type RunRow } from "./logic";

/** The sentence a chip's tooltip carries: whether it answers, its role in each active run, and its memory. */
function nodeTip(node: ClusterNode, roles: ReturnType<typeof rolesOn>): string {
  const lines = [node.online ? T.nodeOnline : `${T.nodeOffline}${node.error ? ` ${node.error}` : ""}`];
  if (roles.length === 0) lines.push(T.nodeIdle);
  for (const r of roles) {
    const words = r.roles.split(", ").map((x) => (x === T.coordinatorRole ? T.nodeCoordinator.toLowerCase() : T.nodeWorker.toLowerCase()));
    lines.push(fill(T.nodeIn, { roles: words.join(" and "), run: r.run }));
  }
  if (node.memory_bytes > 0) lines.push(fill(node.device === "cuda" ? T.nodeMemoryVideo : T.nodeMemoryUnified, { gb: gb(node.memory_bytes) }));
  return lines.join(" ");
}

/** One chip per machine: name · device · its role in each active run. Utilisation is not in the node survey, so it is not shown. */
export function NodeChips({ nodes, active }: { nodes: ClusterNode[]; active: RunRow[] }) {
  return (
    <div className="flex flex-wrap items-center gap-2" aria-label={T.machines}>
      {nodes.map((node) => {
        const name = nodeName(node, nodes);
        const roles = rolesOn(name, active);
        return (
          <Hint key={node.address} tip={nodeTip(node, roles)} plain>
            <Badge variant="outline" data-testid={`chip-${node.address}`} className={cn("h-7 px-3 text-sm font-normal", !node.online && "opacity-60")}>
              <span aria-hidden className={cn("size-2 rounded-full", node.online ? "bg-success" : "bg-muted-foreground")} />
              {[name, node.device, ...roles.map((r) => `${r.roles} ${r.run}`)].filter(Boolean).join(" · ")}
            </Badge>
          </Hint>
        );
      })}
      <Link href="/cluster/machines" className="text-sm text-highlight underline-offset-2 hover:underline">
        {T.machinesLink}
      </Link>
    </div>
  );
}

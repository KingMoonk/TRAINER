import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { T } from "./copy";
import { Hint } from "./hint";

/** A machine in a run: its name and its role, coordinator in the warning colour and worker in the info colour. */
export function RoleChip({ name, coordinator, worker }: { name: string; coordinator: boolean; worker: boolean }) {
  const roles = [coordinator ? T.coordinatorRole : "", worker ? T.workerRole : ""].filter(Boolean).join(", ");
  const tip = coordinator && worker ? T.coordinatorTip : coordinator ? T.nodeCoordinator : T.workerTip;
  return (
    <Hint tip={tip} plain>
      <Badge
        variant="outline"
        className={cn("h-6 px-2.5 text-sm font-normal", coordinator ? "border-warning/50 bg-warning/10 text-warning" : "border-info/50 bg-info/10 text-info")}
      >
        {name} · {roles}
      </Badge>
    </Hint>
  );
}

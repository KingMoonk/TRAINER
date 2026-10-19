// The limits of each machine chosen for a run, read from its queue plan: what a member job on it may use and when it pauses.
import type { ClusterQueue, QueueLimits } from "@/lib/cluster";
import { JOIN } from "@/lib/copy/join";
import { fill } from "@/lib/members";
import type { Readiness } from "./readiness";

/** The plan as plain sentences: memory, sleep, battery, idle time and hours, then what is holding the machine right now. */
export function limitLines(limits: QueueLimits): string[] {
  const { plan } = limits;
  const hours =
    plan.hours === "nights"
      ? JOIN.limitsHoursNights
      : plan.hours === "set"
        ? fill(JOIN.limitsHoursSet, { start: String(plan.start_hour).padStart(2, "0"), end: String(plan.end_hour).padStart(2, "0") })
        : JOIN.limitsHoursAny;
  const lines = [
    JOIN.limitsMemory[plan.memory ?? "most"],
    plan.keep_awake ? JOIN.limitsAwake : JOIN.limitsMaySleep,
    plan.pause_on_battery ? JOIN.limitsBattery : JOIN.limitsOnBattery,
    plan.idle_only ? JOIN.limitsIdle : JOIN.limitsAnytime,
    hours,
  ];
  if (limits.blocked) lines.push(fill(JOIN.limitsHeld, { why: limits.blocked }));
  else if (limits.paused) lines.push(JOIN.limitsPaused);
  return lines;
}

/** Where to change a machine's limits: its own Queue page, which its address serves. */
export function queuePage(paste: string): string {
  return `${paste}/queue/`;
}

export function Limits({ items, cluster }: { items: Readiness[]; cluster: ClusterQueue | undefined }) {
  if (items.length === 0) return null;
  return (
    <section aria-label={JOIN.limitsTitle} className="flex flex-col gap-2" data-testid="limits">
      <h3 className="text-sm font-medium">{JOIN.limitsTitle}</h3>
      <p className="text-xs text-muted-foreground">{JOIN.limitsLead}</p>
      <ul className="flex flex-col gap-2">
        {items.map((item) => {
          const limits = cluster?.nodes.find((n) => n.address === item.address)?.limits;
          return (
            <li key={item.address} className="flex flex-col gap-0.5 text-sm" data-testid={`limits-${item.address}`}>
              <span className="font-medium">{fill(JOIN.limitsFor, { node: item.name })}</span>
              {limits ? (
                <ul className="list-disc pl-5 text-xs text-muted-foreground">
                  {limitLines(limits).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              ) : (
                <span className="text-xs text-muted-foreground">{fill(JOIN.limitsUnknown, { node: item.name })}</span>
              )}
              <a href={queuePage(item.paste)} target="_blank" rel="noreferrer" className="w-fit text-xs underline underline-offset-2">
                {fill(JOIN.limitsChange, { node: item.name })}
              </a>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

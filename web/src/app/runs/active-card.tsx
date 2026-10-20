import Link from "next/link";
import type { ReactNode } from "react";
import { useJoinRun } from "@/app/cluster/training/use-training";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { T, fill } from "./copy";
import { Hint } from "./hint";
import { etaText, lossText, progressPercent, titleText, type RunRow } from "./logic";
import { RoleChip } from "./role-chip";

/** The machines of a run as chips: a coordinator that is also a worker is one chip "c, w". */
function Machines({ row }: { row: RunRow }) {
  const names = [...new Set([row.coordinator, ...row.workers].filter(Boolean))];
  return (
    <div className="flex flex-wrap gap-2">
      {names.map((name) => (
        <RoleChip key={name} name={name} coordinator={row.coordinator === name} worker={row.workers.includes(name)} />
      ))}
    </div>
  );
}

function Numbers({ row }: { row: RunRow }) {
  const parts: ReactNode[] = [];
  if (row.trainLoss != null) parts.push(<Hint key="train" tip={T.trainLossTip}>{fill(T.trainLoss, { value: lossText(row.trainLoss) })}</Hint>);
  if (row.valLoss != null) parts.push(<Hint key="val" tip={T.valLossTip}>{fill(T.valLoss, { value: lossText(row.valLoss) })}</Hint>);
  if (row.etaS != null) parts.push(<Hint key="eta" tip={T.etaTip}>{fill(T.eta, { time: etaText(row.etaS) })}</Hint>);
  return (
    <span className="text-sm text-muted-foreground">
      {parts.flatMap((part, i) => (i ? [" · ", part] : [part]))}
    </span>
  );
}

/** Join a running run another computer coordinates, from its next round. Beside the card's link, not in it. */
function Join({ row, title }: { row: RunRow; title: string }) {
  const join = useJoinRun();
  return (
    <div className="flex justify-end">
      <Hint tip={fill(T.joinTip, { coordinator: row.coordinator })}>
        <Button size="sm" aria-label={`${T.join}: ${title}`} disabled={join.isPending} onClick={() => join.mutate(row.id)}>
          {T.join}
        </Button>
      </Hint>
    </div>
  );
}

export function ActiveCard({ row }: { row: RunRow }) {
  const { version, name } = titleText(row);
  const title = [version, name].filter(Boolean).join(" · ");
  const percent = progressPercent(row);
  const progress = row.distributed
    ? row.rounds
      ? fill(T.round, { round: row.round, rounds: row.rounds })
      : T.starting
    : row.total
      ? fill(T.step, { step: row.step.toLocaleString("en-US"), total: row.total.toLocaleString("en-US") })
      : T.starting;
  const card = (
    <Link href={`/runs/view/?id=${encodeURIComponent(row.id)}`} data-testid={`active-${row.id}`} className="block rounded-xl outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
      <Card className="transition-colors hover:bg-muted/40">
        <CardContent className="flex flex-col gap-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <span className="font-semibold">{title}</span>
            <span className="text-sm text-muted-foreground">
              {row.kindText} · {progress}
              {row.distributed && row.innerSteps > 0 && (
                <>
                  {" · "}
                  <Hint tip={T.innerStepsTip}>{fill(T.innerSteps, { steps: row.innerSteps })}</Hint>
                </>
              )}
              {row.forOwner && (
                <>
                  {" · "}
                  <Hint tip={fill(T.forOwnerTip, { owner: row.forOwner })}>{fill(T.forOwner, { owner: row.forOwner })}</Hint>
                </>
              )}
            </span>
          </div>
          {row.state === "waiting" && <p className="text-sm text-muted-foreground">{row.note || T.waitingForMembers}</p>}
          <Machines row={row} />
          <div className="flex flex-wrap items-center gap-3">
            <div
              role="progressbar"
              aria-label={name}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(percent)}
              className="h-2 min-w-32 flex-1 overflow-hidden rounded-full bg-muted"
            >
              <div className="h-full rounded-full bg-primary" style={{ width: `${percent}%` }} />
            </div>
            <Numbers row={row} />
          </div>
        </CardContent>
      </Card>
    </Link>
  );
  if (!row.canJoin) return card;
  return (
    <div className="flex flex-col gap-2">
      {card}
      <Join row={row} title={title} />
    </div>
  );
}

import type { Mark, Series } from "@/components/series-chart";
import type { Round, Run } from "@/app/cluster/training/logic";
import { TRAINING as T } from "@/lib/copy/training";

/** The paper plots quality against work. Here the outer step's accepted target tokens are the work unit when every round has them. */
export function outcomeAxis(history: Round[]): { label: string; note: string; x: number[] } {
  const measured = history.length > 0 && history.every((row) => row.accepted_tokens != null && row.accepted_tokens > 0);
  if (!measured) return {
    label: "Planned global steps",
    note: "Accepted-token counts are unavailable for at least one round. Planned steps are not the sum of every worker's steps.",
    x: history.map((row) => row.step),
  };
  let total = 0;
  return {
    label: "Accepted target tokens",
    note: "Cumulative target tokens used by outer steps, including permitted one-round-late contributions. This is training work, not wall time.",
    x: history.map((row) => (total += row.accepted_tokens ?? 0)),
  };
}

/** Quality and participation use the same work axis, so a loss change can be read alongside changing cluster membership. */
export function outcomeSeries(run: Run): { axis: ReturnType<typeof outcomeAxis>; loss: Series[]; participation: Series[]; marks: Mark[] } {
  const axis = outcomeAxis(run.history);
  const at = (pick: (row: Round) => number) => run.history.map((row, i) => ({ x: axis.x[i], y: pick(row) }));
  const loss: Series[] = [{
    id: "validation", label: "Held-out validation loss", color: "var(--chart-1)", dots: true,
    points: run.history.flatMap((row, i) => row.val_loss == null || !Number.isFinite(row.val_loss) ? [] : [{ x: axis.x[i], y: row.val_loss }]),
  }];
  const participation: Series[] = [
    { id: "invited", label: "Members in round", color: "var(--chart-2)", dash: "5 4", points: at((row) => row.members) },
    { id: "contributed", label: "Current-round contributors", color: "var(--chart-1)", points: at((row) => row.contributed) },
  ];
  if (run.history.some((row) => (row.late_contributed ?? 0) > 0)) participation.push({
    id: "late", label: "Prior-round contributions used", color: "var(--chart-4)", dots: true,
    points: run.history.flatMap((row, i) => row.late_contributed == null ? [] : [{ x: axis.x[i], y: row.late_contributed }]),
  });
  const best = run.history.find((row) => row.round === run.best_round);
  const polishColor: Record<string, string> = { accepted: "var(--chart-3)", rejected: "var(--chart-4)", skipped: "var(--muted-foreground)" };
  const marks: Mark[] = [
    ...run.history.flatMap((row, i) => [
      ...(row.pivot ? [{ id: `pivot-${row.round}`, x: axis.x[i], label: T.pivots[row.pivot] ?? T.pivotFallback, dash: "4 3" }] : []),
      ...(row.polish ? [{ id: `polish-${row.round}`, x: axis.x[i], label: T.polishKinds[row.polish], color: polishColor[row.polish], dash: "1 3" }] : []),
    ]),
    ...(best ? [{ id: "best", x: axis.x[run.history.indexOf(best)], label: "Best", color: "var(--chart-2)" }] : []),
  ];
  return { axis, loss: loss[0].points.length ? loss : [], participation, marks };
}

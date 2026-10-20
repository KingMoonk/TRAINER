import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { T, fill } from "./copy";
import { overviewCounts, setShare, type Summary } from "./logic";

/** The kind at a glance: the counts that matter, then its sets as small cards with a share bar of the kind's total. */
export function Overview({ summary, loading }: { summary: Summary | undefined; loading: boolean }) {
  if (loading) return <Skeleton className="h-24 w-full" aria-label="Reading the sets" />;
  if (!summary) return null;
  return (
    <section className="flex flex-col gap-3" data-testid="overview">
      <dl className="grid grid-cols-3 gap-2">
        {overviewCounts(summary).map((stat) => (
          <div key={stat.key} className="min-w-0 rounded-lg border px-3 py-2" data-testid={`stat-${stat.key}`}>
            <dt className="truncate text-xs text-muted-foreground">{stat.label}</dt>
            <dd className="text-lg font-semibold">{stat.value}</dd>
          </div>
        ))}
      </dl>
      {summary.sets.length === 0 ? (
        <p className="text-sm text-muted-foreground">{T.noSets}</p>
      ) : (
        <ul aria-label={T.sets} className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {summary.sets.map((set) => {
            const share = setShare(set.count, summary.count);
            return (
              <li key={set.id} className="min-w-0">
                <Card size="sm" className="h-full" data-testid={`set-${set.id}`}>
                  <CardContent className="flex flex-col gap-1.5">
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="min-w-0 truncate text-sm font-medium" title={set.name}>
                        {set.name}
                      </span>
                      <span className="shrink-0 text-sm tabular-nums">{set.count.toLocaleString("en-US")}</span>
                    </div>
                    <div
                      role="progressbar"
                      aria-label={fill(T.shareOf, { name: set.name })}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={share}
                      className="h-1.5 overflow-hidden rounded-full bg-muted"
                    >
                      <div className="h-full bg-primary" style={{ width: `${share}%` }} />
                    </div>
                  </CardContent>
                </Card>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

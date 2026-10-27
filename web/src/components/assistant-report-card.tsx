import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type Entry = { value: number | null; band: [number, number] | null; items: number; method: string };
type CardData = { suite: string; entries: Record<string, Entry>; seconds?: number; decontam?: string };
type TrendPoint = { version: number; entries: Record<string, Entry> };

const METRICS = [
  ["choice_accuracy", "Multiple-choice accuracy", true],
  ["checked_accuracy", "Checked-answer accuracy", true],
  ["heldout_bpc", "Held-back text error", false],
  ["chat_win_rate", "Checked reply win rate vs previous version", true],
] as const;

function percent(value: number, rate: boolean) {
  return rate ? `${(value * 100).toFixed(1)}%` : `${value.toFixed(3)} bits/character`;
}

function asCard(value: unknown): CardData | null {
  if (!value || typeof value !== "object" || !("entries" in value) || !("suite" in value)) return null;
  return value as CardData;
}

export function AssistantReportCard({ card, trend }: { card: unknown; trend?: unknown }) {
  const measured = asCard(card);
  if (!measured) return null;
  const points = Array.isArray(trend) ? trend as TrendPoint[] : [];

  return (
    <Card data-testid="assistant-report-card">
      <CardHeader><CardTitle>Assistant report card</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">Frozen held-back suite {measured.suite} · {measured.decontam === "checked" ? "training overlap checked" : "training overlap unknown"}{measured.seconds == null ? "" : ` · scored in ${measured.seconds.toFixed(1)} seconds`}</p>
        <div className="grid gap-3 sm:grid-cols-2">
          {METRICS.map(([key, label, rate]) => {
            const entry = measured.entries[key];
            return <div key={key} className="rounded-lg border p-3">
              <div className="text-sm font-medium">{label}</div>
              <div className="font-mono text-xl tabular-nums">{entry?.value == null ? "Not measured" : percent(entry.value, rate)}</div>
              {entry?.band && <div className="text-xs text-muted-foreground">{key === "heldout_bpc" ? "10–90% resample interval" : "95% interval"}: {percent(entry.band[0], rate)} to {percent(entry.band[1], rate)}</div>}
              <div className="text-xs text-muted-foreground">{entry?.items ?? 0} items · {entry?.method ?? "no eligible items"}</div>
            </div>;
          })}
        </div>
        {points.length > 1 && <div className="overflow-x-auto">
          <h3 className="mb-2 text-sm font-medium">Across versions on this suite</h3>
          <table className="w-full text-left text-xs" data-testid="report-card-trend">
            <thead><tr><th className="pr-3">Version</th><th className="pr-3">Choices</th><th className="pr-3">Checked answers</th><th>Held-back error</th></tr></thead>
            <tbody>{points.map((point) => <tr key={point.version}>
              <th className="py-1 pr-3 font-medium">{point.version}</th>
              {(["choice_accuracy", "checked_accuracy", "heldout_bpc"] as const).map((key) => <td key={key} className="pr-3 tabular-nums">
                {point.entries[key]?.value == null ? "—" : percent(point.entries[key].value, key !== "heldout_bpc")}
              </td>)}
            </tr>)}</tbody>
          </table>
        </div>}
      </CardContent>
    </Card>
  );
}

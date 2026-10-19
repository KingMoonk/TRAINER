// Page logic as plain functions, ported from studio_ia/pages/benchmark.py and studio_bench.chart_data.
import type { Series } from "@/components/series-chart";
import { BENCH } from "./copy";

// The jobs' results are not in the OpenAPI schema (a job answers with its id, and the result is `Any`), so the shapes of
// studio_api/routers/benchmark.py's BenchmarkOut and FewShotOut are written out here.
export type BenchmarkRow = {
  number: number;
  name?: string;
  lineage?: string;
  continues?: number | null;
  tokens?: number | null;
  stale?: boolean;
  result?: Record<string, unknown> | null;
};
/** A paired test of a version against the one before it (studio_eval.paired_gain / paired_hits): a gain counts only where `better` is true. */
export type Gain = { diff: number; better: boolean; docs?: number; items?: number; reason?: string } | null;
export type TrendPoint = {
  number: number;
  tokens?: number | null;
  bpc?: number | null;
  bpc_band?: number[] | null;
  few_shot?: number | null;
  few_shot_band?: number[] | null;
  problems?: number | null;
  dropped?: Record<string, number>;
  gain?: { bpc?: Gain; few_shot?: Gain; problems?: Gain } | null;
};
export type Lineage = { lineage: string; versions?: number[]; points?: TrendPoint[] };
export type BenchmarkOut = {
  suite: string;
  sizes?: Record<string, number>;
  dropped?: Record<string, number>;
  sources?: string[];
  rows?: BenchmarkRow[];
  lineages?: Lineage[];
};
export type FewShotOut = {
  items: number;
  tasks?: number;
  held_back?: boolean;
  chance?: number | null;
  rows?: { shots: number; used: number; correct: number; accuracy?: number | null }[];
};

/** The worked examples per question the multiple-choice chart follows (studio_bench.SHOTS[-1]). */
export const SHOTS = 4;

export type Metric = "bpc" | "few_shot" | "problems";

export const CHARTS: { metric: Metric; title: string; desc: (shots: string) => string; what: string; axis: string }[] = [
  { metric: "bpc", title: BENCH.bpcTitle, desc: () => BENCH.bpcDesc, what: BENCH.whatBpc, axis: BENCH.bpcAxis },
  { metric: "few_shot", title: BENCH.fewShotTitle, desc: BENCH.fewShotDesc, what: BENCH.whatFewShot, axis: BENCH.fewShotAxis },
  { metric: "problems", title: BENCH.problemsTitle, desc: () => BENCH.problemsDesc, what: BENCH.whatProblems, axis: BENCH.problemsAxis },
];

type SuiteResult = {
  bpc?: number | null;
  few_shot?: { accuracy?: number | null; chance?: number | null; rows?: { used?: number }[] } | null;
  problems?: { rate?: number | null } | null;
};

const suite = (row: BenchmarkRow) => (row.result ?? null) as SuiteResult | null;

export function tokensText(n: number): string {
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}K` : `${n.toFixed(0)}`;
}

/** Bits per character with two decimals; the other two checks as a share. */
export function valueText(metric: Metric, value: number): string {
  return metric === "bpc" ? value.toFixed(2) : `${Math.round(value * 100)}%`;
}

const byNumber = (rows: BenchmarkRow[]) => [...rows].sort((a, b) => a.number - b.number);

/** The trained versions' designs, lettered in the order their first version appears. */
export function designs(rows: BenchmarkRow[]): Map<string, { letter: string; numbers: number[] }> {
  const out = new Map<string, { letter: string; numbers: number[] }>();
  for (const row of byNumber(rows)) {
    const design = out.get(row.lineage ?? "") ?? { letter: String.fromCharCode(65 + out.size), numbers: [] };
    design.numbers.push(row.number);
    out.set(row.lineage ?? "", design);
  }
  return out;
}

/** Version numbers whose suite result is missing, oldest first. */
export function pending(state: BenchmarkOut | undefined): number[] {
  if (!state || !Object.values(state.sizes ?? {}).some(Boolean)) return [];
  return byNumber(state.rows ?? []).filter((r) => r.result == null).map((r) => r.number);
}

/** The worked examples per question the multiple-choice chart used: the number asked for, or "up to" it when some questions had fewer fit. */
export function shotsText(state: BenchmarkOut): string {
  const used = (state.rows ?? []).map((r) => suite(r)?.few_shot?.rows?.at(-1)?.used).filter((u): u is number => typeof u === "number");
  return used.some((u) => u < SHOTS - 1e-9) ? BENCH.upTo(SHOTS) : String(SHOTS);
}

export function hasItems(state: BenchmarkOut, metric: Metric): boolean {
  const sizes = state.sizes ?? {};
  return Boolean(sizes[metric === "bpc" ? "docs" : metric === "few_shot" ? "choice" : "problems"]);
}

/** One line under the header: how many trained versions still need checking (and how many of those were checked under an older method), or that none do. */
export function statusText(state: BenchmarkOut): string {
  const rows = state.rows ?? [];
  if (!rows.length) return BENCH.noVersions;
  if (!Object.values(state.sizes ?? {}).some(Boolean)) return BENCH.noData;
  const left = rows.filter((r) => r.result == null).length;
  const old = rows.filter((r) => r.result == null && r.stale).length;
  return left ? BENCH.waiting(left, rows.length) + (old ? BENCH.waitingStale(old) : "") : BENCH.allDone;
}

/** A value with the range it could be given the items asked: "2.20 (2.15 to 2.25)", or just the value without a range. */
export function bandText(metric: Metric, value: number | null | undefined, band: number[] | null | undefined): string {
  if (typeof value !== "number") return "–";
  const range = band && band.length === 2 && band.every((b) => Number.isFinite(b)) ? ` (${valueText(metric, band[0])} to ${valueText(metric, band[1])})` : "";
  return valueText(metric, value) + range;
}

/** What the paired test says about a version against the one before it: a real gain, or not sure. "–" for the first of a lineage. */
export function gainText(metric: Metric, gain: Gain | undefined): string {
  if (gain === undefined || gain === null) return "–";
  if (Number.isNaN(gain.diff)) return BENCH.gainNone;
  const items = gain.docs ?? gain.items ?? 0;
  const sign = gain.diff > 0 ? "+" : "";
  const change = metric === "bpc" ? `${sign}${gain.diff.toFixed(2)}` : `${sign}${Math.round(gain.diff * 100)} points`;
  if (gain.reason) return BENCH.gainFew(change, items);
  return gain.better ? BENCH.gainBetter(change) : BENCH.gainUnsure(change);
}

/** How many items were left out of one version's score for sharing a 13-word run with its training text. */
export function leftOut(dropped: Record<string, number> | undefined): string {
  return String(Object.values(dropped ?? {}).reduce((a, b) => a + b, 0));
}

/** The trend tables, one per lineage: a row per version with its numbers, their ranges, what the paired tests say against the version before, and what was left out. */
export function trendTables(state: BenchmarkOut): { lineage: string; title: string; rows: string[][] }[] {
  const names = designs(state.rows ?? []);
  const lineages = state.lineages ?? [];
  return lineages.map((line, index) => {
    const points = line.points ?? [];
    return {
      lineage: line.lineage,
      title: BENCH.design(names.get(line.lineage)?.letter ?? String.fromCharCode(65 + index), line.versions ?? points.map((p) => p.number)),
      rows: points.map((p) => [
        String(p.number),
        p.tokens == null ? "–" : tokensText(p.tokens),
        bandText("bpc", p.bpc, p.bpc_band),
        gainText("bpc", p.gain?.bpc),
        bandText("few_shot", p.few_shot, p.few_shot_band),
        gainText("few_shot", p.gain?.few_shot),
        leftOut(p.dropped),
      ]),
    };
  });
}

function metricValue(row: BenchmarkRow, metric: Metric): number | null {
  const result = suite(row);
  const got = metric === "bpc" ? result?.bpc : metric === "few_shot" ? result?.few_shot?.accuracy : result?.problems?.rate;
  return typeof got === "number" ? got : null;
}

/** Points for one check, a line per design with a point per version: x is tokens trained on. Versions without a token count are unplaced. */
export function chartFor(state: BenchmarkOut, metric: Metric): { series: Series[]; chance: number | null; unplaced: number[] } {
  const names = designs(state.rows ?? []);
  const lines = new Map<string, { x: number; y: number }[]>();
  const unplaced: number[] = [];
  let chance: number | null = null;
  for (const row of byNumber(state.rows ?? [])) {
    const value = metricValue(row, metric);
    if (value === null) continue;
    if (row.tokens == null) {
      unplaced.push(row.number);
      continue;
    }
    lines.set(row.lineage ?? "", [...(lines.get(row.lineage ?? "") ?? []), { x: row.tokens, y: value }]);
    if (metric === "few_shot" && chance === null) chance = suite(row)?.few_shot?.chance ?? null;
  }
  const series: Series[] = [...lines].map(([lineage, points]) => {
    const design = names.get(lineage)!;
    return { id: `design-${design.letter}`, label: BENCH.design(design.letter, design.numbers), points: points.sort((a, b) => a.x - b.x) };
  });
  if (series.length && chance !== null) {
    const last = Math.max(...series.flatMap((s) => s.points.map((p) => p.x)));
    series.push({ id: "chance", label: BENCH.chance(valueText(metric, chance)), points: [{ x: 0, y: chance }, { x: last, y: chance }] });
  }
  return { series, chance, unplaced };
}

/** How a count of right answers compares with a guesser's: better | about the same | worse, outside two standard deviations. */
export function againstChance(correct: number, items: number, chance: number): string {
  const spread = 2 * Math.sqrt(items * chance * (1 - chance));
  return correct > items * chance + spread ? BENCH.shotsAbove : correct < items * chance - spread ? BENCH.shotsBelow : BENCH.shotsNear;
}

/** A row per number of worked examples: examples shown, right, score, guessing gets, against guessing. [] when nothing was asked. */
export function fewShotRows(table: FewShotOut | undefined): string[][] {
  if (!table || !table.items || table.chance == null) return [];
  const chance = table.chance;
  return (table.rows ?? []).map((r) => {
    const shown = (r.shots === 0 ? BENCH.shotsZero : r.shots === 1 ? BENCH.shotsOne : String(r.shots)) + (r.used < r.shots ? BENCH.shotsFitted(r.used.toFixed(1)) : "");
    return [
      shown,
      `${r.correct} of ${table.items}`,
      r.accuracy == null ? "–" : `${Math.round(r.accuracy * 100)}%`,
      `${Math.round(chance * 100)}%`,
      againstChance(r.correct, table.items, chance),
    ];
  });
}

/** The plain lines under the table: what was asked, where it came from, what guessing scores. */
export function fewShotNote(table: FewShotOut): string {
  const lines = [BENCH.shotsAsked(table.items, table.tasks ?? 1)];
  if (table.held_back === false) lines.push(BENCH.shotsTraining);
  return [...lines, BENCH.shotsChance(((table.chance ?? 0) * 100).toFixed(0))].join(" ");
}

export const versionName = (row: BenchmarkRow) => row.name || `Version ${row.number}`;

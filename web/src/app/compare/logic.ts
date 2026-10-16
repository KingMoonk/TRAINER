// Page logic as plain functions, ported from studio_ia/pages/compare.py.
import type { Series } from "@/components/series-chart";
import { COMPARE } from "./copy";

/** A point of a run's curve: [step, bpc, trained tokens?, elapsed seconds?]. */
export type CurvePoint = (number | null)[];

/** The summary of one variant (studio_experiments.summarize); the base is variant 0. */
export type VariantStats = {
  variant: number;
  bpc: number[];
  mean: number;
  min: number;
  max: number;
  spread: number;
  delta?: number | null;
  inside_spread?: boolean | null;
  pieces_per_s: number;
  peak_mb: number;
  curve: CurvePoint[];
};

// A finished run is the engine's saved results.json; the API passes it through untyped (the router's CompareRunOut keeps extra keys).
export type RunResults = { variants?: VariantStats[]; references?: Record<string, number>; cancelled?: boolean };
export type CompareRun = {
  base?: unknown;
  variants?: unknown[];
  mode?: string | null;
  names?: string[];
  question?: string;
  guess?: string;
  results?: RunResults;
  finished?: number | string | null;
};

export type AxisId = (typeof COMPARE.axes)[number]["id"];
const AXIS_COLUMN: Record<AxisId, { col: number; div: number }> = { steps: { col: 0, div: 1 }, tokens: { col: 2, div: 1 }, time: { col: 3, div: 60 } };

/** [(x, bpc)] on the chosen axis, or null when any point lacks it. */
export function axisPoints(points: CurvePoint[], axis: AxisId): { x: number; y: number }[] | null {
  const { col, div } = AXIS_COLUMN[axis];
  if (points.some((p) => p.length <= col || p[col] == null || p[1] == null)) return null;
  return points.map((p) => ({ x: (p[col] as number) / div, y: p[1] as number }));
}

/** The axis actually used (the asked-for one when every curve has it, else steps) and the axis that was missing. */
export function axisFor(curves: CurvePoint[][], axis: AxisId): { axis: AxisId; missing: AxisId | null } {
  return axis !== "steps" && curves.some((c) => axisPoints(c, axis) === null) ? { axis: "steps", missing: axis } : { axis, missing: null };
}

export const letter = (variant: number) => String.fromCharCode(65 + variant);

/** The name of variant n in a run: its saved name, else Base / Variant n. */
export const variantName = (names: string[] | undefined, n: number) => names?.[n] || (n === 0 ? "Base" : `Variant ${n}`);

/** One series per variant, on the chosen axis; plus the reference levels the engine measured, as flat lines across the chart. */
export function curveSeries(
  curves: Map<number, CurvePoint[]>,
  names: string[] | undefined,
  axis: AxisId,
  references: Record<string, number> | undefined,
): Series[] {
  const series: Series[] = [];
  for (const [n, points] of [...curves].sort((a, b) => a[0] - b[0])) {
    const placed = axisPoints(points, axis);
    if (placed) series.push({ id: `variant-${letter(n)}`, label: `${letter(n)} · ${variantName(names, n)}`, points: placed });
  }
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  if (xs.length) {
    const [lo, hi] = [Math.min(...xs), Math.max(...xs)];
    for (const [key, value] of Object.entries(references ?? {})) {
      if (typeof value === "number") series.push({ id: `reference-${key}`, label: COMPARE.referenceNames[key] ?? key, points: [{ x: lo, y: value }, { x: hi, y: value }] });
    }
  }
  return series;
}

/** "mean · spread x" (or one seed) and, under it, speed and peak memory; a dash before results. */
export function finalCell(s: VariantStats | undefined): string {
  if (!s) return "–";
  const head = s.bpc.length > 1 ? `${s.mean.toFixed(3)} · ${COMPARE.spread} ${s.spread.toFixed(3)}` : `${s.mean.toFixed(3)} · ${COMPARE.oneSeed}`;
  return `${head}\n${Math.round(s.pieces_per_s).toLocaleString("en-US")} pieces/s · ${Math.round(s.peak_mb)} MB peak`;
}

/** Difference from the base; it counts only when larger than the spread across seeds. */
export function deltaCell(s: VariantStats | undefined): string {
  if (!s) return "–";
  if (s.delta == null) return "—";
  const note = s.inside_spread === true ? COMPARE.insideSpread : s.inside_spread === false ? COMPARE.beyondSpread : COMPARE.oneSeedNoSpread;
  return `${s.delta >= 0 ? "+" : "-"}${Math.abs(s.delta).toFixed(3)} · ${note}`;
}

/** better | about the same | worse, or null when no verdict is possible (a difference counts only beyond the spread). */
export function variantOutcome(row: VariantStats | undefined): string | null {
  if (!row || row.delta == null || row.inside_spread == null) return null;
  return row.inside_spread ? "about the same" : row.delta < 0 ? "better" : "worse";
}

/** What happened to the guessed variant next to the guess. */
export function guessLine(guess: string, name: string, row: VariantStats | undefined): string {
  const outcome = variantOutcome(row);
  if (outcome === null) return `You guessed ${guess.toLowerCase()}. No verdict on ${name}: ${row ? "it needs at least two seeds to have a spread." : "it has no result yet."}`;
  return `You guessed ${guess.toLowerCase()}; ${name} came out ${outcome}${outcome === "about the same" ? " (inside the spread)." : " (beyond the spread)."}`;
}

/** Why the runs cannot be queued yet, in words; "" when they can. */
export function queueBlock(variants: number, guess: string): string {
  return variants === 0 ? COMPARE.needVariant : guess === "" ? COMPARE.needGuess : "";
}

const million = (n: number) => `${(n / 1e6).toFixed(2)}M`;

type PlanVariant = { variant: number; adjustments?: unknown[]; steps?: number | null; parameters: number; relative_error?: number | null };

/** The "Matching" cell of a planned variant. */
export function matchingText(row: PlanVariant, baseSteps: number | null | undefined, mode: string, wallSeconds: number, retune: boolean): string {
  if (!row.variant) return "—";
  let text: string;
  if (mode === "params") {
    text = (row.adjustments?.length ? COMPARE.widthResolved : COMPARE.noneNeeded)(million(row.parameters)) + ` · within ${((row.relative_error ?? 0) * 100).toFixed(1)}%`;
  } else if (mode === "compute") {
    const more = row.steps && baseSteps ? row.steps / baseSteps - 1 : 0;
    text = more > 0.005 ? COMPARE.stopsAtBase(`${Math.round(more * 100)}%`) : COMPARE.sameCompute;
  } else if (mode === "tokens") {
    text = COMPARE.parameters(million(row.parameters));
  } else {
    text = COMPARE.minBudget(String(+(wallSeconds / 60).toFixed(2)));
  }
  return text + (retune ? "" : COMPARE.notRetuned);
}

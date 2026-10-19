// Training speed (studio_mlperf): plain functions for the leaderboard, history and compare views.
import type { Series } from "@/components/series-chart";

export type Spread = { median: number; min: number; max: number; n: number };
export type Summary = {
  id: string;
  created_at: number;
  version: number;
  division: string;
  status: "converged" | "did_not_converge" | "failed" | "stopped";
  label: string;
  app_version?: string | null;
  time_to_target?: Spread | null;
  best_bpc?: number | null;
  tokens_per_s?: number | null;
  steps_per_s?: number | null;
  mfu?: number | null;
  peak_kind?: "datasheet" | "measured_gemm" | null;
  peak_memory_bytes?: number | null;
  energy_joules?: number | null;
  attempt_count?: number;
  target_bpc?: number;
  error?: string | null;
  relative_time?: number | null;
};
export type Board = {
  task: string;
  version: number;
  title: string;
  target_bpc: number;
  divisions: Record<string, Summary[]>;
  other_versions: { version: number; results: number }[];
};
export type HistoryPoint = { id: string; created_at: number; app_version?: string | null; label?: string; status: string; median_s?: number | null };

const STATUS: Record<string, string> = {
  converged: "Reached the target",
  did_not_converge: "Did not converge",
  failed: "Failed",
  stopped: "Stopped",
};
export const statusText = (status: string) => STATUS[status] ?? status;

export function secondsText(seconds: number | null | undefined): string {
  if (seconds == null) return "-";
  return seconds >= 100 ? `${Math.round(seconds)} s` : `${seconds.toFixed(1)} s`;
}

/** "42.0 s", with the spread over attempts when there was more than one: "42.0 s (40.1 to 44.9 s, 3 attempts)". */
export function timeText(spread: Spread | null | undefined): string {
  if (!spread) return "-";
  const base = secondsText(spread.median);
  return spread.n > 1 ? `${base} (${secondsText(spread.min).replace(" s", "")} to ${secondsText(spread.max)}, ${spread.n} attempts)` : base;
}

export const mfuText = (mfu: number | null | undefined) => (mfu == null ? "-" : `${(mfu * 100).toFixed(mfu < 0.1 ? 2 : 1)}%`);
export const rateText = (value: number | null | undefined) => (value == null ? "-" : Math.round(value).toLocaleString("en-US"));
export const memoryText = (bytes: number | null | undefined) => (bytes == null ? "-" : `${(bytes / 1024 ** 3).toFixed(2)} GB`);

/** What an MFU is a share of: the card's rating, or the best matrix multiply torch reached (a different, lower-trust yardstick). */
export function peakNote(kind: string | null | undefined): string {
  if (kind === "datasheet") return "of the datasheet peak";
  if (kind === "measured_gemm") return "of the best matrix multiply torch reached here";
  return "";
}

export const relativeText = (r: number | null | undefined) => (r == null ? "-" : r === 1 ? "fastest" : `${r.toFixed(2)}x slower`);

/** The history chart's line: median time to target per result, oldest first (results that missed the target are left out). */
export function historySeries(points: HistoryPoint[]): Series[] {
  const done = points.filter((p) => p.median_s != null);
  return done.length ? [{ id: "time", label: "Time to target (s)", dots: true, points: done.map((p, i) => ({ x: i + 1, y: p.median_s as number })) }] : [];
}

/** Labels under the history chart: the app version each converged point ran under, where it first appears or changes. */
export function versionChanges(points: HistoryPoint[]): string[] {
  const out: string[] = [];
  let last = "";
  points
    .filter((p) => p.median_s != null)
    .forEach((p, i) => {
      if (p.app_version && p.app_version !== last) out.push(`Result ${i + 1}: ${p.app_version}`);
      if (p.app_version) last = p.app_version;
    });
  return out;
}

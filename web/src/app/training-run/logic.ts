// Page logic that stays in the browser. phase_at, history_of and the switch note are served by GET /training-run (they live in
// studio_ia/pages/training_run.py); the rows under "what it writes now" are built here from copy this page owns.

import type { Band, Series } from "@/components/series-chart";
import { fill, GATE_SEEDS, SWITCH, T } from "./copy";

/** A [step, value] point; the schema types a pair as number[]. */
export type Pair = readonly number[];
export type SampleRow = { label: string; text: string };

/** (label, text) lines under "what it writes now": step 0, the logged samples, the newest marked "now"; before the first sample, a wait note. */
export function sampleRows(samples: { step: number; text: string }[], step: number | null, parent: number | null): SampleRow[] {
  const rows: SampleRow[] = [
    { label: T.step0, text: parent ? fill(T.identicalTo, { parent: `version ${parent}` }) : T.randomStart },
    ...samples.map((s) => ({ label: `step ${s.step}`, text: s.text })),
  ];
  if (samples.length) {
    rows[rows.length - 1] = { ...rows[rows.length - 1], label: fill(T.nowStep, { step: samples[samples.length - 1].step }) };
  } else {
    rows.push({ label: step !== null ? fill(T.nowStep, { step: step.toLocaleString("en-US") }) : "now", text: T.noSampleYet });
  }
  return rows;
}

export const bits = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(3));
export const last = (points: readonly Pair[]) => (points.length ? points[points.length - 1][1] : null);

/** A horizontal reference line over the x range of the curve, drawn as a two-point series. */
function level(id: string, label: string, y: number | null | undefined, from: number, to: number): Series[] {
  return y == null ? [] : [{ id, label, points: [{ x: from, y }, { x: to, y }] }];
}

/** Training error and references share the left axis; learning rate has its own scale. */
export function chartSeries(
  curve: { steps: readonly Pair[]; smooth: readonly Pair[]; heldout: readonly Pair[]; lr: readonly Pair[] },
  refs: { uniform_bpc?: number | null; unigram_bpc?: number | null; best_so_far_bpc?: number | null },
  labels: { train: string; raw: string; lr: string; held: string; uniform: string; unigram: string; best: string },
): Series[] {
  const xs = [...curve.steps, ...curve.smooth, ...curve.heldout, ...curve.lr].map((p) => p[0]);
  const from = xs.length ? Math.min(...xs) : 0;
  const to = xs.length ? Math.max(...xs) : 0;
  return [
    { id: "steps", label: labels.raw, points: curve.steps.map(([x, y]) => ({ x, y })), color: "var(--chart-1)", opacity: 0.2, strokeWidth: 1, interpolation: "linear" },
    { id: "smooth", label: labels.train, points: curve.smooth.map(([x, y]) => ({ x, y })), color: "var(--chart-1)" },
    { id: "held", label: labels.held, points: curve.heldout.map(([x, y]) => ({ x, y })), color: "var(--chart-2)" },
    ...level("uniform", labels.uniform, refs.uniform_bpc, from, to),
    ...level("unigram", labels.unigram, refs.unigram_bpc, from, to),
    ...level("best", labels.best, refs.best_so_far_bpc, from, to),
    ...(curve.lr.length ? [{ id: "lr", label: labels.lr, axis: "right" as const, points: curve.lr.map(([x, y]) => ({ x, y })),
      color: "var(--chart-5)", dash: "4 3", interpolation: "linear" as const }] : []),
  ];
}

export function phaseBands(phases: readonly (readonly (number | string)[])[]): Band[] {
  return phases.flatMap(([from, to, label], index) =>
    typeof from === "number" && typeof to === "number" && typeof label === "string" &&
    Number.isFinite(from) && Number.isFinite(to) && to > from ? [{ id: `${label}-${index}`, label, from, to }] : []);
}

export function switchText(
  note: { key: string; gap?: string | null; docs?: number | null; need?: number | null },
  version: number | null,
  parent: number | null,
  gateVs: number | null,
  running: boolean,
): string {
  const words = {
    Version: version ? `Version ${version}` : "This version",
    version: version ? `version ${version}` : "this version",
    parent: gateVs && !running ? `version ${gateVs}` : parent ? `version ${parent}` : "the earlier best",
    gap: note.gap,
    docs: note.docs,
    need: note.need,
  };
  const seeds = note.key === "gate_not_sure" || note.key === "gate_not_better" ? ` ${GATE_SEEDS}` : "";
  return fill(SWITCH[note.key] ?? "", words) + seeds;
}

export type JobParams = { [key: string]: unknown; config?: { batch_size?: number; context?: number }; pieces_per_pass?: number };

/** The one-line progress read: step k of n, pass, pieces seen, schedule phase, learning rate and time left. */
export function progressLine(q: {
  step: number;
  total: number;
  phase: string;
  lr?: number | null;
  eta_s?: number | null;
  stopping: boolean;
  blocked?: string | null;
  params: JobParams;
}): string {
  const rows = q.params.config?.batch_size ?? 16;
  const seen = q.step * rows * (q.params.config?.context ?? 256);
  const perPass = q.params.pieces_per_pass;
  const parts = [q.total ? `${T.step} ${q.step.toLocaleString("en-US")} ${T.of} ${q.total.toLocaleString("en-US")}` : T.starting];
  parts.push((perPass ? `${T.pass} ${Math.floor(seen / perPass) + 1} ${T.overText} · ` : "") + `${seen.toLocaleString("en-US")} ${T.piecesSeen}`);
  if (q.phase) parts.push(`${T.schedule} ${q.phase}`);
  parts.push(T.lrAuto.replace("auto", q.lr == null ? "auto" : `auto (${Number(q.lr.toPrecision(2))})`));
  if (q.eta_s != null) parts.push(`about ${Math.max(1, Math.round(q.eta_s / 60))} min ${T.left}`);
  if (q.stopping) parts.push(T.coolingDown);
  if (q.blocked) parts.push(`${T.waiting} ${q.blocked}`);
  return parts.join(" · ");
}

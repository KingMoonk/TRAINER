// Page logic that stays in the browser. phase_at, history_of and the switch note are served by GET /training-run (they live in
// studio_ia/pages/training_run.py); the rows under "what it writes now" are built here from copy this page owns.

import type { Series } from "@/components/series-chart";
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

/** The series the shared chart draws: smoothed training error, held-back error and the reference lines, all in bits per character. */
export function chartSeries(
  curve: { smooth: readonly Pair[]; heldout: readonly Pair[] },
  refs: { uniform_bpc?: number | null; unigram_bpc?: number | null; best_so_far_bpc?: number | null },
  labels: { train: string; held: string; uniform: string; unigram: string; best: string },
): Series[] {
  const xs = [...curve.smooth, ...curve.heldout].map((p) => p[0]);
  const from = xs.length ? Math.min(...xs) : 0;
  const to = xs.length ? Math.max(...xs) : 0;
  return [
    { id: "smooth", label: labels.train, points: curve.smooth.map(([x, y]) => ({ x, y })) },
    { id: "held", label: labels.held, points: curve.heldout.map(([x, y]) => ({ x, y })) },
    ...level("uniform", labels.uniform, refs.uniform_bpc, from, to),
    ...level("unigram", labels.unigram, refs.unigram_bpc, from, to),
    ...level("best", labels.best, refs.best_so_far_bpc, from, to),
  ];
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

// The Run view's rules, apart from the page so a test can check them without a browser.
import type { Series } from "@/components/series-chart";
import type { Round, Run } from "@/app/cluster/training/logic";
import { T, fill } from "./copy";

/** The part of GET /pretraining/jobs/{id}/curve this page reads: series of [step, value]. */
export type Curve = { steps: number[][]; smooth: number[][]; heldout: number[][] };
type Pair = number[];

/** The Run view of one run: the page reads `?id=`. */
export function runViewHref(id: string): string {
  return `/runs/view/?id=${encodeURIComponent(id)}`;
}

/** `pretrain · local · W = {BEAST} · queue job on BEAST`. */
export function soloShape(machine: string): string {
  return fill(T.shapeSolo, { machine });
}

/** `pretrain · DiLoCo · W = {2} · c = Mac · H = 100 · outer lr 0.7 · μ 0.9`. */
export function dilocoShape(run: Run): string {
  return fill(T.shapeDiloco, { count: run.members.length, coordinator: run.owner, steps: run.inner_steps, lr: run.outer_lr, momentum: run.outer_momentum });
}

/**
 * The body of POST /distributed/runs/{id}/extend for what the form holds: `add` more rounds on top of the run's, and `steps` (empty keeps
 * H). Null when it holds nothing the API would accept (not whole numbers, nothing to change).
 */
export function extendRequest(run: Pick<Run, "rounds" | "inner_steps">, add: string, steps: string): { rounds?: number; inner_steps?: number } | null {
  const more = add.trim() === "" ? 0 : Number(add);
  const h = steps.trim() === "" ? run.inner_steps : Number(steps);
  if (!Number.isInteger(more) || more < 0 || !Number.isInteger(h) || h < 1 || h > 1_000_000 || run.rounds + more > 100_000) return null;
  const body: { rounds?: number; inner_steps?: number } = {};
  if (more > 0) body.rounds = run.rounds + more;
  if (h !== run.inner_steps) body.inner_steps = h;
  return body.rounds === undefined && body.inner_steps === undefined ? null : body;
}

/** 0 to 100 for the bar, 0 when the total is not known. */
export function percentOf(done: number, all: number): number {
  return all > 0 ? Math.max(0, Math.min(100, (done / all) * 100)) : 0;
}

export function lastOf(points: Pair[]): number | null {
  return points.length ? points[points.length - 1][1] : null;
}

/** The training loss closest at or before `step`, else the first one. */
function trainAt(train: Pair[], step: number): number | null {
  let found: number | null = train.length ? train[0][1] : null;
  for (const [x, y] of train) {
    if (x > step) break;
    found = y;
  }
  return found;
}

/** The loss curve's training line: the smoothed one, else the raw one. */
export function trainLine(curve: Curve): Pair[] {
  return curve.smooth.length ? curve.smooth : curve.steps;
}

/** L_val minus L_train at each check of L_val. */
export function gaps(curve: Curve): Pair[] {
  const train = trainLine(curve);
  return curve.heldout.flatMap(([step, val]) => {
    const t = trainAt(train, step);
    return t == null ? [] : [[step, val - t] as Pair];
  });
}

export function gapNow(curve: Curve): number | null {
  return lastOf(gaps(curve));
}

/** Checks of L_val needed before the rule can say anything: three to compare against the one before them. */
export const CHECKS_NEEDED = 4;

export type Overfit = { state: "early" | "ok" | "flat" | "stalled" | "widening"; checks: number };

/**
 * The overfitting rule, computed here: it fires when L_val has not improved for 3 checks while L_train fell over them, or when the gap
 * has grown for 3 checks in a row. Before 4 checks of L_val there is not enough to say.
 */
export function overfit(curve: Curve): Overfit {
  const val = curve.heldout;
  const checks = val.length;
  if (checks < CHECKS_NEEDED) return { state: "early", checks };
  const train = trainLine(curve);
  const tail = val.slice(-CHECKS_NEEDED);
  const [before, ...recent] = tail;
  const trainBefore = trainAt(train, before[0]);
  const trainNow = trainAt(train, tail[tail.length - 1][0]);
  const notImproved = recent.every(([, v]) => v >= before[1]);
  if (notImproved && trainBefore != null && trainNow != null && trainNow < trainBefore) return { state: "stalled", checks };
  const g = gaps(curve).slice(-CHECKS_NEEDED);
  if (g.length === CHECKS_NEEDED && g.every((point, i) => i === 0 || point[1] > g[i - 1][1])) return { state: "widening", checks };
  return { state: recent[recent.length - 1][1] < before[1] ? "ok" : "flat", checks };
}

export function overfitText(result: Overfit): string {
  switch (result.state) {
    case "early": return fill(T.overfitEarly, { n: result.checks });
    case "ok": return T.overfitNo;
    case "flat": return T.overfitNoFlat;
    case "stalled": return T.overfitStalled;
    default: return T.overfitWidening;
  }
}

/** Whether the pill is a warning. */
export function overfitBad(result: Overfit): boolean {
  return result.state === "stalled" || result.state === "widening";
}

const points = (pairs: Pair[]) => pairs.map(([x, y]) => ({ x, y }));

/** Loss against step for a single job: L_train solid, L_val dashed with a dot at each check. */
export function stepSeries(curve: Curve | undefined): Series[] {
  if (!curve) return [];
  const train = points(trainLine(curve));
  const val = points(curve.heldout);
  return [
    ...(train.length ? [{ id: "train", label: T.lTrain, points: train, color: "var(--info)" }] : []),
    ...(val.length ? [{ id: "val", label: T.lVal, points: val, color: "var(--warning)", dash: "5 4", dots: true }] : []),
  ];
}

/** Loss against round for a DiLoCo run: the token-weighted mean inner loss solid, the held-out loss dashed. */
export function roundSeries(history: Round[]): Series[] {
  const line = (id: string, label: string, pick: (row: Round) => number | null, rest: Partial<Series>): Series[] => {
    const pts = history.flatMap((row) => (pick(row) == null ? [] : [{ x: row.round + 1, y: pick(row) as number }]));
    return pts.length ? [{ id, label, points: pts, ...rest }] : [];
  };
  return [
    ...line("loss", T.lInner, (row) => row.loss, { color: "var(--info)" }),
    ...line("val_loss", T.lVal, (row) => row.val_loss, { color: "var(--warning)", dash: "5 4" }),
  ];
}

/** [round, value] for each finished round that has a value. */
export function roundPairs(history: Round[], pick: (row: Round) => number | null): Pair[] {
  return history.flatMap((row) => {
    const value = pick(row);
    return value == null ? [] : [[row.round + 1, value] as Pair];
  });
}

export function lossText(value: number | null): string {
  return value == null ? "" : value.toFixed(2);
}

/** "+0.09" or "-0.02": the sign always shown. */
export function gapText(value: number | null): string {
  return value == null ? "" : `${value >= 0 ? "+" : "-"}${Math.abs(value).toFixed(2)}`;
}

/** "2:10": a time into the run from seconds, `m:ss` (or `h:mm:ss` past an hour). */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  const ss = String(s % 60).padStart(2, "0");
  return m >= 60 ? `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

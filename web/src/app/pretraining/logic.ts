import type { components } from "@/lib/api-schema";
import { T, fill } from "./copy";

export type Plan = components["schemas"]["PretrainPlan"];
export type Readout = NonNullable<Plan["readout"]>;
export type TextSet = components["schemas"]["TextSet"];
export type Preview = components["schemas"]["PreviewModel"];
export type RangeResult = components["schemas"]["RangeResult"];
export type StepRate = components["schemas"]["StepRate"];

export const ROWS = ["8", "16", "32"] as const;
export const HOLD = ["2%", "5%", "10%"] as const;
export const LENGTHS = { Short: 1, Medium: 2, Long: 4 } as const; // passes over the chosen text
export type Length = keyof typeof LENGTHS;
export const PREVIEW_ROWS = 6; // rows of the size preview the page draws
// Theme tokens (globals.css), as CSS values for an inline style.
export const PALETTE = ["var(--viz-1)", "var(--viz-2)", "var(--viz-3)", "var(--viz-4)", "var(--viz-5)"];
export const END = "var(--viz-end)";

/** The request /pretraining/plan takes for these choices; its answer carries the job to post to /queue/jobs. */
export function planBody(c: {
  version: number | null;
  datasetIds: string[];
  mode: "catch_up" | "from_scratch";
  rows: (typeof ROWS)[number];
  hold: (typeof HOLD)[number];
  length: Length;
  hours?: Hours;
  steps?: number | null; // a step count chosen from the suggestion, instead of the length
  stopWhenFlat?: StopWhenFlat;
  fromVersion?: number | null; // the earlier version a catch-up starts from (one of the plan's sources); null: the server's default
}) {
  return {
    version: c.version,
    dataset_ids: c.datasetIds,
    mode: c.mode,
    rows: Number(c.rows),
    hold: Number(c.hold.slice(0, -1)) / 100,
    passes: LENGTHS[c.length],
    preview_rows: PREVIEW_ROWS,
    hours: HOURS[c.hours ?? DEFAULT_HOURS],
    steps: c.steps ?? null,
    plateau: STOP_WHEN_FLAT[c.stopWhenFlat ?? "Auto"],
    from_version: c.fromVersion ?? null,
  };
}
export type PlanBody = ReturnType<typeof planBody>;

/** End the run when its validation loss goes flat: by length (runs over an hour), always, or never. */
export const STOP_WHEN_FLAT = { Auto: null, Always: true, Never: false } as const;
export type StopWhenFlat = keyof typeof STOP_WHEN_FLAT;

export const HOURS = { "1 h": 1, "4 h": 4, "12 h": 12, "24 h": 24, "3 days": 72 } as const; // the time the person gives the run
export type Hours = keyof typeof HOURS;
export const DEFAULT_HOURS: Hours = "4 h";
export type Sizing = NonNullable<Plan["sizing"]>;

/** Hours in plain words, as the server writes them (studio_ia.sizing.hours_text). */
export function hoursText(hours: number): string {
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))} min`;
  if (hours >= 48) return `${(hours / 24).toFixed(1)} days`;
  return hours < 10 ? `${hours.toFixed(1)} h` : `${Math.round(hours)} h`;
}

/** The lines of the "Is it worth the time?" card for a plan: what the run costs here, where the curve goes flat, what to use instead. */
export function sizingView(plan: Plan | undefined): {
  verdict: Sizing["verdict"];
  warning: string;
  time: string;
  expected: string;
  suggestion: { text: string; steps: number; designChange: boolean; useSteps: boolean } | null;
  gpu: string;
} | null {
  const s = plan?.sizing;
  if (!plan || !s) return null;
  const device = s.device ? fill(T.sizingDevice, { device: s.device.toUpperCase() }) : "";
  const time = s.measured && s.run_hours != null ? fill(T.sizingTime, { steps: n(plan.steps ?? 0), time: hoursText(s.run_hours), device }) : T.sizingTiming;
  const e = s.expected;
  const expected =
    e?.basis === "fit" && e.loss_planned != null && e.loss_flat != null && e.flat_steps != null
      ? fill(T.sizingFit, { flat: n(e.flat_steps), atFlat: e.loss_flat.toFixed(2), atEnd: e.loss_planned.toFixed(2) })
      : "";
  const found = s.suggestion;
  return {
    verdict: s.verdict,
    warning: s.warning,
    time,
    expected,
    suggestion: found ? { text: found.text, steps: found.steps, designChange: !found.same_design, useSteps: found.same_design && found.steps !== plan.steps } : null,
    gpu: s.gpu_note,
  };
}

const n = (value: number) => value.toLocaleString("en-US");

/** The sets the run reads: each set is ticked as the person left it, else as the server starts it (held-out sets off). */
export function chosenIds(sets: TextSet[] | undefined, use: Record<string, boolean>): string[] {
  return (sets ?? []).filter((s) => use[s.id] ?? s.on).map((s) => s.id);
}

export function selectedLine(count: number): string {
  return count ? `${count} ${count === 1 ? "source" : "sources"} selected` : T.nothingSelected;
}

/** The size verdict under "Enough text?": the headline and the sentence behind it. */
export function enoughView(plan: Plan | undefined, estimating: boolean): { headline: string; tip: string } {
  if (estimating) return { headline: T.estimating, tip: T.wordPiecesPlain };
  if (plan?.enough) {
    const tip = plan.enough.verdict === "plenty" ? T.wordPiecesPlenty : T.wordPiecesRule;
    return { headline: plan.enough.headline, tip: fill(tip, { ratio: plan.enough.ratio_text }) };
  }
  return { headline: T.textUnknown, tip: T.wordPiecesPlain };
}

/** A count the way a person says it: 133k, 4.2M, 545M (as the server's studio_ia.sizing.count_text). */
export function countText(value: number): string {
  for (const [size, mark] of [[1e9, "B"], [1e6, "M"], [1e3, "k"]] as const) {
    if (value >= size) return value >= 10 * size ? `${Math.round(value / size)}${mark}` : `${(value / size).toFixed(1)}${mark}`;
  }
  return String(Math.round(value));
}

/** The quick readout under "Enough text?": the sizes of the text and the design, and the advice, ready before the size preview is. */
export function readoutView(plan: Plan | undefined): { size: string; source: string; advice: string; verdict: Readout["verdict"] } | null {
  const r = plan?.readout;
  if (!r) return null;
  const about = r.counted ? "" : T.readoutAbout;
  const advice = { few: T.adviceFew, right: T.adviceRight, many: T.adviceMany }[r.verdict];
  return {
    size: fill(T.readoutSize, { characters: countText(r.characters), pieces: about + countText(r.pieces), params: countText(r.params) }),
    source: r.counted ? T.readoutCounted : fill(T.readoutEstimated, { chars: Number(r.chars_per_piece.toFixed(1)).toString() }),
    advice: fill(advice, { ratio: about + r.ratio_text, target: countText(r.target_params) }),
    verdict: r.verdict,
  };
}

export function perStepLine(plan: Plan | undefined, estimating: boolean): string {
  if (!plan) return "";
  const pieces = n(plan.pieces_per_step);
  if (estimating || !plan.pieces || !plan.passes_over_text) return fill(T.perStepOnly, { pieces });
  return fill(T.perStepPieces, { pieces, passes: `${plan.estimated ? "about " : ""}${n(plan.passes_over_text)}` });
}

export function stepsLine(plan: Plan | undefined, estimating: boolean): string {
  if (plan?.steps && plan.minutes) return fill(T.stepsTime, { steps: n(plan.steps), minutes: n(plan.minutes) });
  if (plan?.steps) return fill(T.stepsOnly, { steps: n(plan.steps) });
  return estimating ? T.stepsEstimating : T.stepsNone;
}

export function lrLine(lr: number | null | undefined): string {
  return `${T.learningRate} ${T.lrAuto}${lr ? " " + fill(T.lrInfo, { lr: Number(lr.toPrecision(2)).toString() }) : ""}`;
}

/** The two modes with their titles and the sentences under them, as Tk's paint_modes wrote them. */
export function modeTexts(plan: Plan): Record<"catch_up" | "from_scratch", { title: string; note: string }> {
  const parent = plan.parent;
  const version = plan.version ? `version ${plan.version}` : T.thisVersion;
  const blocked = plan.continue_blocked ? fill(T.mustStartOver, { why: plan.continue_blocked === "word_list_changed" ? T.wordListChanged : T.noTrainedParent, version }) : "";
  return {
    catch_up: {
      title: parent ? fill(T.continueFrom, { n: parent }) : T.continueFromEarlier,
      note: blocked || (parent ? fill(T.continueRecommended, { n: parent, version }) : T.continueNone),
    },
    from_scratch: { title: T.fromScratch, note: T.fromScratchNote },
  };
}

export function queueNote(plan: Plan): string {
  return fill(T.queueNote, {
    parent: plan.parent ? `version ${plan.parent}` : "its current version",
    version: plan.version ? `version ${plan.version}` : T.thisVersion,
  });
}

const g = (value: number) => Number(value.toPrecision(2)).toString();

/** The sentence under "Find a learning rate" for a finished (or failed or stopped) range test; "" before one has run. */
export function lrText(result: RangeResult | { error: string } | null, current?: number | null): string {
  if (!result) return "";
  if ("error" in result) return fill(T.lrFailed, { error: result.error });
  if (result.cancelled) return T.lrStopped;
  const failed = result.failed_at ?? (result.rates.length ? result.rates[result.rates.length - 1] : null);
  if (!result.suggestion) return result.diverged && failed ? fill(T.lrBlewUpEarly, { rate: g(failed) }) : T.lrFlat;
  const text = fill(T.lrResult, {
    rate: g(result.suggestion),
    steepest: g(result.rates[result.steepest ?? 0]),
    current: current ? g(current) : "its own default",
  });
  return result.diverged && failed ? `${text} ${fill(T.lrBlewUp, { rate: g(failed) })}` : text;
}

/** The learning rate a version's spec trains with, if it names one. */
export function specLr(spec: unknown): number | null {
  const lr = (spec as { train?: { optimizer?: { config?: { lr?: unknown } } } } | null)?.train?.optimizer?.config?.lr;
  return typeof lr === "number" ? lr : null;
}

/** The sentence for a failed call: the server's own `detail`, else what the browser said. */
export function errorText(error: unknown): string {
  const body = (error as { body?: string })?.body;
  if (body) {
    try {
      const detail = (JSON.parse(body) as { detail?: unknown }).detail;
      if (typeof detail === "string") return detail;
    } catch {
      // not JSON: show it as is
    }
    return body;
  }
  return error instanceof Error ? error.message : String(error);
}

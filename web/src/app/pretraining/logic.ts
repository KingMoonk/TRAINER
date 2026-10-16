import type { components } from "@/lib/api-schema";
import { T, fill } from "./copy";

export type Plan = components["schemas"]["PretrainPlan"];
export type TextSet = components["schemas"]["TextSet"];
export type Preview = components["schemas"]["PreviewModel"];
export type RangeResult = components["schemas"]["RangeResult"];

export const ROWS = ["8", "16", "32"] as const;
export const HOLD = ["2%", "5%", "10%"] as const;
export const LENGTHS = { Short: 1, Medium: 2, Long: 4 } as const; // passes over the chosen text
export type Length = keyof typeof LENGTHS;
export const PREVIEW_ROWS = 6; // rows of the size preview the page draws
export const PALETTE = ["#3a3560", "#2a4558", "#4a3a2a", "#2f4a36", "#4a2f45"];
export const END = "#0a0b0e";

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

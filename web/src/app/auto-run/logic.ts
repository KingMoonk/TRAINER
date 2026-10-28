import type { components } from "@/lib/api-schema";
import { ROUND2_SETTINGS, settingValue } from "../post-training/logic";
import { isKept, type KeptGate } from "../post-training/kept-logic";
import { T, fill } from "./copy";

export type Plan = components["schemas"]["AutoRunPlanModel"];
export type PhasePlan = components["schemas"]["AutoRunPhasePlan"];
export type PhaseState = components["schemas"]["AutoRunPhaseState"];
export type RunView = components["schemas"]["AutoRunRunView"];
export type RunSummary = components["schemas"]["AutoRunRunSummary"];
export type Progress = components["schemas"]["AutoRunJobProgress"];
export type Tuning = components["schemas"]["AutoRunTuning"];
export type RunStatus = RunView["state"]["status"];

/** The post-training round each phase runs (studio_ia/autorun.py ROUND). */
export const ROUND: Record<string, number> = { sft: 1, rlvr: 2, distill: 3, dpo: 4 };

/** A setting a phase accepts in its plan: the key the server reads, the range it accepts, and what an empty value means. */
export type SettingSpec = { key: string; label: string; hint: string; min: number; max: number; integer: boolean; fallback: number | null };

const STEPS: SettingSpec = {
  key: "steps",
  label: "Steps",
  hint: "Updates to the model. More steps learn more and take longer.",
  min: 1,
  max: 5000,
  integer: true,
  fallback: null,
};

export type PhaseSpec = { title: string; blurb: string; href: string; link: string; settings: readonly SettingSpec[] };

/** What the page says about each phase, in the order the server runs them. Settings are only the ones the server reads for that phase. */
export const PHASES: Record<string, PhaseSpec> = {
  data: {
    title: "Check the data",
    blurb: "Checks that the phases you chose have what they train on: text for pretraining and mid-training, chats marked Good for Round 1. It trains nothing.",
    href: "/your-data",
    link: "Open Your data",
    settings: [],
  },
  pretrain: {
    title: "Pretraining",
    blurb: "Learn from your text by predicting the next word.",
    href: "/pretraining",
    link: "Open Pretraining",
    settings: [
      { key: "hours", label: "Hours to train", hint: "The time it may use. It is cut to what is left of the overall budget.", min: 0.1, max: 1440, integer: false, fallback: null },
      { key: "passes", label: "Passes over the text", hint: "How many times it reads all the chosen text.", min: 1, max: 100, integer: true, fallback: 2 },
    ],
  },
  midtrain: {
    title: "Mid-training",
    blurb: "Keep training the best weights on your text mixed with chat and reasoning sets. Only weights that pass the check replace them.",
    href: "/runs/new",
    link: "Open New run",
    settings: [{ ...STEPS, max: 100000, hint: "Updates to the model. Empty: sized to the model (2,000 to 5,000)." }],
  },
  sft: { title: "Round 1 · Learn to answer", blurb: "Train on good conversations so it answers in the shape of a chat.", href: "/post-training", link: "Open Post-training", settings: [STEPS] },
  rlvr: {
    title: "Round 2 · Practice with checks",
    blurb: "Practise on problems a rule can check, and favour the answers that pass.",
    href: "/post-training",
    link: "Open Post-training",
    settings: ROUND2_SETTINGS,
  },
  distill: { title: "Round 3 · Learn from a teacher", blurb: "Move its answers toward a teacher's.", href: "/post-training", link: "Open Post-training", settings: [STEPS] },
  dpo: { title: "Round 4 · Prefer the better answer", blurb: "Learn from the answer pairs you rated.", href: "/post-training", link: "Open Post-training", settings: [STEPS] },
};

export function phaseTitle(kind: string): string {
  return PHASES[kind]?.title ?? kind;
}

export const isFinished = (status: string): boolean => ["done", "stopped", "failed"].includes(status);

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

/** The version a new run starts from: the newest one with trained weights, else the newest. */
export function defaultVersion(versions: { number: number; best_checkpoint?: string | null }[] | undefined): number | null {
  if (!versions?.length) return null;
  return ([...versions].reverse().find((v) => v.best_checkpoint) ?? versions[versions.length - 1]).number;
}

// ---- the plan editor

/** The key of a setting in the editor's text state. */
export const textKey = (kind: string, key: string): string => `${kind}.${key}`;

/** The settings of a plan as the text the editor shows. */
export function settingTexts(plan: Plan): Record<string, string> {
  const out: Record<string, string> = {};
  for (const phase of plan.phases) {
    for (const spec of PHASES[phase.kind]?.settings ?? []) {
      const value = phase.settings[spec.key];
      if (typeof value === "number") out[textKey(phase.kind, spec.key)] = String(value);
    }
  }
  return out;
}

/** The limits on the controller's automatic choices, with what each bounds. */
export const BOUNDS = [
  { key: "learning_rate", label: "Learning rate" },
  { key: "steps", label: "Steps" },
  { key: "beta", label: "DPO beta" },
] as const;

export const boundKey = (key: string, end: "min" | "max") => `bound.${key}.${end}`;

/** The bounds of a plan as the text the editor shows. */
export function boundTexts(plan: Plan): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, bound] of Object.entries(plan.bounds ?? {})) {
    out[boundKey(key, "min")] = String(bound.min);
    out[boundKey(key, "max")] = String(bound.max);
  }
  return out;
}

function buildBounds(texts: Record<string, string>): { bounds: NonNullable<Plan["bounds"]> } | { error: string } {
  const bounds: NonNullable<Plan["bounds"]> = {};
  for (const { key, label } of BOUNDS) {
    const low = (texts[boundKey(key, "min")] ?? "").trim();
    const high = (texts[boundKey(key, "max")] ?? "").trim();
    if (!low && !high) continue;
    const min = Number(low);
    const max = Number(high);
    if (!low || !high || !(min > 0) || !(max > 0) || !Number.isFinite(min) || !Number.isFinite(max)) return { error: fill(T.badBound, { label }) };
    if (min > max) return { error: fill(T.boundOrder, { label }) };
    bounds[key] = { min, max };
  }
  return { bounds };
}

/** What the controller chose for a phase (its latest attempt), or undefined when it has chosen nothing. */
export function tuningFor(run: RunView, kind: string): Tuning | undefined {
  return [...(run.state.tuning ?? [])].reverse().find((t) => t.phase === kind);
}

export function budgetText(plan: Plan): string {
  return plan.budget.hours == null ? "" : String(plan.budget.hours);
}

/**
 * The plan the editor's choices make, or the sentence that says what to fix. A setting left empty is dropped so the server uses its default;
 * one outside what the server accepts is refused here rather than silently ignored.
 */
export function buildPlan(base: Plan, texts: Record<string, string>, budget: string, bounds: Record<string, string> = {}): { plan: Plan } | { error: string } {
  if (!base.name.trim()) return { error: T.needName };
  if (!base.phases.some((p) => p.enabled)) return { error: T.atLeastOne };
  let hours: number | null = null;
  if (budget.trim()) {
    hours = Number(budget);
    if (!Number.isFinite(hours) || hours <= 0) return { error: T.badBudget };
  }
  const built = buildBounds(bounds);
  if ("error" in built) return built;
  const phases: PhasePlan[] = [];
  for (const phase of base.phases) {
    const settings: Record<string, unknown> = { ...phase.settings };
    for (const spec of PHASES[phase.kind]?.settings ?? []) {
      const text = texts[textKey(phase.kind, spec.key)] ?? "";
      const value = settingValue(text, spec);
      if (text.trim() && value === null) {
        return { error: fill(T.badSetting, { label: spec.label, phase: phaseTitle(phase.kind), kind: spec.integer ? T.whole : T.number, min: spec.min, max: spec.max }) };
      }
      if (value === null) delete settings[spec.key];
      else settings[spec.key] = value;
    }
    phases.push({ ...phase, settings });
  }
  return { plan: { ...base, name: base.name.trim(), budget: { hours }, bounds: built.bounds, phases } };
}

// ---- a run

export function currentPhase(run: RunView): PhaseState | undefined {
  const name = run.state.current;
  return name ? run.state.phases.find((p) => p.kind === name) : undefined;
}

/** The phases the timeline shows: the ones the plan includes, with their state (a phase the plan leaves out is never shown). */
export function timelinePhases(run: RunView): { plan: PhasePlan; state: PhaseState }[] {
  return run.plan.phases
    .map((plan, i) => ({ plan, state: run.state.phases[i] }))
    .filter((p): p is { plan: PhasePlan; state: PhaseState } => !!p.state && p.plan.enabled);
}

export type Controls = {
  pause: boolean;
  stop: boolean;
  /** "Continue": a paused run with nothing wrong goes on. */
  resume: boolean;
  /** The run waits for a decision about the current phase: try again, skip it, adopt a kept checkpoint, or stop. */
  decide: boolean;
  /** The current phase is a failed round that kept a checkpoint: Try in chat and Adopt. */
  kept: { round: number; gate: KeptGate } | null;
  edit: boolean;
};

const NEEDS_YOU = ["failed", "stopped", "blocked"];

/** The kept round of a run waiting on a failed post-training round: its gate record and the checkpoint it left behind. */
export function keptRound(run: RunView): { round: number; gate: KeptGate } | null {
  if (!["paused", "waiting_user"].includes(run.state.status)) return null;
  const phase = currentPhase(run);
  if (!phase || phase.status !== "failed" || !(phase.kind in ROUND)) return null;
  const gate: KeptGate = { ...((phase.result ?? {}) as KeptGate), checkpoint: phase.checkpoint ?? undefined };
  return isKept(gate) ? { round: ROUND[phase.kind], gate } : null;
}

export function controls(run: RunView): Controls {
  const status = run.state.status;
  const waiting = status === "paused" || status === "waiting_user";
  const decide = waiting && NEEDS_YOU.includes(currentPhase(run)?.status ?? "");
  return {
    pause: status === "running",
    stop: ["running", "pausing", "paused", "waiting_user"].includes(status),
    resume: waiting && !decide,
    decide,
    kept: decide ? keptRound(run) : null,
    edit: status === "draft" || status === "paused",
  };
}

/** The checks a finished phase's result carries (a round's gate), with the sentence it ended on. */
export function resultChecks(result: PhaseState["result"]): { name: string; detail: string; ok: boolean | null }[] {
  const checks = (result as { checks?: unknown } | null | undefined)?.checks;
  return Array.isArray(checks) ? (checks as { name: string; detail: string; ok: boolean | null }[]) : [];
}

export function resultVerdict(result: PhaseState["result"]): string {
  const verdict = (result as { verdict?: unknown } | null | undefined)?.verdict;
  return typeof verdict === "string" ? verdict : "";
}

export function checkText(ok: boolean | null): string {
  return ok === null ? T.checkUnmeasured : ok ? T.checkPassed : T.checkFailed;
}

/** What the live job says about itself, in plain words: its step, or why it has not started. */
export function progressLine(progress: Progress): string {
  if (progress.total) {
    const parts = [fill(T.stepLine, { step: (progress.step ?? 0).toLocaleString("en-US"), total: progress.total.toLocaleString("en-US") })];
    if (progress.eta_s != null) parts.push(fill(T.etaMin, { m: Math.max(1, Math.round(progress.eta_s / 60)) }));
    return parts.join(" · ");
  }
  if (progress.status === "held") return T.held;
  if (progress.status === "waiting") return T.queued;
  return T.preparing;
}

/** The run a page opens on: the one asked for, else the newest one that is not over. */
export function pickRun(runs: RunSummary[] | undefined, chosen: string | null): string | null {
  if (chosen) return chosen;
  return runs?.find((r) => !isFinished(r.status))?.id ?? null;
}

export function runLabel(run: Pick<RunSummary, "name" | "version">): string {
  return `${run.name} · ${fill(T.versionN, { n: run.version })}`;
}

/** A journal time (an ISO string) as 'date time' to the second. */
export function when(iso: string): string {
  return iso.replace("T", " ").replace(/(\.\d+)?(Z|[+-]\d\d:?\d\d)?$/, "");
}

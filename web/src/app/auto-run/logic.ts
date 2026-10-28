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
export type RunStatus = RunView["state"]["status"];
export type Controller = components["schemas"]["AutoRunController"];
export type Choice = components["schemas"]["AutoRunChoiceRecord"];
export type CycleDecision = components["schemas"]["AutoRunCycleDecision"];

/** The post-training round each phase runs (studio_ia/autorun.py ROUND). */
export const ROUND: Record<string, number> = { sft: 1, rlvr: 2, distill: 3, dpo: 4 };

/** A setting a phase accepts in its plan: the key the server reads, the range it accepts, and what an empty value means. */
export type SettingSpec = { key: string; label: string; hint: string; min: number; max: number; integer: boolean; fallback: number | null; text?: boolean };

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
  rlhf: {
    title: "RLHF loop · Keep what the report card confirms",
    blurb:
      "Starts from the Round 1 weights and repeats preference rounds on its own. A round is kept only if nothing got worse, including the report card, and it stops when the report card has stopped improving. Off unless you add it: it can run for hours.",
    href: "/post-training",
    link: "Open Post-training",
    settings: [
      { key: "max_rounds", label: "Most rounds", hint: "It stops after this many rounds, kept or not.", min: 1, max: 1000, integer: true, fallback: 20 },
      { key: "card_patience", label: "Rounds without a better report card", hint: "It stops when the report card has not clearly improved for this many rounds in a row.", min: 1, max: 1000, integer: true, fallback: 3 },
      { key: "patience", label: "Rounds in a row not kept", hint: "It also stops after this many rounds in a row that were rolled back.", min: 1, max: 1000, integer: true, fallback: 4 },
      { key: "judge_model", label: "Judge model", hint: "An Ollama model name that picks the better answer. Empty: the workspace teacher, or none.", min: 0, max: 0, integer: false, fallback: null, text: true },
    ],
  },
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
      if (typeof value === "number" || (spec.text && typeof value === "string")) out[textKey(phase.kind, spec.key)] = String(value);
    }
  }
  return out;
}

/** The limits the parameter controller keeps to (studio_autotune.KEYS), with how the editor shows them. */
export const BOUND_KEYS = [
  { key: "learning_rate", label: "Learning rate" },
  { key: "steps", label: "Steps" },
  { key: "beta", label: "DPO beta" },
  { key: "kl", label: "KL pull-back" },
] as const;
export type BoundKey = (typeof BOUND_KEYS)[number]["key"];
export type BoundTexts = Record<BoundKey, { min: string; max: string }>;

/** The controller's limits as the text the editor shows, or null for a plan that has no controller. */
export function boundTexts(plan: Plan): BoundTexts | null {
  const bounds = plan.controller?.bounds;
  if (!bounds) return null;
  return Object.fromEntries(BOUND_KEYS.map(({ key }) => [key, { min: String(bounds[key].min), max: String(bounds[key].max) }])) as BoundTexts;
}

/** The controller the editor's choices make: its switch, and the limits read from text (each a number, min not above max), or the sentence to fix. */
export function buildController(base: Controller, texts: BoundTexts): { controller: Controller } | { error: string } {
  const bounds = {} as Controller["bounds"];
  for (const { key, label } of BOUND_KEYS) {
    const min = Number(texts[key].min);
    const max = Number(texts[key].max);
    if (!texts[key].min.trim() || !texts[key].max.trim() || !Number.isFinite(min) || !Number.isFinite(max) || min < 0 || max < min || (key !== "kl" && max <= 0)) {
      return { error: fill(T.badBound, { label }) };
    }
    bounds[key] = { min, max };
  }
  return { controller: { ...base, bounds } };
}

/** The cycle limit the plan allows (studio_autotune.MAX_CYCLES) as the text the editor shows. */
export const MAX_CYCLES = 5;
export function cyclesText(plan: Plan): string {
  return String(plan.controller?.max_cycles ?? 1);
}

/** The cycle limit read from text: a whole number from 1 to MAX_CYCLES, or null. */
export function parseCycles(text: string): number | null {
  const value = Number(text);
  return text.trim() && Number.isInteger(value) && value >= 1 && value <= MAX_CYCLES ? value : null;
}

export function budgetText(plan: Plan): string {
  return plan.budget.hours == null ? "" : String(plan.budget.hours);
}

/**
 * The plan the editor's choices make, or the sentence that says what to fix. A setting left empty is dropped so the server uses its default;
 * one outside what the server accepts is refused here rather than silently ignored.
 */
export function buildPlan(base: Plan, texts: Record<string, string>, budget: string, bounds?: BoundTexts | null, cycles?: string): { plan: Plan } | { error: string } {
  if (!base.name.trim()) return { error: T.needName };
  if (!base.phases.some((p) => p.enabled)) return { error: T.atLeastOne };
  let hours: number | null = null;
  if (budget.trim()) {
    hours = Number(budget);
    if (!Number.isFinite(hours) || hours <= 0) return { error: T.badBudget };
  }
  const phases: PhasePlan[] = [];
  for (const phase of base.phases) {
    const settings: Record<string, unknown> = { ...phase.settings };
    for (const spec of PHASES[phase.kind]?.settings ?? []) {
      const text = texts[textKey(phase.kind, spec.key)] ?? "";
      const value = spec.text ? text.trim() || null : settingValue(text, spec);
      if (!spec.text && text.trim() && value === null) {
        return { error: fill(T.badSetting, { label: spec.label, phase: phaseTitle(phase.kind), kind: spec.integer ? T.whole : T.number, min: spec.min, max: spec.max }) };
      }
      if (value === null) delete settings[spec.key];
      else settings[spec.key] = value;
    }
    phases.push({ ...phase, settings });
  }
  let controller = base.controller;
  if (controller && bounds) {
    const built = buildController(controller, bounds);
    if ("error" in built) return built;
    controller = built.controller;
  }
  if (controller && cycles !== undefined) {
    const max = parseCycles(cycles);
    if (max === null) return { error: fill(T.badCycles, { max: MAX_CYCLES }) };
    controller = { ...controller, max_cycles: max };
  }
  return { plan: { ...base, name: base.name.trim(), budget: { hours }, phases, ...(controller ? { controller } : {}) } };
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

/** A number as the page says it: 0.00005 as 5e-5, 1500 as 1,500. */
export function valueText(value: number | null | undefined): string {
  if (value == null) return "";
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential();
  return value.toLocaleString("en-US", { maximumFractionDigits: 6 });
}

/** The label of what a choice set, e.g. "Learning rate"; a design proposal is "Bigger design". */
export function choiceLabel(choice: Choice): string {
  return choice.key === "design" ? T.biggerDesign : (BOUND_KEYS.find((b) => b.key === choice.key)?.label ?? choice.key);
}

/** What the controller decided after each cycle (another cycle or finish), oldest first, for the run page. */
export function runDecisions(run: RunView): CycleDecision[] {
  return run.state.decisions ?? [];
}

/** The choices the controller made, oldest first, for the run page. */
export function runChoices(run: RunView): Choice[] {
  return run.state.choices ?? [];
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

/** One round of the RLHF loop as its phase keeps it (studio_loop.phase_result). */
export type LoopRound = { round: number; status: string; reason: string; judge_win_rate: number | null; card_better: string[] };

/** The rounds an rlhf phase's result lists, in order (while the loop runs, the rounds so far). */
export function loopRounds(result: PhaseState["result"]): LoopRound[] {
  const rounds = (result as { rounds?: unknown } | null | undefined)?.rounds;
  return Array.isArray(rounds) ? (rounds as LoopRound[]) : [];
}

/** The line a round gets: its number and how it ended, then what the judge and the report card said. */
export function roundText(round: LoopRound): string {
  const parts = [fill(T.roundLine, { n: round.round, status: T.roundStatus[round.status] ?? round.status })];
  if (typeof round.judge_win_rate === "number") parts.push(fill(T.judgeWin, { pct: Math.round(round.judge_win_rate * 100) }));
  if (round.card_better?.length) parts.push(fill(T.cardBetter, { what: round.card_better.join(", ") }));
  return parts.join(" · ");
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

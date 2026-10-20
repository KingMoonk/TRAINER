// The start form's stall rules (a plateau window, the changes to try, replay of earlier text), apart from the page so a test can check them
// without a browser. They map onto POST /distributed/runs: plateau_checks, plateau_min_improvement, pivots and stages.
import { fill } from "@/lib/members";
import { TRAINING as T } from "@/lib/copy/training";
import type { StartRun } from "./logic";

/** The changes a stalled run can try, as the API names them. */
export const PIVOTS = ["halve_lr", "next_dataset", "drop_lessons"] as const;
export type Pivot = (typeof PIVOTS)[number];

/** The form's rule fields as typed: numbers stay text. `pivots` is in the order they were ticked, which is the order the run tries them. */
export type RulesForm = { pivots: Pivot[]; plateauChecks: string; plateauGain: string; replay: string };

/** The server's own defaults (plateau_min_improvement 1%, studio_stages.REPLAY 20%), shown so the person sees what a blank would mean. */
export function initialRules(): RulesForm {
  return { pivots: [], plateauChecks: "", plateauGain: "1", replay: "20" };
}

/** A whole number of at least `low`, or null. */
function wholeFrom(text: string, low: number): number | null {
  const n = Number(text);
  return text.trim() !== "" && Number.isInteger(n) && n >= low ? n : null;
}

/** A percentage in [low, 100) (above `low` when `strict`), or null. */
function percent(text: string, low: number, strict: boolean): number | null {
  const n = Number(text);
  return text.trim() !== "" && Number.isFinite(n) && (strict ? n > low : n >= low) && n < 100 ? n : null;
}

/** A list with `pivot` added at the end when `on`, removed when not: the order of ticking is the order of trying. */
export function withPivot(pivots: Pivot[], pivot: Pivot, on: boolean): Pivot[] {
  return on ? (pivots.includes(pivot) ? pivots : [...pivots, pivot]) : pivots.filter((p) => p !== pivot);
}

/** Why the rules cannot be started with, in a sentence, or "" when they can (or are not used). Mirrors what the API refuses. */
export function rulesProblem(rules: RulesForm, polishEvery: string, rounds: string): string {
  const windowBlank = rules.plateauChecks.trim() === "";
  if (windowBlank) return rules.pivots.length > 0 ? T.pivotsNeedWindow : "";
  const window = wholeFrom(rules.plateauChecks, 2);
  if (window === null) return T.plateauBad;
  const total = wholeFrom(rounds, 1);
  if (total !== null && window >= total) return T.plateauTooLate;
  if (percent(rules.plateauGain, 0, true) === null) return T.gainBad;
  if (rules.pivots.includes("next_dataset") && percent(rules.replay, 0, false) === null) return T.replayBad;
  if (rules.pivots.includes("drop_lessons") && wholeFrom(polishEvery, 1) === null) return T.dropLessonsNeedsPolish;
  return "";
}

/** The rules' part of the request: nothing while no window is set. Only called when `rulesProblem` is empty. */
export function rulesParams(rules: RulesForm): Pick<StartRun, "plateau_checks" | "plateau_min_improvement" | "pivots" | "stages"> | undefined {
  const window = wholeFrom(rules.plateauChecks, 2);
  if (window === null) return undefined;
  return {
    plateau_checks: window,
    plateau_min_improvement: Number(rules.plateauGain) / 100,
    ...(rules.pivots.length > 0 ? { pivots: rules.pivots } : {}),
    ...(rules.pivots.includes("next_dataset") ? { stages: { replay: Number(rules.replay) / 100 } } : {}),
  };
}

/** When the run will move on by itself, in plain sentences, or "" when no window is set or the rules are not usable yet. */
export function planText(rules: RulesForm, polishEvery: string, rounds: string): string {
  if (rules.plateauChecks.trim() === "" || rulesProblem(rules, polishEvery, rounds) !== "") return "";
  const stalls = fill(T.planStalls, { gain: Number(rules.plateauGain), rounds: Number(rules.plateauChecks) });
  if (rules.pivots.length === 0) return `${stalls} ${T.planEnds}`;
  const steps = rules.pivots.map((pivot, index) => {
    const step = fill(T.planStep[pivot], { replay: Number(rules.replay) });
    return rules.pivots.length === 1 ? step : fill(index === 0 ? T.planFirst : T.planNext, { step });
  });
  return `${stalls} ${fill(T.planThen, { steps: steps.join(", ") })}`;
}

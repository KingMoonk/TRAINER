import type { components } from "@/lib/api-schema";

export type Loop = components["schemas"]["LoopOut"];
export type LoopRound = components["schemas"]["LoopRoundOut"];
export type LoopCheck = components["schemas"]["LoopCheckOut"];
export type LoopStartBody = components["schemas"]["LoopStart"];

export type LoopForm = { rounds: string; patience: string; hours: string; judge: string };
export const NO_JUDGE = "none";
export const LOOP_DEFAULTS: LoopForm = { rounds: "20", patience: "4", hours: "", judge: NO_JUDGE };

/** The newest loop that works on `version` (the API lists newest first), else the newest of all when no version is known. */
export function loopFor(loops: Loop[] | undefined, version: number | undefined): Loop | undefined {
  return (loops ?? []).find((l) => version === undefined || l.version === version);
}

/** A whole number from a text field within [low, high], else null. */
export function wholeNumber(text: string, low: number, high: number): number | null {
  if (!/^\d+$/.test(text.trim())) return null;
  const n = Number(text);
  return n >= low && n <= high ? n : null;
}

/** The start request for the form, or the sentence saying which field is wrong. */
export function startRequest(form: LoopForm, version: number | null, resume: string | null): { body: LoopStartBody } | { error: string } {
  const rounds = wholeNumber(form.rounds, 1, 100000);
  if (rounds === null) return { error: "Rounds must be a whole number from 1 to 100,000." };
  const patience = wholeNumber(form.patience, 1, 1000);
  if (patience === null) return { error: "Stop-after must be a whole number from 1 to 1,000." };
  let hours: number | null = null;
  if (form.hours.trim() !== "") {
    hours = Number(form.hours);
    if (!Number.isFinite(hours) || hours <= 0) return { error: "Hours must be more than 0, or left empty." };
  }
  return {
    body: {
      version,
      resume,
      judge_model: form.judge === NO_JUDGE ? null : form.judge,
      config: { max_rounds: rounds, patience, ...(hours !== null ? { max_hours: hours } : {}) },
    },
  };
}

/** What the loop is doing, in a sentence. */
export function headline(loop: Loop, active: boolean): string {
  const tried = (loop.rounds ?? []).length;
  const rounds = `${loop.kept} of ${tried} round${tried === 1 ? "" : "s"} kept`;
  if (active) {
    if (loop.status === "waiting_judge") return `Waiting for the judge before round ${loop.round}. ${rounds}.`;
    return `${loop.phase || `Round ${loop.round}`}. ${rounds}.`;
  }
  switch (loop.status) {
    case "finished":
      return `Finished. ${rounds}.`;
    case "stopped":
      return `Stopped before round ${loop.round}. ${rounds}.`;
    case "failed":
      return `Failed. ${rounds}.`;
    default:
      // "running" on disk with no job: the app went away; it carries on by itself when the queue runs the job again
      return `Interrupted before round ${loop.round}: it carries on when the queue runs it again, or press Continue. ${rounds}.`;
  }
}

/** Solved problems and judge result of a round, as short text. */
export function roundEvidence(round: LoopRound): string {
  const parts: string[] = [];
  const after = round.after as { pass_k?: number; pass_n?: number } | null;
  const before = round.before as { pass_k?: number; pass_n?: number } | null;
  if (after?.pass_n) parts.push(`solved ${after.pass_k} of ${after.pass_n}${before?.pass_n ? ` (was ${before.pass_k})` : ""}`);
  const judge = round.judge as { win_rate?: number | null } | null;
  if (judge?.win_rate != null) parts.push(`judge preferred it ${Math.round(judge.win_rate * 100)}%`);
  if (round.length_ratio != null) parts.push(`${round.length_ratio.toFixed(2)}x as long`);
  return parts.join(" · ");
}

export const STATUS_WORDS: Record<string, string> = { kept: "kept", rolled_back: "rolled back", too_few_pairs: "too few pairs" };

/** How the pairs of a round were made, as text. */
export function pairsLine(round: LoopRound): string {
  const p = round.pairs as { total?: number; verified?: number; comparison?: number; rewrite?: number; skipped?: Record<string, number> };
  const kinds = [
    p.verified ? `${p.verified} checked by a rule` : "",
    p.comparison ? `${p.comparison} judged` : "",
    p.rewrite ? `${p.rewrite} teacher rewrites` : "",
  ].filter(Boolean);
  const skipped = Object.entries(p.skipped ?? {}).map(([why, n]) => `${why} ×${n}`);
  return `${p.total ?? 0} pairs${kinds.length ? ` (${kinds.join(", ")})` : ""}${skipped.length ? `; skipped: ${skipped.join(", ")}` : ""}`;
}

export function elapsed(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

import type { components } from "@/lib/api-schema";
import { T, fill } from "./copy";

export type Options = components["schemas"]["OptionsOut"];
export type Status = components["schemas"]["PostTrainingOut"];
export type JobOut = components["schemas"]["studio_api__models_work__JobOut"];
export type VersionRow = components["schemas"]["Version"];
export type Queue = components["schemas"]["QueueStateModel"];

export type Gate = {
  round?: number;
  ok?: boolean;
  checks?: { name: string; detail: string; ok: boolean | null }[];
  verdict?: string;
  judge?: [number, number] | null;
  taught?: number;
  groups?: number;
};

export type GateState = "passed" | "failed" | "unmeasured";

/** How a finished round reads: failed when the gate or any check failed, unmeasured when no check measured anything, else passed. */
export function gateState(gate: Gate): GateState {
  const checks = gate.checks ?? [];
  if (gate.ok === false || checks.some((c) => c.ok === false)) return "failed";
  if (checks.length === 0 ? gate.ok !== true : checks.every((c) => c.ok == null)) return "unmeasured";
  return "passed";
}

/** studio_api/routers/post_training.py round_blocker: why a round cannot be queued yet (plain words), or null. */
export function roundBlocker(
  round: number,
  counts: Record<string, number>,
  teacherIsVersion: boolean,
  hasTeacher: boolean,
  checkpoint: boolean,
): string | null {
  if (!checkpoint) return "Train this version in Pretraining first.";
  if (round === 3) {
    if (teacherIsVersion) return hasTeacher ? null : "There is no other trained version to learn from yet.";
    return counts.corrections ? null : "Teach a correction in Chat first.";
  }
  const need: Record<number, [string, number, string]> = {
    1: ["conversations", 1, "Add conversations in Your data first."],
    2: ["problems", 1, "Make problems a rule can check first."],
    4: ["pairs", 2, "Rate at least 2 pairs in Chat first."],
  };
  const [key, least, why] = need[round];
  return (counts[key] ?? 0) >= least ? null : why;
}

export function count(value: number | null | undefined): string {
  return value == null ? "–" : value.toLocaleString("en-US");
}

/** 'step k of n · about m min left' from the running queue state; only the parts that are known. */
export function stepLine(q: Pick<Queue, "step" | "total" | "eta_s"> | undefined): string {
  if (!q) return "starting";
  const parts = [q.total ? `step ${q.step.toLocaleString("en-US")} of ${q.total.toLocaleString("en-US")}` : "starting"];
  if (q.eta_s != null) parts.push(`about ${Math.max(1, Math.round(q.eta_s / 60))} min left`);
  return parts.join(" · ");
}

export function spreadText(taught: number, groups: number): string {
  if (!groups) return "";
  const groupsWord = groups === 1 ? "group" : "groups";
  return fill(taught ? T.spreadSome : T.spreadNone, { taught: count(taught), groups: count(groups), groupsWord });
}

/** The first round that is not done (the page opens on it), else the first. */
export function firstOpenRound(status: Status | undefined): number {
  for (let n = 1; n <= 4; n++) if ((status?.rounds?.[String(n)]?.status ?? "idle") !== "done") return n;
  return 1;
}

/** The text of a failed API call: the server's own sentence (`detail`) when it sent one. */
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

/** A Round 2 setting the person can change, with the default the server applies when it is left empty (studio_rl.ROUND2_DEFAULTS). */
export const ROUND2_SETTINGS = [
  { key: "steps", label: "Steps", hint: "Updates to the model. More steps learn more and take longer.", fallback: 200, min: 1, max: 5000, integer: true },
  { key: "prompts_per_step", label: "Prompts per step", hint: "Different prompts trained on in each step. Prompts whose answers all pass or all fail are swapped for others.", fallback: 8, min: 1, max: 64, integer: true },
  { key: "group", label: "Answers per prompt", hint: "How many answers it writes to each prompt, to compare them with each other.", fallback: 8, min: 2, max: 32, integer: true },
  { key: "temperature", label: "Temperature", hint: "How adventurous its answers are. Higher gives more varied answers to compare.", fallback: 1, min: 0.1, max: 2, integer: false },
  { key: "max_tokens", label: "Longest answer (tokens)", hint: "Empty sizes it by the rule being checked.", fallback: null, min: 1, max: 2048, integer: true },
] as const;
export type Round2Key = (typeof ROUND2_SETTINGS)[number]["key"];

/** The number typed into a Round 2 setting, or null when it is empty or outside what the server accepts (then the default applies). */
export function settingValue(text: string, spec: { min: number; max: number; integer: boolean }): number | null {
  if (!text.trim()) return null;
  const value = Number(text);
  if (!Number.isFinite(value) || value < spec.min || value > spec.max || (spec.integer && !Number.isInteger(value))) return null;
  return value;
}

/** One row of the data behind a round, as short text: a string as is, an object as 'key: value' lines. */
export function rowText(row: unknown): string {
  if (typeof row === "string") return row;
  if (row && typeof row === "object") {
    return Object.entries(row as Record<string, unknown>)
      .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
      .join("\n");
  }
  return JSON.stringify(row);
}

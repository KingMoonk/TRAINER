// Page logic for Chat as plain functions (the Tk page's score_rows, verdict_view and friends), so the page stays markup.
import { CHAT as T } from "./copy";
import type { components } from "@/lib/api-schema";
import type { ApiEvent } from "@/lib/api";

export type Message = components["schemas"]["Message"];
export type Version = components["schemas"]["Version"];

export type VerdictResult = { comparable: boolean; percent?: number | null; reason?: string };

export const versionName = (v: { number: number }) => `Version ${v.number}`;

/** The versions Chat can talk to: those with a trained checkpoint, oldest first. */
export function trainedVersions(all: Version[]): Version[] {
  return all.filter((v) => v.best_checkpoint).sort((a, b) => a.number - b.number);
}

/** The trained version beside `number` for the compare slot: the one just before it, else the one just after, else null. */
export function otherVersion(numbers: number[], number: number | null): number | null {
  const before = numbers.filter((n) => n < (number ?? 0));
  const after = numbers.filter((n) => n > (number ?? 0));
  return before.length ? Math.max(...before) : after.length ? Math.min(...after) : null;
}

/** The (left, right) versions compare mode starts from: the picked one (else the newest) and the one beside it. */
export function compareStart(numbers: number[], picked: number | null): [number, number] | null {
  const left = picked !== null && numbers.includes(picked) ? picked : Math.max(...numbers);
  const right = otherVersion(numbers, left);
  return Number.isFinite(left) && right !== null ? [left, right] : null;
}

/** Rows of the scorecard: [label, value]. */
export function scoreRows(v: Version): [string, string][] {
  const s = v.scorecard ?? {};
  const kn = (key: string) => {
    const k = s[key];
    return Array.isArray(k) && k.length === 2 ? `${k[0]} of ${k[1]}` : T.valKofn;
  };
  const bpc = typeof s.bpc === "number" ? s.bpc : null;
  const size = typeof s.params_m === "number" ? `${s.params_m}M` : "";
  const speed = typeof s.pieces_per_s === "number" ? `${s.pieces_per_s} pieces/s` : "";
  return [
    [T.errorHeldBack, bpc === null ? T.valBpc : `${bpc.toFixed(2)} bits/char`],
    [T.answersPrompts, s.answers === true || s.answers === "yes" ? "yes" : "not yet"],
    [T.checkedProblems, kn("problems")],
    [T.judgePreferred, kn("judge")],
    [T.sizeSpeed, [size, speed].filter(Boolean).join(" · ") || T.valSizeSpeed],
  ];
}

/** True while a version's run is mid-schedule, so it has no fair verdict yet. */
export const unfinished = (v: Version) => v.status === "queued" || v.status === "training";

/** (line, sub, comparable) for two versions; `result` is the server's verdict for (older, newer), unused while either is unfinished. */
export function verdictView(a: Version, b: Version, result?: VerdictResult | null): { line: string; sub: string; ok: boolean } {
  const [older, newer] = a.number < b.number ? [a, b] : [b, a];
  const open = [a, b].find(unfinished);
  if (open) return { line: T.noVerdictYet(versionName(open).toLowerCase()), sub: T.noVerdictText, ok: false };
  if (result?.comparable && typeof result.percent === "number") {
    return { line: T.verdictLine(versionName(newer), versionName(older), 100 - result.percent), sub: T.verdictSub, ok: true };
  }
  return { line: `No verdict yet: ${result?.reason || "the runs are not comparable."}`, sub: T.noVerdictText, ok: false };
}

/** Whether a rated pair can be saved: both replies are in, differ, and nothing is streaming. */
export function canRate(replies: [string, string], busy: boolean): boolean {
  return !busy && replies[0] !== "" && replies[1] !== "" && replies[0] !== replies[1];
}

// These job result shapes are documented by the routes, but JobOut.result is unknown in OpenAPI.
export type FewShot = {
  items: number; tasks: number; chance: number | null; held_back: boolean;
  rows: { shots: number; used: number; correct: number; accuracy: number | null }[];
};
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const count = (value: unknown): value is number => finite(value) && Number.isInteger(value) && value >= 0;
const share = (value: unknown): value is number | null => value === null || (finite(value) && value >= 0 && value <= 1);

export function bestNumber(value: unknown): number | null {
  if (value === null) return null;
  if (!record(value) || !count(value.number) || !value.number || typeof value.best_checkpoint !== "string" || !value.best_checkpoint) {
    throw new Error(T.invalidBest);
  }
  return value.number;
}

export function checkpointStep(value: unknown): number | null {
  if (value === null || count(value)) return value;
  throw new Error(T.invalidStep);
}

export function fewShotResult(value: unknown): FewShot {
  if (!record(value) || !count(value.items) || !share(value.chance) || !Array.isArray(value.rows) ||
      (value.tasks !== undefined && !count(value.tasks)) || (value.held_back !== undefined && typeof value.held_back !== "boolean")) {
    throw new Error(T.invalidFewShot);
  }
  const items = value.items;
  const rows = value.rows.map((row: unknown) => {
    if (!record(row) || !count(row.shots) || !finite(row.used) || row.used < 0 || !count(row.correct) || row.correct > items || !share(row.accuracy)) {
      throw new Error(T.invalidFewShot);
    }
    return { shots: row.shots, used: row.used, correct: row.correct, accuracy: row.accuracy };
  });
  if (new Set(rows.map((row) => row.shots)).size !== rows.length) throw new Error(T.invalidFewShot);
  return { items, tasks: value.tasks === undefined ? 0 : value.tasks as number, chance: value.chance, held_back: value.held_back !== false, rows };
}

export function fewShotRows(table: FewShot): { key: number; cells: string[] }[] {
  if (!table.items) return [];
  return table.rows.map((row) => {
    let shown = row.shots === 0 ? T.fewShotZero : row.shots === 1 ? T.fewShotOne : String(row.shots);
    if (row.used < row.shots) shown += T.fewShotFitted(row.used.toFixed(1));
    const chance = table.chance;
    const spread = chance === null ? null : 2 * Math.sqrt(table.items * chance * (1 - chance));
    const against = chance === null || spread === null ? T.notMeasured : row.correct > table.items * chance + spread ? T.fewShotAbove : row.correct < table.items * chance - spread ? T.fewShotBelow : T.fewShotNear;
    const percent = (value: number | null) => value === null ? T.notMeasured : `${(100 * value).toFixed(0)}%`;
    return { key: row.shots, cells: [shown, `${row.correct} of ${table.items}`, percent(row.accuracy), percent(chance), against] };
  });
}

export function fewShotNote(table: FewShot): string {
  return [T.fewShotAsked(table.items, table.tasks), !table.held_back ? T.fewShotTraining : "",
    table.chance === null ? T.fewShotChanceUnknown : T.fewShotChance((table.chance * 100).toFixed(0))].filter(Boolean).join(" ");
}

/** API read jobs emit `id`; training queue jobs emit `job`. Never invalidate ourselves on a read finishing. */
export function changesCheckpoints(event: ApiEvent): boolean {
  if (event.kind === "versions") return true;
  if (!record(event.payload)) return false;
  if (event.kind === "checkpoint") return event.payload.phase === "done";
  return event.kind === "job" && typeof event.payload.job === "string" && !("id" in event.payload) &&
    ["done", "failed", "stopped", "cancelled"].includes(String(event.payload.status));
}

export type LiveCheckpoint = components["schemas"]["LiveCheckpoint"];

/** "just now", "3 min ago", "2 h ago" for a save `seconds` old. */
export function ageText(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return T.justNow;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}

/** Which step a live checkpoint answers from and how old it is: "step 12,000 of 50,000 · saved 3 min ago". */
export function liveLabel(live: LiveCheckpoint, nowSeconds: number): string {
  const step = live.step === null || live.step === undefined ? T.liveStepUnknown : `step ${live.step.toLocaleString()}${live.total ? ` of ${live.total.toLocaleString()}` : ""}`;
  return `${step} · saved ${ageText(nowSeconds - live.saved_at)}`;
}

/** The live checkpoints of versions that have no trained checkpoint of their own, oldest version first. */
export function liveOnly(live: LiveCheckpoint[], trained: Version[]): LiveCheckpoint[] {
  const have = new Set(trained.map((v) => v.number));
  return live.filter((l) => !have.has(l.version)).sort((a, b) => a.version - b.version);
}

export type LiveScore = { version: number; step: number | null; bpc: number; documents: number };

export function liveScore(value: unknown): LiveScore {
  if (!record(value) || !count(value.version) || !finite(value.bpc) || !count(value.documents) || (value.step !== null && !count(value.step))) {
    throw new Error(T.invalidLiveScore);
  }
  return { version: value.version, step: value.step as number | null, bpc: value.bpc, documents: value.documents };
}

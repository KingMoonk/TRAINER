// Page logic for Chat as plain functions (the Tk page's score_rows, verdict_view and friends), so the page stays markup.
import { CHAT as T } from "./copy";

export type Message = { role: "system" | "user" | "assistant"; content: string };

export type Version = {
  number: number;
  name?: string;
  change?: string;
  parent?: number | null;
  status: string;
  verdict?: string;
  best_checkpoint?: string | null;
  scorecard?: { [key: string]: unknown };
};

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

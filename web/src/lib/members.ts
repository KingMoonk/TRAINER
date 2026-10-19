// One vocabulary for a machine in a run, on both computers: invited, waiting, joining, training, behind, left, unreachable. The server
// computes it (studio_distributed.member_rows on the owner, studio_membership.invite_row on the machine that was added) and every cluster
// page speaks it. The rest of this file is the pure rules the join flow shares, apart from the pages so a test can check them.
import type { components } from "@/lib/api-schema";
import { JOIN } from "@/lib/copy/join";
import { TRAINING } from "@/lib/copy/training";

export type MemberState = components["schemas"]["Member"]["state"];
export type BlockingJob = components["schemas"]["BlockingJob"];

/** Fill `{name}` holes in a copy string. */
export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_m, key: string) => String(values[key] ?? ""));
}

/** The word for a state, the same on every page. */
export function stateLabel(state: MemberState): string {
  return TRAINING.memberStates[state];
}

/** A state that needs a person: the machine cannot join by itself until something changes. */
export function isBlocked(state: MemberState): boolean {
  return state === "waiting" || state === "unreachable";
}

/** ", about 12 min left", or "" when the job cannot say. */
export function etaText(seconds: number | null | undefined): string {
  return seconds == null ? "" : fill(JOIN.etaLeft, { minutes: Math.max(1, Math.round(seconds / 60)) });
}

/** The https:// address to put in a Machines list for a machine at `address`: a URL as it is, else its host (a port dropped) behind https://, where `tailscale serve` listens. */
export function pasteAddress(address: string): string {
  const text = address.trim();
  if (/^https?:\/\//i.test(text)) return text.replace(/\/+$/, "");
  return `https://${text.replace(/\/.*$/, "").replace(/:\d+$/, "")}`;
}

/** The state of one machine in one run as a page last read it. `here` is true when the machine is this computer. */
export type Seen = { key: string; node: string; run: string; state: MemberState; here: boolean };
export type Change = { id: string; kind: "success" | "warning" | "info"; text: string };

const LIVE: (MemberState | undefined)[] = ["training", "behind", "joining", "invited", "waiting"];
const NOT_YET: (MemberState | undefined)[] = [undefined, "invited", "waiting", "joining", "unreachable"];

/**
 * What changed since `before` (the states last seen, null on the first read, which only records): a machine joined, fell behind, lapsed (the
 * run let it go) or left. Returns the states to compare with next time and one toast per change, in the vocabulary's words.
 */
export function memberChanges(before: Map<string, Seen> | null, seen: Seen[]): { next: Map<string, Seen>; changes: Change[] } {
  const next = new Map(seen.map((s) => [s.key, s] as const));
  const changes: Change[] = [];
  if (before === null) return { next, changes };
  // An invite that is gone while it was live: the machine left (or the run ended and took its job with it).
  for (const [key, old] of before) {
    if (old.here && !next.has(key) && LIVE.includes(old.state)) {
      changes.push({ id: `${key}:gone`, kind: "info", text: fill(JOIN.leftHere, { node: old.node, run: old.run }) });
    }
  }
  for (const s of seen) {
    const was = before.get(s.key)?.state;
    if (was === s.state) continue;
    const values = { node: s.node, run: s.run };
    const say = (kind: Change["kind"], mine: string, theirs: string) => changes.push({ id: `${s.key}:${s.state}`, kind, text: fill(s.here ? mine : theirs, values) });
    if (s.state === "training" && NOT_YET.includes(was)) say("success", JOIN.joinedHere, JOIN.joined);
    else if (s.state === "behind") say("warning", JOIN.fellBehindHere, JOIN.fellBehind);
    else if (s.state === "dropped") say("warning", JOIN.lapsedHere, JOIN.lapsed);
    else if ((s.state === "left" || s.state === "stopped") && LIVE.includes(was)) say("info", JOIN.leftHere, JOIN.leftToast);
  }
  return { next, changes };
}

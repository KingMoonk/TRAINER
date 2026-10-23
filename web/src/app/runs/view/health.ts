// Is the cluster helping? The run's measured health (studio_run_health), turned into what the charts and the verdict line show.
import type { components } from "@/lib/api-schema";
import type { Series } from "@/components/series-chart";
import { T } from "./copy";

export type RunHealth = components["schemas"]["RunHealth"];
export type RoundHealth = components["schemas"]["RoundHealth"];
export type VerdictState = components["schemas"]["HealthVerdict"]["state"];

/** The rounds the bars of the round-time split show. */
export const SPLIT_ROUNDS = 12;

const MB = 2 ** 20;
const round2 = (value: number) => Math.round(value * 100) / 100;

/** The members that appear in the rounds, in the order they first do. */
export function healthMembers(health: RunHealth): string[] {
  return [...new Set(health.rounds.flatMap((r) => Object.keys(r.members)))];
}

function perMember(health: RunHealth, pick: (m: RoundHealth["members"][string]) => number | null): Series[] {
  return healthMembers(health).flatMap((node) => {
    const points = health.rounds.flatMap((r) => {
      const value = r.members[node] ? pick(r.members[node]) : null;
      return value == null ? [] : [{ x: r.round + 1, y: round2(value) }];
    });
    return points.length ? [{ id: node, label: node, points }] : [];
  });
}

/** The outer loss (the shared weights on held-back text) and each member's inner loss, by round. */
export function lossSeries(health: RunHealth): Series[] {
  const outer = health.rounds.flatMap((r) => (r.val_loss == null ? [] : [{ x: r.round + 1, y: r.val_loss }]));
  return [
    ...(outer.length ? [{ id: "outer", label: T.hOuterLoss, points: outer, color: "var(--warning)", dash: "5 4", dots: true }] : []),
    ...perMember(health, (m) => m.loss),
  ];
}

/** Each member's own speed (tokens per second of its compute), by round. */
export function throughputSeries(health: RunHealth): Series[] {
  return perMember(health, (m) => m.tokens_per_s);
}

/** How fast each member's weights came down and its change went up, in MB/s, by round (the rounds that measured it). */
export function linkSeries(health: RunHealth): Series[] {
  const way = (suffix: string, pick: (m: RoundHealth["members"][string]) => number | null | undefined, dash?: string) =>
    perMember(health, (m) => pick(m) ?? null).map((s) => ({ ...s, id: `${s.id} ${suffix}`, label: `${s.label} ${suffix}`, ...(dash ? { dash } : {}) }));
  return [...way(T.hLinkDown, (m) => m.down_mb_s), ...way(T.hLinkUp, (m) => m.up_mb_s, "5 4")];
}

/** What share of the entries the latest round's changes carried each way ("Sent up: the largest 5% of each tensor's entries. Down: all entries."), or "" with no rounds. */
export function sparsityText(health: RunHealth): string {
  const last = health.rounds[health.rounds.length - 1];
  if (!last) return "";
  const way = (share: number | undefined) => (share && share > 0 && share < 1 ? T.hSparsityShare.replace("{pct}", `${+(share * 100).toFixed(2)}%`) : T.hSparsityAll);
  return T.hSparsity.replace("{up}", way(last.sparsity_up)).replace("{down}", way(last.sparsity_down));
}

/** What went over the wire each round, in MB. */
export function bytesSeries(health: RunHealth): Series[] {
  const mb = (pick: (r: RoundHealth) => number) => health.rounds.map((r) => ({ x: r.round + 1, y: round2(pick(r) / MB) }));
  return [
    { id: "up", label: T.hUp, points: mb((r) => r.bytes_up) },
    { id: "down", label: T.hDown, points: mb((r) => r.bytes_down) },
  ];
}

export type SplitRow = { round: number; compute: number; wait: number; sync: number };

/** The latest rounds' time split: the average member's compute, wait for the slowest, and sync (seconds that add up to the round). */
export function splitRows(health: RunHealth): SplitRow[] {
  return health.rounds.slice(-SPLIT_ROUNDS).map((r) => {
    const members = Object.values(r.members);
    const mean = (pick: (m: (typeof members)[number]) => number) => (members.length ? members.reduce((sum, m) => sum + pick(m), 0) / members.length : 0);
    return { round: r.round + 1, compute: mean((m) => m.compute_s), wait: mean((m) => m.wait_s ?? 0), sync: mean((m) => m.sync_s) };
  });
}

/** "1.80×" or "" while there is not enough data. */
export function speedupText(health: RunHealth | null | undefined): string {
  return health?.speedup == null ? "" : `${health.speedup.toFixed(2)}×`;
}

/** The share of a round as a whole percentage, "" when unknown. */
export function percentText(share: number | null | undefined): string {
  return share == null ? "" : `${Math.round(share * 100)}%`;
}

/** The compact speed reading: "5.1k tok/s". */
export function rateText(rate: number | null | undefined): string {
  if (rate == null) return "";
  return rate >= 1000 ? `${(rate / 1000).toFixed(1)}k tok/s` : `${Math.round(rate)} tok/s`;
}

/** The classes of the verdict: good, bad, and the quiet one for not enough data. */
export function verdictTone(state: VerdictState): string {
  return state === "working" ? "border-success text-success" : state === "not_enough_data" ? "border-border text-muted-foreground" : "border-destructive text-destructive";
}

/** How much smaller a change is than the weights, from the latest round that recorded it: "2×", or "". */
export function compressionText(health: RunHealth): string {
  const ratio = [...health.rounds].reverse().find((r) => r.compression != null)?.compression;
  return ratio == null ? "" : `${ratio.toFixed(ratio < 10 ? 1 : 0)}×`;
}

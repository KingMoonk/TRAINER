import type { components } from "@/lib/api-schema";
import { fill } from "@/app/storage/logic";
import { COLLECT as T } from "./copy";

export { errorText, fill, sizeText } from "@/app/storage/logic";

export type Config = components["schemas"]["CollectConfig"];
export type Update = components["schemas"]["ConfigUpdate"];
export type State = components["schemas"]["CollectState"];
export type Options = components["schemas"]["CollectOptions"];
export type Site = components["schemas"]["Site"];
export type Dropped = components["schemas"]["Dropped"];

/** The settings as the form edits them: numbers and lists as text until they are saved. */
export type Draft = {
  topics: string[];
  customTopics: string;
  sources: Config["sources"];
  licences: Config["licences"];
  limits: Record<keyof Config["limits"], string>;
  idle: { only_when_idle: boolean; only_on_mains: boolean; unmetered_confirmed: boolean; cpu_percent: string };
  stackSites: string;
  stackScore: string;
  githubQueries: string;
  githubMax: string;
  web: Site[];
};

const LIMIT_FIELDS: [keyof Config["limits"], string][] = [
  ["disk_mb", T.diskMb],
  ["bandwidth_mb_per_day", T.bandwidthMb],
  ["requests_per_host_per_day", T.requestsPerHost],
  ["seconds_between_requests", T.secondsBetween],
  ["max_dataset_mb", T.maxDatasetMb],
  ["max_file_mb", T.maxFileMb],
  ["auto_approve_mb", T.approveMb],
];
export { LIMIT_FIELDS };

export function parseList(text: string): string[] {
  return text.split(",").map((s) => s.trim()).filter(Boolean);
}

export function toDraft(c: Config): Draft {
  return {
    topics: [...c.topics],
    customTopics: c.custom_topics.join(", "),
    sources: { ...c.sources },
    licences: { ...c.licences },
    limits: Object.fromEntries(Object.entries(c.limits).map(([k, v]) => [k, String(v)])) as Draft["limits"],
    idle: { ...c.idle, cpu_percent: String(c.idle.cpu_percent) },
    stackSites: c.stackexchange.sites.join(", "),
    stackScore: String(c.stackexchange.min_score),
    githubQueries: c.github.queries.join(", "),
    githubMax: String(c.github.max_repos),
    web: c.web.sites.map((s) => ({ ...s })),
  };
}

/** The update the form means, or the sentence for the first number that is not one. */
export function toUpdate(d: Draft): { update: Update } | { error: string } {
  const number = (text: string, field: string): number | string => {
    const n = Number(text.trim());
    return text.trim() === "" || Number.isNaN(n) ? fill(T.needNumber, { field }) : n;
  };
  const limits: Record<string, number> = {};
  for (const [key, label] of LIMIT_FIELDS) {
    const n = number(d.limits[key], label);
    if (typeof n === "string") return { error: n };
    limits[key] = n;
  }
  const cpu = number(d.idle.cpu_percent, T.cpuPercent);
  const score = number(d.stackScore, T.stackScore);
  const max = number(d.githubMax, T.githubMax);
  for (const n of [cpu, score, max]) if (typeof n === "string") return { error: n };
  return {
    update: {
      topics: d.topics,
      custom_topics: parseList(d.customTopics),
      sources: d.sources,
      licences: d.licences,
      limits,
      idle: { ...d.idle, cpu_percent: cpu as number },
      stackexchange: { sites: parseList(d.stackSites), min_score: score as number },
      github: { queries: parseList(d.githubQueries), max_repos: max as number },
      web: { sites: d.web.filter((s) => s.host.trim()).map((s) => ({ ...s, host: s.host.trim(), licence: s.licence.trim() })) },
    },
  };
}

/** Whether the form differs from the saved settings. */
export function isDirty(d: Draft, saved: Config): boolean {
  return JSON.stringify(toDraft(saved)) !== JSON.stringify(d);
}

/** The status sentence for the top of the page. */
export function statusLine(s: State): string {
  if (!s.enabled) return T.off;
  if (s.paused) return T.paused;
  if (s.status === "error") return fill(T.error, { reason: s.reason });
  if (!s.running) return T.starting;
  if (s.status === "running") return T.collecting;
  return s.reason || T.starting;
}

export const DROP_LABELS: Record<keyof Dropped, string> = {
  licence: "Licence not allowed",
  quality: "Low quality",
  pii: "Mostly personal data",
  duplicate: "Duplicate",
  near_duplicate: "Near-duplicate",
  decontam: "Overlaps held-out test text",
  invalid: "Unusable",
};

export function dropRows(dropped: Partial<Dropped> | undefined): { key: keyof Dropped; label: string; n: number }[] {
  return (Object.keys(DROP_LABELS) as (keyof Dropped)[]).map((key) => ({ key, label: DROP_LABELS[key], n: dropped?.[key] ?? 0 }));
}

export function percent(used: number, cap: number): number {
  return cap <= 0 ? 0 : Math.min(100, Math.round((100 * used) / cap));
}

export function bytesText(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.round(bytes / 1e3)} kB`;
}

/** The ids to send as `approved` after approving one more. */
export function approving(c: Config, id: string): string[] {
  return c.approved.includes(id) ? c.approved : [...c.approved, id];
}

/** The ids to send as `blocked` after skipping (or un-skipping) one. */
export function blocking(c: Config, id: string, blocked: boolean): string[] {
  return blocked ? (c.blocked.includes(id) ? c.blocked : [...c.blocked, id]) : c.blocked.filter((b) => b !== id);
}

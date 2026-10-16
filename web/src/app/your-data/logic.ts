import type { components } from "@/lib/api-schema";
import { COLLECTION, KEYED, RULES, RULE_LABELS, T, fill, type Kind } from "./copy";

export type Source = components["schemas"]["Source"];
export type DataRow = components["schemas"]["DataRow"];
export type Summary = components["schemas"]["SummaryModel"];

/** studio_ia/pages/your_data.py back_link: text trains in Pretraining, the other two in Post-training. */
export function backLink(kind: Kind): { label: string; href: string } {
  return kind === "text" ? { label: T.backPretraining, href: "/pretraining" } : { label: T.backPost, href: "/post-training" };
}

export function visibleRows(rows: DataRow[], onlyDuplicates: boolean): DataRow[] {
  return onlyDuplicates ? rows.filter((r) => r.status === "Near-duplicate") : rows;
}

export function dupLabel(found: number, onlyDuplicates: boolean): string {
  return fill(T.nearDuplicates, { n: found.toLocaleString("en-US") }) + (onlyDuplicates ? T.showing : "");
}

/** 'Showing the first N of M.' (of the matches, when searching) while more rows wait behind Load more, else ''. */
export function shownLine(loaded: number, count: number, query: string, more: boolean): string {
  if (!more) return "";
  return query.trim()
    ? fill(T.shownMatches, { shown: loaded.toLocaleString("en-US") })
    : fill(T.shown, { shown: loaded.toLocaleString("en-US"), count: count.toLocaleString("en-US") });
}

/** The recommended collection is first-run guidance: it shows while the open kind has no sets. */
export function collectionShown(summary: Summary | undefined): boolean {
  return !!summary && summary.sets.length === 0;
}

export const STATUS_CLASS: Record<string, string> = {
  "In use": "text-foreground/80",
  "Held back": "text-sky-400",
  "Near-duplicate": "text-amber-400",
  Good: "text-violet-400",
  Review: "text-muted-foreground",
  "From Chat": "text-pink-400",
  Ready: "text-foreground/80",
};

/** studio_ia/pretraining_plan.py source_action: the button for a collection row; null while it works or once it is in your sources. */
export function sourceAction(row: Pick<Source, "state">): { label: string; action: "download" | "import" } | null {
  if (row.state === "not downloaded") return { label: COLLECTION.download, action: "download" };
  if (row.state === "downloaded") return { label: COLLECTION.import, action: "import" };
  return null;
}

/** The plain-words line under a collection row: a failure first, then live progress, then the state. */
export function sourceStatus(row: Pick<Source, "state" | "error">, note = ""): string {
  if (row.error) return fill(COLLECTION.failed, { error: `${row.error.replace(/\.+$/, "")}.` });
  if (row.state === "downloading") return note || COLLECTION.stateDownloading;
  const states: Record<string, string> = {
    "not downloaded": COLLECTION.stateNotDownloaded,
    downloaded: COLLECTION.stateDownloaded,
    imported: COLLECTION.stateImported,
  };
  return states[row.state] ?? row.state;
}

/** A rule key or its plain label (either case) to the key, else null. `allowed` are the keys the form may take. */
export function parseRule(text: string, allowed: string[]): string | null {
  const word = text.trim().toLowerCase();
  return allowed.find((k) => word === k || word === k.replace(/_/g, " ") || word === RULE_LABELS[k].toLowerCase()) ?? null;
}

/** The rules a problem form may name: a keyed rule is kept only when the problem already has it (an import stored its key). */
export function allowedRules(current: string | undefined): string[] {
  return current && KEYED.includes(current) ? [...RULES, ...KEYED] : RULES;
}

/** The fields to send, or the sentence that says what is missing. */
export function checkFields(
  kind: Kind,
  values: Record<string, string>,
  current?: string,
): { fields: Record<string, string> } | { error: string } {
  const need = { text: ["text"], conversations: ["user", "assistant"], problems: ["prompt", "rule"] }[kind];
  if (need.some((name) => !(values[name] ?? "").trim())) return { error: T.fillAll };
  if (kind !== "problems") return { fields: Object.fromEntries(need.map((name) => [name, values[name]])) };
  const allowed = allowedRules(current);
  const rule = parseRule(values.rule, allowed);
  if (!rule) return { error: fill(T.unknownRule, { rules: allowed.join(", ") }) };
  return { fields: { prompt: values.prompt, rule } };
}

export function kindOf(value: string | null | undefined): Kind {
  return value === "text" || value === "problems" || value === "conversations" ? value : "conversations";
}

export function setsLine(summary: Summary | undefined): string {
  return summary ? summary.sets.map((s) => `${s.name} (${s.count.toLocaleString("en-US")})`).join(" · ") || T.noSets : "";
}

/** The sentence for a failed call: the server's own `detail`, else what the browser said. */
export function errorText(error: unknown): string {
  const body = (error as { body?: string })?.body;
  if (body) {
    try {
      const detail = (JSON.parse(body) as { detail?: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        return detail
          .map((d) => (d as { msg?: string }).msg ?? "")
          .filter(Boolean)
          .join("; ");
      }
    } catch {
      // not JSON: show it as is
    }
    return body;
  }
  return error instanceof Error ? error.message : String(error);
}

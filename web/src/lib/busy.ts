import { BUSY } from "@/lib/copy/busy";

/** One thing in progress: server work (a `busy` event) or a read the browser has waited on for a while. `total` null = no measurable steps. */
export type BusyItem = { id: string; label: string; done: number; total: number | null };

/** How long a read may take before the card shows it. */
export const SLOW_MS = 400;

/** The `busy` event's payload, or an empty list when it is not one (a malformed frame shows nothing). */
export function busyItems(payload: unknown): BusyItem[] {
  const items = (payload as { items?: unknown } | null)?.items;
  if (!Array.isArray(items)) return [];
  return items.flatMap((raw) => {
    const item = raw as Partial<BusyItem> | null;
    if (!item || typeof item.id !== "string" || typeof item.label !== "string") return [];
    return [{ id: item.id, label: item.label, done: Number(item.done) || 0, total: typeof item.total === "number" ? item.total : null }];
  });
}

/** 0 to 1 for a bar with known steps, null for an indeterminate one. */
export function fraction(item: BusyItem): number | null {
  if (item.total === null || item.total <= 0) return null;
  return Math.min(1, Math.max(0, item.done / item.total));
}

/** "Loading runs" for a read keyed ["distributed", "runs"]. Ids in the key (numbers, objects) are not words. */
export function queryLabel(key: readonly unknown[]): string {
  const words = key.filter((part): part is string => typeof part === "string");
  const named = BUSY.queries[words.join(" ")];
  const what = named ?? (words.join(" ").replace(/[-_]/g, " ") || BUSY.somethingElse);
  return BUSY.loading.replace("{what}", what);
}

const ROW_REM = 2.5;
const GAP_REM = 0.75;
const PAD_REM = 0.75;
const EDGE_REM = 1;

/** How far above the bottom edge toasts sit while the card shows `rows` rows, so they stack above it instead of under it. */
export function toastOffset(rows: number): string {
  if (rows <= 0) return "";
  const card = rows * ROW_REM + (rows - 1) * GAP_REM + 2 * PAD_REM;
  return `${EDGE_REM + card + GAP_REM}rem`;
}
export const ROW = { height: `${ROW_REM}rem`, gap: `${GAP_REM}rem`, padding: `${PAD_REM}rem`, edge: `${EDGE_REM}rem` };

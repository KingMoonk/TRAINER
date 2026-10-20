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

/** 0 to 1 across everything busy (each item counts the same), or null when any of it has no measurable steps. */
export function overall(items: BusyItem[]): number | null {
  if (!items.length) return null;
  const parts = items.map(fraction);
  if (parts.some((part) => part === null)) return null;
  return (parts as number[]).reduce((sum, part) => sum + part, 0) / parts.length;
}

/** The bar's title: the first item, with its step count when it has several, and how many more are busy. */
export function busyTitle(items: BusyItem[]): string {
  const [first, ...rest] = items;
  if (!first) return "";
  const steps = fraction(first) !== null && first.total !== null && first.total > 1
    ? ` (${BUSY.stepOf.replace("{done}", String(first.done)).replace("{total}", String(first.total))})`
    : "";
  return first.label + steps + (rest.length ? BUSY.more.replace("{count}", String(rest.length)) : "");
}

const BAR_REM = 2.25;
const GAP_REM = 0.75;

/** The bar's height, also set on the page as `--busy-bar` so the traffic console keeps its last line above it. */
export const BAR_HEIGHT = `${BAR_REM}rem`;

/** How far above the bottom edge toasts sit while the bar shows, so they stack above it instead of under it. */
export function toastOffset(busy: boolean): string {
  return busy ? `${BAR_REM + GAP_REM}rem` : "";
}

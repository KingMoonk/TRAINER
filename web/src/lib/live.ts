"use client";

import { useQueryClient, type QueryKey } from "@tanstack/react-query";
import { useLiveRefresh } from "@/lib/events";

/** Options for `useLive`: which events re-read the query and how. */
export type LiveOptions = {
  /** Only the events this accepts count (for `run` and `train`, the run or job the page shows). */
  match?: (kind: string, payload: unknown) => boolean;
  /** The slow safety net for a lost event, in milliseconds, or `false` for none. */
  backstopMs?: number | false;
  /** The least time between two re-reads, for kinds sent at every training step. */
  throttleMs?: number;
  /** Re-read only while this is true (a query that is not enabled has nothing to read). */
  enabled?: boolean;
};

/**
 * Re-read the queries under `queryKey` when the server says what they show changed (any event of `kinds`), when the stream comes
 * back after a drop, when the tab is shown again, and every minute as a backstop: the page follows the server without polling.
 */
export function useLive(queryKey: QueryKey, kinds: readonly string[], options: LiveOptions = {}): void {
  const client = useQueryClient();
  const enabled = options.enabled ?? true;
  useLiveRefresh(
    enabled ? kinds : [],
    () => (enabled ? client.invalidateQueries({ queryKey }) : undefined),
    enabled ? options.backstopMs : false,
    { throttleMs: options.throttleMs, match: options.match },
  );
}

/** A `match` that accepts an event whose payload's `field` is `value`, and every event of the kinds not in `kinds`. */
export function payloadIs(field: string, value: unknown, kinds: readonly string[]): (kind: string, payload: unknown) => boolean {
  return (kind, payload) => !kinds.includes(kind) || (typeof payload === "object" && payload !== null && (payload as Record<string, unknown>)[field] === value);
}

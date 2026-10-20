"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { BUSY } from "@/lib/copy/busy";
import { busyItems, fraction, queryLabel, ROW, SLOW_MS, toastOffset, type BusyItem } from "@/lib/busy";
import { useEvent } from "@/lib/events";

const FADE_MS = 300;

/** Reads this page's React Query cache is still waiting on after SLOW_MS, as indeterminate items labelled from the query's key. */
function useSlowReads(): BusyItem[] {
  const client = useQueryClient();
  const [slow, setSlow] = useState<BusyItem[]>([]);
  useEffect(() => {
    const cache = client.getQueryCache();
    const since = new Map<string, number>();
    const timers = new Map<string, ReturnType<typeof setTimeout>>();
    let shown = "";
    const sync = () => {
      const now = Date.now();
      const fetching = cache.findAll({ fetchStatus: "fetching" });
      const live = new Set(fetching.map((query) => query.queryHash));
      for (const hash of [...since.keys()]) {
        if (live.has(hash)) continue;
        since.delete(hash);
        clearTimeout(timers.get(hash));
        timers.delete(hash);
      }
      for (const hash of live) {
        if (since.has(hash)) continue;
        since.set(hash, now);
        timers.set(hash, setTimeout(sync, SLOW_MS));
      }
      const items = fetching
        .filter((query) => now - (since.get(query.queryHash) ?? now) >= SLOW_MS)
        .map((query) => ({ id: `read:${query.queryHash}`, label: queryLabel(query.queryKey), done: 0, total: null }));
      const signature = items.map((item) => item.id).join("|");
      if (signature === shown) return;
      shown = signature;
      setSlow(items);
    };
    const off = cache.subscribe(sync);
    sync();
    return () => {
      off();
      for (const timer of timers.values()) clearTimeout(timer);
    };
  }, [client]);
  return slow;
}

function Bar({ item }: { item: BusyItem }) {
  const part = fraction(item);
  return (
    <div className="flex flex-col justify-center gap-1.5" style={{ height: ROW.height }} data-testid="busy-item">
      <div
        role="progressbar"
        aria-label={item.label}
        aria-valuemin={part === null ? undefined : 0}
        aria-valuemax={part === null ? undefined : 100}
        aria-valuenow={part === null ? undefined : Math.round(part * 100)}
        className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
      >
        {part === null ? (
          <div className="busy-indeterminate h-full rounded-full bg-primary" data-testid="busy-indeterminate" />
        ) : (
          <div className="h-full rounded-full bg-primary transition-[width] duration-200" style={{ width: `${part * 100}%` }} />
        )}
      </div>
      <p className="truncate text-xs text-muted-foreground">
        {item.label}
        {part !== null && item.total !== null && item.total > 1 ? ` (${BUSY.stepOf.replace("{done}", String(item.done)).replace("{total}", String(item.total))})` : ""}
      </p>
    </div>
  );
}

/**
 * The card at the bottom right while anything is busy: the server's heavy work (`busy` events, GET /busy) and reads the browser has
 * waited on for SLOW_MS. A bar for each (filled to its real progress, or sliding when its steps are unknown) with what it is under it.
 * It fades out when idle, and tells the toasts how high to sit so it never covers them (`--busy-offset`, read by the Toaster).
 */
export function BusyIndicator() {
  const [server, setServer] = useState<BusyItem[]>([]);
  useEvent("busy", (payload) => setServer(busyItems(payload)));
  const slow = useSlowReads();
  const items = [...server, ...slow];
  const count = items.length;
  const signature = JSON.stringify(items);
  // The last non-empty list stays on screen while the card fades out, then goes.
  const [held, setHeld] = useState<{ signature: string; items: BusyItem[] }>({ signature: "", items: [] });
  if (count && held.signature !== signature) setHeld({ signature, items });
  useEffect(() => {
    if (count) return;
    const timer = setTimeout(() => setHeld({ signature: "", items: [] }), FADE_MS);
    return () => clearTimeout(timer);
  }, [count]);
  useEffect(() => {
    const root = document.documentElement.style;
    if (count) root.setProperty("--busy-offset", toastOffset(count));
    else root.removeProperty("--busy-offset");
    return () => {
      root.removeProperty("--busy-offset");
    };
  }, [count]);
  const shown = count ? items : held.items;
  if (!shown.length) return null;
  return (
    <section
      role="status"
      aria-live="polite"
      aria-label={BUSY.region}
      data-testid="busy-indicator"
      className={`fixed z-40 flex w-80 max-w-[calc(100vw-2rem)] flex-col rounded-xl border border-border bg-popover text-popover-foreground shadow-card transition-opacity duration-300 ${count ? "opacity-100" : "opacity-0"}`}
      style={{ right: ROW.edge, bottom: ROW.edge, gap: ROW.gap, padding: ROW.padding }}
    >
      {shown.map((item) => (
        <Bar key={item.id} item={item} />
      ))}
    </section>
  );
}

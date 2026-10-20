"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { BUSY } from "@/lib/copy/busy";
import { BAR_HEIGHT, busyItems, busyTitle, overall, queryLabel, SLOW_MS, toastOffset, type BusyItem } from "@/lib/busy";
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

/**
 * The loading bar along the bottom edge while anything is busy: the server's heavy work (`busy` events, GET /busy) and reads the browser
 * has waited on for SLOW_MS. One bar across the top of the strip (filled to the combined progress, or sliding when any step count is unknown)
 * and a title naming the first thing busy and how many more. It fades out when idle, and tells the toasts how high to sit so it never covers
 * them (`--busy-offset`, read by the Toaster) and the traffic console how much room to leave (`--busy-bar`).
 */
export function BusyIndicator() {
  const [server, setServer] = useState<BusyItem[]>([]);
  useEvent("busy", (payload) => setServer(busyItems(payload)));
  const slow = useSlowReads();
  const items = [...server, ...slow];
  const count = items.length;
  const signature = JSON.stringify(items);
  // The last non-empty list stays on screen while the bar fades out, then goes.
  const [held, setHeld] = useState<{ signature: string; items: BusyItem[] }>({ signature: "", items: [] });
  if (count && held.signature !== signature) setHeld({ signature, items });
  useEffect(() => {
    if (count) return;
    const timer = setTimeout(() => setHeld({ signature: "", items: [] }), FADE_MS);
    return () => clearTimeout(timer);
  }, [count]);
  const busy = count > 0;
  useEffect(() => {
    const root = document.documentElement.style;
    if (busy) {
      root.setProperty("--busy-offset", toastOffset(true));
      root.setProperty("--busy-bar", BAR_HEIGHT);
    } else {
      root.removeProperty("--busy-offset");
      root.removeProperty("--busy-bar");
    }
    return () => {
      root.removeProperty("--busy-offset");
      root.removeProperty("--busy-bar");
    };
  }, [busy]);
  const shown = count ? items : held.items;
  if (!shown.length) return null;
  const part = overall(shown);
  const title = busyTitle(shown);
  const everything = shown.map((item) => item.label).join(", ");
  return (
    <section
      role="status"
      aria-live="polite"
      aria-label={BUSY.region}
      data-testid="busy-indicator"
      className={`fixed inset-x-0 bottom-0 z-40 flex flex-col border-t border-border bg-popover text-popover-foreground shadow-card transition-opacity duration-300 md:left-60 ${count ? "opacity-100" : "opacity-0"}`}
      style={{ height: BAR_HEIGHT }}
    >
      <div
        role="progressbar"
        aria-label={everything}
        aria-valuemin={part === null ? undefined : 0}
        aria-valuemax={part === null ? undefined : 100}
        aria-valuenow={part === null ? undefined : Math.round(part * 100)}
        className="h-1 w-full shrink-0 overflow-hidden bg-muted"
      >
        {part === null ? (
          <div className="busy-indeterminate h-full bg-primary" data-testid="busy-indeterminate" />
        ) : (
          <div className="h-full bg-primary transition-[width] duration-200" style={{ width: `${part * 100}%` }} />
        )}
      </div>
      <p className="truncate px-4 text-xs leading-8 text-muted-foreground" title={everything} data-testid="busy-title">
        {title}
      </p>
    </section>
  );
}

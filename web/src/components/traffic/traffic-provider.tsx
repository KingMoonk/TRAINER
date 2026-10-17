"use client";

// The state of the API traffic console, kept above the pages (in app/layout.tsx) so the drawer survives navigation. While the console is
// open it reads GET /traffic (backfill, then live) and mirrors the server's ring buffer; closed, the stream is shut and only the alert
// rules on /events events keep working (the traffic rules need the stream, so they fire while the console is open).

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ApiError, api, unwrap } from "@/lib/api";
import { useAnyEvent } from "@/lib/events";
import { getStream } from "@/lib/sse-post";
import {
  NO_FILTERS, alertDetail, pushAlert, ruleFires, upsert,
  type Alert, type Filters, type Rule, type RuleMemory, type Signal, type TrafficRecord,
} from "@/lib/traffic";

export type Connection = "connecting" | "live" | "offline";
export type Hello = { node: string; size: number; count: number; started: number };
type Rules = { rules: Rule[] };

const RETRY_MS = 3000;
const FLUSH_MS = 100;
const EVALUATED_LIMIT = 10_000;
const TOAST = { success: toast.success, info: toast.info, warning: toast.warning, destructive: toast.error } as const;

export type TrafficState = {
  open: boolean;
  setOpen: (open: boolean) => void;
  toggle: () => void;
  /** Open the console from anywhere, optionally filtered (a path prefix, or free text such as a run id). */
  openConsole: (filter?: Partial<Pick<Filters, "prefix" | "search" | "node">>) => void;
  records: TrafficRecord[];
  hello: Hello | null;
  connection: Connection;
  paused: boolean;
  setPaused: (paused: boolean) => void;
  /** Records that arrived while paused and are held back. */
  held: number;
  clear: () => void;
  filters: Filters;
  setFilters: (filters: Filters) => void;
  alerts: Alert[];
  dismissAlert: (key: string) => void;
  clearAlerts: () => void;
  rules: Rule[] | undefined;
  rulesFailed: boolean;
  saveRules: (rules: Rule[]) => void;
  savingRules: boolean;
};

const TrafficContext = createContext<TrafficState | null>(null);

export function useTraffic(): TrafficState {
  const state = useContext(TrafficContext);
  if (!state) throw new Error("useTraffic is used outside TrafficProvider");
  return state;
}

export function TrafficProvider({ children }: { children: React.ReactNode }) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [records, setRecords] = useState<TrafficRecord[]>([]);
  const [hello, setHello] = useState<Hello | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [paused, setPausedState] = useState(false);
  const [held, setHeld] = useState(0);
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [alerts, setAlerts] = useState<Alert[]>([]);

  const pausedRef = useRef(false);
  const heldRecords = useRef<TrafficRecord[]>([]);
  const queue = useRef<TrafficRecord[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const evaluated = useRef(new Set<number>());
  const memory = useRef<RuleMemory>(new Map());
  const openRef = useRef(false);
  useEffect(() => {
    openRef.current = open;
  }, [open]);

  const rulesQuery = useQuery({ queryKey: ["traffic-rules"], queryFn: () => unwrap(api.GET("/traffic/rules")) as Promise<Rules>, retry: false, staleTime: Infinity });
  const rules = rulesQuery.data?.rules;
  const rulesRef = useRef<Rule[] | undefined>(undefined);
  useEffect(() => {
    rulesRef.current = rules;
  }, [rules]);

  const save = useMutation({
    mutationFn: (next: Rule[]) => unwrap(api.PUT("/traffic/rules", { body: { rules: next } })) as Promise<Rules>,
    onSuccess: (saved) => client.setQueryData(["traffic-rules"], saved),
    onError: () => toast.error("The alert rules could not be saved."),
  });

  const raise = useCallback((rule: Rule, signal: Signal) => {
    const detail = alertDetail(signal);
    setAlerts((current) => pushAlert(current, { rule: rule.id, label: rule.label, severity: rule.severity, detail, at: Date.now() }));
    if (!openRef.current) TOAST[rule.severity](rule.label, { id: `traffic-${rule.id}`, description: detail });
  }, []);

  const evaluate = useCallback((signal: Signal) => {
    for (const rule of rulesRef.current ?? []) if (ruleFires(rule, signal, memory.current)) raise(rule, signal);
  }, [raise]);

  useAnyEvent((event) => {
    if (event.kind === "traffic_rules") {
      const next = (event.payload as Rules | undefined)?.rules;
      if (next) client.setQueryData(["traffic-rules"], { rules: next });
      return;
    }
    evaluate({ source: "event", kind: event.kind, payload: event.payload });
  });

  const flush = useCallback(() => {
    timer.current = undefined;
    const batch = queue.current;
    queue.current = [];
    if (!batch.length) return;
    for (const record of batch) {
      if (record.state !== "closed" || evaluated.current.has(record.id)) continue;
      evaluated.current.add(record.id);
      evaluate({ source: "traffic", record });
    }
    if (evaluated.current.size > EVALUATED_LIMIT) evaluated.current.clear();
    if (pausedRef.current) {
      heldRecords.current = upsert(heldRecords.current, batch);
      setHeld(heldRecords.current.length);
    } else {
      setRecords((current) => upsert(current, batch));
    }
  }, [evaluate]);

  useEffect(() => {
    if (!open) return;
    const abort = new AbortController();
    void (async () => {
      while (!abort.signal.aborted) {
        let backfill = 0;
        try {
          await getStream("/traffic", (event) => {
            if (event.kind === "hello") {
              const next = event.payload as Hello;
              backfill = next.count;
              setHello(next);
            } else if (event.kind === "record") {
              const record = event.payload as TrafficRecord;
              if (backfill > 0) {  // what happened before the console opened is shown, never alerted on
                backfill -= 1;
                if (record.state === "closed") evaluated.current.add(record.id);
              }
              queue.current.push(record);
              timer.current ??= setTimeout(flush, FLUSH_MS);
            }
          }, abort.signal, () => setConnection("live"));
        } catch (error) {
          if (error instanceof ApiError && error.status === 401) {
            setConnection("offline");
            return;
          }
        }
        if (abort.signal.aborted) return;
        setConnection("offline");
        await new Promise((resolve) => setTimeout(resolve, RETRY_MS));
      }
    })();
    return () => {
      abort.abort();
      setConnection("connecting");
      clearTimeout(timer.current);
      timer.current = undefined;
      flush();
    };
  }, [open, flush]);

  const setPaused = useCallback((next: boolean) => {
    pausedRef.current = next;
    setPausedState(next);
    if (!next && heldRecords.current.length) {
      const release = heldRecords.current;
      heldRecords.current = [];
      setHeld(0);
      setRecords((current) => upsert(current, release));
    }
  }, []);

  const clear = useCallback(() => {
    queue.current = [];
    heldRecords.current = [];
    setHeld(0);
    setRecords([]);
  }, []);

  const openConsole = useCallback((filter?: Partial<Pick<Filters, "prefix" | "search" | "node">>) => {
    if (filter) setFilters((current) => ({ ...current, ...filter, hideNoise: filter.search || filter.prefix ? false : current.hideNoise }));
    setOpen(true);
  }, []);
  const toggle = useCallback(() => setOpen((current) => !current), []);

  // Ctrl+` toggles from any page. `?console=<text>` opens it filtered by that text, for links from other pages.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey && !event.metaKey && !event.altKey && (event.key === "`" || event.code === "Backquote")) {
        event.preventDefault();
        setOpen((current) => !current);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    const text = new URLSearchParams(window.location.search).get("console");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the URL can only be read after hydration (the page is a static export)
    if (text) openConsole(text.startsWith("/") ? { prefix: text } : { search: text });
  }, [openConsole]);

  const value = useMemo<TrafficState>(() => ({
    open, setOpen, toggle, openConsole, records, hello, connection, paused, setPaused, held, clear, filters, setFilters, alerts,
    dismissAlert: (key) => setAlerts((current) => current.filter((a) => a.key !== key)),
    clearAlerts: () => setAlerts([]),
    rules, rulesFailed: rulesQuery.isError, saveRules: (next) => save.mutate(next), savingRules: save.isPending,
  }), [open, toggle, openConsole, records, hello, connection, paused, setPaused, held, clear, filters, alerts, rules, rulesQuery.isError, save]);

  return <TrafficContext.Provider value={value}>{children}</TrafficContext.Provider>;
}

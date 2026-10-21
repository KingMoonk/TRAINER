"use client";

import { BellRing, Info, Pause, Play, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { AlertStack, RuleList } from "@/components/traffic/alerts-panel";
import { TrafficRow } from "@/components/traffic/traffic-row";
import { useTraffic } from "@/components/traffic/traffic-provider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TRAFFIC } from "@/lib/copy/traffic";
import { RING, duration, filterChoices, isSlow, matchesFilters, p95, reqPerSecond, type Filters, type StatusClass } from "@/lib/traffic";
import { cn } from "@/lib/utils";

const SHOWN = 300;  // rows drawn at once; the filters and search still cover the whole buffer
const STATUSES: ("all" | StatusClass)[] = ["all", "2xx", "3xx", "4xx", "5xx", "failed"];

function Pick({ label, value, all, options, onChange }: { label: string; value: string; all: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger size="sm" aria-label={label} className="h-7 w-auto min-w-24 text-xs"><SelectValue /></SelectTrigger>
      <SelectContent>
        <SelectItem value="all">{all}</SelectItem>
        {options.map((option) => <SelectItem key={option} value={option}>{option}</SelectItem>)}
      </SelectContent>
    </Select>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <span className="whitespace-nowrap text-muted-foreground">{label} <span className="font-mono text-foreground">{value}</span></span>;
}

export function TrafficConsole() {
  const t = useTraffic();
  const [showRules, setShowRules] = useState(false);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [shown, setShown] = useState(SHOWN);
  useEffect(() => {
    if (!t.open) return;
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(tick);
  }, [t.open]);

  const { records, filters } = t;
  const choices = useMemo(() => filterChoices(records), [records]);
  const matching = useMemo(() => records.filter((r) => matchesFilters(r, filters)), [records, filters]);
  const limit = useMemo(() => p95(records), [records]);
  const rows = useMemo(() => matching.slice(-shown).reverse(), [matching, shown]);
  if (!t.open) return null;

  const set = (patch: Partial<Filters>) => t.setFilters({ ...filters, ...patch });
  const size = t.hello?.size ?? RING;
  const live = t.connection === "live" && !t.paused;

  return (
    <section aria-label={TRAFFIC.title} className="isolate fixed inset-x-0 bottom-0 z-30 flex h-[45dvh] min-h-64 flex-col border-t pb-[var(--busy-bar,0px)] border-surface-border-strong bg-popover text-sm text-popover-foreground shadow-card md:left-60">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-surface-border px-3 py-1.5">
        <h2 className="flex items-center gap-2 font-medium">
          <span aria-hidden="true" className={cn("size-2 rounded-full", live ? "bg-success" : t.paused ? "bg-warning" : "bg-muted-foreground")} />
          {TRAFFIC.title}
          <span className="text-xs font-normal text-muted-foreground">{t.paused ? "paused" : t.connection === "live" ? "live" : t.connection}</span>
        </h2>
        <Stat label="req/s" value={reqPerSecond(records, now).toFixed(1)} />
        <Stat label="p95" value={limit === null ? "–" : duration(limit)} />
        <Stat label="buffer" value={`${records.length} / ${size}`} />
        {t.hello && <Stat label="node" value={t.hello.node} />}
        <div className="ml-auto flex items-center gap-1">
          <Button variant="outline" size="xs" onClick={() => t.setPaused(!t.paused)} aria-pressed={t.paused}>
            {t.paused ? <Play aria-hidden="true" /> : <Pause aria-hidden="true" />}{t.paused ? `Resume${t.held ? ` (${t.held} held)` : ""}` : "Pause"}
          </Button>
          <Button variant="outline" size="xs" onClick={t.clear}><Trash2 aria-hidden="true" />Clear</Button>
          <Button variant="outline" size="xs" onClick={() => setShowRules((v) => !v)} aria-pressed={showRules}><BellRing aria-hidden="true" />Alert rules</Button>
          <Button variant="ghost" size="icon-xs" onClick={() => t.setOpen(false)} aria-label="Close the traffic console"><X aria-hidden="true" /></Button>
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-2 border-b border-surface-border px-3 py-1.5">
        <div role="group" aria-label="Status" className="flex gap-1">
          {STATUSES.map((status) => (
            <Button key={status} size="xs" variant={filters.status === status ? "secondary" : "ghost"} aria-pressed={filters.status === status} onClick={() => set({ status })}>
              {status === "all" ? "All" : status}
            </Button>
          ))}
        </div>
        <Pick label="Method" value={filters.method} all="Any method" options={choices.methods} onChange={(method) => set({ method })} />
        <Pick label="Node" value={filters.node} all="Any node" options={choices.nodes} onChange={(node) => set({ node })} />
        <Pick label="Path" value={filters.prefix || "all"} all="Any path" options={choices.prefixes} onChange={(prefix) => set({ prefix: prefix === "all" ? "" : prefix })} />
        <div className="flex min-w-48 flex-1 items-center gap-1.5">
          <Input value={filters.search} onChange={(e) => set({ search: e.target.value })} placeholder={TRAFFIC.searchPlaceholder} aria-label="Search" className="h-7 max-w-md text-xs" />
          <Tooltip>
            <TooltipTrigger asChild><Info className="size-3.5 shrink-0 text-muted-foreground" aria-label={TRAFFIC.searchHint} tabIndex={0} /></TooltipTrigger>
            <TooltipContent>{TRAFFIC.searchHint}</TooltipContent>
          </Tooltip>
        </div>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch size="sm" checked={filters.hideNoise} onCheckedChange={(hideNoise) => set({ hideNoise })} />Hide health and polls
        </label>
      </div>

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div className="min-h-0 min-w-0 flex-1 overflow-y-auto px-3" data-testid="traffic-list">
          {t.connection === "offline" && <p role="status" className="py-2 text-xs text-warning">{TRAFFIC.offline}</p>}
          {rows.map((record) => <TrafficRow key={record.id} record={record} slow={isSlow(record, limit)} />)}
          {matching.length > shown && (
            <div className="py-2 text-center"><Button variant="ghost" size="xs" onClick={() => setShown(shown + SHOWN)}>Show older ({matching.length - shown} more)</Button></div>
          )}
          {rows.length === 0 && <p className="py-6 text-center text-xs text-muted-foreground">{records.length ? TRAFFIC.emptyFiltered : TRAFFIC.empty}</p>}
        </div>
        <aside aria-label="Alerts and rules" className="max-h-1/2 w-full shrink-0 space-y-4 overflow-y-auto border-t border-surface-border p-3 md:max-h-none md:w-80 md:border-t-0 md:border-l">
          <AlertStack alerts={t.alerts} onDismiss={t.dismissAlert} onClear={t.clearAlerts} />
          {showRules && <RuleList rules={t.rules} failed={t.rulesFailed} saving={t.savingRules} onChange={t.saveRules} />}
        </aside>
      </div>
    </section>
  );
}

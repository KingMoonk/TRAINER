"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Copy, Radio } from "lucide-react";
import { memo, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trace, TraceCollapsible, TraceCollapsibleContent, TraceCollapsibleTrigger, TraceDescription, TraceLine, TraceTitle } from "@/components/ui/trace";
import { fetchJson } from "@/lib/api";
import { BASE } from "@/lib/api-core";
import { TRAFFIC } from "@/lib/copy/traffic";
import { bytes, clock, curl, duration, route, statusClass, type BodyView, type StatusClass, type TrafficRecord } from "@/lib/traffic";
import { fields, modelSchema, operationId, problems, type Schema, type Spec } from "@/lib/traffic-schema";
import { cn } from "@/lib/utils";

const STATUS_TONE: Record<StatusClass, string> = {
  "2xx": "text-muted-foreground", "3xx": "text-muted-foreground", "4xx": "text-warning", "5xx": "text-destructive", failed: "text-destructive",
};

/** The OpenAPI document, fetched once when the first row is opened (it is too big to bundle). */
function useSpec(): Spec | undefined {
  return useQuery({ queryKey: ["openapi"], queryFn: () => fetchJson<Spec>("/openapi.json"), staleTime: Infinity, retry: false }).data;
}

function Value({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null) return <span className="text-muted-foreground">null</span>;
  if (typeof value === "number") return <span className="font-mono text-info">{value}</span>;
  if (typeof value === "boolean") return <span className="font-mono text-warning">{String(value)}</span>;
  if (typeof value === "string") return <span className="font-mono break-all text-foreground">{JSON.stringify(value)}</span>;
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v] as const) : Object.entries(value as Record<string, unknown>);
  const label = Array.isArray(value) ? `list of ${entries.length}` : `${entries.length} field${entries.length === 1 ? "" : "s"}`;
  if (!entries.length) return <span className="text-muted-foreground">{Array.isArray(value) ? "[]" : "{}"}</span>;
  return (
    <details open={depth < 1 && entries.length <= 6} className="min-w-0">
      <summary className="cursor-pointer text-muted-foreground">{label}</summary>
      <dl className="ml-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 border-l border-surface-border pl-3">
        {entries.slice(0, 200).map(([key, item]) => (
          <div key={key} className="contents"><dt className="font-mono text-muted-foreground">{key}</dt><dd className="min-w-0"><Value value={item} depth={depth + 1} /></dd></div>
        ))}
      </dl>
    </details>
  );
}

function Typed({ view, model, spec }: { view: BodyView; model: string | null; spec: Spec | undefined }) {
  if (view.json === undefined && !view.form) {
    const why = view.binary ? TRAFFIC.binary : view.invalid ? TRAFFIC.invalid : view.truncated ? TRAFFIC.oversize : "";
    return <p className="text-muted-foreground">{view.type || "body"} · {bytes(view.bytes)}{why && ` · ${why}`}</p>;
  }
  const body = view.json ?? view.form;
  const schema: Schema | undefined = spec ? modelSchema(model, spec) : undefined;
  const wrong = spec && schema ? problems(body, schema, spec) : [];
  return (
    <div className="space-y-1.5">
      {wrong.length > 0 && (
        <p role="note" className="flex items-start gap-1.5 text-warning"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
          <span>Does not match {model}: {wrong.join("; ")}</span></p>
      )}
      {body && typeof body === "object" && !Array.isArray(body) ? (
        <dl className="grid grid-cols-[auto_auto_1fr] gap-x-3 gap-y-0.5">
          {fields(body as Record<string, unknown>, schema, spec ?? {}).map((f) => (
            <div key={f.name} className="contents">
              <dt className="font-mono text-muted-foreground">{f.name}</dt>
              <dd className="text-[11px] text-muted-foreground/70">{f.type}</dd>
              <dd className="min-w-0"><Value value={f.value} /></dd>
            </div>
          ))}
        </dl>
      ) : <Value value={body} />}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="space-y-1"><h4 className="text-[11px] font-medium tracking-widest text-muted-foreground uppercase">{title}</h4>{children}</section>;
}

function Headers({ headers }: { headers: Record<string, string> }) {
  const entries = Object.entries(headers);
  if (!entries.length) return null;
  return (
    <details>
      <summary className="cursor-pointer text-muted-foreground">Headers ({entries.length})</summary>
      <dl className="ml-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 pl-3">
        {entries.map(([k, v]) => <div key={k} className="contents"><dt className="font-mono text-muted-foreground">{k}</dt><dd className="font-mono break-all">{v}</dd></div>)}
      </dl>
    </details>
  );
}

function Detail({ record }: { record: TrafficRecord }) {
  const spec = useSpec();
  const id = spec ? operationId(record, spec) : null;
  const query = Object.entries(record.request.query);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(curl(record, BASE || window.location.origin));
      toast.success("Copied as curl. Secrets are masked: set $GROUNDWORK_TOKEN.");
    } catch {
      toast.error("The clipboard is not available here.");
    }
  };
  return (
    <div className="space-y-3 py-1 text-xs">
      <p className="flex flex-wrap gap-x-4 gap-y-1 text-muted-foreground">
        <span>{route(record)}</span>
        {(id ?? record.handler) && <span>operation <span className="font-mono text-foreground">{id ?? record.handler}</span></span>}
        {record.route && record.route !== record.path && <span>route <span className="font-mono text-foreground">{record.route}</span></span>}
        {record.request_model && <span>request <span className="font-mono text-foreground">{record.request_model}</span></span>}
        {record.model && <span>response <span className="font-mono text-foreground">{record.model}</span></span>}
        {record.synthetic && <Badge variant="outline">simulated (--fake)</Badge>}
      </p>
      {record.error && <p className="text-destructive">{record.error}</p>}
      {query.length > 0 && <Section title="Query"><Value value={Object.fromEntries(query)} /></Section>}
      {record.request.body && <Section title="Request body"><Typed view={record.request.body} model={record.request_model} spec={spec} /></Section>}
      <Headers headers={record.request.headers} />
      {record.response.body && <Section title="Response body"><Typed view={record.response.body} model={record.model} spec={spec} /></Section>}
      <Headers headers={record.response.headers} />
      <Button variant="outline" size="xs" onClick={copy}><Copy aria-hidden="true" />Copy as curl</Button>
    </div>
  );
}

function RowView({ record, slow }: { record: TrafficRecord; slow: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const cls = statusClass(record);
  const failed = cls === "5xx" || cls === "failed";
  return (
    <Trace className="border-b border-surface-border py-1 font-mono text-xs" data-testid="traffic-row">
      <TraceCollapsible open={expanded} onOpenChange={setExpanded}>
        <TraceCollapsibleTrigger className="w-full" aria-label={`${record.method} ${record.path}`}>
          <TraceLine className="w-full gap-3 text-xs" status={failed ? "error" : record.state === "open" ? "running" : undefined}>
            <TraceDescription className="w-[88px] shrink-0">{clock(record.ts)}</TraceDescription>
            <span className={cn("w-12 shrink-0 font-medium", record.direction === "out" && "text-info")}>{record.method}</span>
            <TraceTitle className="flex-1 text-foreground">{record.path}</TraceTitle>
            {record.streaming && <Radio className="size-3.5 shrink-0 text-info" aria-label="stream" />}
            <span className="hidden w-40 shrink-0 truncate text-muted-foreground lg:block">{route(record)}</span>
            <span className={cn("w-12 shrink-0 text-right", cls && STATUS_TONE[cls])}>{record.status ?? (record.state === "open" ? "open" : "–")}</span>
            <span className={cn("w-16 shrink-0 text-right text-muted-foreground", slow && "text-warning")}>{slow && "slow "}{duration(record.ms)}</span>
          </TraceLine>
        </TraceCollapsibleTrigger>
        <TraceCollapsibleContent><Detail record={record} /></TraceCollapsibleContent>
      </TraceCollapsible>
    </Trace>
  );
}

export const TrafficRow = memo(RowView);

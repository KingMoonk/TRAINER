"use client";

import { CircleAlert, CircleCheck, Info, TriangleAlert, X } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { TRAFFIC } from "@/lib/copy/traffic";
import type { Alert as AlertItem, Rule, Severity } from "@/lib/traffic";
import { cn } from "@/lib/utils";

const ICON = { success: CircleCheck, info: Info, warning: TriangleAlert, destructive: CircleAlert } as const;
const TONE: Record<Severity, string> = { success: "text-success", info: "text-info", warning: "text-warning", destructive: "text-destructive" };

export function AlertStack({ alerts, onDismiss, onClear }: { alerts: AlertItem[]; onDismiss: (key: string) => void; onClear: () => void }) {
  return (
    <section aria-label="Alerts" className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-medium tracking-widest text-muted-foreground uppercase">Alerts</h3>
        {alerts.length > 0 && <Button variant="ghost" size="xs" onClick={onClear}>Clear</Button>}
      </div>
      {alerts.length === 0 && <p className="text-xs text-muted-foreground">{TRAFFIC.noAlerts}</p>}
      {alerts.map((alert) => {
        const Icon = ICON[alert.severity];
        return (
          <Alert key={alert.key} variant={alert.severity === "destructive" ? "destructive" : "default"} className="relative pr-8">
            <Icon aria-hidden="true" className={TONE[alert.severity]} />
            <AlertTitle>{alert.label}{alert.count > 1 && <span className="ml-1.5 text-xs font-normal text-muted-foreground">×{alert.count}</span>}</AlertTitle>
            <AlertDescription className="text-xs break-words">{alert.detail}</AlertDescription>
            <Button variant="ghost" size="icon-xs" className="absolute top-1.5 right-1.5" aria-label={`Dismiss ${alert.label}`} onClick={() => onDismiss(alert.key)}>
              <X aria-hidden="true" />
            </Button>
          </Alert>
        );
      })}
    </section>
  );
}

function describe(rule: Rule): string {
  if (rule.source === "event") return [`event ${rule.kind}`, rule.where && (rule.new_min ? `${rule.where} new low` : `${rule.where} = ${String(rule.equals)}`)].filter(Boolean).join(", ");
  return [rule.method, rule.path ?? "any path", rule.status, rule.where && `${rule.where} = ${String(rule.equals)}`].filter(Boolean).join(" ");
}

export function RuleList({ rules, failed, saving, onChange }: { rules: Rule[] | undefined; failed: boolean; saving: boolean; onChange: (rules: Rule[]) => void }) {
  return (
    <section aria-label="Alert rules" className="space-y-2">
      <h3 className="text-[11px] font-medium tracking-widest text-muted-foreground uppercase">{TRAFFIC.rulesTitle}</h3>
      <p className="text-xs text-muted-foreground">{TRAFFIC.rulesHint}</p>
      {failed && <p className="text-xs text-destructive">The rules could not be loaded from this node.</p>}
      <ul className="space-y-1.5">
        {(rules ?? []).map((rule) => (
          <li key={rule.id} className="flex items-start gap-2 text-xs">
            <Switch size="sm" className="mt-0.5" checked={rule.enabled} disabled={saving}
              aria-label={rule.label} onCheckedChange={(enabled) => onChange((rules ?? []).map((r) => (r.id === rule.id ? { ...r, enabled } : r)))} />
            <span className="min-w-0">
              <span className={cn("block", TONE[rule.severity])}>{rule.label}</span>
              <span className="block truncate text-muted-foreground" title={describe(rule)}>{describe(rule)}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

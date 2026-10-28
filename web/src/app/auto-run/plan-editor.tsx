"use client";

import { useState } from "react";
import { Disclosure } from "@/components/disclosure";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { T, fill } from "./copy";
import { PHASES, buildPlan, budgetText, phaseTitle, settingTexts, textKey, type PhasePlan, type Plan } from "./logic";

const FIELD = "h-9 w-fit max-w-full rounded-md border border-input bg-background px-2 text-sm text-foreground";

export type EditorAction = { label: string; primary?: boolean; run: (plan: Plan) => void };

/**
 * The plan of a draft or paused run: which phases to include, the settings the server reads for each, what to do when one passes or
 * fails, and the overall time budget. `actions` are the buttons; each gets the plan the choices make, or the editor shows what to fix.
 */
export function PlanEditor({ initial, actions, pending }: { initial: Plan; actions: EditorAction[]; pending: boolean }) {
  const [plan, setPlan] = useState<Plan>(initial);
  const [texts, setTexts] = useState<Record<string, string>>(() => settingTexts(initial));
  const [budget, setBudget] = useState(() => budgetText(initial));
  const [problem, setProblem] = useState<string | null>(null);

  const change = (kind: string, patch: Partial<PhasePlan>) => {
    setProblem(null);
    setPlan((p) => ({ ...p, phases: p.phases.map((phase) => (phase.kind === kind ? { ...phase, ...patch } : phase)) }));
  };

  function submit(action: EditorAction) {
    const built = buildPlan(plan, texts, budget);
    if ("error" in built) setProblem(built.error);
    else {
      setProblem(null);
      action.run(built.plan);
    }
  }

  return (
    <div className="flex flex-col gap-4" data-testid="plan-editor">
      <div className="flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-sm">
          {T.nameLabel}
          <Input value={plan.name} onChange={(e) => { setProblem(null); setPlan({ ...plan, name: e.target.value }); }} className="w-64 max-w-full" />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          {T.budgetLabel}
          <Input value={budget} inputMode="decimal" onChange={(e) => { setProblem(null); setBudget(e.target.value); }} className="w-40" placeholder="no limit" />
        </label>
      </div>
      <p className="max-w-prose text-xs text-muted-foreground">{T.budgetHint}</p>

      <h2 className="text-sm font-medium">{T.phasesTitle}</h2>
      <ul className="flex flex-col gap-3">
        {plan.phases
          .map((phase) => {
            const spec = PHASES[phase.kind];
            const title = phaseTitle(phase.kind);
            return (
              <li key={phase.kind}>
                <Card data-testid={`plan-${phase.kind}`}>
                  <CardHeader className="flex flex-row items-start justify-between gap-3">
                    <div className="flex min-w-0 flex-col gap-1">
                      <CardTitle className="text-sm">{title}</CardTitle>
                      <p className="text-sm text-muted-foreground">{spec?.blurb}</p>
                    </div>
                    <Switch aria-label={fill(T.enable, { phase: title })} checked={phase.enabled} onCheckedChange={(enabled) => change(phase.kind, { enabled })} />
                  </CardHeader>
                  {phase.enabled && (
                    <CardContent className="flex flex-col gap-3">
                      <div className="flex flex-wrap gap-4">
                        <label className="flex flex-col gap-1 text-sm">
                          {T.onPassLabel}
                          <select aria-label={`${T.onPassLabel}, ${title}`} className={FIELD} value={phase.on_pass} onChange={(e) => change(phase.kind, { on_pass: e.target.value })}>
                            {Object.entries(T.onPass).map(([value, label]) => (
                              <option key={value} value={value}>{label}</option>
                            ))}
                          </select>
                        </label>
                        <label className="flex flex-col gap-1 text-sm">
                          {T.onFailLabel}
                          <select aria-label={`${T.onFailLabel}, ${title}`} className={FIELD} value={phase.on_fail} onChange={(e) => change(phase.kind, { on_fail: e.target.value })}>
                            {Object.entries(T.onFail).map(([value, label]) => (
                              <option key={value} value={value}>{label}</option>
                            ))}
                          </select>
                        </label>
                      </div>
                      {spec && spec.settings.length > 0 && (
                        <Disclosure title="Settings" hint="Leave a setting empty to use the default.">
                          <div className="grid gap-3 sm:grid-cols-2">
                            {spec.settings.map((setting) => (
                              <label key={setting.key} className="flex flex-col gap-1 text-sm">
                                {setting.label}
                                <Input
                                  aria-label={`${setting.label}, ${title}`}
                                  inputMode="decimal"
                                  value={texts[textKey(phase.kind, setting.key)] ?? ""}
                                  placeholder={setting.fallback == null ? T.settingEmptyDefault : fill(T.settingEmpty, { fallback: setting.fallback })}
                                  onChange={(e) => { setProblem(null); setTexts({ ...texts, [textKey(phase.kind, setting.key)]: e.target.value }); }}
                                />
                                <span className="text-xs text-muted-foreground">{setting.hint}</span>
                              </label>
                            ))}
                          </div>
                        </Disclosure>
                      )}
                    </CardContent>
                  )}
                </Card>
              </li>
            );
          })}
      </ul>

      {problem && (
        <p role="alert" className="text-sm text-destructive">
          {problem}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {actions.map((action) => (
          <Button key={action.label} variant={action.primary ? "default" : "outline"} disabled={pending} onClick={() => submit(action)}>
            {action.label}
          </Button>
        ))}
      </div>
    </div>
  );
}

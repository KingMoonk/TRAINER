"use client";

// The start form's "when the run stalls" section: the plateau window and gain, the changes to try in order, and the replay share that
// goes with moving on to new text. What it sends is built in rules.ts.
import { TRAINING as T } from "@/lib/copy/training";
import { PIVOTS, planText, rulesProblem, withPivot, type RulesForm } from "./rules";

const FIELD = "h-9 w-32 rounded-md border border-input bg-background px-2 text-sm text-foreground";

function NumberField({ id, label, help, value, onChange, step }: { id: string; label: string; help: string; value: string; onChange: (value: string) => void; step: string }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      <input id={id} inputMode="decimal" type="number" min="0" step={step} value={value} onChange={(e) => onChange(e.target.value)} className={FIELD} />
      <p className="max-w-prose text-xs text-muted-foreground">{help}</p>
    </div>
  );
}

export function RulesFields({ rules, polishEvery, rounds, onChange }: { rules: RulesForm; polishEvery: string; rounds: string; onChange: (rules: RulesForm) => void }) {
  const set = (patch: Partial<RulesForm>) => onChange({ ...rules, ...patch });
  const why = rulesProblem(rules, polishEvery, rounds);
  const plan = planText(rules, polishEvery, rounds);
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="text-sm font-medium">{T.moveOn}</legend>
      <p className="max-w-prose text-xs text-muted-foreground">{T.moveOnLead}</p>
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <NumberField id="training-plateau-window" label={T.plateauWindow} help={T.plateauWindowHelp} value={rules.plateauChecks} onChange={(plateauChecks) => set({ plateauChecks })} step="1" />
        <NumberField id="training-plateau-gain" label={T.plateauGain} help={T.plateauGainHelp} value={rules.plateauGain} onChange={(plateauGain) => set({ plateauGain })} step="0.1" />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-sm font-medium">{T.pivotsTitle}</span>
        <p className="max-w-prose text-xs text-muted-foreground">{T.pivotsHelp}</p>
        {PIVOTS.map((pivot) => (
          <label key={pivot} className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={rules.pivots.includes(pivot)} onChange={(e) => set({ pivots: withPivot(rules.pivots, pivot, e.target.checked) })} />
            <span>{T.pivotChoices[pivot]}</span>
            {rules.pivots.includes(pivot) && <span className="text-xs text-muted-foreground">{rules.pivots.indexOf(pivot) + 1}</span>}
          </label>
        ))}
      </div>
      {rules.pivots.includes("next_dataset") && (
        <NumberField id="training-replay" label={T.replay} help={T.replayHelp} value={rules.replay} onChange={(replay) => set({ replay })} step="1" />
      )}
      {plan && <p role="status" className="max-w-prose text-sm">{plan}</p>}
      {why && <p role="status" className="max-w-prose text-sm text-destructive">{why}</p>}
    </fieldset>
  );
}

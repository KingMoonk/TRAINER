"use client";

// The training settings of the model setup page: batch size, accumulation, learning rate, weight decay, warm-up, dropout,
// evaluate-every, seed and precision. They edit the draft like the model controls; a value that is not allowed stays in the box
// with a sentence under it and is never sent on, so the draft always holds what the server can write.

import { useId, useState } from "react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { Choices, Registry } from "@/lib/builder";
import { BUILDER, fill } from "@/lib/copy/builder";
import { TRAINING_FIELDS, checkSetting, settingValue, showNumber, type TrainingField } from "@/lib/training-settings";
import { Panel } from "./side";

type SettingProps = {
  field: TrainingField;
  value: number | null;
  saved: number | null;
  onChange: (value: number) => void;
};

function Setting({ field, value, saved, onChange }: SettingProps) {
  const id = useId();
  const [text, setText] = useState(value === null ? "" : showNumber(value));
  const [error, setError] = useState("");
  // The draft changed from outside (discard, a preset, an earlier version): show it, unless the box already says it.
  const [seen, setSeen] = useState(value);
  if (value !== seen) {
    setSeen(value);
    const checked = checkSetting(field, text);
    if (value !== null && !("value" in checked && checked.value === value)) setText(showNumber(value));
    setError("");
  }
  const unavailable = value === null;
  const changed = value !== null && saved !== null && value !== saved;
  return (
    <div className="flex min-w-0 flex-col gap-1" data-testid={`training-${field.key}`}>
      <label htmlFor={id} className="text-xs font-medium">
        {field.label}
      </label>
      <Input
        id={id}
        type="text"
        inputMode="decimal"
        value={unavailable ? "" : text}
        placeholder={unavailable ? "—" : undefined}
        disabled={unavailable}
        aria-invalid={error ? true : undefined}
        aria-describedby={`${id}-help`}
        className="h-9"
        onChange={(event) => {
          const next = event.currentTarget.value;
          setText(next);
          const checked = checkSetting(field, next);
          if ("error" in checked) setError(checked.error);
          else {
            setError("");
            if (checked.value !== value) onChange(checked.value);
          }
        }}
      />
      <p id={`${id}-help`} className="text-xs text-muted-foreground">
        {unavailable ? BUILDER.training_no_warmup : field.help}
      </p>
      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}
      {changed && !error && <p className="text-xs text-muted-foreground">{fill(BUILDER.control_saved, { value: showNumber(saved) })}</p>}
    </div>
  );
}

type Props = {
  registry: Registry;
  cur: Choices;
  draft: Choices;
  onChange: (patch: Partial<Choices>) => void;
};

export function TrainingSettings({ registry, cur, draft, onChange }: Props) {
  const id = useId();
  const precisions = registry.precisions;
  const picked = precisions.find((p) => p.short === draft.precision);
  const savedPrecision = precisions.find((p) => p.short === cur.precision);
  return (
    <Panel title={BUILDER.training_settings} testId="training-settings">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
        {TRAINING_FIELDS.map((field) => (
          <Setting
            key={field.key}
            field={field}
            value={settingValue(draft, field.key)}
            saved={settingValue(cur, field.key)}
            onChange={(value) => onChange({ [field.key]: value })}
          />
        ))}
        <div className="flex min-w-0 flex-col gap-1" data-testid="training-precision">
          <label htmlFor={id} className="text-xs font-medium">
            {BUILDER.training_precision}
          </label>
          <Select value={draft.precision} onValueChange={(precision) => onChange({ precision })}>
            <SelectTrigger id={id} size="sm" className="w-full" aria-label={BUILDER.training_precision}>
              <SelectValue placeholder={draft.precision} />
            </SelectTrigger>
            <SelectContent>
              {precisions.map((p) => (
                <SelectItem key={p.short} value={p.short} title={p.blurb}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">{picked?.blurb ?? BUILDER.training_precision_help}</p>
          {savedPrecision && draft.precision !== cur.precision && (
            <p className="text-xs text-muted-foreground">{fill(BUILDER.control_saved, { value: savedPrecision.name })}</p>
          )}
        </div>
      </div>
      <p className="text-xs text-muted-foreground">{BUILDER.training_note}</p>
    </Panel>
  );
}

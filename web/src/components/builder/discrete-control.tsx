"use client";

import { useId } from "react";
import { Button } from "@/components/ui/button";
import { BUILDER, fill } from "@/lib/copy/builder";
import { cn } from "@/lib/utils";

type Props = {
  label: string;
  help: string;
  options: readonly (string | number)[];
  value: number;
  saved: number;
  disabled?: boolean;
  onChange: (index: number) => void;
  testId: string;
};

/** The range uses registry indices so every step is a supported model choice. */
export function DiscreteControl({ label, help, options, value, saved, disabled, onChange, testId }: Props) {
  const id = useId();
  const valid = Number.isInteger(value) && value >= 0 && value < options.length;
  const unavailable = disabled || !valid || options.length < 2;
  const changed = valid && saved !== value && options[saved] !== undefined;
  const choose = (index: number) => {
    if (!unavailable && Number.isInteger(index) && index >= 0 && index < options.length) onChange(index);
  };
  return (
    <div className="min-w-0 py-1" data-testid={testId}>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-sm font-medium">{label}</label>
        <span className="text-lg font-semibold tabular-nums" data-testid={`${testId}-value`}>{valid ? options[value] : "—"}</span>
      </div>
      <p id={`${id}-help`} className="mt-1 text-xs leading-relaxed text-muted-foreground">{help}</p>
      <div className="mt-2 flex items-center gap-2">
        <Button type="button" variant="outline" size="icon" className="size-11" aria-label={fill(BUILDER.control_decrease, { name: label })} disabled={unavailable || value === 0} onClick={() => choose(value - 1)}>−</Button>
        <input
          id={id}
          type="range"
          min={0}
          max={Math.max(0, options.length - 1)}
          step={1}
          value={valid ? value : 0}
          aria-valuetext={valid ? String(options[value]) : BUILDER.control_unavailable}
          aria-describedby={`${id}-help`}
          disabled={unavailable}
          onChange={(event) => choose(Number(event.target.value))}
          className="h-11 w-full min-w-0 cursor-pointer accent-primary focus-visible:rounded focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-40"
        />
        <Button type="button" variant="outline" size="icon" className="size-11" aria-label={fill(BUILDER.control_increase, { name: label })} disabled={unavailable || value === options.length - 1} onClick={() => choose(value + 1)}>+</Button>
      </div>
      <div className="mt-1 flex flex-wrap gap-1" role="group" aria-label={fill(BUILDER.control_choices, { name: label })}>
        {options.map((option, index) => (
          <Button key={index} type="button" variant={index === value ? "secondary" : "ghost"} size="sm" className={cn("min-h-9 min-w-9 tabular-nums", changed && index === saved && "border-part-own")} aria-pressed={index === value} disabled={disabled || !valid} onClick={() => choose(index)}>{option}</Button>
        ))}
      </div>
      {changed && <p className="mt-2 text-xs text-muted-foreground">{fill(BUILDER.control_saved, { value: options[saved] })}</p>}
    </div>
  );
}

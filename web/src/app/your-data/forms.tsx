"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FIELDS, RULE_LABELS, T, type Kind } from "./copy";
import { allowedRules, checkFields } from "./logic";

export const INPUT = "w-full rounded-md border bg-background px-2 py-1.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50";

/**
 * The fields of one item (a document, a conversation or a problem): the Tk page asked for them one dialog at a time. `initial`
 * is the item's current fields when editing. `onSubmit` gets the fields; the card shows `error` (the server's sentence) under them.
 */
export function ItemForm({
  kind,
  title,
  initial,
  submitLabel,
  busy,
  error,
  onSubmit,
  onCancel,
}: {
  kind: Kind;
  title: string;
  initial?: Record<string, string>;
  submitLabel: string;
  busy: boolean;
  error?: string;
  onSubmit: (fields: Record<string, string>) => void;
  onCancel: () => void;
}) {
  const [values, setValues] = useState<Record<string, string>>(() => ({ ...Object.fromEntries(FIELDS[kind].map((f) => [f.name, ""])), ...initial }));
  const [problem, setProblem] = useState<string | null>(null);
  const rules = allowedRules(initial?.rule);
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const checked = checkFields(kind, values, initial?.rule);
    if ("error" in checked) setProblem(checked.error);
    else {
      setProblem(null);
      onSubmit(checked.fields);
    }
  };
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="flex flex-col gap-3">
          {FIELDS[kind].map((field) => (
            <label key={field.name} className="flex flex-col gap-1 text-sm">
              <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{field.label}</span>
              {kind === "problems" && field.name === "rule" ? (
                <select className={INPUT} value={values.rule} onChange={(e) => setValues({ ...values, rule: e.target.value })}>
                  <option value="" disabled>
                    Pick a rule
                  </option>
                  {rules.map((key) => (
                    <option key={key} value={key}>
                      {RULE_LABELS[key]}
                    </option>
                  ))}
                </select>
              ) : field.long ? (
                <textarea className={`${INPUT} min-h-24`} value={values[field.name]} onChange={(e) => setValues({ ...values, [field.name]: e.target.value })} />
              ) : (
                <input className={INPUT} value={values[field.name]} onChange={(e) => setValues({ ...values, [field.name]: e.target.value })} />
              )}
            </label>
          ))}
          {(problem || error) && (
            <p role="alert" className="text-sm text-destructive">
              {problem ?? error}
            </p>
          )}
          <div className="flex gap-2">
            <Button type="submit" disabled={busy}>
              {submitLabel}
            </Button>
            <Button type="button" variant="outline" onClick={onCancel}>
              {T.cancel}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

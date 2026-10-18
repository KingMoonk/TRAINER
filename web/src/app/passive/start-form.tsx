"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, unwrap } from "@/lib/api";
import { problem } from "@/lib/cluster";
import { PASSIVE } from "./copy";
import { INITIAL_FORM, UNITS, startBody, toggled, type StartBody, type StartForm } from "./start";

const FIELD = "h-9 rounded-md border border-input bg-background px-2 text-sm text-foreground";

const message = problem;

function Number_({ id, label, help, value, onChange, step }: { id: string; label: string; help?: string; value: string; onChange: (value: string) => void; step: string }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      <input id={id} type="number" inputMode="decimal" min="0" step={step} value={value} onChange={(e) => onChange(e.target.value)} className={`${FIELD} w-32`} />
      {help && <p className="text-xs text-muted-foreground">{help}</p>}
    </div>
  );
}

function Check({ id, label, checked, onChange }: { id: string; label: string; checked: boolean; onChange: (on: boolean) => void }) {
  return (
    <label htmlFor={id} className="flex items-start gap-2 text-sm">
      <input id={id} type="checkbox" className="mt-1" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

/** The New session form: what to train, on what, when to stop, and the machine's limits. Posts to /passive; the server checks the plan. */
export function StartSession({ onStarted }: { onStarted: () => void }) {
  const versions = useQuery({ queryKey: ["versions"], queryFn: () => unwrap(api.GET("/versions")) });
  const sets = useQuery({ queryKey: ["pretraining", "text_sets"], queryFn: () => unwrap(api.GET("/pretraining/text_sets")) });
  const [form, setForm] = useState<StartForm>(INITIAL_FORM);
  const [failure, setFailure] = useState("");
  const change = (patch: Partial<StartForm>) => {
    setForm((f) => ({ ...f, ...patch }));
    setFailure("");
  };
  const start = useMutation({
    mutationFn: (body: StartBody) => unwrap(api.POST("/passive", { body })),
    onSuccess: onStarted,
    onError: (error) => setFailure(message(error)),
  });
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const sent = startBody(form);
    if ("problem" in sent) setFailure(sent.problem);
    else start.mutate(sent.body);
  };

  const trainable = (sets.data ?? []).filter((s) => !s.held);
  const loading = versions.isPending || sets.isPending;
  const noVersions = versions.isSuccess && versions.data.length === 0;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{PASSIVE.formTitle}</CardTitle>
        <CardDescription>{PASSIVE.formLead}</CardDescription>
      </CardHeader>
      <CardContent>
        {versions.isError || sets.isError ? (
          <p role="alert" className="text-sm text-destructive">
            {PASSIVE.formFailed} {message(versions.error ?? sets.error)}
          </p>
        ) : loading ? (
          <p className="text-sm text-muted-foreground">{PASSIVE.formLoading}</p>
        ) : (
          <form onSubmit={submit} aria-label={PASSIVE.formTitle} className="flex flex-col gap-5">
            <div className="flex flex-col gap-1">
              <label htmlFor="passive-version" className="text-sm font-medium">
                {PASSIVE.version}
              </label>
              {noVersions ? (
                <p className="text-sm text-muted-foreground">{PASSIVE.noVersions}</p>
              ) : (
                <select id="passive-version" value={form.version} onChange={(e) => change({ version: e.target.value })} className={`${FIELD} w-fit max-w-full`}>
                  <option value="">{PASSIVE.versionCurrent}</option>
                  {versions.data!.map((v) => (
                    <option key={v.number} value={String(v.number)}>
                      {PASSIVE.versionOption(v.number, v.name ?? "")}
                    </option>
                  ))}
                </select>
              )}
            </div>

            <fieldset className="flex flex-col gap-1">
              <legend className="text-sm font-medium">{PASSIVE.sources}</legend>
              {trainable.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  {PASSIVE.noSources} <Link href="/your-data" className="underline underline-offset-2">Your data</Link>
                </p>
              ) : (
                <>
                  <Check id="passive-all" label={PASSIVE.allSources} checked={form.all} onChange={(all) => change({ all })} />
                  {!form.all &&
                    trainable.map((s) => (
                      <label key={s.id} className="ml-6 flex items-center gap-2 text-sm">
                        <input type="checkbox" checked={form.sources.includes(s.id)} onChange={(e) => change({ sources: toggled(form.sources, s.id, e.target.checked) })} />
                        <span>{s.name}</span>
                        <span className="text-xs text-muted-foreground">{s.count.toLocaleString("en-US")}</span>
                      </label>
                    ))}
                </>
              )}
            </fieldset>

            <fieldset className="flex flex-col gap-2">
              <legend className="text-sm font-medium">{PASSIVE.stopRule}</legend>
              <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
                <div className="flex flex-col gap-1">
                  <label htmlFor="passive-duration" className="text-sm font-medium">
                    {PASSIVE.duration}
                  </label>
                  <div className="flex gap-2">
                    <input id="passive-duration" type="number" inputMode="decimal" min="0" step="any" value={form.duration} onChange={(e) => change({ duration: e.target.value })} className={`${FIELD} w-24`} />
                    <select aria-label="Time unit" value={form.unit} onChange={(e) => change({ unit: e.target.value as StartForm["unit"] })} className={FIELD}>
                      {UNITS.map((unit) => (
                        <option key={unit} value={unit}>
                          {PASSIVE[unit]}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <Number_ id="passive-patience" label={PASSIVE.patience} help={PASSIVE.patienceHelp} value={form.patience} onChange={(patience) => change({ patience })} step="1" />
                <Number_ id="passive-target" label={PASSIVE.target} help={PASSIVE.targetHelp} value={form.target} onChange={(target) => change({ target })} step="any" />
              </div>
            </fieldset>

            <div className="flex flex-col gap-2">
              <Check id="passive-teacher" label={PASSIVE.teacher} checked={form.teacher} onChange={(teacher) => change({ teacher })} />
              <Check id="passive-awake" label={PASSIVE.keepAwake} checked={form.keepAwake} onChange={(keepAwake) => change({ keepAwake })} />
              <Check id="passive-battery" label={PASSIVE.pauseOnBattery} checked={form.pauseOnBattery} onChange={(pauseOnBattery) => change({ pauseOnBattery })} />
              <Number_ id="passive-memory" label={PASSIVE.memory} value={form.memory} onChange={(memory) => change({ memory })} step="any" />
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button type="submit" disabled={noVersions || start.isPending}>
                {start.isPending ? PASSIVE.starting : PASSIVE.start}
              </Button>
              {failure && (
                <p role="alert" data-testid="passive-start-error" className="text-sm text-destructive">
                  {failure}
                </p>
              )}
            </div>
          </form>
        )}
      </CardContent>
    </Card>
  );
}

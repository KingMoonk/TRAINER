"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { problem, useCluster } from "@/lib/cluster";
import { JOIN } from "@/lib/copy/join";
import { fill } from "@/lib/members";
import { deviceText } from "../machines/logic";
import { TRAINING as T } from "@/lib/copy/training";
import { initialForm, otherMachines, startBody, teacherProblem, toggled, type Defaults, type Form, type Run, type TextSet } from "./logic";
import { allReady, readiness, waitForAll } from "./readiness";
import { Limits } from "./limits";
import { Roster } from "./roster";
import { useDefaults, useMachines, useStart, useTeachers, useTextSets, useVersions } from "./use-training";

const FIELD = "h-9 rounded-md border border-input bg-background px-2 text-sm text-foreground";

function Setting({ id, label, help, value, onChange, step, min = "0" }: { id: string; label: string; help: string; value: string; onChange: (value: string) => void; step: string; min?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      <input id={id} inputMode="decimal" type="number" min={min} step={step} value={value} onChange={(e) => onChange(e.target.value)} className={`${FIELD} w-32`} />
      <p className="max-w-prose text-xs text-muted-foreground">{help}</p>
    </div>
  );
}

/** The form fills itself from the server once the defaults and text sets have arrived, then belongs to the person. */
function Fields({ defaults, sets, onStarted }: { defaults: Defaults; sets: TextSet[]; onStarted: (run: Run) => void }) {
  const versions = useVersions().data ?? [];
  const nodes = otherMachines(useMachines().data);
  const cluster = useCluster().data;
  const [teacherOpen, setTeacherOpen] = useState(false);
  const teachers = useTeachers(teacherOpen);
  const [form, setForm] = useState<Form>(() => initialForm(defaults, sets));
  const teacherWhy = teacherProblem(form);
  const [failure, setFailure] = useState("");
  const [asking, setAsking] = useState(false);
  const start = useStart((run) => {
    setFailure("");
    onStarted(run);
  }, setFailure);
  const set = (patch: Partial<Form>) => {
    setForm((f) => ({ ...f, ...patch }));
    setFailure("");
  };
  const body = startBody(form);
  const chosen = readiness(nodes, form.machines, cluster);
  const ready = chosen.filter((item) => item.state === "ready");
  const notReady = chosen.filter((item) => item.state !== "ready");
  // Start is plain when every chosen machine is ready; otherwise a dialog asks whether to go without the others or wait for them.
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!body) return;
    if (allReady(chosen)) start.mutate(body);
    else setAsking(true);
  };
  return (
    <form onSubmit={submit} className="flex flex-col gap-5" aria-label={T.startTitle}>
      <div className="flex flex-col gap-1">
        <label htmlFor="training-design" className="text-sm font-medium">{T.design}</label>
        {versions.length === 0 ? (
          <p className="text-sm text-muted-foreground">{T.noDesign}</p>
        ) : (
          <select id="training-design" value={form.version} onChange={(e) => set({ version: e.target.value })} className={`${FIELD} w-fit max-w-full`}>
            <option value="">{T.designLatest}</option>
            {versions.map((v) => (
              <option key={v.number} value={String(v.number)}>{[T.version.replace("{number}", String(v.number)), v.name].filter(Boolean).join(" · ")}</option>
            ))}
          </select>
        )}
      </div>

      <fieldset className="flex flex-col gap-1">
        <legend className="text-sm font-medium">{T.data}</legend>
        <p className="text-xs text-muted-foreground">{T.dataLead}</p>
        {sets.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {T.noData} <Link href="/your-data" className="underline underline-offset-2">{T.yourData}</Link>
          </p>
        ) : (
          sets.map((s) => (
            <label key={s.id} className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={form.datasets.includes(s.id)} onChange={(e) => set({ datasets: toggled(form.datasets, s.id, e.target.checked) })} />
              <span>{s.name}</span>
              <span className="text-xs text-muted-foreground">{s.count.toLocaleString("en-US")}</span>
            </label>
          ))
        )}
        <label className="mt-1 flex items-center gap-2 text-sm font-medium">
          <input type="checkbox" checked={form.reviewedOnly} onChange={(e) => set({ reviewedOnly: e.target.checked })} />
          <span>{T.reviewedOnly}</span>
        </label>
        <p className="max-w-prose text-xs text-muted-foreground">{T.reviewedOnlyHelp}</p>
      </fieldset>

      <fieldset className="flex flex-col gap-1">
        <legend className="text-sm font-medium">{T.machines}</legend>
        <p className="text-xs text-muted-foreground">{T.machinesLead}</p>
        {nodes.length === 0 && <p className="text-sm text-muted-foreground">{T.noMachines}</p>}
        {nodes.map((node) => (
          <label key={node.address} className="flex flex-wrap items-center gap-2 text-sm">
            <input type="checkbox" checked={form.machines.includes(node.address)} onChange={(e) => set({ machines: toggled(form.machines, node.address, e.target.checked) })} />
            <span>{node.hostname || node.address}</span>
            {deviceText(node) && <span className="text-xs text-muted-foreground">{deviceText(node)}</span>}
            {!node.online && <span className="text-xs text-muted-foreground" title={node.error || undefined}>{T.notAnswering}</span>}
          </label>
        ))}
      </fieldset>

      {form.machines.length > 0 && (
        <div className="flex flex-col gap-1">
          <label htmlFor="training-coordinator" className="text-sm font-medium">{T.coordinator}</label>
          <select id="training-coordinator" value={form.machines.includes(form.coordinator) ? form.coordinator : ""} onChange={(e) => set({ coordinator: e.target.value })} className={`${FIELD} w-fit max-w-full`}>
            <option value="">{T.coordinatorHere}</option>
            {nodes.filter((node) => form.machines.includes(node.address)).map((node) => (
              <option key={node.address} value={node.address} disabled={!node.online}>{node.hostname || node.address}</option>
            ))}
          </select>
          <p className="max-w-prose text-xs text-muted-foreground">{T.coordinatorHelp}</p>
        </div>
      )}

      <Roster items={chosen} />
      <Limits items={chosen} cluster={cluster} />

      <fieldset className="flex flex-col gap-2">
        <legend className="text-sm font-medium">{T.shape}</legend>
        <p className="text-xs text-muted-foreground">{T.shapeLead}</p>
        <div className="flex flex-wrap gap-x-6 gap-y-3">
          <Setting id="training-min-machines" label={T.minMachines} help={T.minMachinesHelp} value={form.minMachines} onChange={(minMachines) => set({ minMachines })} step="1" min="1" />
          <Setting id="training-max-machines" label={T.maxMachines} help={T.maxMachinesHelp} value={form.maxMachines} onChange={(maxMachines) => set({ maxMachines })} step="1" min="1" />
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-2">
        <legend className="text-sm font-medium">{T.settings}</legend>
        <p className="text-xs text-muted-foreground">{T.settingsLead}</p>
        <div className="flex flex-wrap gap-x-6 gap-y-3">
          <Setting id="training-rounds" label={T.rounds} help={T.roundsHelp} value={form.rounds} onChange={(rounds) => set({ rounds })} step="1" />
          <Setting id="training-steps" label={T.innerSteps} help={T.innerStepsHelp} value={form.innerSteps} onChange={(innerSteps) => set({ innerSteps })} step="1" />
          <Setting id="training-lr" label={T.outerLr} help={T.outerLrHelp} value={form.outerLr} onChange={(outerLr) => set({ outerLr })} step="0.1" />
          <Setting id="training-momentum" label={T.outerMomentum} help={T.outerMomentumHelp} value={form.outerMomentum} onChange={(outerMomentum) => set({ outerMomentum })} step="0.1" />
        </div>
        <label className="flex items-center gap-2 text-sm font-medium">
          <input type="checkbox" checked={form.overlap} onChange={(e) => set({ overlap: e.target.checked })} />
          <span>{T.overlap}</span>
        </label>
        <p className="max-w-prose text-xs text-muted-foreground">{T.overlapHelp}</p>
      </fieldset>

      <details className="flex flex-col gap-2" onToggle={(e) => e.currentTarget.open && setTeacherOpen(true)}>
        <summary className="cursor-pointer text-sm font-medium">{T.teacherTitle}</summary>
        <div className="mt-2 flex flex-col gap-3">
          <p className="max-w-prose text-xs text-muted-foreground">{T.teacherLead}</p>
          <div className="flex flex-col gap-1">
            <label htmlFor="training-teacher" className="text-sm font-medium">{T.teacher}</label>
            <select id="training-teacher" value={form.teacher} onChange={(e) => set({ teacher: e.target.value })} className={`${FIELD} w-fit max-w-full`}>
              <option value="">{T.teacherNone}</option>
              {(teachers.data?.models ?? []).map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
            {(teachers.isError || teachers.data?.error) && (
              <p className="max-w-prose text-xs text-muted-foreground">{T.teacherUnread.replace("{why}", teachers.isError ? problem(teachers.error) : teachers.data?.error ?? "")}</p>
            )}
          </div>
          <div className="flex flex-wrap gap-x-6 gap-y-3">
            <Setting id="training-polish-every" label={T.polishEvery} help={T.polishEveryHelp} value={form.polishEvery} onChange={(polishEvery) => set({ polishEvery })} step="1" min="1" />
            <Setting id="training-win-rate" label={T.targetWinRate} help={T.targetWinRateHelp} value={form.targetWinRate} onChange={(targetWinRate) => set({ targetWinRate })} step="1" min="1" />
          </div>
        </div>
      </details>

      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" disabled={!body || start.isPending}>{start.isPending ? T.starting : T.start}</Button>
        {teacherWhy && <p role="status" className="text-sm text-destructive">{teacherWhy}</p>}
        {failure && <p role="alert" className="text-sm text-destructive">{failure}</p>}
      </div>

      <AlertDialog open={asking} onOpenChange={setAsking}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{JOIN.startAnyway}</AlertDialogTitle>
            <AlertDialogDescription>
              {fill(JOIN.startAnywayLead, { ready: ready.map((item) => item.name).join(", ") || "none", notReady: notReady.map((item) => item.name).join(", ") })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{JOIN.goBack}</AlertDialogCancel>
            {notReady.some((item) => item.state === "busy") && (
              <AlertDialogAction className="border border-border bg-background text-foreground hover:bg-muted" title={JOIN.waitAllHelp} onClick={() => body && start.mutate(waitForAll(body, chosen))}>
                {JOIN.waitAll}
              </AlertDialogAction>
            )}
            <AlertDialogAction onClick={() => body && start.mutate(body)}>{JOIN.startReady}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </form>
  );
}

export function StartForm({ onStarted }: { onStarted: (run: Run) => void }) {
  const defaults = useDefaults();
  const sets = useTextSets();
  return (
    <Card>
      <CardHeader>
        <CardTitle>{T.startTitle}</CardTitle>
        <CardDescription>{T.startLead}</CardDescription>
      </CardHeader>
      <CardContent>
        {defaults.data && sets.data ? (
          <Fields defaults={defaults.data} sets={sets.data} onStarted={onStarted} />
        ) : (
          <p className="text-sm text-muted-foreground" role={defaults.isError || sets.isError ? "alert" : undefined}>
            {defaults.isError || sets.isError ? T.couldNotLoad.replace("{why}", problem(defaults.error ?? sets.error)) : T.loading}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

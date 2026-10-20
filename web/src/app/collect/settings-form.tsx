"use client";

import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { COLLECT as T } from "./copy";
import { LIMIT_FIELDS, type Draft, type Options, type Site } from "./logic";

export function Section({ kicker, desc, children }: { kicker: string; desc?: string; children: ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium tracking-wider text-muted-foreground">{kicker}</CardTitle>
        {desc && <p className="max-w-prose text-sm text-muted-foreground">{desc}</p>}
      </CardHeader>
      <CardContent className="flex flex-col gap-3">{children}</CardContent>
    </Card>
  );
}

function Check({ label, checked, onChange, hint, disabled }: { label: string; checked: boolean; onChange: (on: boolean) => void; hint?: string; disabled?: boolean }) {
  return (
    <label className="flex flex-col gap-0.5 text-sm">
      <span className="flex items-center gap-2">
        <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} className="size-4 accent-primary" />
        {label}
      </span>
      {hint && <span className="ml-6 max-w-prose text-xs text-muted-foreground">{hint}</span>}
    </label>
  );
}

function Field({ label, value, onChange, wide }: { label: string; value: string; onChange: (v: string) => void; wide?: boolean }) {
  return (
    <label className={`flex flex-col gap-1 text-sm ${wide ? "w-full" : "w-full sm:w-56"}`}>
      {label}
      <Input value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

/** Topics, sources, licences, limits and when to run: one draft, one Save. */
export function SettingsForm({
  draft,
  options,
  dirty,
  saving,
  onChange,
  onSave,
}: {
  draft: Draft;
  options: Options;
  dirty: boolean;
  saving: boolean;
  onChange: (next: Draft) => void;
  onSave: () => void;
}) {
  const set = (patch: Partial<Draft>) => onChange({ ...draft, ...patch });
  const toggleTopic = (id: string, on: boolean) => set({ topics: on ? [...draft.topics, id] : draft.topics.filter((t) => t !== id) });
  const setSite = (i: number, patch: Partial<Site>) => set({ web: draft.web.map((s, j) => (j === i ? { ...s, ...patch } : s)) });

  return (
    <>
      <Section kicker={T.topicsKicker} desc={T.topicsDesc}>
        <fieldset className="grid gap-2 sm:grid-cols-2">
          <legend className="sr-only">{T.topicsKicker}</legend>
          {options.topics.map((t) => (
            <Check key={t.id} label={t.label} checked={draft.topics.includes(t.id)} onChange={(on) => toggleTopic(t.id, on)} />
          ))}
        </fieldset>
        <Field wide label={T.customTopics} value={draft.customTopics} onChange={(v) => set({ customTopics: v })} />
      </Section>

      <Section kicker={T.sourcesKicker} desc={T.sourcesDesc}>
        {options.sources.map((s) => {
          const id = s.id as keyof Draft["sources"];
          return (
            <div key={s.id} className="flex flex-col gap-2 rounded-md border p-3">
              <Check label={s.label} checked={draft.sources[id]} onChange={(on) => set({ sources: { ...draft.sources, [id]: on } })} hint={s.what} />
              <p className="ml-6 max-w-prose text-xs text-muted-foreground">{s.terms}</p>
              {s.note && <p className="ml-6 max-w-prose text-xs text-muted-foreground">{s.note}</p>}
              {s.id === "stackexchange" && draft.sources.stackexchange && (
                <div className="ml-6 flex flex-wrap gap-3">
                  <Field label={T.stackSites} value={draft.stackSites} onChange={(v) => set({ stackSites: v })} wide />
                  <Field label={T.stackScore} value={draft.stackScore} onChange={(v) => set({ stackScore: v })} />
                </div>
              )}
              {s.id === "github" && draft.sources.github && (
                <div className="ml-6 flex flex-wrap gap-3">
                  <Field label={T.githubQueries} value={draft.githubQueries} onChange={(v) => set({ githubQueries: v })} wide />
                  <Field label={T.githubMax} value={draft.githubMax} onChange={(v) => set({ githubMax: v })} />
                </div>
              )}
              {s.id === "web" && draft.sources.web && (
                <div className="ml-6 flex flex-col gap-2">
                  <p className="max-w-prose text-xs text-muted-foreground">{T.webTerms}</p>
                  {draft.web.map((site, i) => (
                    <div key={i} className="flex flex-wrap items-end gap-2">
                      <Field label={T.webHost} value={site.host} onChange={(v) => setSite(i, { host: v, seeds: [] })} />
                      <Field label={T.webLicence} value={site.licence} onChange={(v) => setSite(i, { licence: v })} />
                      <Field label={T.webPages} value={String(site.pages)} onChange={(v) => setSite(i, { pages: Number(v) || 0 })} />
                      <Button type="button" variant="outline" aria-label={T.webRemove.replace("{host}", site.host || "site")} onClick={() => set({ web: draft.web.filter((_, j) => j !== i) })}>
                        Remove
                      </Button>
                    </div>
                  ))}
                  <div>
                    <Button type="button" variant="outline" onClick={() => set({ web: [...draft.web, { host: "", seeds: [], licence: "", pages: 200 }] })}>
                      {T.webAdd}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </Section>

      <Section kicker={T.licencesKicker} desc={T.licencesDesc}>
        {options.licences.map((l) => {
          const key = l.opt_in as keyof Draft["licences"] | "";
          return key ? (
            <Check key={l.id} label={`Also keep: ${l.label}`} checked={draft.licences[key]} onChange={(on) => set({ licences: { ...draft.licences, [key]: on } })} />
          ) : (
            <Check key={l.id} label={l.label} checked disabled onChange={() => undefined} />
          );
        })}
      </Section>

      <Section kicker={T.limitsKicker} desc={T.limitsDesc}>
        <div className="flex flex-wrap gap-3">
          {LIMIT_FIELDS.map(([key, label]) => (
            <Field key={key} label={label} value={draft.limits[key]} onChange={(v) => set({ limits: { ...draft.limits, [key]: v } })} />
          ))}
        </div>
        <Check label={T.onlyIdle} checked={draft.idle.only_when_idle} onChange={(on) => set({ idle: { ...draft.idle, only_when_idle: on } })} />
        {draft.idle.only_when_idle && <Field label={T.cpuPercent} value={draft.idle.cpu_percent} onChange={(v) => set({ idle: { ...draft.idle, cpu_percent: v } })} />}
        <Check label={T.onlyMains} checked={draft.idle.only_on_mains} onChange={(on) => set({ idle: { ...draft.idle, only_on_mains: on } })} />
        <Check label={T.unmetered} hint={T.unmeteredHint} checked={draft.idle.unmetered_confirmed} onChange={(on) => set({ idle: { ...draft.idle, unmetered_confirmed: on } })} />
        <p className="text-xs text-muted-foreground">{T.agent.replace("{agent}", options.user_agent)}</p>
      </Section>

      <div className="flex items-center gap-3">
        <Button onClick={onSave} disabled={!dirty || saving}>
          {saving ? T.saving : T.save}
        </Button>
      </div>
    </>
  );
}

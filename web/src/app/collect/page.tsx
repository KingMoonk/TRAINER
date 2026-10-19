"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { api, subscribe, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { COLLECT as T } from "./copy";
import {
  approving,
  blocking,
  bytesText,
  dropRows,
  errorText,
  fill,
  isDirty,
  percent,
  statusLine,
  toDraft,
  toUpdate,
  type Config,
  type Draft,
  type Options,
  type State,
  type Update,
} from "./logic";
import { Section, SettingsForm } from "./settings-form";

export default function Page() {
  const route = routeFor("collect");
  const queryClient = useQueryClient();

  // The collector runs in another process: the page asks again every few seconds, and at once when the server says something changed.
  useEffect(
    () =>
      subscribe((event) => {
        if (event.kind === "collect") queryClient.invalidateQueries({ queryKey: ["collect"] });
      }),
    [queryClient],
  );
  const state = useQuery({ queryKey: ["collect", "state"], queryFn: () => unwrap(api.GET("/collect")), refetchInterval: 5000 });
  const config = useQuery({ queryKey: ["collect", "config"], queryFn: () => unwrap(api.GET("/collect/config")) });
  const options = useQuery({ queryKey: ["collect", "options"], queryFn: () => unwrap(api.GET("/collect/options")), staleTime: Infinity });

  const save = useMutation({
    mutationFn: (update: Update) => unwrap(api.PUT("/collect/config", { body: update })),
    onSuccess: (saved) => {
      queryClient.setQueryData(["collect", "config"], saved);
      queryClient.invalidateQueries({ queryKey: ["collect", "state"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  const control = useMutation({
    mutationFn: (action: "pause" | "resume") => unwrap(api.POST(action === "pause" ? "/collect/pause" : "/collect/resume")),
    onSuccess: (saved) => {
      queryClient.setQueryData(["collect", "config"], saved);
      queryClient.invalidateQueries({ queryKey: ["collect", "state"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={route.title} lead={route.lead} />
      {state.isError || config.isError || options.isError ? (
        <p className="text-sm text-destructive">{T.loadFailed}</p>
      ) : !state.data || !config.data || !options.data ? (
        <Skeleton className="h-40 w-full" aria-label="Loading the collector" />
      ) : (
        <Loaded state={state.data} config={config.data} options={options.data} save={save} control={control} />
      )}
    </main>
  );
}

type Mutation<V> = { mutate: (v: V, options?: { onSuccess?: () => void }) => void; isPending: boolean };

function Loaded({ state, config, options, save, control }: { state: State; config: Config; options: Options; save: Mutation<Update>; control: Mutation<"pause" | "resume"> }) {
  const [draft, setDraft] = useState<Draft>(() => toDraft(config));
  const [problem, setProblem] = useState("");
  const dirty = isDirty(draft, config);

  function submit() {
    const built = toUpdate(draft);
    if ("error" in built) {
      setProblem(built.error);
      return;
    }
    setProblem("");
    save.mutate(built.update, {
      onSuccess: () => toast.success(T.saved),
    });
  }

  const running = state.enabled && !state.paused;
  return (
    <>
      <StatusCard state={state} running={running} busy={control.isPending} onToggle={() => control.mutate(running ? "pause" : "resume")} />
      <Drops state={state} />
      <Plan state={state} config={config} onChange={(update) => save.mutate(update)} />
      <Progress state={state} />
      <SettingsForm draft={draft} options={options} dirty={dirty} saving={save.isPending} onChange={setDraft} onSave={submit} />
      {problem && <p role="alert" className="text-sm text-destructive">{problem}</p>}
      <Section kicker={T.excludedKicker} desc={T.excludedDesc}>
        <ul className="flex flex-col gap-1 text-sm">
          {options.excluded.map((e) => (
            <li key={e.host}>
              <span className="font-medium">{e.host}</span> <span className="text-muted-foreground">{e.reason}</span>
            </li>
          ))}
        </ul>
      </Section>
      <Section kicker={T.journalKicker}>
        {state.journal.length === 0 ? (
          <p className="text-sm text-muted-foreground">{T.journalNone}</p>
        ) : (
          <ol className="flex flex-col gap-1 text-xs text-muted-foreground">
            {[...state.journal].reverse().map((row, i) => (
              <li key={`${row.at}-${i}`}>
                <time dateTime={row.at}>{row.at.replace("T", " ").replace(/\+.*$/, "")}</time> {row.text}
              </li>
            ))}
          </ol>
        )}
      </Section>
    </>
  );
}

function Meter({ label, used, cap }: { label: string; used: number; cap: number }) {
  return (
    <div className="flex min-w-48 flex-col gap-1 text-sm">
      <span>
        {label}: {bytesText(used)} of {bytesText(cap)}
      </span>
      <div className="h-1.5 w-full rounded-full bg-muted" role="presentation">
        <div className="h-full rounded-full bg-primary" style={{ width: `${percent(used, cap)}%` }} />
      </div>
    </div>
  );
}

function StatusCard({ state, running, busy, onToggle }: { state: State; running: boolean; busy: boolean; onToggle: () => void }) {
  const label = busy ? T.working : !state.enabled ? T.start : running ? T.pause : T.resume;
  return (
    <Section kicker={T.statusKicker} desc={T.statusDesc}>
      <p aria-live="polite" className="text-base">
        {statusLine(state)}
      </p>
      <div>
        <Button onClick={onToggle} disabled={busy} variant={running ? "outline" : "default"}>
          {label}
        </Button>
      </div>
      <div className="flex flex-wrap gap-x-8 gap-y-3">
        <Meter label={T.disk} used={state.disk_used} cap={state.disk_cap} />
        <Meter label={T.bandwidth} used={state.bandwidth_today} cap={state.bandwidth_cap} />
      </div>
      <p className="text-sm">
        {T.kept}: <span className="font-medium">{state.stats.kept.toLocaleString()}</span> <span className="text-muted-foreground">{T.keptHint}</span>
      </p>
      {state.stats.redactions > 0 && <p className="text-xs text-muted-foreground">{fill(T.redactions, { n: state.stats.redactions.toLocaleString() })}</p>}
    </Section>
  );
}

function Drops({ state }: { state: State }) {
  const rows = dropRows(state.stats.dropped);
  const total = rows.reduce((n, r) => n + r.n, 0);
  return (
    <Section kicker={T.dropsKicker} desc={T.dropsDesc}>
      {state.stats.seen === 0 ? (
        <p className="text-sm text-muted-foreground">{T.nothingYet}</p>
      ) : (
        <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2" aria-label={`${total} records left out`}>
          {rows.map((r) => (
            <div key={r.key} className="flex justify-between gap-3">
              <dt>{r.label}</dt>
              <dd className="tabular-nums">{r.n.toLocaleString()}</dd>
            </div>
          ))}
        </dl>
      )}
    </Section>
  );
}

function Plan({ state, config, onChange }: { state: State; config: Config; onChange: (update: Update) => void }) {
  return (
    <Section kicker={T.planKicker} desc={T.planDesc}>
      {state.plan.length === 0 ? (
        <p className="text-sm text-muted-foreground">{T.planNone}</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {state.plan.map((item) => (
            <li key={`${item.source}:${item.id}`} className="flex flex-col gap-1 rounded-md border p-2 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="min-w-0 flex-1 truncate font-medium">{item.id}</span>
                <Badge variant="secondary">{item.source}</Badge>
                {item.licence && <Badge variant="outline">{item.licence}</Badge>}
                {item.bytes > 0 && <span className="text-xs text-muted-foreground">{bytesText(item.bytes)}</span>}
                {item.status === "needs_approval" && !item.approved && (
                  <Button size="sm" onClick={() => onChange({ approved: approving(config, item.id) })} aria-label={`${T.approve} ${item.id}`}>
                    {T.approve}
                  </Button>
                )}
                {item.status !== "skipped" && (
                  <Button size="sm" variant="outline" onClick={() => onChange({ blocked: blocking(config, item.id, !item.blocked) })} aria-label={`${item.blocked ? T.unskip : T.skip} ${item.id}`}>
                    {item.blocked ? T.unskip : T.skip}
                  </Button>
                )}
              </div>
              {item.status === "skipped" && <p className="text-xs text-muted-foreground">{fill(T.skipped, { reason: item.reason })}</p>}
              {item.status === "needs_approval" && !item.approved && <p className="text-xs text-muted-foreground">{fill(T.needsApproval, { reason: item.reason })}</p>}
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}

function Progress({ state }: { state: State }) {
  const active = state.sources.filter((s) => !s.done);
  return (
    <>
      <Section kicker={T.progressKicker}>
        {active.length === 0 ? (
          <p className="text-sm text-muted-foreground">{T.progressNone}</p>
        ) : (
          <ul className="flex flex-col gap-1 text-sm">
            {active.map((s) => (
              <li key={s.key}>
                <span className="font-medium">{s.name}</span> <span className="text-muted-foreground">{[s.status, s.detail, s.downloaded ? bytesText(s.downloaded) : ""].filter(Boolean).join(" · ")}</span>
                {s.error && <span className="text-destructive"> {s.error}</span>}
              </li>
            ))}
          </ul>
        )}
      </Section>
      <Section kicker={T.landedKicker} desc={T.landedDesc}>
        {state.waiting_shards > 0 && <p className="text-xs text-muted-foreground">{fill(T.waitingShards, { n: state.waiting_shards })}</p>}
        {state.sets.length === 0 ? (
          <p className="text-sm text-muted-foreground">{T.landedNone}</p>
        ) : (
          <ul className="flex flex-col gap-1 text-sm">
            {state.sets.map((s) => (
              <li key={s.id}>
                <span className="font-medium">{s.name}</span> <span className="text-muted-foreground">{fill(T.landedRow, { records: s.records.toLocaleString(), kept: s.kept.toLocaleString() })}</span>
              </li>
            ))}
          </ul>
        )}
      </Section>
    </>
  );
}

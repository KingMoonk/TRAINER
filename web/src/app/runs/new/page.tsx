"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { toast } from "sonner";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { buttonVariants, Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { TooltipProvider } from "@/components/ui/tooltip";
import { api, unwrap } from "@/lib/api";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { problem, useCluster } from "@/lib/cluster";
import { nodeName } from "@/lib/node-name";
import { jobRows } from "../../cluster/machines/logic";
import { useEvent } from "@/lib/events";
import { cn } from "@/lib/utils";
import { initialForm, startBody, toggled, type Defaults, type Form, type TextSet } from "../../cluster/training/logic";
import { useDefaults, useMachines, useTextSets, useVersions } from "../../cluster/training/use-training";
import { AllowedHours } from "../allowed-hours";
import { T as PRETRAINING } from "../../pretraining/copy";
import { planBody } from "../../pretraining/logic";
import { QUICKTEST } from "../../quick-test/copy";
import { Coordinator, FIELD, Hint, OuterLoop, Workers, selectedText } from "./fields";
import { fill, NEW_RUN as T } from "./copy";
import { busyPeers, ctaLabel, defaultSftVersion, localForced, nameList, peersOf, runHref, shapeOf, sourceFacts, workersOf, type Node, type Source, type Stage } from "./logic";


const SLOW_MS = 6_000;
const NO_NODES: Node[] = [];
const NO_SETS: TextSet[] = [];

/** Whether a read has been going on for a while (from mount), so its section can say it is still waiting. */
function useSlow(): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setSlow(true), SLOW_MS);
    return () => clearTimeout(timer);
  }, []);
  return slow;
}

function Waiting({ text }: { text: string }) {
  const slow = useSlow();
  return (
    <div className="flex flex-col gap-2" aria-busy="true" data-testid="section-loading">
      <p className="text-sm text-muted-foreground">{slow ? `${text} ${T.slow}` : text}</p>
      <Skeleton className="h-16 w-full" />
    </div>
  );
}

/** One section's own read: its content once it is in, else a loading line or the reason it failed with a way to try again. */
function Slot<V>({ query, title, loading, failed, children }: { query: UseQueryResult<V>; title: ReactNode; loading: string; failed: string; children: (data: V) => ReactNode }) {
  if (query.data !== undefined) return children(query.data);
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-xl font-semibold">{title}</h2>
      {query.isError && !query.isFetching ? (
        <div role="alert" className="flex flex-wrap items-center gap-3 text-sm text-destructive">
          <p>{fill(failed, { why: problem(query.error) })}</p>
          <Button type="button" variant="outline" size="sm" onClick={() => void query.refetch()}>{T.retry}</Button>
        </div>
      ) : (
        <Waiting text={loading} />
      )}
    </section>
  );
}

function Fields({ defaults, sets, machines }: { defaults: UseQueryResult<Defaults>; sets: UseQueryResult<TextSet[]>; machines: UseQueryResult<Node[]> }) {
  const router = useRouter();
  const client = useQueryClient();
  const versions = useVersions().data ?? [];
  const nodes = machines.data ?? NO_NODES;
  const textSets = sets.data ?? NO_SETS;
  const [stage, setStage] = useState<Stage>("pretrain");
  const [form, setForm] = useState<Form>(() => initialForm(undefined, undefined));
  const [picked, setPicked] = useState<string[]>([]);
  // Each read fills in its own part of the form when it arrives; the rest of the form is usable before then.
  const [seeded, setSeeded] = useState({ defaults: false, sets: false, nodes: false });
  if (defaults.data && !seeded.defaults) {
    const { rounds, innerSteps, outerLr, outerMomentum } = initialForm(defaults.data, undefined);
    setSeeded((now) => ({ ...now, defaults: true }));
    setForm((f) => ({ ...f, rounds, innerSteps, outerLr, outerMomentum }));
  }
  if (sets.data && !seeded.sets) {
    setSeeded((now) => ({ ...now, sets: true }));
    setForm((f) => ({ ...f, datasets: initialForm(undefined, sets.data).datasets }));
  }
  if (machines.data && !seeded.nodes) {
    setSeeded((now) => ({ ...now, nodes: true }));
    setPicked(machines.data.filter((n) => n.local).map((n) => n.address));
  }
  const [mode, setMode] = useState<"catch_up" | "from_scratch">("from_scratch");
  const [fromVersion, setFromVersion] = useState<number | null>(null);
  const [dataset, setDataset] = useState<"stories" | "slice" | "problems">("stories");
  const [minutes, setMinutes] = useState<"1" | "3" | "10">("3");
  const [failure, setFailure] = useState("");
  const [confirming, setConfirming] = useState(false);
  const rows = jobRows(useCluster().data, undefined);
  const change = (patch: Partial<Form>) => {
    setForm((f) => ({ ...f, ...patch }));
    if (patch.version !== undefined) setFromVersion(null); // another version has its own list of earlier runs
    setFailure("");
  };

  const workers = workersOf(stage, picked, nodes);
  const peers = peersOf(workers, nodes);
  const several = stage === "pretrain" && workers.length > 1;
  const single = stage === "pretrain" && workers.length === 1;
  const shape = shapeOf(workers, nodes);
  const ids = form.datasets;
  const version = form.version === "" ? null : Number(form.version);
  const sftVersion = version ?? defaultSftVersion(versions)?.number ?? null;
  const pin = workers.length === 1 && peers.length === 1 ? peers[0] : null;

  // The plan is the Pretraining page's: the server turns the choices into the job to queue.
  const body = planBody({ version, datasetIds: ids, mode, rows: "16", hold: "5%", length: "Medium", fromVersion });
  const plan = useQuery({
    queryKey: ["pretraining", "plan", "new-run", version, ids, mode, fromVersion],
    queryFn: ({ signal }) => unwrap(api.POST("/pretraining/plan", { body, signal })),
    enabled: single && ids.length > 0,
    placeholderData: keepPreviousData,
  });
  // Start never waits for the size of the text: POST /pretraining/start queues the run now, and the queue works the size out first.
  const sizeFirst = single && !!plan.data?.estimating;
  const outlook = useQuery({ queryKey: ["queue", "outlook"], queryFn: ({ signal }) => unwrap(api.GET("/queue/outlook", { signal })), retry: false });
  useEvent("queue", () => void client.invalidateQueries({ queryKey: ["queue", "outlook"] }));

  const submit = useMutation({
    mutationFn: async (): Promise<string> => {
      if (several) {
        const body = startBody({ ...form, machines: peers });
        if (!body) throw new Error(T.needSettings);
        return (await unwrap(api.POST("/distributed/runs", { body }))).id;
      }
      const id =
        stage === "pretrain"
          ? (await unwrap(api.POST("/pretraining/start", { body: { ...body, version: plan.data?.version ?? version } }))).id
          : stage === "sft"
          ? (
              await unwrap(
                api.POST("/post-training/rounds", { body: { round: 1, version: sftVersion, lora: null, lora_rank: 8, teacher: "version", teacher_version: null, leash: "Medium" } }),
              )
            ).id
          : (await unwrap(api.POST("/quicktest/run", { body: { dataset, minutes: Number(minutes) as 1 | 3 | 10, shrink: false } }))).job_id;
      if (pin) {
        try {
          await unwrap(api.POST("/queue/jobs/{job_id}/pin", { params: { path: { job_id: id } }, body: { pinned: pin } }));
        } catch (error) {
          toast.error(fill(T.pinFailed, { node: nodes.find((n) => n.address === pin)?.hostname || pin, why: problem(error) }));
        }
      }
      return id;
    },
    onSuccess: (id) => {
      void client.invalidateQueries({ queryKey: ["queue"] });
      void client.invalidateQueries({ queryKey: ["distributed", "runs"] });
      void client.invalidateQueries({ queryKey: ["cluster"] });
      router.push(runHref(id));
    },
    onError: (error) => setFailure(fill(T.couldNotStart, { why: problem(error) })),
  });

  const toggle = (address: string) => {
    setFailure("");
    setPicked((now) => (stage === "pretrain" ? toggled(now, address, !now.includes(address)) : [address]));
    if (stage === "pretrain" && mode === "catch_up") setMode("from_scratch");
  };
  const pickStage = (next: Stage) => {
    setStage(next);
    setFailure("");
    if (next !== "pretrain") setPicked((now) => workersOf(next, now, nodes));
  };

  // Why the button is off, in a sentence under it; "" when it is on.
  const blocker = (() => {
    if (!machines.data) return machines.isError ? T.machinesDown : T.loadingMachines;
    if (workers.length === 0) return T.needWorker;
    if (stage === "pretrain" && !sets.data) return sets.isError ? T.textDown : T.loadingText;
    if (stage === "pretrain" && ids.length === 0) return T.needText;
    if (several && !defaults.data) return defaults.isError ? T.settingsDown : T.loadingSettings;
    if (several) return startBody({ ...form, machines: peers }) ? "" : T.needSettings;
    if (stage === "pretrain") return plan.data && !plan.data.can_add && !plan.data.estimating ? plan.data.reason : ""; // the server checks the rest on Start
    return sftVersion === null ? T.noVersions : "";
  })();

  // Chosen machines that are busy get no slot: say so before the run starts without them.
  const left = several ? busyPeers(workers, nodes) : [];
  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (blocker) return;
    if (left.length > 0) setConfirming(true);
    else submit.mutate();
  };
  const latest = stage === "pretrain" ? T.versionNewest : T.versionBest;

  return (
    <form onSubmit={onSubmit} aria-label={T.title} className="flex flex-col gap-8">
      <section className="flex flex-wrap gap-4">
        <label className="flex min-w-60 flex-1 flex-col gap-2 text-sm text-muted-foreground">
          <Hint tip={T.stageTip} className="self-start">{T.stage}</Hint>
          <select value={stage} onChange={(e) => pickStage(e.target.value as Stage)} className={FIELD}>
            {T.stages.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
        </label>
        {stage !== "ablation" && (
          <>
            <label className="flex min-w-60 flex-1 flex-col gap-2 text-sm text-muted-foreground">
              <Hint tip={T.versionTip} className="self-start">{T.version}</Hint>
              <select value={form.version} onChange={(e) => change({ version: e.target.value })} className={FIELD}>
                <option value="">{latest}</option>
                {versions.map((v) => (
                  <option key={v.number} value={String(v.number)}>{[fill(T.versionName, { number: v.number }), v.name].filter(Boolean).join(" · ")}</option>
                ))}
              </select>
            </label>
            {stage === "pretrain" && (
              <label className="flex min-w-60 flex-1 flex-col gap-2 text-sm text-muted-foreground">
                <Hint tip={T.startTip} className="self-start">{T.start}</Hint>
                <select value={several ? "from_scratch" : mode} disabled={several || workers.length === 0} onChange={(e) => setMode(e.target.value as typeof mode)} className={FIELD}>
                  <option value="from_scratch">{T.startScratch}</option>
                  <option value="catch_up">{T.startContinue}</option>
                </select>
              </label>
            )}
          </>
        )}
      </section>

      {single && mode === "catch_up" && (
        <Sources plan={plan.data} onPick={(n) => { setFromVersion(n); setFailure(""); }} />
      )}

      {single && stage === "pretrain" && plan.data?.scaling && plan.data.scaling.notes.length > 0 && (
        <section className="flex flex-col gap-1" data-testid="scaling-notes">
          <p className="text-sm font-medium">{plan.data.scaling.scaled ? PRETRAINING.scalingScaled : PRETRAINING.scalingHeadsUp}</p>
          <ul className="list-disc pl-5 text-sm text-muted-foreground">
            {plan.data.scaling.notes.map((note, i) => (
              <li key={i}>{note.text}</li>
            ))}
          </ul>
          <Link href="/pretraining" className="w-fit text-sm text-highlight hover:underline">
            {PRETRAINING.scalingKeep} ›
          </Link>
        </section>
      )}

      {stage === "pretrain" && (
        <Slot query={sets} title={T.text} loading={T.loadingText} failed={T.textFailed}>
          {() => (
            <fieldset className="flex flex-col gap-2">
              <legend className="mb-2 text-xl font-semibold">
                <Hint tip={T.textTip}>{T.text}</Hint> <span className="text-sm font-normal text-muted-foreground">{selectedText(ids.length)}</span>
              </legend>
              {textSets.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  {T.noText} <Link href="/your-data" className="underline underline-offset-2">{T.yourData}</Link>
                </p>
              ) : (
                textSets.map((s) => (
                  <label key={s.id} className="flex min-h-9 items-center gap-2 text-base">
                    <input type="checkbox" checked={form.datasets.includes(s.id)} onChange={(e) => change({ datasets: toggled(form.datasets, s.id, e.target.checked) })} />
                    <span>{s.name}</span>
                    <span className="text-sm text-muted-foreground">{s.count.toLocaleString("en-US")}</span>
                  </label>
                ))
              )}
            </fieldset>
          )}
        </Slot>
      )}

      {stage === "ablation" && (
        <fieldset className="flex flex-col gap-4">
          <Pick label={QUICKTEST.tinyDataset} value={dataset} onChange={(v) => setDataset(v as typeof dataset)} options={QUICKTEST.datasets} />
          <Pick label={QUICKTEST.timeEach} value={minutes} onChange={(v) => setMinutes(v as typeof minutes)} options={QUICKTEST.minutes} />
          <p className="text-sm text-muted-foreground">
            {T.quickNeedsDraft} <Link href="/builder" className="text-highlight hover:underline">{T.quickDesigner}</Link>
          </p>
        </fieldset>
      )}

      <Slot query={machines} title={T.workers} loading={T.loadingMachines} failed={T.machinesFailed}>
        {() => (
          <>
            <Workers nodes={nodes} rows={rows} workers={workers} forced={localForced(stage, picked, nodes)} multiple={stage === "pretrain"} onToggle={toggle} />
            <div role="status" className="flex items-center">
              <Hint tip={shape.tip} className="rounded-full border border-border px-3.5 py-2 font-mono text-sm text-highlight">{shape.label}</Hint>
            </div>
          </>
        )}
      </Slot>
      <Coordinator />

      {several && (
        <Slot query={defaults} title={T.outerLoop} loading={T.loadingSettings} failed={T.settingsFailed}>
          {(d) => <OuterLoop values={form} deltaDtype={d.delta_dtype} onChange={change} />}
        </Slot>
      )}

      <section className="flex flex-col gap-2" aria-label={T.schedule}>
        <h2 className="text-xl font-semibold"><Hint tip={T.scheduleTip}>{T.schedule}</Hint></h2>
        <p className="text-base" data-testid="outlook">
          {outlook.data?.outlook ? `${outlook.data.outlook}. ` : outlook.isPending ? `${T.outlookLoading} ` : ""}
          <AllowedHours>
            <button type="button" className="text-highlight hover:underline">{T.allowedHours}</button>
          </AllowedHours>
        </p>
      </section>

      <div className="flex flex-wrap items-center justify-end gap-3 border-t pt-6">
        {failure && <p role="alert" className="mr-auto text-sm text-destructive">{failure}</p>}
        {!failure && blocker && <p className="mr-auto text-sm text-muted-foreground" data-testid="blocker">{blocker}</p>}
        {!failure && !blocker && sizeFirst && <p className="mr-auto text-sm text-muted-foreground" data-testid="size-first">{PRETRAINING.sizeFirst}</p>}
        <Link href="/runs" className={cn(buttonVariants({ variant: "outline", size: "lg" }))}>{T.cancel}</Link>
        <Button type="submit" size="lg" disabled={!!blocker || submit.isPending}>
          {submit.isPending ? T.working : ctaLabel(outlook.data?.outlook)}
        </Button>
      </div>

      <AlertDialog open={confirming && left.length > 0} onOpenChange={setConfirming}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{fill(T.busyTitle, { names: nameList(left.map((n) => nodeName(n, nodes))) })}</AlertDialogTitle>
            <AlertDialogDescription>
              {fill(left.length === 1 ? T.busyBodyOne : T.busyBodyMany, { names: nameList(left.map((n) => nodeName(n, nodes))) })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{T.cancel}</AlertDialogCancel>
            <AlertDialogAction onClick={() => submit.mutate()}>{T.launchAnyway}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </form>
  );
}

function Pick<V extends string>({ label, value, onChange, options }: { label: string; value: V; onChange: (value: V) => void; options: readonly { id: V; label: string }[] }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-sm font-medium">{label}</span>
      <ToggleGroup type="single" variant="outline" value={value} onValueChange={(v) => v && onChange(v as V)} aria-label={label} className="flex-wrap justify-start">
        {options.map((o) => (
          <ToggleGroupItem key={o.id} value={o.id}>{o.label}</ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

export default function NewRunPage() {
  const defaults = useDefaults();
  const sets = useTextSets();
  const machines = useMachines();
  return (
    <TooltipProvider>
      <main className={PAGE_MAIN}>
        <PageHeader
          title={T.title}
          breadcrumb={
            <Link href="/runs" className="w-fit text-sm text-highlight hover:underline">
              ← {T.back}
            </Link>
          }
        />
        <Fields defaults={defaults} sets={sets} machines={machines} />
      </main>
    </TooltipProvider>
  );
}

/** The earlier runs a catch-up may start from, each with what it did, as radio rows; the plan's choice is ticked. */
function Sources({ plan, onPick }: { plan: { sources?: Source[]; continue_from?: number | null; continue_blocked?: string | null } | undefined; onPick: (version: number) => void }) {
  if (!plan) return <p className="text-sm text-muted-foreground">{T.sourcesLoading}</p>;
  const sources = plan.sources ?? [];
  if (sources.length === 0) return <p className="text-sm text-muted-foreground">{plan.continue_blocked === "word_list_changed" ? T.noSourceWordList : T.noSource}</p>;
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="mb-2 text-xl font-semibold">
        <Hint tip={T.sourcesTip}>{T.sources}</Hint>
      </legend>
      {sources.map((s) => {
        const { title, when, facts, design } = sourceFacts(s);
        const on = plan.continue_from === s.version;
        return (
          <label key={s.version} data-testid={`source-${s.version}`} className={cn("flex cursor-pointer gap-3 rounded-lg border p-3", on ? "border-primary bg-accent" : "border-border")}>
            <input type="radio" name="continue-from" className="mt-1" checked={on} onChange={() => onPick(s.version)} />
            <span className="flex min-w-0 flex-col gap-1">
              <span className="text-base font-medium text-foreground">
                {title}
                {when && <span className="ml-2 text-sm font-normal text-muted-foreground">{when}</span>}
              </span>
              <span className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted-foreground tabular-nums">
                {facts.map((f) => (
                  <span key={f}>{f}</span>
                ))}
              </span>
              <span className="text-sm text-muted-foreground">{design}</span>
            </span>
          </label>
        );
      })}
    </fieldset>
  );
}

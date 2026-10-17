"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { PAGE_MAIN } from "@/components/page-header";
import { buttonVariants, Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { TooltipProvider } from "@/components/ui/tooltip";
import { api, unwrap } from "@/lib/api";
import { problem } from "@/lib/cluster";
import { useEvent } from "@/lib/events";
import { cn } from "@/lib/utils";
import { initialForm, startBody, toggled, type Defaults, type Form, type TextSet } from "../../cluster/training/logic";
import { useDefaults, useMachines, useTextSets, useVersions } from "../../cluster/training/use-training";
import { planBody } from "../../pretraining/logic";
import { QUICKTEST } from "../../quick-test/copy";
import { Coordinator, FIELD, Hint, OuterLoop, Workers, selectedText } from "./fields";
import { fill, NEW_RUN as T } from "./copy";
import { ctaLabel, defaultSftVersion, localForced, peersOf, pinnedJob, runHref, shapeOf, workersOf, type Node, type Stage } from "./logic";

type Guess = (typeof QUICKTEST.guesses)[number];

function Fields({ defaults, sets, nodes }: { defaults: Defaults; sets: TextSet[]; nodes: Node[] }) {
  const router = useRouter();
  const client = useQueryClient();
  const versions = useVersions().data ?? [];
  const [stage, setStage] = useState<Stage>("pretrain");
  const [form, setForm] = useState<Form>(() => initialForm(defaults, sets));
  const [picked, setPicked] = useState<string[]>(() => nodes.filter((n) => n.local).map((n) => n.address));
  const [mode, setMode] = useState<"catch_up" | "from_scratch">("from_scratch");
  const [dataset, setDataset] = useState<"stories" | "slice" | "problems">("stories");
  const [minutes, setMinutes] = useState<"1" | "3" | "10">("3");
  const [guess, setGuess] = useState<Guess | "">("");
  const [failure, setFailure] = useState("");
  const change = (patch: Partial<Form>) => {
    setForm((f) => ({ ...f, ...patch }));
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
  const plan = useQuery({
    queryKey: ["pretraining", "plan", "new-run", version, ids, mode],
    queryFn: ({ signal }) => unwrap(api.POST("/pretraining/plan", { body: planBody({ version, datasetIds: ids, mode, rows: "16", hold: "5%", length: "Medium" }), signal })),
    enabled: single && ids.length > 0,
    placeholderData: keepPreviousData,
  });
  const outlook = useQuery({ queryKey: ["queue", "outlook"], queryFn: ({ signal }) => unwrap(api.GET("/queue/outlook", { signal })), retry: false });
  useEvent("queue", () => void client.invalidateQueries({ queryKey: ["queue", "outlook"] }));

  const submit = useMutation({
    mutationFn: async (): Promise<string> => {
      if (several) {
        const body = startBody({ ...form, machines: peers });
        if (!body) throw new Error(T.needSettings);
        return (await unwrap(api.POST("/distributed/runs", { body }))).id;
      }
      if (stage === "pretrain") {
        if (!plan.data?.job) throw new Error(plan.data?.reason || T.needText);
        return (await unwrap(api.POST("/queue/jobs", { body: pinnedJob(plan.data.job, pin) }))).id;
      }
      const id =
        stage === "sft"
          ? (
              await unwrap(
                api.POST("/post-training/rounds", { body: { round: 1, version: sftVersion, lora: null, lora_rank: 8, teacher: "version", teacher_version: null, leash: "Medium" } }),
              )
            ).id
          : (await unwrap(api.POST("/quicktest/run", { body: { dataset, minutes: Number(minutes) as 1 | 3 | 10, shrink: false, guess: guess as Guess } }))).job_id;
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
    if (workers.length === 0) return T.needWorker;
    if (stage === "pretrain" && ids.length === 0) return T.needText;
    if (several) return startBody({ ...form, machines: peers }) ? "" : T.needSettings;
    if (stage === "pretrain") return plan.data && !plan.data.can_add ? plan.data.reason : plan.data?.job ? "" : T.loading;
    if (stage === "ablation") return guess ? "" : T.needGuess;
    return sftVersion === null ? T.noVersions : "";
  })();

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (!blocker) submit.mutate();
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

      {stage === "pretrain" && (
        <fieldset className="flex flex-col gap-2">
          <legend className="mb-2 text-xl font-semibold">
            <Hint tip={T.textTip}>{T.text}</Hint> <span className="text-sm font-normal text-muted-foreground">{selectedText(ids.length)}</span>
          </legend>
          {sets.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {T.noText} <Link href="/your-data" className="underline underline-offset-2">{T.yourData}</Link>
            </p>
          ) : (
            sets.map((s) => (
              <label key={s.id} className="flex min-h-9 items-center gap-2 text-base">
                <input type="checkbox" checked={form.datasets.includes(s.id)} onChange={(e) => change({ datasets: toggled(form.datasets, s.id, e.target.checked) })} />
                <span>{s.name}</span>
                <span className="text-sm text-muted-foreground">{s.count.toLocaleString("en-US")}</span>
              </label>
            ))
          )}
        </fieldset>
      )}

      {stage === "ablation" && (
        <fieldset className="flex flex-col gap-4">
          <Pick label={QUICKTEST.tinyDataset} value={dataset} onChange={(v) => setDataset(v as typeof dataset)} options={QUICKTEST.datasets} />
          <Pick label={QUICKTEST.timeEach} value={minutes} onChange={(v) => setMinutes(v as typeof minutes)} options={QUICKTEST.minutes} />
          <div className="flex flex-col gap-1.5">
            <span className="text-sm font-medium"><Hint tip={T.quickGuessTip}>{T.quickGuess}</Hint></span>
            <ToggleGroup type="single" variant="outline" value={guess} onValueChange={(v) => setGuess(v as Guess | "")} aria-label={T.quickGuess} className="justify-start">
              {QUICKTEST.guesses.map((g) => (
                <ToggleGroupItem key={g} value={g}>{g}</ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
          <p className="text-sm text-muted-foreground">
            {T.quickNeedsDraft} <Link href="/builder" className="text-highlight hover:underline">{T.quickDesigner}</Link>
          </p>
        </fieldset>
      )}

      <Workers nodes={nodes} workers={workers} forced={localForced(stage, picked, nodes)} multiple={stage === "pretrain"} onToggle={toggle} />
      <Coordinator />

      <div role="status" className="flex items-center">
        <Hint tip={shape.tip} className="rounded-full border border-border px-3.5 py-2 font-mono text-sm text-highlight">{shape.label}</Hint>
      </div>

      {several && <OuterLoop values={form} deltaDtype={defaults.delta_dtype} onChange={change} />}

      <section className="flex flex-col gap-2" aria-label={T.schedule}>
        <h2 className="text-xl font-semibold"><Hint tip={T.scheduleTip}>{T.schedule}</Hint></h2>
        <p className="text-base" data-testid="outlook">
          {outlook.data?.outlook ? `${outlook.data.outlook}. ` : outlook.isPending ? `${T.outlookLoading} ` : ""}
          <Link href="/queue" className="text-highlight hover:underline">{T.allowedHours}</Link>
        </p>
      </section>

      <div className="flex flex-wrap items-center justify-end gap-3 border-t pt-6">
        {failure && <p role="alert" className="mr-auto text-sm text-destructive">{failure}</p>}
        {!failure && blocker && <p className="mr-auto text-sm text-muted-foreground" data-testid="blocker">{blocker}</p>}
        <Link href="/runs" className={cn(buttonVariants({ variant: "outline", size: "lg" }))}>{T.cancel}</Link>
        <Button type="submit" size="lg" disabled={!!blocker || submit.isPending}>
          {submit.isPending ? T.working : ctaLabel(outlook.data?.outlook)}
        </Button>
      </div>
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
  const failed = defaults.isError || sets.isError || machines.isError;
  return (
    <TooltipProvider>
      <main className={PAGE_MAIN}>
        <header className="flex flex-col gap-2">
          <Link href="/runs" className="w-fit text-sm text-highlight hover:underline">← {T.back}</Link>
          <h1 className="text-2xl font-semibold tracking-tight">{T.title}</h1>
        </header>
        {defaults.data && sets.data && machines.data ? (
          <Fields defaults={defaults.data} sets={sets.data} nodes={machines.data} />
        ) : failed ? (
          <p role="alert" className="text-sm text-destructive">{fill(T.couldNotLoad, { why: problem(defaults.error ?? sets.error ?? machines.error) })}</p>
        ) : (
          <div className="flex flex-col gap-4" aria-label={T.loading}>
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        )}
      </main>
    </TooltipProvider>
  );
}

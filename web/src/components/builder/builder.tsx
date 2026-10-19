"use client";

// The Builder's design half (studio_ia/pages/builder.py BuilderPage, edit phase): the tower of blocks, where to start
// from, the knobs, what the draft costs and what to expect. The draft is this page's state, kept in localStorage so a
// reload finds it; every sum is one of the stateless /design routes, cached by TanStack Query on its inputs. Saving
// makes a version (POST /versions); the `versions` event then moves the comparison to it.

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { VersionTray } from "@/app/builder/version-tray";
import { Skeleton } from "@/components/ui/skeleton";
import { Controls, Costs, Expect, StartFrom, Tray, TryIt, VersionList } from "@/components/builder/side";
import { Experiments } from "@/components/builder/experiments";
import { Answers } from "@/components/builder/answers";
import { Explain } from "@/components/builder/explain";
import { GuessCheck, GuessPanel, type GuessStep } from "@/components/builder/guess";
import { LabNotebook } from "@/components/builder/notebook";
import { LrCard } from "@/app/pretraining/lr-card";
import { Tower } from "@/components/builder/tower";
import {
  anchoredDraft,
  changeText,
  design,
  detailOf,
  guessOf,
  isVersion,
  jobFor,
  pretraining,
  queueJobs,
  quicktest,
  sameChoices,
  specBase,
  specOf,
  loadDraft,
  storeDraft,
  upsertVersion,
  versionName,
  versions,
  wordChanged,
  type Choices,
  type Draft,
  type Explaining,
  type Guess,
  type Line,
  type Spec,
  type Try,
  type Version,
} from "@/lib/builder";
import { BUILDER, fill } from "@/lib/copy/builder";
import { useEvent } from "@/lib/events";
import { routeFor } from "@/lib/routes";

const MAX_BLOCKS = 12;

export function Builder() {
  const route = routeFor("builder");
  const client = useQueryClient();

  // What the server knows: the registry and presets never change while it runs; the versions do.
  const registryQ = useQuery({ queryKey: ["design", "registry"], queryFn: design.registry, staleTime: Infinity, retry: false });
  const presetsQ = useQuery({ queryKey: ["design", "presets"], queryFn: design.presets, staleTime: Infinity, retry: false });
  const versionsQ = useQuery({ queryKey: ["versions"], queryFn: versions.list, retry: false });
  const currentQ = useQuery({ queryKey: ["versions", "current"], queryFn: versions.current, retry: false });
  const previewQ = useQuery({ queryKey: ["pretraining", "preview"], queryFn: pretraining.preview, retry: false });

  useEvent("versions", (payload) => {
    if (isVersion(payload)) client.setQueryData<Version[]>(["versions"], (list) => upsertVersion(list, payload));
    else client.invalidateQueries({ queryKey: ["versions"], exact: true });
    client.invalidateQueries({ queryKey: ["versions", "current"], exact: true });
  });

  const registry = registryQ.data;
  const current = currentQ.data ?? null;
  const base = current?.number ?? null;
  const baseSpec = useMemo(() => specOf(current), [current]);
  const curQ = useQuery({
    queryKey: ["design", "read", baseSpec],
    queryFn: () => design.read(baseSpec),
    enabled: currentQ.isSuccess,
    staleTime: Infinity,
    retry: false,
  });
  const cur = curQ.data;

  // The draft: against the current version. A new current version (a save, here or elsewhere) starts a fresh draft from
  // it, as the Tk page's refresh does; a reload finds the one localStorage kept when it is against the same version.
  const [store, setStore] = useState<Draft | null>(null);
  // A split saves several versions one after another; each moves the current version, which must not reset the draft until it is done.
  const [splitting, setSplitting] = useState(false);
  if (cur && !splitting && (!store || store.base !== base)) {
    // Set during render, as React suggests for state that follows a prop: the next render has the draft.
    const kept = loadDraft();
    setStore(kept && kept.base === base ? kept : { base, draft: cur, anchor: null });
  }
  useEffect(() => {
    if (store) storeDraft(store);
  }, [store]);

  const ready = Boolean(registry && cur && store && store.base === base);
  const draft = store?.draft;
  const { base: writeBase, was } = store && cur ? specBase(store, baseSpec, cur) : { base: baseSpec, was: null };

  // The draft as a spec: what every cost and expectation is read from. A draft the maths cannot write (a part that
  // does not fit) keeps the last good spec's numbers on screen and says why in the tray.
  const writeQ = useQuery({
    queryKey: ["design", "write", writeBase, draft, was],
    queryFn: () => design.write(writeBase, draft!, was),
    enabled: ready,
    placeholderData: keepPreviousData,
    retry: false,
  });
  const draftSpec: Spec | null = writeQ.isError ? null : (writeQ.data ?? null);
  const lrSpec: Spec | null = draftSpec ?? baseSpec;
  const cannotWrite = writeQ.isError ? detailOf(writeQ.error) : null;

  const curTowerQ = useQuery({ queryKey: ["design", "tower", cur], queryFn: () => design.tower(cur!), enabled: !!cur, staleTime: Infinity, retry: false });
  const draftTowerQ = useQuery({
    queryKey: ["design", "tower", draft],
    queryFn: () => design.tower(draft!),
    enabled: ready,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    retry: false,
  });

  const sizesNowQ = useQuery({
    queryKey: ["design", "sizes", baseSpec, cur?.V],
    queryFn: () => design.sizes(baseSpec, cur!.V),
    enabled: !!cur,
    staleTime: Infinity,
    retry: false,
  });
  const sizesNewQ = useQuery({
    queryKey: ["design", "sizes", draftSpec, draft?.V],
    queryFn: () => design.sizes(draftSpec!, draft!.V),
    enabled: ready && !!draftSpec,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    retry: false,
  });
  const changesQ = useQuery({
    queryKey: ["design", "changes", baseSpec, draftSpec, cur, draft],
    queryFn: () => design.changes(baseSpec, draftSpec!, cur!, draft!),
    enabled: ready && !!draftSpec,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    retry: false,
  });
  const expectQ = useQuery({
    queryKey: ["design", "expectations", cur, draft, baseSpec, draftSpec],
    queryFn: () => design.expectations(cur!, draft!, baseSpec, draftSpec),
    enabled: ready && !writeQ.isPending,
    placeholderData: keepPreviousData,
    retry: false,
  });
  const recount = Boolean(cur && draft && wordChanged(cur, draft));
  const computeSpec = draftSpec ?? baseSpec;
  const preview = previewQ.data ?? null;
  const linesQ = useQuery({
    queryKey: ["design", "compute-lines", computeSpec, draft?.V, preview],
    queryFn: () => design.computeLines(computeSpec, draft!.V, preview),
    enabled: ready && !recount && previewQ.isFetched,
    placeholderData: keepPreviousData,
    retry: false,
  });
  const lines: Line[] = recount ? [{ text: BUILDER.compute_recount, bad: false }] : (linesQ.data ?? []);
  const chips = draftSpec ? (changesQ.data?.chips ?? []) : [];
  const warm = draftSpec ? (changesQ.data?.warm ?? null) : null;

  // An unsaved change is staged for Quick test and Compare runs, and taken back when the draft is discarded or saved.
  const staged = useRef<string | null>(null);
  const toStage = ready && !writeQ.isPlaceholderData && draftSpec && cur && draft && !sameChoices(cur, draft) ? draftSpec : null;
  const stageKey = toStage ? JSON.stringify(toStage) : null;
  useEffect(() => {
    if (!ready || writeQ.isPlaceholderData || stageKey === staged.current) return;
    staged.current = stageKey;
    quicktest.stage(toStage).catch(() => {
      staged.current = null; // not staged: try again on the next change
    });
    // toStage is stageKey's source; the key is what says it changed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stageKey, ready, writeQ.isPlaceholderData]);

  // Editing: every change is a new draft; the queries above re-run on it.
  const edit = (next: Choices) => setStore((s) => (s ? { ...s, draft: next } : s));
  const run = async (work: () => Promise<void>) => {
    try {
      await work();
    } catch (reason) {
      toast.error(detailOf(reason));
    }
  };
  const pick = (slot: Parameters<typeof design.choose>[1], short: string, block: number | null) =>
    run(async () => edit(await design.choose(draft!, slot, short, block)));
  const resize = (n: number) => {
    if (n < 1 || n > MAX_BLOCKS) return;
    void run(async () => edit(await design.resize(draft!, n)));
  };
  const knob = (key: "H" | "W", value: number) => {
    if (draftTowerQ.data?.knobs[key]) edit({ ...draft!, [key]: value });
  };
  const setWords = (v: number) => edit({ ...draft!, V: v });
  const loadSpec = (spec: Spec) =>
    run(async () => {
      const choices = await design.read(spec);
      setStore(anchoredDraft(base, spec, choices));
    });
  const discard = () => setStore({ base, draft: cur!, anchor: null });

  // The side panel explains the part last clicked; a starting question sets up one change and goes with the version it is saved as.
  const [explaining, setExplaining] = useState<Explaining | null>(null);
  const [experiment, setExperiment] = useState<{ question: string; draft: Choices } | null>(null);
  const heldQuestion = experiment && JSON.stringify(experiment.draft) === JSON.stringify(draft) ? experiment.question : null;
  const tryQuestion = (key: Try["key"]) =>
    run(async () => {
      const tried = await design.experiment(cur!, key);
      if (!tried.choices) {
        toast.error(tried.why);
        return;
      }
      setStore({ base, draft: tried.choices, anchor: null });
      setExperiment({ question: tried.question, draft: tried.choices });
    });

  // Saving: version 1 has nothing to compare, so it is saved at once. Any other save first asks for a guess at what the
  // change will do (or Skip), as the Tk page does; a draft of several changes can instead be split into one version
  // (and one queued job) per change. `asking` is only live while the draft is the one it was opened on.
  const draftKey = JSON.stringify([base, store?.draft, store?.anchor]);
  const [asked, setAsked] = useState<{ kind: "one" | "split"; key: string } | null>(null);
  const [picks, setPicks] = useState<Guess[]>([]);
  if (asked && asked.key !== draftKey) setAsked(null); // the draft moved on: the question is about a draft that is gone
  const asking = asked ? asked.kind : null;
  const ask = (kind: "one" | "split") => {
    setPicks([]);
    setAsked({ kind, key: draftKey });
  };

  const textSetsQ = useQuery({ queryKey: ["pretraining", "text_sets"], queryFn: queueJobs.textSets, retry: false });
  const datasets = useMemo(() => (textSetsQ.data ?? []).filter((set) => set.on).map((set) => set.id), [textSetsQ.data]);
  const questionsQ = useQuery({
    queryKey: ["design", "questions", baseSpec, draftSpec],
    queryFn: () => design.questions(baseSpec, draftSpec!),
    enabled: asking === "one" && !!draftSpec,
    staleTime: Infinity,
    retry: false,
  });
  // One version per change: only for a draft written onto the current version (a design loaded from elsewhere is saved whole).
  const splittable = ready && !!cur && !!draftSpec && !store?.anchor && base !== null && chips.length > 1 && textSetsQ.isSuccess;
  const splitQ = useQuery({
    queryKey: ["design", "split-plan", baseSpec, cur, draft, base, datasets],
    queryFn: () => design.splitPlan(baseSpec, cur!, cur!, draft!, base, datasets),
    enabled: splittable && !writeQ.isPlaceholderData,
    staleTime: Infinity,
    retry: false,
  });
  const plan = splitQ.data?.steps.length ? splitQ.data : null;

  const saved = (version: Version) => client.setQueryData<Version[]>(["versions"], (list) => upsertVersion(list, version));
  const save = useMutation({
    mutationFn: ({ guess, question }: { guess: Guess; question: string | null }) =>
      versions.save(draftSpec!, changeText(chips, base), question ? { ...guess, question } : guess),
    onSuccess: (version) => {
      saved(version);
      void client.invalidateQueries({ queryKey: ["versions", "current"], exact: true });
      setAsked(null);
      setExperiment(null);
      toast.success(fill(BUILDER.saved, { version: versionName(version.number) }));
    },
    onError: (reason) => toast.error(detailOf(reason)),
  });
  const saveSplit = useMutation({
    mutationFn: async (guesses: Guess[]) => {
      if (!plan) throw new Error(BUILDER.one_change_cannot);
      if (!datasets.length) throw new Error(BUILDER.no_text);
      const made: Version[] = [];
      let queued = 0;
      try {
        for (const [i, step] of plan.steps.entries()) {
          const version = await versions.save(step.spec, step.chips.join(", "), guesses[i]);
          saved(version);
          made.push(version);
        }
        for (const [i, job] of plan.jobs.entries()) {
          await queueJobs.enqueue(jobFor(job, Number(job.params?.version), made[i].number));
          queued++;
        }
      } catch (reason) {
        if (made.length) {
          throw new Error(
            fill(BUILDER.split_partial, {
              saved: made.map((v) => versionName(v.number)).join(", "),
              queued,
              total: plan.jobs.length,
              reason: detailOf(reason),
            }),
          );
        }
        throw reason;
      }
      return made;
    },
    onMutate: () => setSplitting(true),
    onSettled: () => {
      setSplitting(false);
      void client.invalidateQueries({ queryKey: ["versions", "current"], exact: true });
      void client.invalidateQueries({ queryKey: ["queue"] });
    },
    onSuccess: (made) => {
      setAsked(null);
      toast.success(fill(BUILDER.split_note, { first: made[0].number, last: made[made.length - 1].number }));
    },
    onError: (reason) => toast.error(detailOf(reason)),
  });
  // Back at the tray (Back, a save, or the draft moving on), keyboard focus returns to the button that opened the panel.
  const opener = useRef<"one" | "split" | null>(null);
  useEffect(() => {
    if (asking) {
      opener.current = asking;
      return;
    }
    if (!opener.current) return;
    const id = opener.current === "split" ? "builder-split" : "builder-save";
    opener.current = null;
    document.getElementById(id)?.focus();
  }, [asking]);
  const startSave = () => (base === null || !draftSpec ? save.mutate({ guess: {}, question: null }) : ask("one"));
  const finish = (skip: boolean) => {
    const guesses = (plan && asking === "split" ? plan.steps : [null]).map((_, i) => (skip ? {} : guessOf(picks[i])));
    if (asking === "split") saveSplit.mutate(guesses);
    else save.mutate({ guess: guesses[0], question: heldQuestion });
  };
  const guessSteps: GuessStep[] | null =
    asking === "split"
      ? plan
        ? plan.steps.map((step, i) => ({
            title: fill(BUILDER.predict_step_title, { n: (base ?? 0) + 1 + i, change: step.chips.join(", ") }),
            questions: step.questions,
          }))
        : null
      : questionsQ.data
        ? [{ title: "", questions: questionsQ.data }]
        : null;
  const guessError = asking === "split" ? (splitQ.error ?? null) : questionsQ.error;
  const pickGuess = (step: number, key: string, value: string) =>
    setPicks((all) => {
      const next = [...all];
      next[step] = { ...next[step], [key]: value };
      return next;
    });
  const before = current && (current.parent ?? current.number - 1) > 0 ? (current.parent ?? current.number - 1) : null;
  const checkBefore = before !== null && (versionsQ.data ?? []).some((v) => v.number === before) ? before : null;

  const [note, setNote] = useState("");
  const importGpt2 = () => {
    if (!window.confirm(BUILDER.import_gpt2_ask)) return;
    versions.importPretrained("gpt2").then(
      () => setNote(BUILDER.import_gpt2_started),
      (reason) => setNote(detailOf(reason)),
    );
  };

  const failed = registryQ.error ?? currentQ.error ?? curQ.error;

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-3 p-8">
      <h1 className="text-2xl font-semibold tracking-tight">{route.title}</h1>
      <p className="text-sm text-muted-foreground">{route.lead}</p>
      {failed ? (
        <p role="alert" className="rounded-md border border-destructive/40 p-2 text-sm text-destructive">
          {BUILDER.loading_failed} {detailOf(failed)}
        </p>
      ) : !ready || !registry || !cur || !draft ? (
        <Loading />
      ) : (
        <>
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <Tower
              registry={registry}
              cur={cur}
              draft={draft}
              curTower={curTowerQ.data}
              draftTower={draftTowerQ.data}
              writeBase={writeBase}
              was={was}
              onPick={pick}
              onResize={resize}
              onWords={setWords}
              explaining={explaining}
              onExplain={(part, block) => setExplaining({ part, block })}
            />
            <div className="flex flex-col gap-3">
              <Explain draft={draft} explaining={explaining} onClose={() => setExplaining(null)} />
              <StartFrom
                presets={presetsQ.data ?? []}
                versions={versionsQ.data ?? []}
                onLoad={loadSpec}
                onGpt2={importGpt2}
                note={note}
              />
              <Experiments cur={cur} asking={heldQuestion} busy={save.isPending || saveSplit.isPending} onTry={tryQuestion} />
              <Controls registry={registry} cur={cur} draft={draft} knobs={draftTowerQ.data?.knobs} onKnob={knob} />
              <Costs now={sizesNowQ.data} next={draftSpec ? sizesNewQ.data : sizesNowQ.data} warm={warm} lines={lines} />
              <LrCard version={lrSpec ? JSON.stringify(lrSpec) : null} spec={lrSpec} />
              <Expect expectations={expectQ.data} />
              <VersionList versions={versionsQ.data} current={base} />
            </div>
          </div>
          {asking ? (
            <GuessPanel
              steps={guessSteps}
              error={guessError ? detailOf(guessError) : null}
              picks={picks}
              question={heldQuestion}
              saveLabel={asking === "split" && plan ? fill(BUILDER.predict_split_save, { count: plan.steps.length }) : BUILDER.predict_save}
              saving={save.isPending || saveSplit.isPending}
              onPick={pickGuess}
              onSave={() => finish(false)}
              onSkip={() => finish(true)}
              onBack={() => setAsked(null)}
            />
          ) : (
            <Tray
              base={base}
              chips={chips}
              cannotWrite={cannotWrite}
              saving={save.isPending || saveSplit.isPending}
              canSave={!!draftSpec && !writeQ.isPlaceholderData}
              splitCount={plan?.steps.length ?? 0}
              onSave={startSave}
              onSplit={() => ask("split")}
              onDiscard={discard}
            />
          )}
          {textSetsQ.isError && chips.length > 1 && base !== null && (
            <p role="alert" className="text-xs text-destructive">
              {fill(BUILDER.text_sets_failed, { reason: detailOf(textSetsQ.error) })}
            </p>
          )}
          <Answers versions={versionsQ.data} current={base} />
          {current && <GuessCheck before={checkBefore} version={current} />}
        </>
      )}
      {/* Outside the ready branch: a save moves the current version and the page reads its design again; the notebook keeps its rows. */}
      {!failed && <LabNotebook versions={versionsQ.data} />}
      {!failed && <TryIt />}
      <VersionTray />
    </main>
  );
}

function Loading() {
  return (
    <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_22rem]" aria-busy="true">
      <div className="flex flex-col gap-2">
        {Array.from({ length: 6 }, (_, i) => (
          <Skeleton key={i} className="h-10 w-full" />
        ))}
      </div>
      <div className="flex flex-col gap-3">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-28 w-full" />
        ))}
      </div>
    </div>
  );
}

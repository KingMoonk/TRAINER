"use client";

// The Builder's design half (studio_ia/pages/builder.py BuilderPage, edit phase): the tower of blocks, where to start
// from, the knobs, what the draft costs and what to expect. The draft is this page's state, kept in localStorage so a
// reload finds it; every sum is one of the stateless /design routes, cached by TanStack Query on its inputs. Saving
// makes a version (POST /versions); the `versions` event then moves the comparison to it.

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Skeleton } from "@/components/ui/skeleton";
import { Controls, Costs, Expect, StartFrom, Tray, VersionList } from "@/components/builder/side";
import { Tower } from "@/components/builder/tower";
import {
  anchoredDraft,
  changeText,
  design,
  detailOf,
  isVersion,
  pretraining,
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
  type Line,
  type Spec,
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
  if (cur && (!store || store.base !== base)) {
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

  const save = useMutation({
    mutationFn: () => versions.save(draftSpec!, changeText(chips, base)),
    onSuccess: (version) => {
      client.setQueryData<Version[]>(["versions"], (list) => upsertVersion(list, version));
      void client.invalidateQueries({ queryKey: ["versions", "current"], exact: true });
      toast.success(fill(BUILDER.saved, { version: versionName(version.number) }));
    },
    onError: (reason) => toast.error(detailOf(reason)),
  });

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
            />
            <div className="flex flex-col gap-3">
              <StartFrom
                presets={presetsQ.data ?? []}
                versions={versionsQ.data ?? []}
                onLoad={loadSpec}
                onGpt2={importGpt2}
                note={note}
              />
              <Controls registry={registry} cur={cur} draft={draft} knobs={draftTowerQ.data?.knobs} onKnob={knob} />
              <Costs now={sizesNowQ.data} next={draftSpec ? sizesNewQ.data : sizesNowQ.data} warm={warm} lines={lines} />
              <Expect expectations={expectQ.data} />
              <VersionList versions={versionsQ.data} current={base} />
            </div>
          </div>
          <Tray
            base={base}
            chips={chips}
            cannotWrite={cannotWrite}
            saving={save.isPending}
            canSave={!!draftSpec && !writeQ.isPlaceholderData}
            onSave={() => save.mutate()}
            onDiscard={discard}
          />
        </>
      )}
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

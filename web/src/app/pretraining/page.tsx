"use client";

import Link from "next/link";

import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap, type ApiEvent } from "@/lib/api";
import { CollectionCard } from "../your-data/collection-card";
import { useApiEvents } from "../your-data/events";
import { runJob, type JobLike } from "../your-data/jobs";
import { BatchesCard, EnoughCard, SetsCard } from "./cards";
import { T } from "./copy";
import { DEFAULT_HOURS, HOLD, PREVIEW_ROWS, DEFAULT_ROWS, chosenIds, errorText, planBody, type Chat, type Hours, type Length, type PreparedStatus, type Preview, type StepRate, savedRows, type StopWhenFlat } from "./logic";
import { LrCard } from "./lr-card";
import { ScalingCard } from "./scaling-card";
import { SizingCard } from "./sizing-card";
import { TrainCard } from "./train-card";
import { Disclosure } from "@/components/disclosure";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

export default function PretrainingPage() {
  const queryClient = useQueryClient();
  // The draft choices live here, in the browser; the server turns them into steps, a config and a job.
  const [use, setUse] = useState<Record<string, boolean>>({});
  const [mode, setMode] = useState<"catch_up" | "from_scratch">("catch_up");
  const [hold, setHold] = useState<(typeof HOLD)[number]>("5%");
  const [length, setLength] = useState<Length>("Medium");
  const [chat, setChat] = useState<Chat>("Off"); // off unless the person mixes some conversations in
  const [hours, setHours] = useState<Hours>(DEFAULT_HOURS);
  const [stopWhenFlat, setStopWhenFlat] = useState<StopWhenFlat>("Auto");
  const [keepMine, setKeepMine] = useState(false); // "use my settings anyway", on the fitted-to-this-computer card
  const [chosenSteps, setChosenSteps] = useState<number | null>(null); // from the sizing suggestion; a length chosen afterwards clears it

  const versions = useQuery({ queryKey: ["versions"], queryFn: () => unwrap(api.GET("/versions")) });
  // Rows per step start at the newest version's saved batch size (Model setup) until one is picked here.
  const [pickedRows, setRows] = useState<string | null>(null);
  const saved = savedRows(versions.data?.at(-1));
  const rows = pickedRows ?? saved ?? DEFAULT_ROWS;
  const sets = useQuery({ queryKey: ["pretraining", "text_sets"], queryFn: () => unwrap(api.GET("/pretraining/text_sets")) });
  const ids = useMemo(() => chosenIds(sets.data, use), [sets.data, use]);

  const body = planBody({ version: null, datasetIds: ids, mode, rows, hold, length, hours, steps: chosenSteps, stopWhenFlat, keepMine, chat });
  const plan = useQuery({
    queryKey: ["pretraining", "plan", body],
    queryFn: () => unwrap(api.POST("/pretraining/plan", { body })),
    enabled: sets.isSuccess,
    placeholderData: keepPreviousData, // a changed choice keeps the last numbers on screen until the new ones are in
  });

  // The size of the chosen text: the cached answer at once, else a job (it samples the text and builds a word list).
  const previewKey = ["pretraining", "preview", ids, PREVIEW_ROWS];
  const cached = useQuery({
    queryKey: previewKey,
    queryFn: () => unwrap(api.GET("/pretraining/preview", { params: { query: { dataset_ids: ids, rows: PREVIEW_ROWS } } })),
    enabled: sets.isSuccess && ids.length > 0,
  });
  const version = plan.data?.version ?? null;
  const asking = cached.isSuccess && cached.data === null && ids.length > 0;
  const previewJob = useQuery({
    queryKey: ["pretraining", "preview-job", ids, version],
    queryFn: async ({ signal }) => {
      const found = await runJob<Preview>(
        () => unwrap(api.POST("/pretraining/preview", { body: { dataset_ids: ids, rows: PREVIEW_ROWS } })) as Promise<JobLike>,
        signal,
        { cancel: true }, // sets chosen again: the old read is not needed
      );
      queryClient.setQueryData(previewKey, found);
      queryClient.invalidateQueries({ queryKey: ["pretraining", "plan"] });
      return found;
    },
    enabled: asking,
    staleTime: Infinity,
    retry: false,
  });
  useEffect(() => {
    if (previewJob.error) toast.error(errorText(previewJob.error));
  }, [previewJob.error]);
  const estimating = ids.length > 0 && (plan.data?.estimating ?? true);

  // Whether binary token shards are already prepared on disk for these choices.
  const preparedKey = useMemo(() => ["pretraining", "prepared", ids, version, hold, rows], [ids, version, hold, rows]);
  const prepared = useQuery({
    queryKey: preparedKey,
    queryFn: () =>
      unwrap(
        api.GET("/pretraining/prepared", {
          params: {
            query: {
              dataset_ids: ids,
              version: version ?? undefined,
              hold: Number(hold.slice(0, -1)) / 100,
              rows: Number(rows),
            },
          },
        }),
      ),
    enabled: sets.isSuccess && ids.length > 0 && version !== null,
    staleTime: 10_000,
  });

  const [preparing, setPreparing] = useState(false);
  const [prepareProgress, setPrepareProgress] = useState<{ note: string; fraction: number | null } | null>(null);

  const prepare = useCallback(async () => {
    if (ids.length === 0 || version === null) return;
    setPreparing(true);
    setPrepareProgress({ note: T.preparingData, fraction: 0 });
    try {
      const result = await runJob<PreparedStatus>(
        () =>
          unwrap(
            api.POST("/pretraining/prepare", {
              body: {
                dataset_ids: ids,
                version,
                hold: Number(hold.slice(0, -1)) / 100,
                rows: Number(rows),
              },
            }),
          ) as Promise<JobLike>,
      );
      queryClient.setQueryData(preparedKey, result);
      queryClient.invalidateQueries({ queryKey: ["pretraining", "prepared"] });
      toast.success(T.dataPreparedToast);
    } catch (err) {
      toast.error(errorText(err));
    } finally {
      setPreparing(false);
      setPrepareProgress(null);
    }
  }, [ids, version, hold, rows, preparedKey, queryClient]);

  // How long a step takes on this computer: a few timed steps of the design, as a job, once the size of the text is known.
  const needsRate = plan.data?.sizing != null && !plan.data.sizing.measured && version !== null;
  const rateJob = useQuery({
    queryKey: ["pretraining", "step-rate", version, rows],
    queryFn: async ({ signal }) => {
      const found = await runJob<StepRate>(
        () => unwrap(api.POST("/pretraining/step_rate", { body: { version, dataset_ids: ids, rows: Number(rows), preview_rows: PREVIEW_ROWS } })) as Promise<JobLike>,
        signal,
      );
      queryClient.invalidateQueries({ queryKey: ["pretraining", "plan"] });
      return found;
    },
    enabled: needsRate,
    staleTime: Infinity,
    retry: false,
  });
  useEffect(() => {
    if (rateJob.error) toast.error(errorText(rateJob.error));
  }, [rateJob.error]);

  const onEvent = useCallback(
    (event: ApiEvent) => {
      if (event.kind === "data" || event.kind === "downloads_finished" || event.kind === "prepared") {
        queryClient.invalidateQueries({ queryKey: ["pretraining"] });
      } else if (event.kind === "job") {
        const payload = event.payload as { kind?: string; progress?: string; fraction?: number | null } | undefined;
        if (payload?.kind === "pretrain_prepare") {
          setPrepareProgress({
            note: payload.progress || T.preparingData,
            fraction: payload.fraction ?? null,
          });
        }
      } else if (event.kind === "versions") {
        queryClient.invalidateQueries({ queryKey: ["versions"] });
        queryClient.invalidateQueries({ queryKey: ["pretraining"] });
      } else if (event.kind === "queue" || event.kind === "blocked") {
        queryClient.invalidateQueries({ queryKey: ["queue"] });
        queryClient.invalidateQueries({ queryKey: ["pretraining", "plan"] });
      }
    },
    [queryClient],
  );
  useApiEvents(onEvent);

  const current = versions.data?.find((v) => v.number === version) ?? null;
  const toggle = (set: { id: string }) => setUse((now) => ({ ...now, [set.id]: !(now[set.id] ?? sets.data?.find((s) => s.id === set.id)?.on) }));

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={T.h1} lead={T.lead}>
        <Link href="/your-data/?kind=text#make-extra-data" className="w-fit text-sm text-highlight hover:underline">{T.makeData} ›</Link>
      </PageHeader>

      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.modelLabel}</div>
        {plan.isLoading || !plan.data ? (
          <Skeleton className="mt-1 h-6 w-72" />
        ) : (
          <p className="mt-1 text-base" data-testid="model-line">
            {plan.data.version ? `Version ${plan.data.version} · from Build your model` : T.noDesign}
          </p>
        )}
      </div>

      <SetsCard
        sets={sets.data}
        loading={sets.isLoading}
        chosen={ids}
        onToggle={toggle}
        prepared={prepared.data}
        preparing={preparing}
        onPrepare={prepare}
        progress={prepareProgress}
      />
      <Disclosure title="Download more text" testId="collection-disclosure">
        <CollectionCard kind="text" description={T.collectionDesc} />
      </Disclosure>
      <EnoughCard plan={plan.data} estimating={estimating} />
      <BatchesCard plan={plan.data} preview={cached.data} estimating={estimating} rows={rows} savedRows={saved} hold={hold} chat={chat} onRows={setRows} onHold={setHold} onChat={setChat} />
      <SizingCard plan={plan.data} estimating={estimating} hours={hours} onHours={setHours} chosenSteps={chosenSteps} onUseSteps={setChosenSteps} />
      <TrainCard
        plan={plan.data}
        body={body}
        estimating={estimating}
        mode={mode}
        onMode={setMode}
        length={length}
        stopWhenFlat={stopWhenFlat}
        onStopWhenFlat={setStopWhenFlat}
        prepared={prepared.data}
        onLength={(next) => {
          setChosenSteps(null);
          setLength(next);
        }}
      />
      <Disclosure title="Advanced: learning rate and scaling" testId="advanced-disclosure" reveal={plan.data?.scaling?.refusal}>
        <ScalingCard plan={plan.data} keepMine={keepMine} onKeepMine={setKeepMine} />
        <LrCard version={version} spec={current?.spec ?? null} />
      </Disclosure>
    </main>
  );
}

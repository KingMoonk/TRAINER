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
import { DEFAULT_HOURS, HOLD, PREVIEW_ROWS, ROWS, chosenIds, errorText, planBody, type Hours, type Length, type Preview, type StepRate, type StopWhenFlat } from "./logic";
import { LrCard } from "./lr-card";
import { SizingCard } from "./sizing-card";
import { TrainCard } from "./train-card";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

export default function PretrainingPage() {
  const queryClient = useQueryClient();
  // The draft choices live here, in the browser; the server turns them into steps, a config and a job.
  const [use, setUse] = useState<Record<string, boolean>>({});
  const [mode, setMode] = useState<"catch_up" | "from_scratch">("catch_up");
  const [rows, setRows] = useState<(typeof ROWS)[number]>("16");
  const [hold, setHold] = useState<(typeof HOLD)[number]>("5%");
  const [length, setLength] = useState<Length>("Medium");
  const [hours, setHours] = useState<Hours>(DEFAULT_HOURS);
  const [stopWhenFlat, setStopWhenFlat] = useState<StopWhenFlat>("Auto");
  const [chosenSteps, setChosenSteps] = useState<number | null>(null); // from the sizing suggestion; a length chosen afterwards clears it

  const versions = useQuery({ queryKey: ["versions"], queryFn: () => unwrap(api.GET("/versions")) });
  const sets = useQuery({ queryKey: ["pretraining", "text_sets"], queryFn: () => unwrap(api.GET("/pretraining/text_sets")) });
  const ids = useMemo(() => chosenIds(sets.data, use), [sets.data, use]);

  const body = planBody({ version: null, datasetIds: ids, mode, rows, hold, length, hours, steps: chosenSteps, stopWhenFlat });
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
      if (event.kind === "data" || event.kind === "downloads_finished") {
        queryClient.invalidateQueries({ queryKey: ["pretraining"] });
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
        <Link href="/make-data/?kind=text" className="w-fit text-sm text-highlight hover:underline">{T.makeData} ›</Link>
      </PageHeader>

      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.modelLabel}</div>
        {plan.isLoading || !plan.data ? (
          <Skeleton className="mt-1 h-6 w-72" />
        ) : (
          <p className="mt-1 text-base" data-testid="model-line">
            {plan.data.version ? `Version ${plan.data.version} · from the Network designer` : T.noDesign}
          </p>
        )}
      </div>

      <SetsCard sets={sets.data} loading={sets.isLoading} chosen={ids} onToggle={toggle} />
      <CollectionCard kind="text" description={T.collectionDesc} />
      <EnoughCard plan={plan.data} estimating={estimating} />
      <BatchesCard plan={plan.data} preview={cached.data} estimating={estimating} rows={rows} hold={hold} onRows={setRows} onHold={setHold} />
      <SizingCard plan={plan.data} estimating={estimating} hours={hours} onHours={setHours} chosenSteps={chosenSteps} onUseSteps={setChosenSteps} />
      <TrainCard
        plan={plan.data}
        estimating={estimating}
        mode={mode}
        onMode={setMode}
        length={length}
        stopWhenFlat={stopWhenFlat}
        onStopWhenFlat={setStopWhenFlat}
        onLength={(next) => {
          setChosenSteps(null);
          setLength(next);
        }}
      />
      <LrCard version={version} spec={current?.spec ?? null} />
    </main>
  );
}

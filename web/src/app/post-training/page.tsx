"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ROUNDS, STATUS_TEXT, T } from "./copy";
import { GateCard } from "./gate-card";
import { ROUND2_SETTINGS, count, errorText, firstOpenRound, roundBlocker, settingValue, type Gate, type JobOut, type Status } from "./logic";
import { BaseWarning, KeptCard } from "./kept-card";
import { baseWarning, isKept, type KeptGate } from "./kept-logic";
import { LoopCard } from "./loop-card";
import { RoundCard, type Choices, type Live } from "./round-card";
import { ImportCard, RowsCard, VersionsCard } from "./side-cards";
import { useEventStream, useJob } from "./use-events";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { Disclosure } from "@/components/disclosure";
import { optionBox } from "@/components/option-box";

const REFRESH_MS = 600;

export default function PostTrainingPage() {
  const queryClient = useQueryClient();
  const [picked, setPicked] = useState<number | null>(null);
  const [choices, setChoices] = useState<Choices>({ lora: null, rank: 8, teacher: "version", teacherVersion: null, leash: "Medium", method: "practice", round2: { steps: "", prompts_per_step: "", group: "", temperature: "", max_tokens: "", kl: "" } });
  const [live, setLive] = useState<Live>({});
  const [liveGates, setLiveGates] = useState<Record<number, Gate>>({});

  // The version the rounds start from: the best trained one, else the latest.
  const versionsQuery = useQuery({ queryKey: ["versions"], queryFn: () => unwrap(api.GET("/versions")) });
  const versions = versionsQuery.data;
  const version = useMemo(() => [...(versions ?? [])].reverse().find((v) => v.best_checkpoint) ?? versions?.at(-1), [versions]);

  const optionsQuery = useQuery({
    queryKey: ["post-training", "options", version?.number ?? null],
    queryFn: () => unwrap(api.GET("/post-training/options", { params: { query: { version: version?.number } } })),
    enabled: versionsQuery.isSuccess,
  });
  const options = optionsQuery.data;

  const queueQuery = useQuery({ queryKey: ["queue"], queryFn: () => unwrap(api.GET("/queue")) });
  const outlookQuery = useQuery({ queryKey: ["queue", "outlook"], queryFn: () => unwrap(api.GET("/queue/outlook")) });

  // What each round has done: a job, so the first read shows a skeleton and later reads keep the last answer on screen.
  const statusRequest = useCallback(
    () => unwrap(api.POST("/post-training/status", { body: { version: version?.number ?? null } })) as Promise<JobOut>,
    [version?.number],
  );
  const requestedFor = useRef<number | null>(null);
  const [status, setStatus] = useState<Status | undefined>();
  const [statusFor, setStatusFor] = useState<number | null | undefined>();
  const statusJob = useJob(statusRequest, (job) => {
    if (job.status === "done") {
      setStatus(job.result as Status);
      setStatusFor(requestedFor.current);
    } else if (job.status === "failed") {
      toast.error(job.error);
    }
  });

  const refreshRef = useRef<() => void>(() => {});
  useEffect(() => {
    refreshRef.current = () => {
      requestedFor.current = version?.number ?? null;
      statusJob.start().catch((error) => toast.error(errorText(error)));
    };
  });
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const refreshSoon = useCallback(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => refreshRef.current(), REFRESH_MS);
  }, []);
  useEffect(() => {
    if (versionsQuery.isSuccess) refreshRef.current();
    return () => clearTimeout(timer.current);
  }, [versionsQuery.isSuccess, version?.number]);

  const lastQueueRefresh = useRef(0);
  useEventStream((event) => {
    if (event.kind === "queue") {
      queryClient.invalidateQueries({ queryKey: ["queue"] });
      refreshSoon();
    } else if (event.kind === "versions") {
      queryClient.invalidateQueries({ queryKey: ["versions"] });
      queryClient.invalidateQueries({ queryKey: ["post-training", "options"] });
      refreshSoon();
    } else if (event.kind === "train") {
      const p = event.payload as { type?: string } & Record<string, unknown>;
      if (p.type === "group") setLive((l) => ({ ...l, group: p as unknown as Live["group"] }));
      else if (p.type === "heat") setLive((l) => ({ ...l, heat: p as unknown as Live["heat"] }));
      else if (p.type === "moved") setLive((l) => ({ ...l, moved: p as unknown as Live["moved"] }));
      else if (p.type === "metric" && "groups_total" in p) setLive((l) => ({ ...l, spread: p as unknown as Live["spread"] }));
      else if (p.type === "gate") {
        setLiveGates((g) => ({ ...g, [Number(p.round)]: p as Gate }));
        refreshSoon();
      }
      if (Date.now() - lastQueueRefresh.current > 1000) {
        lastQueueRefresh.current = Date.now();
        queryClient.invalidateQueries({ queryKey: ["queue"] });
      }
    }
  });

  const rounds = status?.rounds ?? {};
  const counts = (status?.counts ?? {}) as Record<string, number>;
  const selected = picked ?? firstOpenRound(status);
  const spec = ROUNDS[selected - 1];
  const roundStatus = rounds[String(selected)]?.status ?? "idle";
  const gate = (liveGates[selected] ?? status?.gates?.[String(selected)]) as Gate | undefined;
  const warning = baseWarning(selected, status?.gates as Record<string, KeptGate> | undefined, rounds["1"]?.status, version?.number);
  const reading = !status || statusFor !== (version?.number ?? null);
  const set = (patch: Partial<Choices>) => setChoices((c) => ({ ...c, ...patch }));

  const blocker = version
    ? roundBlocker(selected, counts, choices.teacher === "version", (options?.teachers.length ?? 0) > 0, !!version.best_checkpoint)
    : "There is no version yet. Build one first.";

  const queueRound = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/post-training/rounds", {
          body: {
            round: selected,
            version: version?.number ?? null,
            lora: selected === 1 && !options?.lora_blocker ? (choices.lora ?? options?.lora_default ?? false) : null,
            lora_rank: choices.rank,
            teacher: choices.teacher,
            teacher_version: choices.teacher === "version" ? choices.teacherVersion : null,
            leash: choices.leash,
            method: selected === 2 ? choices.method : "practice",
            ...(selected === 2
              ? Object.fromEntries(ROUND2_SETTINGS.map((spec) => [spec.key, settingValue(choices.round2[spec.key], spec)]))
              : {}),
          },
        }),
      ),
    onSuccess: (queued) => {
      toast.success(`Added to the queue: ${queued.label}`);
      queryClient.invalidateQueries({ queryKey: ["queue"] });
      refreshSoon();
    },
    onError: (error) => toast.error(errorText(error)),
  });

  const outlook = outlookQuery.data?.outlook;
  const running = queueQuery.data?.job?.kind === "post_round" ? queueQuery.data.job : null;

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={T.title} lead={T.lead} />

      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.startsFrom}</div>
        {versionsQuery.isLoading ? (
          <Skeleton className="mt-1 h-6 w-72" />
        ) : (
          <p className="mt-1 text-base">
            {version
              ? `Version ${version.number}${version.best_checkpoint ? " · best checkpoint from Pretraining" : " · no checkpoint yet"}`
              : T.noVersion}
          </p>
        )}
        {running && <p className="mt-1 text-sm text-highlight">Running now: {running.label}</p>}
      </div>

      <div role="tablist" aria-label="Rounds" className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {ROUNDS.map((r) => {
          const s = rounds[String(r.n)]?.status ?? "idle";
          const note = rounds[String(r.n)]?.note;
          const on = r.n === selected;
          const label =
            r.n === 4 && s === "idle" ? `optional · ${count(counts.pairs)} pairs rated` : STATUS_TEXT[s] + (note ? ` · ${note}` : "");
          return (
            <button
              key={r.n}
              role="tab"
              aria-selected={on}
              onClick={() => setPicked(r.n)}
              className={optionBox(on, "flex items-center gap-3 p-3")}
            >
              <span
                className={cn(
                  "flex size-7 shrink-0 items-center justify-center rounded font-bold",
                  s === "done" ? "bg-info text-background" : s === "running" ? "bg-highlight-strong text-background" : "text-muted-foreground",
                )}
              >
                {s === "done" ? "✓" : r.n}
              </span>
              <span className="min-w-0">
                <span className="block font-semibold">{r.name}</span>
                {reading ? (
                  <Skeleton className="mt-1 h-3 w-24" />
                ) : (
                  <span className={cn("block text-xs", s === "done" ? "text-info" : s === "running" ? "text-highlight" : "text-muted-foreground")}>
                    {label}
                  </span>
                )}
              </span>
            </button>
          );
        })}
      </div>
      <p className="text-xs text-muted-foreground">{T.atThisSize} Click a round to see it.</p>

      {reading ? (
        <div className="flex flex-col gap-3" aria-label="Reading your data">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      ) : (
        <>
          {warning && <BaseWarning text={warning} />}
          <RoundCard
            spec={spec}
            props={{ counts, options, choices, set, live, gate, loading: optionsQuery.isLoading }}
            example={status?.example}
            blocker={blocker}
            outlook={outlook ? `If you add it: ${outlook[0].toLowerCase()}${outlook.slice(1)}` : undefined}
            status={roundStatus}
            queuing={queueRound.isPending}
            onQueue={() => queueRound.mutate()}
          />
          <GateCard
            round={selected}
            status={roundStatus}
            gate={gate}
            queue={queueQuery.data}
            checkpoint={version ? `version ${version.number}` : "its current checkpoint"}
            overlap={status?.overlap}
          />
          {isKept(gate as KeptGate | undefined) && <KeptCard round={selected} version={version?.number} onAdopted={refreshSoon} />}
        </>
      )}

      <Disclosure title={T.importTitle} defaultOpen={versionsQuery.isSuccess && !versions?.some((v) => v.best_checkpoint)} testId="import-disclosure">
        <ImportCard ranks={options?.lora_ranks ?? []} onImported={() => setPicked(null)} />
      </Disclosure>

      <Disclosure title={T.advancedTitle} hint={T.advancedHint} testId="advanced-disclosure">
        <LoopCard version={version?.number} hasCheckpoint={!!version?.best_checkpoint} />
        <VersionsCard versions={versions} startsFrom={version?.number} loading={versionsQuery.isLoading} />
        <RowsCard />
      </Disclosure>
    </main>
  );
}

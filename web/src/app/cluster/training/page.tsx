"use client";

import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { problem } from "@/lib/cluster";
import { TRAINING as T } from "@/lib/copy/training";
import { RunCard } from "./run-view";
import { StartForm } from "./start-form";
import { useRuns, useStopMember, useStopRun } from "./use-training";

export default function TrainingPage() {
  const runs = useRuns();
  const stopRun = useStopRun();
  const stopMember = useStopMember();
  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={T.title} lead={T.lead} />
      <StartForm />
      <section aria-labelledby="training-runs" className="flex flex-col gap-3" data-testid="runs">
        <h2 id="training-runs" className="text-lg font-semibold tracking-tight">{T.runs}</h2>
        {runs.isError ? (
          <p role="alert" className="text-sm text-destructive">{T.couldNotLoad.replace("{why}", problem(runs.error))}</p>
        ) : runs.isPending ? (
          <Skeleton className="h-32 w-full" aria-hidden="true" />
        ) : runs.data.length === 0 ? (
          <p className="text-sm text-muted-foreground">{T.noRuns}</p>
        ) : (
          runs.data.map((run) => (
            <RunCard
              key={run.id}
              run={run}
              stoppingRun={stopRun.isPending && stopRun.variables === run.id}
              stoppingNode={stopMember.isPending && stopMember.variables.id === run.id ? stopMember.variables.node : ""}
              onStopRun={() => stopRun.mutate(run.id)}
              onStopMember={(node) => stopMember.mutate({ id: run.id, node })}
            />
          ))
        )}
      </section>
    </main>
  );
}

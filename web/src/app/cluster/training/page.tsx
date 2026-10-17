"use client";

import { useState } from "react";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { problem } from "@/lib/cluster";
import { TRAINING as T } from "@/lib/copy/training";
import { LiveView } from "./live-view";
import { liveRun } from "./live";
import { RunsList } from "./runs-list";
import { StartForm } from "./start-form";
import { useRuns, useStopMember, useStopRun } from "./use-training";

const FORM = "training-start";

export default function TrainingPage() {
  const runs = useRuns();
  const stopRun = useStopRun();
  const stopMember = useStopMember();
  const [chosen, setChosen] = useState<string | null>(null);
  const [formOpen, setFormOpen] = useState<boolean | null>(null);
  const list = runs.data ?? [];
  const live = liveRun(list, chosen);
  // The form is open on its own only while there is no run to watch; once the person opens or closes it, it stays as they left it.
  const open = formOpen ?? (runs.isSuccess && list.length === 0);
  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={T.title} lead={T.lead} />
      <section aria-label={T.runs} className="flex flex-col gap-3" data-testid="runs">
        {runs.isError ? (
          <p role="alert" className="text-sm text-destructive">{T.couldNotLoad.replace("{why}", problem(runs.error))}</p>
        ) : runs.isPending ? (
          <Skeleton className="h-32 w-full" aria-hidden="true" />
        ) : live ? (
          <LiveView
            run={live}
            stoppingRun={stopRun.isPending && stopRun.variables === live.id}
            stoppingNode={stopMember.isPending && stopMember.variables.id === live.id ? stopMember.variables.node : ""}
            onStopRun={() => stopRun.mutate(live.id)}
            onStopMember={(node) => stopMember.mutate({ id: live.id, node })}
          />
        ) : (
          <p className="text-sm text-muted-foreground">{T.noRuns}</p>
        )}
      </section>
      <RunsList runs={list} live={live} onSelect={setChosen} />
      <div className="flex flex-col gap-3">
        <div>
          <Button variant="outline" aria-expanded={open} aria-controls={FORM} onClick={() => setFormOpen(!open)}>{T.startToggle}</Button>
        </div>
        {open && (
          <div id={FORM}>
            <StartForm
              onStarted={(run) => {
                setChosen(run.id);
                setFormOpen(false);
              }}
            />
          </div>
        )}
      </div>
    </main>
  );
}

"use client";

import Link from "next/link";
import { useState } from "react";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { runViewHref } from "@/app/runs/view/logic";
import { problem } from "@/lib/cluster";
import { TRAINING as T } from "@/lib/copy/training";
import { liveRun } from "@/app/runs/view/live";
import { RunsList } from "./runs-list";
import { StartForm } from "./start-form";
import { useRuns } from "./use-training";

const FORM = "training-start";

export default function TrainingPage() {
  const runs = useRuns();
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
          <Card data-testid={`run-${live.id}`} aria-label={T.liveRunLabel.replace("{label}", live.label)}>
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center gap-2">
                {live.label}
                <Badge variant={live.state === "failed" ? "destructive" : "secondary"} data-testid={`run-state-${live.id}`}>{T.states[live.state]}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 text-sm">
              <p className="text-muted-foreground">{T.roundOf.replace("{round}", String(live.round)).replace("{rounds}", String(live.rounds))} · {T.phases[live.phase]}</p>
              <div className="flex flex-wrap gap-x-6 gap-y-1 tabular-nums">
                <span>{T.validationLoss}: {live.history.at(-1)?.val_loss?.toFixed(3) ?? "—"}</span>
                <span>{T.contributedCol}: {live.contributed} of {live.expected} this round</span>
                <span>{live.members.length} {live.members.length === 1 ? "member" : "members"}</span>
              </div>
              <Button asChild className="w-fit"><Link href={runViewHref(live.id)} data-testid={`open-run-view-${live.id}`}>{T.openRunView}</Link></Button>
            </CardContent>
          </Card>
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

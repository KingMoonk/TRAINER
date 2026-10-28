"use client";

import Link from "next/link";
import { useEffect } from "react";
import { toast } from "sonner";
import { useQueueActions } from "@/app/queue/use-queue";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ActiveCard } from "./active-card";
import { AllowedHours } from "./allowed-hours";
import { CompletedList } from "./completed-list";
import { T } from "./copy";
import { Hint } from "./hint";
import { inSection } from "./logic";
import { NodeChips } from "./node-chips";
import { ScheduledTable } from "./scheduled-table";
import { useRunRows } from "./use-runs";

function Empty({ children }: { children: string }) {
  return <p className="text-base text-muted-foreground">{children}</p>;
}

export default function RunsPage() {
  const { rows, nodes, loading, failed } = useRunRows();
  const actions = useQueueActions();
  useEffect(() => {
    if (failed) toast.error(T.couldNotLoad, { id: "runs-load" });
  }, [failed]);
  const active = inSection(rows, "active");
  const scheduled = inSection(rows, "scheduled");
  const completed = inSection(rows, "completed");

  return (
    <main className={PAGE_MAIN}>
      <PageHeader
        title={T.title}
        lead={T.lead}
        actions={
          <Button asChild>
            <Link href="/runs/new/">{T.newRun}</Link>
          </Button>
        }
      />

      {nodes && nodes.length > 0 && <NodeChips nodes={nodes} active={active} />}

      <section aria-labelledby="runs-active" className="flex flex-col gap-3">
        <h2 id="runs-active" className="text-lg font-semibold">{T.active}</h2>
        {loading ? <Skeleton className="h-28 w-full" /> : active.length === 0 ? <Empty>{T.nothingActive}</Empty> : active.map((row) => <ActiveCard key={row.id} row={row} />)}
      </section>

      <section aria-labelledby="runs-scheduled" className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="runs-scheduled" className="text-lg font-semibold">{T.scheduled}</h2>
          <div className="flex flex-wrap items-center gap-3">
            <Link href="/queue/" className="text-sm text-highlight hover:underline">{T.queueLink}</Link>
            <AllowedHours>
              <Button variant="outline" size="sm">{T.allowedHours}</Button>
            </AllowedHours>
          </div>
        </div>
        {loading ? <Skeleton className="h-20 w-full" /> : scheduled.length === 0 ? <Empty>{T.nothingScheduled}</Empty> : <ScheduledTable rows={scheduled} actions={actions} />}
      </section>

      <section aria-labelledby="runs-completed" className="flex flex-col gap-3">
        <h2 id="runs-completed" className="text-lg font-semibold">
          <Hint tip={T.completedTip} plain>{T.completed}</Hint>
        </h2>
        {loading ? <Skeleton className="h-14 w-full" /> : completed.length === 0 ? <Empty>{T.nothingCompleted}</Empty> : <CompletedList rows={completed} onRestart={(id) => actions.restart.mutate(id)} restarting={actions.restart.isPending} />}
      </section>
    </main>
  );
}

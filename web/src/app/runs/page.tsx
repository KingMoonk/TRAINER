"use client";

import Link from "next/link";
import { useEffect } from "react";
import { toast } from "sonner";
import { PlanCard } from "@/app/queue/plan-card";
import { useQueueActions } from "@/app/queue/use-queue";
import { PAGE_MAIN } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { ActiveCard } from "./active-card";
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
  const { rows, queue, nodes, loading, failed } = useRunRows();
  const actions = useQueueActions();
  useEffect(() => {
    if (failed) toast.error(T.couldNotLoad, { id: "runs-load" });
  }, [failed]);
  const active = inSection(rows, "active");
  const scheduled = inSection(rows, "scheduled");
  const completed = inSection(rows, "completed");

  return (
    <main className={PAGE_MAIN}>
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex flex-col gap-1.5">
          <h1 className="text-2xl font-semibold tracking-tight">{T.title}</h1>
          <p className="max-w-prose text-muted-foreground">{T.lead}</p>
        </div>
        <Button asChild>
          <Link href="/runs/new/">{T.newRun}</Link>
        </Button>
      </header>

      {nodes && nodes.length > 0 && <NodeChips nodes={nodes} active={active} />}

      <section aria-labelledby="runs-active" className="flex flex-col gap-3">
        <h2 id="runs-active" className="text-lg font-semibold">{T.active}</h2>
        {loading ? <Skeleton className="h-28 w-full" /> : active.length === 0 ? <Empty>{T.nothingActive}</Empty> : active.map((row) => <ActiveCard key={row.id} row={row} />)}
      </section>

      <section aria-labelledby="runs-scheduled" className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="runs-scheduled" className="text-lg font-semibold">{T.scheduled}</h2>
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="outline" size="sm">{T.allowedHours}</Button>
            </SheetTrigger>
            <SheetContent>
              <SheetHeader>
                <SheetTitle>{T.allowedHoursTitle}</SheetTitle>
                <SheetDescription>{T.allowedHoursLead}</SheetDescription>
              </SheetHeader>
              <PlanCard queue={queue.data} actions={actions} />
            </SheetContent>
          </Sheet>
        </div>
        {loading ? <Skeleton className="h-20 w-full" /> : scheduled.length === 0 ? <Empty>{T.nothingScheduled}</Empty> : <ScheduledTable rows={scheduled} actions={actions} />}
      </section>

      <section aria-labelledby="runs-completed" className="flex flex-col gap-3">
        <h2 id="runs-completed" className="text-lg font-semibold">
          <Hint tip={T.completedTip} plain>{T.completed}</Hint>
        </h2>
        {loading ? <Skeleton className="h-14 w-full" /> : completed.length === 0 ? <Empty>{T.nothingCompleted}</Empty> : <CompletedList rows={completed} />}
      </section>
    </main>
  );
}

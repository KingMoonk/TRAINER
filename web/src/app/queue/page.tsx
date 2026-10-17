"use client";

import { useEffect } from "react";
import { toast } from "sonner";
import { isCluster, useCluster } from "@/lib/cluster";
import { T } from "./copy";
import { OtherMachinesCard, isElsewhere } from "./nodes";
import { NowCard } from "./now-card";
import { PlanCard } from "./plan-card";
import { UpNextCard } from "./up-next-card";
import { useQueue, useQueueActions } from "./use-queue";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

export default function QueuePage() {
  const queue = useQueue();
  const actions = useQueueActions();
  const merged = useCluster().data;
  const cluster = isCluster(merged) ? merged : undefined; // a cluster of one shows nothing about machines
  // A job still running on a machine that was removed from the list keeps its row (and its note), so the card stays for it.
  const others = cluster ?? (merged && merged.jobs.some(isElsewhere) ? merged : undefined);
  useEffect(() => {
    if (queue.isError) toast.error(T.couldNotLoad, { id: "queue-load" });
  }, [queue.isError]);

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={T.title} lead={T.lead} />
      <NowCard queue={queue.data} actions={actions} cluster={cluster} />
      <UpNextCard queue={queue.data} actions={actions} cluster={cluster} />
      {others && <OtherMachinesCard cluster={others} />}
      <PlanCard queue={queue.data} actions={actions} />
    </main>
  );
}

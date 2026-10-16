"use client";

import { useEffect } from "react";
import { toast } from "sonner";
import { isCluster, useCluster } from "@/lib/cluster";
import { T } from "./copy";
import { OtherMachinesCard } from "./nodes";
import { NowCard } from "./now-card";
import { PlanCard } from "./plan-card";
import { UpNextCard } from "./up-next-card";
import { useQueue, useQueueActions } from "./use-queue";

export default function QueuePage() {
  const queue = useQueue();
  const actions = useQueueActions();
  const merged = useCluster().data;
  const cluster = isCluster(merged) ? merged : undefined; // a cluster of one shows nothing about machines
  useEffect(() => {
    if (queue.isError) toast.error(T.couldNotLoad, { id: "queue-load" });
  }, [queue.isError]);

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-5 p-8">
      <div className="flex flex-col gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">{T.title}</h1>
        <p className="max-w-prose text-muted-foreground">{T.lead}</p>
      </div>
      <NowCard queue={queue.data} actions={actions} cluster={cluster} />
      <UpNextCard queue={queue.data} actions={actions} cluster={cluster} />
      {cluster && <OtherMachinesCard cluster={cluster} />}
      <PlanCard queue={queue.data} actions={actions} />
    </main>
  );
}

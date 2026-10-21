"use client";

import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { useMachines, useRuns } from "@/app/cluster/training/use-training";
import { useQueue } from "@/app/queue/use-queue";
import { api, unwrap } from "@/lib/api";
import { useLive } from "@/lib/live";
import { buildRows } from "./logic";

const CLUSTER_JOBS = ["cluster", "jobs"] as const;

/** Jobs queued here that run on a peer, and jobs this computer runs for another (GET /cluster/jobs). Re-read on every queue, cluster or facts event. */
export function useClusterJobs() {
  useLive(CLUSTER_JOBS, ["queue", "cluster", "facts"]);
  return useQuery({
    queryKey: CLUSTER_JOBS,
    queryFn: ({ signal }) => unwrap(api.GET("/cluster/jobs", { signal })),
    retry: false,
  });
}

/** The page's data: the existing queue, cluster and training reads, merged into one list of runs by id. */
export function useRunRows() {
  const queue = useQueue();
  const runs = useRuns();
  const clusterJobs = useClusterJobs();
  const machines = useMachines();
  const rows = useMemo(
    () => buildRows({ queue: queue.data, runs: runs.data, clusterJobs: clusterJobs.data, nodes: machines.data }),
    [queue.data, runs.data, clusterJobs.data, machines.data],
  );
  return { rows, queue, nodes: machines.data, loading: queue.isPending, failed: queue.isError };
}

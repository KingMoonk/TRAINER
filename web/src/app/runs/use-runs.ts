"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { useMachines, useRuns } from "@/app/cluster/training/use-training";
import { useQueue } from "@/app/queue/use-queue";
import { api, unwrap } from "@/lib/api";
import { useEvent } from "@/lib/events";
import { buildRows } from "./logic";

const CLUSTER_JOBS = ["cluster", "jobs"] as const;
const REFRESH_MS = 5_000;

/** Jobs queued here that run on a peer, and jobs this computer runs for another (GET /cluster/jobs). Re-read on every queue or cluster event. */
export function useClusterJobs() {
  const client = useQueryClient();
  const refresh = () => void client.invalidateQueries({ queryKey: CLUSTER_JOBS });
  useEvent("queue", refresh);
  useEvent("cluster", refresh);
  return useQuery({
    queryKey: CLUSTER_JOBS,
    queryFn: ({ signal }) => unwrap(api.GET("/cluster/jobs", { signal })),
    refetchInterval: REFRESH_MS,
    retry: false,
  });
}

/** The page's data: the existing queue, cluster and training reads, merged into one list of runs by id. No new polling of its own beyond /cluster/jobs. */
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

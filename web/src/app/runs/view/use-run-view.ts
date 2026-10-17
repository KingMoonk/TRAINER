"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { isActive, type Run } from "@/app/cluster/training/logic";
import { useMachines } from "@/app/cluster/training/use-training";
import { thisComputer } from "@/app/runs/logic";
import type { Job, Queue } from "@/app/queue/logic";
import { useQueue } from "@/app/queue/use-queue";
import { api, ApiError, unwrap } from "@/lib/api";
import { useEvent } from "@/lib/events";
import type { Curve } from "./logic";

const ACTIVE_MS = 1_000;
const CURVE_MS = 3_000;
const DISTRIBUTED = "pretrain_distributed";

export type RunView =
  | { kind: "loading" }
  | { kind: "error"; error: unknown }
  | { kind: "missing" }
  | { kind: "diloco"; run: Run; machine: string }
  | { kind: "solo"; job: Job; queue: Queue; running: boolean; machine: string; curve: Curve | undefined };

/** A run this computer coordinates (GET /distributed/runs/{id}), or null when there is none by that id. */
function useDilocoRun(id: string) {
  const client = useQueryClient();
  const refresh = () => void client.invalidateQueries({ queryKey: ["distributed", "run", id] });
  useEvent("queue", refresh);
  useEvent("cluster", refresh);
  return useQuery({
    queryKey: ["distributed", "run", id],
    enabled: id !== "",
    queryFn: async ({ signal }): Promise<Run | null> => {
      try {
        return await unwrap(api.GET("/distributed/runs/{run_id}", { params: { path: { run_id: id } }, signal }));
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    // The members report to the coordinator without telling this page, so it reads the run every second while the run is going.
    refetchInterval: (query) => (query.state.data && isActive(query.state.data) ? ACTIVE_MS : false),
    retry: false,
  });
}

/** What the page shows for `?id=`: a DiLoCo run (resolved first) or a job in this computer's queue. */
export function useRunView(id: string): RunView {
  const run = useDilocoRun(id);
  const queue = useQueue();
  const machines = useMachines();
  const here = thisComputer(machines.data);
  const job = queue.data ? [...(queue.data.job ? [queue.data.job] : []), ...queue.data.up_next].find((j) => j.id === id) : undefined;
  const solo = run.isSuccess && run.data === null && job !== undefined && job.kind !== DISTRIBUTED;
  const running = queue.data?.job?.id === id;
  const curve = useQuery({
    queryKey: ["pretraining", "curve", id],
    enabled: solo && job !== undefined && job.status !== "waiting" && job.status !== "held",
    queryFn: ({ signal }) => unwrap(api.GET("/pretraining/jobs/{job_id}/curve", { params: { path: { job_id: id } }, signal })),
    refetchInterval: running ? CURVE_MS : false,
    retry: false,
  });

  if (id === "") return { kind: "missing" };
  if (run.isError) return { kind: "error", error: run.error };
  if (run.isPending || queue.isPending) return queue.isError ? { kind: "error", error: queue.error } : { kind: "loading" };
  if (run.data) return { kind: "diloco", run: run.data, machine: here };
  if (solo && queue.data && job) return { kind: "solo", job, queue: queue.data, running, machine: job.node || here, curve: curve.data };
  return { kind: "missing" };
}

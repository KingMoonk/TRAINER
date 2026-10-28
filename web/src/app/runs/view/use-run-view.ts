"use client";

import { useQuery } from "@tanstack/react-query";
import type { Run } from "@/app/cluster/training/logic";
import { useMachines } from "@/app/cluster/training/use-training";
import { thisComputer } from "@/app/runs/logic";
import type { Job, Queue } from "@/app/queue/logic";
import { useQueue } from "@/app/queue/use-queue";
import { api, ApiError, unwrap } from "@/lib/api";
import { payloadIs, useLive } from "@/lib/live";
import { inviteFor, useInvites, type Invite } from "@/lib/join";
import type { Curve } from "./logic";

const DISTRIBUTED = "pretrain_distributed";
const MEMBER = "diloco_member";

export type RunView =
  | { kind: "loading" }
  | { kind: "error"; error: unknown }
  | { kind: "missing" }
  | { kind: "diloco"; run: Run; machine: string }
  | { kind: "member"; job: Job; invite: Invite | undefined; run: Run | null; machine: string }
  | { kind: "solo"; job: Job; queue: Queue; running: boolean; machine: string; curve: Curve | undefined };

/** A run this computer coordinates (GET /distributed/runs/{id}), or null when there is none by that id. */
function useDilocoRun(id: string) {
  // `run`: the coordinator here changed; `facts`: the cluster's account of a run another computer coordinates changed
  useLive(["distributed", "run", id], ["queue", "cluster", "facts", "run"], { enabled: id !== "", match: payloadIs("run", id, ["run"]) });
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
    retry: false,
  });
}

/** What the page shows for `?id=`: a DiLoCo run (resolved first) or a job in this computer's queue or failed list. */
export function useRunView(id: string): RunView {
  const run = useDilocoRun(id);
  const queue = useQueue();
  const machines = useMachines();
  const here = thisComputer(machines.data);
  // a job that ended failed stays in `failed` (the same list the Runs page shows under Completed), so its page opens read-only
  const job = queue.data ? [...(queue.data.job ? [queue.data.job] : []), ...queue.data.up_next, ...(queue.data.failed ?? [])].find((j) => j.id === id) : undefined;
  const invites = useInvites();
  const invite = job?.kind === MEMBER ? inviteFor(invites.data, job.id) : undefined;
  const coordinated = useDilocoRun(invite?.run_id ?? ""); // the owner's account of the run this job is a member of, from the replica
  const solo = run.isSuccess && run.data === null && job !== undefined && job.kind !== DISTRIBUTED && job.kind !== MEMBER;
  const running = queue.data?.job?.id === id;
  // a `train` event is sent at every step, so the curve is re-read at most once a second
  useLive(["pretraining", "curve", id], ["train"], { enabled: running, match: payloadIs("job", id, ["train"]), throttleMs: 1_000 });
  const curve = useQuery({
    queryKey: ["pretraining", "curve", id],
    enabled: solo && job !== undefined && job.status !== "waiting" && job.status !== "held",
    queryFn: ({ signal }) => unwrap(api.GET("/pretraining/jobs/{job_id}/curve", { params: { path: { job_id: id } }, signal })),
    retry: false,
  });

  if (id === "") return { kind: "missing" };
  if (run.isError) return { kind: "error", error: run.error };
  if (run.isPending || queue.isPending) return queue.isError ? { kind: "error", error: queue.error } : { kind: "loading" };
  if (run.data) return { kind: "diloco", run: run.data, machine: here };
  if (run.data === null && job?.kind === MEMBER) return { kind: "member", job, invite, run: coordinated.data ?? null, machine: job.node || here };
  if (solo && queue.data && job) return { kind: "solo", job, queue: queue.data, running, machine: job.node || here, curve: curve.data };
  return { kind: "missing" };
}

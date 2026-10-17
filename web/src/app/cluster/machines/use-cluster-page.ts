"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, unwrap } from "@/lib/api";
import { problem } from "@/lib/cluster";
import { CLUSTER as T } from "@/lib/copy/cluster";
import type { QueueRow } from "@/lib/cluster";

export const NODES = ["cluster", "nodes"] as const;
const ACCESS = ["cluster", "access"] as const;

/** The machines (GET /cluster/nodes, re-read every few seconds while others are listed, since a peer changes without telling this one). */
export function useNodes() {
  return useQuery({
    queryKey: NODES,
    queryFn: ({ signal }) => unwrap(api.GET("/cluster/nodes", { signal })),
    refetchInterval: (query) => ((query.state.data?.length ?? 0) > 1 ? 5_000 : false),
    retry: false,
  });
}

/** The allow lists and this computer's Tailscale login (GET /cluster/access). */
export function useAccess() {
  return useQuery({ queryKey: ACCESS, queryFn: ({ signal }) => unwrap(api.GET("/cluster/access", { signal })), retry: false });
}

export function useSaveNodes(onSaved: () => void, onProblem: (why: string) => void) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (addresses: string[]) => unwrap(api.PUT("/cluster/nodes", { body: { addresses } })),
    onSuccess: async () => {
      toast.success(T.saved);
      await client.invalidateQueries({ queryKey: NODES });
      void client.invalidateQueries({ queryKey: ["cluster", "queue"] });
      onSaved();
    },
    onError: (error) => onProblem(problem(error)),
  });
}

export function useSaveAccess(onProblem: (why: string) => void) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: { allow_users: string[]; allow_tags: string[] }) => unwrap(api.PUT("/cluster/access", { body })),
    onSuccess: async () => {
      toast.success(T.accessSaved);
      await client.invalidateQueries({ queryKey: ACCESS });
    },
    onError: (error) => onProblem(problem(error)),
  });
}

/**
 * Stop a running job, or drop a waiting one, on whatever machine queued it: this computer's own queue for a job queued here,
 * and for a job a peer queued, that peer through this computer (POST /cluster/remote/{id}/stop).
 */
export function useStopJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (row: QueueRow) => {
      if (row.queued_here) await unwrap(api.DELETE("/queue/jobs/{job_id}", { params: { path: { job_id: row.id } } }));
      else await unwrap(api.POST("/cluster/remote/{job_id}/stop", { params: { path: { job_id: row.id } }, body: { address: row.queued_address } }));
    },
    onSuccess: (_data, row) => {
      toast.success(T.stopped.replace("{node}", row.hostname || row.queued_on || "the machine").replace("{job}", row.label));
      void client.invalidateQueries({ queryKey: ["cluster", "queue"] });
      void client.invalidateQueries({ queryKey: ["queue"] });
    },
    onError: (error, row) => toast.error(T.stopFailed.replace("{job}", row.label).replace("{why}", problem(error))),
  });
}

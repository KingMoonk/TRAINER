"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, ApiError, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { awaitJob } from "@/lib/job-result";
import { useLive } from "@/lib/live";

export { nodeName } from "@/lib/node-name";

export type ClusterQueue = components["schemas"]["ClusterQueue"];
export type QueueNode = components["schemas"]["QueueNode"];
export type QueueLimits = components["schemas"]["QueueLimits"];
export type QueueRow = components["schemas"]["QueueRow"];
export type PublishedVersion = components["schemas"]["PublishedVersion"];

const CLUSTER = ["cluster", "queue"] as const;

/** More than one machine is listed. A cluster of one shows nothing about machines anywhere. */
export function isCluster(cluster: ClusterQueue | undefined): boolean {
  return (cluster?.nodes.length ?? 0) > 1;
}

/**
 * The jobs of every machine and the machines themselves (GET /cluster/merged). Empty for a cluster of one, so a page asks
 * `isCluster(data)` before it shows anything about machines. Re-read on every `queue` or `cluster` event, and on `facts`: a
 * peer's own queue reaches this one through the shared observations.
 */
export function useCluster() {
  useLive(CLUSTER, ["queue", "cluster", "facts"]);
  return useQuery({
    queryKey: CLUSTER,
    queryFn: ({ signal }) => unwrap(api.GET("/cluster/merged", { signal })),
    retry: false,
  });
}

/** The last time a machine answered, as a clock time, or "" when it never did. */
export function seenText(seconds: number): string {
  return seconds > 0 ? new Date(seconds * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "";
}

/** A node with no version store answers 409 to GET /versions/published: a setup state, not a failure. */
export function isStoreUnset(error: unknown): boolean {
  return error instanceof ApiError && error.status === 409;
}

/** Look at the store once a minute (another machine publishes into it without telling this one), but not while it answers 409:
 * nothing changes until a store is set up. A `versions` event re-reads it at once. */
export function refetchPublished(query: { state: { error: unknown } }): number | false {
  return isStoreUnset(query.state.error) ? false : 60_000;
}

/**
 * The version store's versions (GET /versions/published), only while other machines are listed. A node with no store
 * answers 409, and that is the same as nothing published: the list is empty and says nothing about machines.
 */
export function usePublished(cluster: ClusterQueue | undefined) {
  useLive(["versions", "published"], ["versions"], { enabled: isCluster(cluster), backstopMs: false });
  return useQuery({
    queryKey: ["versions", "published"],
    enabled: isCluster(cluster),
    queryFn: ({ signal }) => unwrap(api.GET("/versions/published", { signal })),
    refetchInterval: refetchPublished,
    retry: false,
  });
}

/** Versions in the store that this workspace does not have yet: available, not local. */
export function availableElsewhere(published: PublishedVersion[] | undefined): PublishedVersion[] {
  return (published ?? []).filter((entry) => !entry.local);
}

/** Where a version here came from: the machine that published it, or "" when this machine did (or it was never published). */
export function cameFrom(published: PublishedVersion[] | undefined, number: number): string {
  const entry = (published ?? []).find((item) => item.local_number === number);
  return entry && !entry.mine ? entry.origin : "";
}

/** The sentence an API error carries (its `detail`), or the status when it has none. */
export function problem(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const body = JSON.parse(error.body) as { detail?: unknown };
      if (typeof body.detail === "string") return body.detail;
    } catch {
      // not JSON: the status line says enough
    }
    return `The server answered ${error.status}.`;
  }
  return error instanceof Error ? error.message : String(error);
}

/** The Pull action: download a published version and add it here (POST /versions/{number}/pull), following its job. */
export function usePull() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (entry: PublishedVersion) => {
      const started = await unwrap(
        api.POST("/versions/{number}/pull", { params: { path: { number: entry.number }, query: { origin: entry.origin } } }),
      );
      await awaitJob<unknown>(started, new AbortController().signal);
      return entry;
    },
    onSuccess: (entry) => {
      toast.success(`Pulled version ${entry.number} from ${entry.origin}.`);
      void client.invalidateQueries({ queryKey: ["versions"] });
    },
    onError: (error, entry) => toast.error(`Could not pull version ${entry.number} from ${entry.origin}: ${problem(error)}`),
  });
}

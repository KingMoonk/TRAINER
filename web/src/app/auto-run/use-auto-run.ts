"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, unwrap } from "@/lib/api";
import { useLive, payloadIs } from "@/lib/live";
import { T } from "./copy";
import { errorText, type Plan, type RunView } from "./logic";

const LIST = ["auto-run", "list"] as const;
const runKey = (id: string) => ["auto-run", "run", id] as const;

/** Every Auto Run, newest first, kept current by the `auto_run` events. */
export function useRuns() {
  const query = useQuery({ queryKey: LIST, queryFn: () => unwrap(api.GET("/auto-runs")) });
  useLive(LIST, ["auto_run"]);
  return query;
}

/** The plan a new run starts from for a version. */
export function useDefaults(version: number | null) {
  return useQuery({
    queryKey: ["auto-run", "defaults", version],
    queryFn: () => unwrap(api.GET("/auto-runs/defaults", { params: { query: { version: version! } } })),
    enabled: version !== null,
  });
}

/** One run. While it is going, its job's step numbers (`train`) and the queue move it too, at most once a second. */
export function useRun(id: string, going: boolean) {
  const query = useQuery({ queryKey: runKey(id), queryFn: () => unwrap(api.GET("/auto-runs/{run_id}", { params: { path: { run_id: id } } })) });
  useLive(runKey(id), ["train", "queue"], { throttleMs: 1000, enabled: going });
  useLive(runKey(id), ["auto_run"], { match: payloadIs("id", id, ["auto_run"]) });
  return query;
}

export type Control = "start" | "pause" | "resume" | "stop";

/** Every control on a run. A reply carries the new run, which replaces the old one at once; a refusal is a toast with its sentence. */
export function useRunActions(id: string) {
  const client = useQueryClient();
  const take = (view: RunView) => {
    client.setQueryData(runKey(id), view);
    client.invalidateQueries({ queryKey: LIST });
  };
  const onError = (e: unknown) => toast.error(errorText(e));
  const path = { params: { path: { run_id: id } } };
  return {
    control: useMutation({
      mutationFn: (what: Control) => {
        const call = { start: "/auto-runs/{run_id}/start", pause: "/auto-runs/{run_id}/pause", resume: "/auto-runs/{run_id}/resume", stop: "/auto-runs/{run_id}/stop" } as const;
        return unwrap(api.POST(call[what], path));
      },
      onSuccess: take,
      onError,
    }),
    choose: useMutation({
      mutationFn: (choice: "next" | "retry" | "stop" | "adopt") => unwrap(api.POST("/auto-runs/{run_id}/continue", { ...path, body: { choice } })),
      onSuccess: (view, choice) => {
        take(view);
        if (choice === "adopt") client.invalidateQueries({ queryKey: ["versions"] });
      },
      onError,
    }),
    savePlan: useMutation({
      mutationFn: (plan: Plan) => unwrap(api.PUT("/auto-runs/{run_id}/plan", { ...path, body: { plan } })),
      onSuccess: (view) => {
        take(view);
        toast.success(T.saved);
      },
      onError,
    }),
  };
}

/** Make a draft from a plan, and optionally start it at once. A start that is refused leaves the draft, and says why. */
export function useCreateRun(onMade: (id: string) => void) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async ({ plan, start }: { plan: Plan; start: boolean }) => {
      const draft = await unwrap(api.POST("/auto-runs", { body: { plan } }));
      client.setQueryData(runKey(draft.id), draft);
      onMade(draft.id);
      if (!start) return draft;
      try {
        return await unwrap(api.POST("/auto-runs/{run_id}/start", { params: { path: { run_id: draft.id } } }));
      } catch (e) {
        toast.error(`${T.startFailed} ${errorText(e)}`);
        return draft;
      }
    },
    onSuccess: (view) => {
      client.setQueryData(runKey(view.id), view);
      client.invalidateQueries({ queryKey: LIST });
    },
    onError: (e) => toast.error(errorText(e)),
  });
}

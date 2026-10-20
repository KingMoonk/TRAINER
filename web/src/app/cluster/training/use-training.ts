"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, unwrap } from "@/lib/api";
import { problem } from "@/lib/cluster";
import { TRAINING as T } from "@/lib/copy/training";
import { useEvent } from "@/lib/events";
import { isActive, type Run, type StartRun } from "./logic";

const RUNS = ["distributed", "runs"] as const;
const ACTIVE_MS = 1_000;
export const READ_TIMEOUT_MS = 20_000;

/**
 * Run a read that gives up with a plain sentence when the server has not answered in READ_TIMEOUT_MS, so a page showing it
 * can say so instead of waiting for ever. An abort from the query itself (it is no longer wanted) passes through unchanged.
 */
export async function withTimeout<T>(signal: AbortSignal, call: (signal: AbortSignal) => Promise<T>, ms = READ_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const stop = () => controller.abort();
  let late = false;
  signal.addEventListener("abort", stop, { once: true });
  const timer = setTimeout(() => {
    late = true;
    controller.abort();
  }, ms);
  try {
    return await call(controller.signal);
  } catch (error) {
    if (late) throw new Error(`The server did not answer within ${Math.round(ms / 1000)} seconds.`);
    throw error;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", stop);
  }
}

/**
 * The runs this computer coordinates (GET /distributed/runs), newest first. Re-read on every `queue` event and every
 * second while one is going (nothing polls otherwise), since the members report to the coordinator without telling this page.
 */
export function useRuns() {
  const client = useQueryClient();
  const refresh = () => void client.invalidateQueries({ queryKey: RUNS });
  useEvent("queue", refresh);
  useEvent("cluster", refresh);
  return useQuery({
    queryKey: RUNS,
    queryFn: ({ signal }) => unwrap(api.GET("/distributed/runs", { signal })),
    refetchInterval: (query) => ((query.state.data ?? []).some(isActive) ? ACTIVE_MS : false),
    retry: false,
  });
}

/** The DiLoCo settings the form starts from (GET /distributed/defaults). */
export function useDefaults() {
  return useQuery({ queryKey: ["distributed", "defaults"], queryFn: ({ signal }) => withTimeout(signal, (s) => unwrap(api.GET("/distributed/defaults", { signal: s }))), retry: false });
}

/** The saved designs, for the form's design choice. */
export function useVersions() {
  return useQuery({ queryKey: ["versions"], queryFn: ({ signal }) => withTimeout(signal, (s) => unwrap(api.GET("/versions", { signal: s }))), retry: false });
}

/** The text sets, for the form's data choice. */
export function useTextSets() {
  return useQuery({ queryKey: ["pretraining", "text_sets"], queryFn: ({ signal }) => withTimeout(signal, (s) => unwrap(api.GET("/pretraining/text_sets", { signal: s }))), retry: false });
}

/** The sets that hold conversations (GET /distributed/chat_sets): where a teacher may take its prompts from, read once the teacher section is opened. */
export function useChatSets(enabled: boolean) {
  return useQuery({ enabled, queryKey: ["distributed", "chat_sets"], queryFn: ({ signal }) => withTimeout(signal, (s) => unwrap(api.GET("/distributed/chat_sets", { signal: s }))), retry: false });
}

/** The Ollama models that can teach (GET /post-training/judges): the form's teacher choice. */
export function useTeachers(enabled: boolean) {
  return useQuery({ enabled, queryKey: ["post-training", "judges"], queryFn: ({ signal }) => withTimeout(signal, (s) => unwrap(api.GET("/post-training/judges", { signal: s }))), retry: false });
}

/** The machines on the Machines tab (GET /cluster/nodes): the ones a run can ask to join. */
export function useMachines() {
  return useQuery({ queryKey: ["cluster", "nodes"], queryFn: ({ signal }) => withTimeout(signal, (s) => unwrap(api.GET("/cluster/nodes", { signal: s }))), retry: false });
}

/** Put a run in the list at once, from the answer to the request that changed it. */
function useRunWrite() {
  const client = useQueryClient();
  return (run: Run) => {
    client.setQueryData<Run[]>(RUNS, (runs) => [run, ...(runs ?? []).filter((r) => r.id !== run.id)].sort((a, b) => b.started_at - a.started_at));
    client.setQueryData<Run | null>(["distributed", "run", run.id], run); // the Run view's own read of this run
    void client.invalidateQueries({ queryKey: RUNS });
    void client.invalidateQueries({ queryKey: ["queue"] });
  };
}

export function useStart(onStarted: (run: Run) => void, onProblem: (why: string) => void) {
  const write = useRunWrite();
  return useMutation({
    mutationFn: (body: StartRun) => unwrap(api.POST("/distributed/runs", { body })),
    onSuccess: (run) => {
      write(run);
      toast.success(T.started);
      onStarted(run);
    },
    onError: (error) => onProblem(T.couldNotStart.replace("{why}", problem(error))),
  });
}

export function useStopRun() {
  const write = useRunWrite();
  return useMutation({
    mutationFn: (id: string) => unwrap(api.POST("/distributed/runs/{run_id}/stop", { params: { path: { run_id: id } } })),
    onSuccess: write,
    onError: (error) => toast.error(T.couldNotStop.replace("{why}", problem(error))),
  });
}

/** Pause or resume a run; the answer is the run, so its buttons and phase change at once. */
export function useHoldRun(verb: "pause" | "resume") {
  const write = useRunWrite();
  return useMutation({
    mutationFn: (id: string) =>
      verb === "pause"
        ? unwrap(api.POST("/distributed/runs/{run_id}/pause", { params: { path: { run_id: id } } }))
        : unwrap(api.POST("/distributed/runs/{run_id}/resume", { params: { path: { run_id: id } } })),
    onSuccess: write,
    onError: (error) => toast.error((verb === "pause" ? T.couldNotPause : T.couldNotResume).replace("{why}", problem(error))),
  });
}

/** Start a new run from an ended one (a new id); `onRestarted` gets it so the page can show it. */
export function useRestartRun(onRestarted: (run: Run) => void) {
  const write = useRunWrite();
  return useMutation({
    mutationFn: (id: string) => unwrap(api.POST("/distributed/runs/{run_id}/restart", { params: { path: { run_id: id } } })),
    onSuccess: (run) => {
      write(run);
      toast.success(T.restarted);
      onRestarted(run);
    },
    onError: (error) => toast.error(T.couldNotRestart.replace("{why}", problem(error))),
  });
}

export function useStopMember() {
  const write = useRunWrite();
  return useMutation({
    mutationFn: ({ id, node }: { id: string; node: string }) =>
      unwrap(api.POST("/distributed/runs/{run_id}/members/{node}/stop", { params: { path: { run_id: id, node } } })),
    onSuccess: write,
    onError: (error) => toast.error(T.couldNotStop.replace("{why}", problem(error))),
  });
}

/** What the teacher gave a run (GET /distributed/runs/{id}/lessons), re-read every few seconds while `enabled`. */
export function useLessons(id: string, enabled: boolean) {
  return useQuery({
    queryKey: [...RUNS, id, "lessons"],
    queryFn: ({ signal }) => unwrap(api.GET("/distributed/runs/{run_id}/lessons", { params: { path: { run_id: id } }, signal })),
    enabled,
    refetchInterval: enabled ? 5_000 : false,
    retry: false,
  });
}

/** Drop a lesson or keep it again; the answer is the lessons as they stand, and the run's counts are re-read. */
export function useChangeLesson(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ lesson, verb }: { lesson: number; verb: "drop" | "keep" }) =>
      verb === "drop"
        ? unwrap(api.POST("/distributed/runs/{run_id}/lessons/{lesson}/drop", { params: { path: { run_id: id, lesson } } }))
        : unwrap(api.POST("/distributed/runs/{run_id}/lessons/{lesson}/keep", { params: { path: { run_id: id, lesson } } })),
    onSuccess: (lessons) => {
      client.setQueryData([...RUNS, id, "lessons"], lessons);
      void client.invalidateQueries({ queryKey: RUNS, exact: true });
    },
    onError: (error) => toast.error(T.couldNotChangeLesson.replace("{why}", problem(error))),
  });
}

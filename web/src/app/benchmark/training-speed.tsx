"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { SeriesChart } from "@/components/series-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import {
  historySeries,
  memoryText,
  mfuText,
  peakNote,
  rateText,
  relativeText,
  statusText,
  timeText,
  versionChanges,
  type Board,
  type HistoryPoint,
  type Summary,
} from "./mlperf-logic";

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));
const DIVISION_TITLE: Record<string, string> = { single: "Single device", cluster: "Cluster" };

/** The Training speed section: how long each machine takes to train a fixed reference task to its quality target. */
export function TrainingSpeed() {
  const client = useQueryClient();
  const [task, setTask] = useState("small");
  const [device, setDevice] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const read = useQuery({
    queryKey: ["mlperf", "board", task],
    queryFn: () => unwrap(api.GET("/mlperf/board", { params: { query: { task } } })),
    retry: false,
  });
  const run = useMutation({
    mutationFn: (body: { task: string; device: string }) => unwrap(api.POST("/mlperf/run", { body })),
    onSuccess: (queued) => {
      toast.success(`${queued.label} is in the queue`);
      void client.invalidateQueries({ queryKey: ["mlperf"] });
    },
    onError: (error) => toast.error(message(error)),
  });
  const compare = useMutation({
    mutationFn: (ids: string[]) => unwrap(api.POST("/mlperf/compare", { body: { ids } })),
    onError: (error) => toast.error(message(error)),
  });

  const data = read.data;
  const board = data?.board as unknown as Board | undefined;
  const history = (data?.history as { points?: HistoryPoint[] } | undefined)?.points ?? [];
  const devices = data?.devices ?? ["cpu"];
  const chosen = device || devices[devices.length - 1];
  const spec = data?.tasks.find((t) => t.id === task);
  const rows = (compare.data as { rows?: Summary[] } | undefined)?.rows ?? [];
  const toggle = (id: string) => {
    compare.reset();
    setPicked((old) => (old.includes(id) ? old.filter((x) => x !== id) : [...old, id]));
  };

  return (
    <section aria-label="Training speed" className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Training speed</CardTitle>
          <CardDescription>
            How long a machine takes to train a fixed reference task until its held-out error reaches a target, the way MLPerf Training scores
            it. The task is frozen: its text and settings are checksummed, and results of different task versions are never ranked together.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {read.isPending && <Skeleton aria-label="Reading the training speed results" className="h-24 w-full" />}
          {read.isError && (
            <p className="text-sm text-warning">
              Could not read the training speed results. {message(read.error)}
            </p>
          )}
          {data && spec && (
            <div className="flex flex-wrap items-center gap-2">
              <select
                aria-label="Reference task"
                className="h-9 rounded-md border bg-background px-2 text-sm"
                value={task}
                onChange={(e) => setTask(e.target.value)}
              >
                {data.tasks.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.title} (v{t.version})
                  </option>
                ))}
              </select>
              <select
                aria-label="Device to time"
                className="h-9 rounded-md border bg-background px-2 text-sm"
                value={chosen}
                onChange={(e) => setDevice(e.target.value)}
              >
                {devices.map((d) => (
                  <option key={d} value={d}>
                    {d.toUpperCase()}
                  </option>
                ))}
              </select>
              <Button disabled={run.isPending} onClick={() => run.mutate({ task, device: chosen })}>
                Run on this machine
              </Button>
              <p className="text-xs text-muted-foreground">
                Target {spec.target_bpc} bits per character within {spec.step_cap} steps, {spec.attempts} attempts. Checksum {spec.checksum.slice(0, 12)}.
                It waits behind any running training.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {board && Object.values(board.divisions).every((d) => d.length === 0) && (
        <p className="text-sm text-muted-foreground">No results for this task yet. Run it on this machine to set the first one.</p>
      )}

      {board &&
        Object.entries(board.divisions)
          .filter(([, results]) => results.length > 0)
          .map(([division, results]) => (
            <Card key={division}>
              <CardHeader>
                <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Leaderboard: {DIVISION_TITLE[division] ?? division}
                </CardTitle>
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs text-muted-foreground">
                    <tr>
                      <th className="py-1 pr-3">Compare</th>
                      <th className="py-1 pr-3">Machine</th>
                      <th className="py-1 pr-3">Result</th>
                      <th className="py-1 pr-3">Time to target</th>
                      <th className="py-1 pr-3">Tokens/s</th>
                      <th className="py-1 pr-3">MFU</th>
                      <th className="py-1 pr-3">Memory</th>
                      <th className="py-1 pr-3">App</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r) => (
                      <tr key={r.id} className="border-t">
                        <td className="py-1 pr-3">
                          <input type="checkbox" aria-label={`Compare ${r.label}`} checked={picked.includes(r.id)} onChange={() => toggle(r.id)} />
                        </td>
                        <td className="py-1 pr-3">{r.label}</td>
                        <td className="py-1 pr-3" title={r.error ?? ""}>
                          {statusText(r.status)}
                        </td>
                        <td className="py-1 pr-3 font-mono">{timeText(r.time_to_target)}</td>
                        <td className="py-1 pr-3 font-mono">{rateText(r.tokens_per_s)}</td>
                        <td className="py-1 pr-3 font-mono" title={peakNote(r.peak_kind)}>
                          {mfuText(r.mfu)}
                        </td>
                        <td className="py-1 pr-3 font-mono">{memoryText(r.peak_memory_bytes)}</td>
                        <td className="py-1 pr-3">{r.app_version ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-2 text-xs text-muted-foreground">
                  MFU is model FLOPs per second over the device peak; hover it to see which peak. A run that missed the target in its step cap has no
                  time.
                </p>
              </CardContent>
            </Card>
          ))}

      {board && board.other_versions.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Results of older task versions are kept apart and not ranked:{" "}
          {board.other_versions.map((o) => `v${o.version} (${o.results})`).join(", ")}.
        </p>
      )}

      {board && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Compare</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            <div>
              <Button variant="outline" disabled={picked.length < 2 || compare.isPending} onClick={() => compare.mutate(picked)}>
                Compare {picked.length} selected
              </Button>
            </div>
            {picked.length < 2 && <p className="text-sm text-muted-foreground">Tick two or more results above to set them side by side.</p>}
            {rows.length > 0 && (
              <ul aria-label="Comparison" className="text-sm">
                {rows.map((r) => (
                  <li key={r.id}>
                    {r.label}: {timeText(r.time_to_target)}, {relativeText(r.relative_time)}, {rateText(r.tokens_per_s)} tokens/s, MFU {mfuText(r.mfu)}
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      )}

      {historySeries(history).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">History</CardTitle>
            <CardDescription>Median time to target of each converged result, oldest first. Lower is faster.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            <SeriesChart series={historySeries(history)} xLabel="Result" yLabel="Seconds" />
            <p className="text-xs text-muted-foreground">{versionChanges(history).join(" · ")}</p>
          </CardContent>
        </Card>
      )}
    </section>
  );
}

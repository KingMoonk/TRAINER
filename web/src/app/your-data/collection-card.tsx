"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap, type ApiEvent } from "@/lib/api";
import { cn } from "@/lib/utils";
import { COLLECTION, type Kind } from "./copy";
import { useApiEvents } from "./events";
import { errorText, sourceAction, sourceStatus, type Source } from "./logic";

const FLUSH_MS = 500; // progress arrives several times a second; the rows repaint at most this often

type Progress = { text: string; completed: number; total: number };

/** The kind's recommended sources, kept current from the event stream: the same query on Your data and on Pretraining. */
export function useCollection(kind: Kind) {
  const queryClient = useQueryClient();
  const query = useQuery({ queryKey: ["data", "collection", kind], queryFn: () => unwrap(api.GET("/data/collections/{kind}", { params: { path: { kind } } })) });
  const [progress, setProgress] = useState<Record<string, Progress>>({});
  const pending = useRef<Record<string, Progress>>({});
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);

  const onEvent = useCallback(
    (event: ApiEvent) => {
      const p = event.payload as { id?: string; text?: string; completed?: number; total?: number } | null;
      if (event.kind === "download" && p?.id) {
        pending.current[p.id] = { text: p.text ?? "", completed: p.completed ?? 0, total: p.total ?? 0 };
        if (timer.current === undefined) {
          timer.current = setTimeout(() => {
            timer.current = undefined;
            setProgress((now) => ({ ...now, ...pending.current }));
          }, FLUSH_MS);
        }
      } else if (event.kind === "download_done" || event.kind === "import_done" || event.kind === "download_failed") {
        if (p?.id) {
          delete pending.current[p.id];
          const id = p.id;
          setProgress((now) => Object.fromEntries(Object.entries(now).filter(([key]) => key !== id)));
        }
        queryClient.invalidateQueries({ queryKey: ["data", "collection"] });
      } else if (event.kind === "downloads_finished") {
        pending.current = {};
        setProgress({});
        queryClient.invalidateQueries({ queryKey: ["data"] });
      }
    },
    [queryClient],
  );
  useApiEvents(onEvent);

  const start = useMutation({
    mutationFn: ({ id, action }: { id: string | null; action: "download" | "import" }) =>
      action === "download"
        ? unwrap(api.POST("/data/collections/{kind}/download", { params: { path: { kind } }, body: { ids: id ? [id] : null, install: true } }))
        : unwrap(api.POST("/data/sources/{source_id}/import", { params: { path: { source_id: id as string } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["data", "collection"] }),
    onError: (error) => toast.error(errorText(error)),
  });

  return { ...query, progress, start: (id: string | null, action: "download" | "import") => start.mutate({ id, action }), starting: start.isPending };
}

function SourceRow({ source, progress, onAct, disabled }: { source: Source; progress?: Progress; onAct: (action: "download" | "import") => void; disabled: boolean }) {
  const action = sourceAction(source);
  const fraction = source.state === "downloading" && progress?.total ? Math.min(1, progress.completed / progress.total) : null;
  return (
    <li data-testid={`source-${source.id}`} className="flex flex-wrap items-center gap-x-3 gap-y-1 py-1.5 text-sm">
      <span className="font-medium">{source.name}</span>
      <span className="text-muted-foreground">
        {source.size} · {source.records}
      </span>
      <span className="flex-1" />
      <span className={cn("max-w-sm text-right", source.error ? "text-destructive" : "text-muted-foreground")} data-testid={`source-${source.id}-status`}>
        {sourceStatus(source, progress?.text)}
      </span>
      {fraction !== null && (
        <span
          role="progressbar"
          aria-label={`${source.name} progress`}
          aria-valuenow={Math.round(fraction * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          className="h-1.5 w-24 overflow-hidden rounded bg-muted"
        >
          <span className="block h-full bg-violet-500" style={{ width: `${fraction * 100}%` }} />
        </span>
      )}
      <span className="min-w-36 text-right">
        {action && (
          <Button size="sm" variant="outline" disabled={disabled} onClick={() => onAct(action.action)}>
            {action.label}
          </Button>
        )}
      </span>
    </li>
  );
}

/**
 * One kind's recommended collection, the same card on Your data and on Pretraining: each source on a line (name, size, status,
 * the button for its next step), keyed by source id so a download's progress updates one line in place and the others keep
 * their DOM nodes.
 */
export function CollectionCard({ kind, description }: { kind: Kind; description: string }) {
  const { data, isLoading, isError, progress, start, starting } = useCollection(kind);
  const downloadable = (data ?? []).some((s) => sourceAction(s)?.action === "download");
  return (
    <Card data-testid={`collection-${kind}`}>
      <CardHeader>
        <CardTitle>{COLLECTION.title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex flex-col gap-2" aria-label="Reading the collection">
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-2/3" />
          </div>
        ) : isError ? (
          <p className="text-sm text-destructive">The collection could not be read. Check that the server is running.</p>
        ) : !data?.length ? (
          <p className="text-sm text-muted-foreground">{COLLECTION.empty}</p>
        ) : (
          <>
            <ul aria-label="Sources" className="flex flex-col divide-y">
              {data.map((source) => (
                <SourceRow key={source.id} source={source} progress={progress[source.id]} disabled={starting} onAct={(action) => start(source.id, action)} />
              ))}
            </ul>
            {downloadable && (
              <Button className="mt-3" disabled={starting} onClick={() => start(null, "download")}>
                {COLLECTION.downloadAll}
              </Button>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

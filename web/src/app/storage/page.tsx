"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, apiUrl, ApiError, subscribe, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { EXPORT, STORAGE as T } from "./copy";
import { AGES, ALL, DEFAULT_CHOICE, choiceDays, dataApplies, downloadName, errorText, fill, sizeText } from "./logic";

type ExportKind = "hf" | "gguf";

export default function Page() {
  const route = routeFor("storage");
  const queryClient = useQueryClient();

  // Disk numbers move when a job finishes or a version trains, so those events refresh them in place.
  useEffect(
    () =>
      subscribe((event) => {
        if (event.kind === "queue" || event.kind === "versions") {
          queryClient.invalidateQueries({ queryKey: ["storage"] });
        }
      }),
    [queryClient],
  );

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-4 p-4 sm:p-8">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold tracking-tight">{route.title}</h1>
        <p className="max-w-prose text-muted-foreground">{route.lead}</p>
      </div>
      <StorageCard />
      <ExportCard />
    </main>
  );
}

function Kicker({ children, desc }: { children: string; desc: string }) {
  return (
    <CardHeader>
      <CardTitle className="text-xs font-medium tracking-wider text-muted-foreground">{children}</CardTitle>
      <p className="max-w-prose text-sm text-muted-foreground">{desc}</p>
    </CardHeader>
  );
}

function StorageCard() {
  const queryClient = useQueryClient();
  const [choice, setChoice] = useState(DEFAULT_CHOICE);
  const [data, setData] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const days = choiceDays(choice);
  const withData = dataApplies(choice, data);

  const free = useQuery({ queryKey: ["storage", "free"], queryFn: () => unwrap(api.GET("/storage/free")) });
  const use = useQuery({
    queryKey: ["storage", "use", days, withData],
    queryFn: () => unwrap(api.GET("/storage", { params: { query: { ...(days === null ? {} : { days }), data: withData } } })),
    placeholderData: keepPreviousData,
  });
  const clean = useMutation({
    mutationFn: () => unwrap(api.POST("/storage/clean", { body: { days, data: withData } })),
    onSuccess: (done) => {
      toast.success(fill(T.freed, { size: sizeText(done.bytes) }));
      queryClient.invalidateQueries({ queryKey: ["storage"] });
    },
    onError: (error) => toast.error(errorText(error)),
    onSettled: () => setConfirming(false),
  });

  const items = use.data?.items ?? 0;
  const bytes = use.data?.bytes ?? 0;
  const choices = [...AGES.map((n) => ({ id: String(n), label: fill(T.olderThan, { days: n }) })), { id: ALL, label: T.all }];
  const line = items === 0 ? T.nothing : fill(items === 1 ? T.freesOne : T.frees, { size: sizeText(bytes), n: items });

  return (
    <Card>
      <Kicker desc={T.desc}>{T.kicker}</Kicker>
      <CardContent className="flex flex-col gap-3">
        {free.isError || use.isError ? (
          <p className="text-sm text-destructive">{T.loadFailed}</p>
        ) : free.isPending ? (
          <Skeleton className="h-5 w-64" aria-label="Loading free space" />
        ) : (
          <p>{fill(T.free, { size: sizeText(free.data.bytes) })}</p>
        )}

        <fieldset className="flex flex-col gap-1" disabled={clean.isPending}>
          <legend className="sr-only">{T.choices}</legend>
          {choices.map((c) => (
            <label key={c.id} className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name="age"
                value={c.id}
                checked={choice === c.id}
                onChange={() => {
                  setChoice(c.id);
                  setConfirming(false);
                }}
                className="size-4 accent-primary"
              />
              {c.label}
            </label>
          ))}
          <label className={`mt-2 flex items-center gap-2 text-sm ${choice === ALL ? "" : "opacity-50"}`}>
            <input
              type="checkbox"
              checked={withData}
              disabled={choice !== ALL}
              onChange={(e) => {
                setData(e.target.checked);
                setConfirming(false);
              }}
              className="size-4 accent-primary"
            />
            {T.data}
          </label>
        </fieldset>

        {use.isPending ? <Skeleton className="h-5 w-48" aria-label="Loading what can be deleted" /> : <p aria-live="polite">{line}</p>}

        {confirming ? (
          <div role="alertdialog" aria-label={T.confirmYes} className="flex flex-col gap-2 rounded-md border p-3">
            <p className="text-sm">{fill(items === 1 ? T.confirmOne : T.confirm, { n: items, size: sizeText(bytes) })}</p>
            <div className="flex gap-2">
              <Button variant="destructive" onClick={() => clean.mutate()} disabled={clean.isPending}>
                {clean.isPending ? T.deleting : T.confirmYes}
              </Button>
              <Button variant="outline" onClick={() => setConfirming(false)} disabled={clean.isPending}>
                {T.confirmNo}
              </Button>
            </div>
          </div>
        ) : (
          <div>
            <Button variant="outline" disabled={items === 0 || use.isPending} onClick={() => setConfirming(true)}>
              {T.delete}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ExportCard() {
  const rows = useQuery({ queryKey: ["storage", "exports"], queryFn: () => unwrap(api.GET("/storage/exports")) });
  const [busy, setBusy] = useState<string | null>(null);

  async function download(number: number, name: string, kind: ExportKind) {
    setBusy(`${number}:${kind}`);
    try {
      const res = await fetch(apiUrl(`/versions/${number}/export?kind=${kind}`));
      if (!res.ok) throw new ApiError(res.status, `/versions/${number}/export`, await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = downloadName(res.headers.get("content-disposition"), `${name}-${kind}`);
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      toast.success(fill(EXPORT.done, { name }));
    } catch (error) {
      toast.error(errorText(error));
    } finally {
      setBusy(null);
    }
  }

  return (
    <Card>
      <Kicker desc={EXPORT.desc}>{EXPORT.kicker}</Kicker>
      <CardContent className="flex flex-col gap-3">
        {rows.isPending ? (
          <div className="flex flex-col gap-2" aria-label="Loading versions">
            <Skeleton className="h-9 w-full max-w-md" />
            <Skeleton className="h-9 w-full max-w-md" />
          </div>
        ) : rows.isError ? (
          <p className="text-sm text-destructive">{EXPORT.loadFailed}</p>
        ) : rows.data.length === 0 ? (
          <p className="text-sm text-muted-foreground">{EXPORT.none}</p>
        ) : (
          <ul className="flex flex-col gap-3">
            {rows.data.map((v) => (
              <li key={v.number} className="flex flex-col gap-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-sm">{v.name}</span>
                  <Button variant="outline" disabled={busy !== null} onClick={() => download(v.number, v.name, "hf")}>
                    {busy === `${v.number}:hf` ? EXPORT.working : EXPORT.hf}
                  </Button>
                  <Button
                    variant="outline"
                    disabled={busy !== null || !!v.gguf_problem}
                    aria-describedby={v.gguf_problem ? `gguf-why-${v.number}` : undefined}
                    onClick={() => download(v.number, v.name, "gguf")}
                  >
                    {busy === `${v.number}:gguf` ? EXPORT.working : EXPORT.gguf}
                  </Button>
                </div>
                {v.gguf_problem && (
                  <p id={`gguf-why-${v.number}`} className="max-w-prose text-xs text-muted-foreground">
                    {v.gguf_problem}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

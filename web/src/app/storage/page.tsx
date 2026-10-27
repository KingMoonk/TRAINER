"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, subscribe, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { LEFTOVERS, STORAGE as T } from "./copy";
import { AGES, ALL, DEFAULT_CHOICE, choiceDays, dataApplies, errorText, fill, sizeText } from "./logic";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

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
    <main className={PAGE_MAIN}>
      <PageHeader title={route.title} lead={route.lead} />
      <StorageCard />
      <LeftoversCard />
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

function LeftoversCard() {
  const queryClient = useQueryClient();
  const [picked, setPicked] = useState<string[]>([]);
  const [confirming, setConfirming] = useState(false);
  const rows = useQuery({ queryKey: ["storage", "leftovers"], queryFn: () => unwrap(api.GET("/storage/leftovers")) });
  const remove = useMutation({
    mutationFn: (ids: string[]) => unwrap(api.POST("/storage/leftovers/remove", { body: { ids } })),
    onSuccess: (done) => {
      toast.success(fill(LEFTOVERS.removed, { size: sizeText(done.bytes) }));
      setPicked([]);
      queryClient.invalidateQueries({ queryKey: ["storage"] });
    },
    onError: (error) => toast.error(errorText(error)),
    onSettled: () => setConfirming(false),
  });

  // Only what is still listed counts: a row another refresh dropped can no longer be sent.
  const listed = rows.data ?? [];
  const chosen = listed.filter((row) => picked.includes(row.id));
  const bytes = chosen.reduce((sum, row) => sum + row.bytes, 0);

  return (
    <Card>
      <Kicker desc={LEFTOVERS.desc}>{LEFTOVERS.kicker}</Kicker>
      <CardContent className="flex flex-col gap-3">
        {rows.isPending ? (
          <Skeleton className="h-9 w-full max-w-md" aria-label="Loading leftovers" />
        ) : rows.isError ? (
          <p className="text-sm text-destructive">{LEFTOVERS.loadFailed}</p>
        ) : listed.length === 0 ? (
          <p className="text-sm text-muted-foreground">{LEFTOVERS.none}</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {listed.map((row) => (
              <li key={row.id}>
                <label className="flex items-start gap-2 text-sm">
                  <input
                    type="checkbox"
                    aria-label={fill(LEFTOVERS.select, { name: row.name })}
                    checked={picked.includes(row.id)}
                    disabled={remove.isPending}
                    onChange={(e) => {
                      setPicked(e.target.checked ? [...picked, row.id] : picked.filter((id) => id !== row.id));
                      setConfirming(false);
                    }}
                    className="mt-0.5 size-4 accent-primary"
                  />
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate">
                      {LEFTOVERS.kinds[row.kind]}: {row.name}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {row.belongs_to} · {sizeText(row.bytes)}
                      {row.shared > 0 && ` · ${fill(LEFTOVERS.shared, { size: sizeText(row.shared) })}`}
                    </span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
        )}
        {confirming ? (
          <div role="alertdialog" aria-label="Confirm removing leftovers" className="flex flex-col gap-2 rounded-md border p-3">
            <p className="text-sm">{fill(chosen.length === 1 ? LEFTOVERS.confirmOne : LEFTOVERS.confirm, { n: chosen.length, size: sizeText(bytes) })}</p>
            <div className="flex gap-2">
              <Button variant="destructive" onClick={() => remove.mutate(chosen.map((row) => row.id))} disabled={remove.isPending}>
                {remove.isPending ? LEFTOVERS.removing : T.confirmYes}
              </Button>
              <Button variant="outline" onClick={() => setConfirming(false)} disabled={remove.isPending}>
                {T.confirmNo}
              </Button>
            </div>
          </div>
        ) : (
          listed.length > 0 && (
            <div>
              <Button variant="outline" disabled={chosen.length === 0} onClick={() => setConfirming(true)}>
                {LEFTOVERS.remove}
              </Button>
            </div>
          )
        )}
      </CardContent>
    </Card>
  );
}

"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap, type ApiEvent } from "@/lib/api";
import { CollectionCard } from "./collection-card";
import { KIND, KINDS, MAKE_LABELS, T, fill, type Kind } from "./copy";
import { useApiEvents } from "./events";
import { ItemForm } from "./forms";
import { ImportCard } from "./import-card";
import { runJob, type JobLike } from "./jobs";
import { backLink, collectionShown, dupLabel, errorText, kindOf, setsLine, type DataRow, type Summary } from "./logic";
import { RowsTable, useRows } from "./rows-table";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { optionBox } from "@/components/option-box";

const SEARCH_MS = 250; // after the last key before a search runs
let refreshed = false; // downloads an older app converted are rebuilt once per page load

function useSummary(kind: Kind) {
  return useQuery({
    queryKey: ["data", "summary", kind],
    queryFn: ({ signal }) =>
      runJob<Summary>(() => unwrap(api.POST("/data/{kind}/summary", { params: { path: { kind } }, body: { query: "" } })) as Promise<JobLike>, signal),
    staleTime: 30_000,
  });
}

function Detail({ row, kind, onChanged, onEdit }: { row: DataRow | undefined; kind: Kind; onChanged: () => void; onEdit: () => void }) {
  const queryClient = useQueryClient();
  const review = useMutation({
    mutationFn: (keep: boolean) => unwrap(api.POST("/data/examples/{example_id}/review", { params: { path: { example_id: row!.id } }, body: { keep } })),
    onSuccess: () => {
      onChanged();
      queryClient.invalidateQueries({ queryKey: ["data"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  return (
    <Card>
      <CardContent className="flex flex-col gap-3" data-testid="detail">
        {row ? (
          <>
            <p className="whitespace-pre-wrap text-sm">
              {row.a}
              {"\n\n"}
              {row.b}
            </p>
            <div className="flex flex-wrap gap-2">
              {row.status === "Review" && (
                <>
                  <Button disabled={review.isPending} onClick={() => review.mutate(true)}>
                    {T.markGood}
                  </Button>
                  <Button variant="outline" disabled={review.isPending} onClick={() => review.mutate(false)}>
                    {T.discard}
                  </Button>
                </>
              )}
              <Button variant="outline" onClick={onEdit}>
                {T.edit}
              </Button>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">{T.pickRow}</p>
        )}
        <p className="text-xs text-muted-foreground">{KIND[kind].note}</p>
      </CardContent>
    </Card>
  );
}

function EditForm({ row, kind, onDone }: { row: DataRow; kind: Kind; onDone: () => void }) {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | undefined>();
  const fields = useQuery({
    queryKey: ["data", "fields", row.id],
    queryFn: () => unwrap(api.GET("/data/examples/{example_id}/fields", { params: { path: { example_id: row.id } } })),
    gcTime: 0,
  });
  const save = useMutation({
    mutationFn: (values: Record<string, string>) =>
      unwrap(api.PUT("/data/examples/{example_id}", { params: { path: { example_id: row.id } }, body: { fields: values } })),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data"] });
      onDone();
    },
    onError: (e) => {
      setError(fill(T.editFailed, { why: errorText(e) }));
      toast.error(fill(T.editFailed, { why: errorText(e) }));
    },
  });
  if (fields.isLoading) return <Skeleton className="h-40 w-full" aria-label="Reading the example" />;
  if (fields.isError) return <p className="text-sm text-destructive">{errorText(fields.error)}</p>;
  return (
    <ItemForm
      kind={kind}
      title={T.edit}
      initial={fields.data?.fields}
      submitLabel={T.save}
      busy={save.isPending}
      error={error}
      onSubmit={(values) => save.mutate(values)}
      onCancel={onDone}
    />
  );
}

function YourData() {
  const queryClient = useQueryClient();
  const [kind, setKind] = useState<Kind>(kindOf(useSearchParams().get("kind")));
  const [typed, setTyped] = useState("");
  const [query, setQuery] = useState("");
  const [onlyDuplicates, setOnlyDuplicates] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [panel, setPanel] = useState<"add" | "edit" | "import" | null>(null);

  const summaries = { text: useSummary("text"), conversations: useSummary("conversations"), problems: useSummary("problems") };
  const summary = summaries[kind].data;
  const rows = useRows(kind, query);
  const row = rows.data?.pages.flatMap((p) => p.rows).find((r) => r.id === selected);

  useEffect(() => {
    const timer = setTimeout(() => setQuery(typed.trim()), SEARCH_MS);
    return () => clearTimeout(timer);
  }, [typed]);

  useEffect(() => {
    if (refreshed) return;
    refreshed = true;
    runJob<string[]>(() => unwrap(api.POST("/data/refresh_downloads")) as Promise<JobLike>)
      .then((ids) => {
        if (ids.length) queryClient.invalidateQueries({ queryKey: ["data"] });
      })
      .catch(() => {});
  }, [queryClient]);

  const onEvent = useCallback(
    (event: ApiEvent) => {
      if (event.kind === "data" || event.kind === "downloads_finished") {
        queryClient.invalidateQueries({ queryKey: ["data", "summary"] });
        queryClient.invalidateQueries({ queryKey: ["data", "rows"] });
      }
    },
    [queryClient],
  );
  useApiEvents(onEvent);

  const show = (next: Kind) => {
    setKind(next);
    setTyped("");
    setQuery("");
    setOnlyDuplicates(false);
    setSelected(null);
    setPanel(null);
  };

  const [addError, setAddError] = useState<string | undefined>();
  const add = useMutation({
    mutationFn: (fields: Record<string, string>) => unwrap(api.POST("/data/{kind}/rows", { params: { path: { kind } }, body: { fields } })),
    onSuccess: (done) => {
      if (done.added) {
        setPanel(null);
        setAddError(undefined);
        toast.success("Added.");
      } else {
        setAddError(T.already);
      }
      queryClient.invalidateQueries({ queryKey: ["data"] });
    },
    onError: (error) => setAddError(errorText(error)),
  });

  const back = backLink(kind);
  const duplicates = summary?.near_duplicates ?? 0;

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={T.h1} lead={T.lead}>
        <div className="flex flex-wrap gap-6 text-sm">
          <Link href="/storage" className="text-highlight hover:underline">
            {T.storageLink}
          </Link>
          <Link href="/passive/" className="text-highlight hover:underline">{T.passiveLink}</Link>
        </div>
      </PageHeader>
      <Link href={back.href} className="w-fit text-sm text-highlight hover:underline">
        ‹ {back.label}
      </Link>

      <div role="tablist" aria-label="Kinds of data" className="grid gap-2 sm:grid-cols-3">
        {KINDS.map((k) => {
          const s = summaries[k];
          return (
            <button
              key={k}
              role="tab"
              aria-selected={k === kind}
              title={KIND[k].lead}
              onClick={() => show(k)}
              className={optionBox(k === kind, "flex flex-col gap-0.5 p-3")}
            >
              <span className="font-semibold">{KIND[k].tab}</span>
              <span className="text-xs text-muted-foreground">{s.data?.used_by || KIND[k].used}</span>
              {s.isLoading ? <Skeleton className="mt-0.5 h-4 w-32" /> : <span className="text-sm">{s.data?.count_line}</span>}
            </button>
          );
        })}
      </div>

      <div className="min-h-5 text-sm text-muted-foreground" data-testid="sets-line">
        {summaries[kind].isLoading ? <Skeleton className="h-4 w-80" /> : setsLine(summary)}
      </div>

      <label className="flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">{T.search}</span>
        <input
          aria-label={KIND[kind].searchLabel}
          className="w-80 rounded-md border bg-background px-2 py-1.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
        />
      </label>

      <RowsTable
        kind={kind}
        query={query}
        count={summary?.count ?? 0}
        onlyDuplicates={onlyDuplicates}
        selected={selected}
        onSelect={setSelected}
        rows={rows}
      />

      <Detail row={row} kind={kind} onChanged={() => setSelected(null)} onEdit={() => setPanel("edit")} />

      <div className="flex flex-wrap gap-2">
        <Button onClick={() => setPanel(panel === "add" ? null : "add")}>{KIND[kind].add}</Button>
        <Link href={`/make-data/?kind=${kind}`} className={buttonVariants({ variant: "outline" })}>{MAKE_LABELS[kind]}</Link>
        <Button variant="outline" onClick={() => setPanel(panel === "import" ? null : "import")}>
          {T.importFiles}
        </Button>
        <Button variant="outline" disabled={!duplicates} onClick={() => setOnlyDuplicates((on) => !on)}>
          {dupLabel(duplicates, onlyDuplicates)}
        </Button>
      </div>

      {panel === "add" && (
        <ItemForm
          key={`add-${kind}`}
          kind={kind}
          title={KIND[kind].add}
          submitLabel={KIND[kind].add}
          busy={add.isPending}
          error={addError}
          onSubmit={(fields) => add.mutate(fields)}
          onCancel={() => setPanel(null)}
        />
      )}
      {panel === "edit" && row && <EditForm key={row.id} row={row} kind={kind} onDone={() => setPanel(null)} />}
      {panel === "import" && <ImportCard key={`import-${kind}`} kind={kind} onClose={() => setPanel(null)} />}

      {collectionShown(summary) && <CollectionCard kind={kind} description={KIND[kind].collectionDesc} />}
    </main>
  );
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <YourData />
    </Suspense>
  );
}

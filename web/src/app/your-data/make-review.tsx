"use client";

import { useInfiniteQuery, useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { errorText } from "./logic";
import { MAKE as T } from "./make-copy";

export function refreshData(cache: QueryClient) {
  return cache.invalidateQueries({ predicate: (query) => query.queryKey[0] === "data" &&
    query.queryKey[1] !== "draft-fields" && query.queryKey[1] !== "problem-rules" }, { cancelRefetch: false });
}

export function Review({ kind, count }: { kind: "text" | "conversations"; count?: number }) {
  const rows = useInfiniteQuery({
    queryKey: ["data", "draft-rows", kind], initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) => unwrap(api.GET("/data/{kind}/rows", { params: { path: { kind }, query: { review_only: true, cursor: pageParam ?? undefined, limit: 50 } }, signal })),
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  });
  const loaded = rows.data?.pages.flatMap((page) => page.rows) ?? [];
  return <Card><CardHeader><CardTitle>{T.review}{count !== undefined ? ` · ${count} across all categories` : ""}</CardTitle></CardHeader><CardContent className="space-y-3">
    {rows.isPending ? <Skeleton className="h-32 w-full" aria-label="Loading drafts" /> : rows.isError ?
      <p role="alert">{errorText(rows.error)} <Button variant="outline" onClick={() => void rows.refetch()}>{T.retry}</Button></p> : <>
      {!loaded.length && <p className="text-sm text-muted-foreground">{T.empty}</p>}
      <ul className="space-y-3" aria-label="Drafts">{loaded.map((row) => <Draft key={row.id} row={row} />)}</ul>
      {rows.hasNextPage && <Button variant="outline" disabled={rows.isFetchingNextPage} onClick={() => void rows.fetchNextPage()}>{T.next}</Button>}
    </>}
  </CardContent></Card>;
}

function Draft({ row }: { row: components["schemas"]["DataRow"] }) {
  const [open, setOpen] = useState(false);
  const cache = useQueryClient();
  const fields = useQuery({ queryKey: ["data", "draft-fields", row.id], enabled: open,
    queryFn: ({ signal }) => unwrap(api.GET("/data/examples/{example_id}/fields", { params: { path: { example_id: row.id } }, signal })),
  });
  const review = useMutation({ mutationFn: (keep: boolean) => unwrap(api.POST("/data/examples/{example_id}/review", { params: { path: { example_id: row.id } }, body: { keep } })),
    onSuccess: () => { void refreshData(cache); },
  });
  return <li className="space-y-3 rounded-lg border bg-background/30 p-3" data-testid={`draft-${row.id}`}>
    <p className="break-words text-sm">{row.a}</p>
    <Button variant="ghost" size="sm" aria-expanded={open} onClick={() => setOpen(!open)}>{T.read}</Button>
    {open && (fields.isPending ? <Skeleton className="h-20 w-full" aria-label="Loading full draft" /> : fields.isError ?
      <p role="alert">{errorText(fields.error)} <Button variant="outline" onClick={() => void fields.refetch()}>{T.retry}</Button></p> : <>
      <dl className="space-y-3">{Object.entries(fields.data.fields).map(([name, text]) => <div key={name}><dt className="text-xs uppercase tracking-wide text-muted-foreground">{name}</dt><dd className="break-words whitespace-pre-wrap text-sm">{text}</dd></div>)}</dl>
      <div className="flex gap-2"><Button size="sm" disabled={review.isPending} onClick={() => review.mutate(true)}>{T.keep}</Button><Button size="sm" variant="outline" disabled={review.isPending} onClick={() => review.mutate(false)}>{T.discard}</Button></div>
    </>)}
    {review.isError && <p role="alert" className="text-sm text-destructive">{errorText(review.error)}</p>}
  </li>;
}

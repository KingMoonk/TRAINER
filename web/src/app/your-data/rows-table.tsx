"use client";

import { keepPreviousData, useInfiniteQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { cn } from "@/lib/utils";
import { T, type Kind } from "./copy";
import { STATUS_CLASS, shownLine, visibleRows, type DataRow } from "./logic";

const PAGE = 200; // rows the list gains per read

/** The rows of one kind, a page at a time. Nothing is read while `enabled` is false (the examples list is closed). */
export function useRows(kind: Kind, query: string, enabled: boolean) {
  return useInfiniteQuery({
    queryKey: ["data", "rows", kind, query],
    initialPageParam: null as string | null,
    enabled,
    queryFn: ({ pageParam }) =>
      unwrap(api.GET("/data/{kind}/rows", { params: { path: { kind }, query: { query, cursor: pageParam ?? undefined, limit: PAGE } } })),
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    placeholderData: keepPreviousData, // a search keeps the last rows on screen until the new ones are in
  });
}

/** One line per row: the prompt or text, cut to the width, and its status. Keyed by row id: an edit or a review changes a row in place. */
export function RowsList({
  kind,
  query,
  count,
  onlyDuplicates,
  selected,
  onSelect,
  rows,
}: {
  kind: Kind;
  query: string;
  count: number;
  onlyDuplicates: boolean;
  selected: string | null;
  onSelect: (id: string) => void;
  rows: ReturnType<typeof useRows>;
}) {
  const loaded: DataRow[] = rows.data?.pages.flatMap((p) => p.rows) ?? [];
  const shown = visibleRows(loaded, onlyDuplicates);
  const mono = kind === "problems";
  const note = shownLine(loaded.length, count, query, !!rows.hasNextPage);
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <ul className="max-h-96 divide-y overflow-y-auto rounded-lg border" aria-label={T.browse}>
        {rows.isPending ? (
          [0, 1, 2, 3].map((i) => (
            <li key={i} aria-hidden className="px-3 py-2">
              <Skeleton className="h-5 w-full" />
            </li>
          ))
        ) : shown.length === 0 ? (
          <li className="px-3 py-4 text-sm text-muted-foreground">{query.trim() || onlyDuplicates ? T.noMatches : T.noRows}</li>
        ) : (
          shown.map((row) => (
            <li
              key={row.id}
              data-testid={`row-${row.id}`}
              tabIndex={0}
              aria-current={row.id === selected}
              onClick={() => onSelect(row.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(row.id);
                }
              }}
              className={cn(
                "flex cursor-pointer items-center gap-3 px-3 py-2 text-sm outline-none hover:bg-muted/30 focus-visible:bg-muted/50",
                row.id === selected && "bg-(--selected-surface)",
              )}
            >
              <span className={cn("min-w-0 flex-1 truncate", mono && "font-mono text-[13px]")}>{row.a}</span>
              <Badge variant="outline" className={STATUS_CLASS[row.status]}>
                {row.status}
              </Badge>
            </li>
          ))
        )}
      </ul>
      {note && <p className="text-xs text-muted-foreground">{note}</p>}
      {rows.hasNextPage && (
        <div>
          <Button size="sm" variant="outline" disabled={rows.isFetchingNextPage} onClick={() => rows.fetchNextPage()}>
            {T.loadMore}
          </Button>
        </div>
      )}
    </div>
  );
}

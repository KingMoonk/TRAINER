"use client";

import { keepPreviousData, useInfiniteQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { cn } from "@/lib/utils";
import { KIND, T, type Kind } from "./copy";
import { STATUS_CLASS, shownLine, visibleRows, type DataRow } from "./logic";

const PAGE = 200; // rows the table gains per read

export function useRows(kind: Kind, query: string) {
  return useInfiniteQuery({
    queryKey: ["data", "rows", kind, query],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) =>
      unwrap(api.GET("/data/{kind}/rows", { params: { path: { kind }, query: { query, cursor: pageParam ?? undefined, limit: PAGE } } })),
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    placeholderData: keepPreviousData, // a search keeps the last rows on screen until the new ones are in
  });
}

/** The rows of one kind, a page at a time. Keyed by row id: an edit or a review changes a row in place, nothing remounts. */
export function RowsTable({
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
    <div className="flex flex-col gap-1">
      <div className="overflow-hidden rounded-lg border">
        <table className="w-full table-fixed text-sm" aria-label={`${KIND[kind].tab} rows`}>
          <thead className="bg-muted/40 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="w-[38%] px-3 py-2 font-medium">{KIND[kind].colA}</th>
              <th className="w-[38%] px-3 py-2 font-medium">{KIND[kind].colB}</th>
              <th className="w-[24%] px-3 py-2 font-medium">{T.status}</th>
            </tr>
          </thead>
          <tbody>
            {rows.isPending ? (
              [0, 1, 2, 3].map((i) => (
                <tr key={i} aria-hidden>
                  <td colSpan={3} className="px-3 py-1.5">
                    <Skeleton className="h-5 w-full" />
                  </td>
                </tr>
              ))
            ) : shown.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-3 py-4 text-muted-foreground">
                  {query.trim() || onlyDuplicates ? T.noMatches : T.noRows}
                </td>
              </tr>
            ) : (
              shown.map((row) => (
                <tr
                  key={row.id}
                  data-testid={`row-${row.id}`}
                  tabIndex={0}
                  aria-selected={row.id === selected}
                  onClick={() => onSelect(row.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(row.id);
                    }
                  }}
                  className={cn("cursor-pointer border-t outline-none hover:bg-muted/30 focus-visible:bg-muted/50", row.id === selected && "bg-(--selected-surface)")}
                >
                  <td className={cn("truncate px-3 py-1.5", mono && "font-mono text-[13px]")}>{row.a}</td>
                  <td className={cn("truncate px-3 py-1.5", mono && "font-mono text-[13px]")}>{row.b}</td>
                  <td className={cn("px-3 py-1.5", STATUS_CLASS[row.status])}>{row.status}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
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

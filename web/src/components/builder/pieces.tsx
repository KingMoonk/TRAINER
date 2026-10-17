"use client";

// The example sentence cut into pieces by the draft's word list (builder.py _paint_pieces, design.pieces_for): one chip per
// piece, the words a list lacks marked as lost, and a line counting every list's pieces side by side. The first read fits
// the lists on your text (a job), so the card shows a skeleton until it arrives.

import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@/components/ui/skeleton";
import { design, piecesFor } from "@/lib/builder";
import { cn } from "@/lib/utils";

/** The word-lists job, shared with the explain panel through one query key. */
export function useWordLists(enabled = true) {
  return useQuery({
    queryKey: ["design", "word-lists"],
    queryFn: ({ signal }) => design.wordLists(signal),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function Pieces({ v }: { v: number }) {
  const listsQ = useWordLists();
  const view = piecesFor(v, listsQ.data);
  return (
    <div className="flex flex-col gap-1 rounded-md border px-3 py-2" data-testid="pieces">
      <p className="text-xs text-muted-foreground" data-testid="pieces-count">
        {view.line}
      </p>
      {listsQ.isPending ? (
        <Skeleton className="h-6 w-full" aria-hidden="true" />
      ) : view.note ? (
        <p className="text-xs text-muted-foreground">{view.note}</p>
      ) : (
        <ul className="flex flex-wrap gap-1" aria-label="The example sentence in pieces">
          {view.chips.map((chip, i) => (
            <li
              key={i}
              data-state={chip.state}
              className={cn("rounded border px-1.5 py-0.5 font-mono text-xs", chip.state === "lost" ? "border-warning text-warning" : "bg-part-mlp/10")}
            >
              {chip.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

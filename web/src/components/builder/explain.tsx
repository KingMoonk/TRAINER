"use client";

// The side panel a click on a part opens (builder.py _paint_explain, design.explain): what the part does, the module it
// runs now, what it does to the example sentence, and the look-inside pages that show it work. The words are the
// server's; the example sentence is cut by the draft's word list, which the first read fits on your text (a job).

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { design, detailOf, type Choices, type Explaining } from "@/lib/builder";
import { BUILDER } from "@/lib/copy/builder";
import { routeFor } from "@/lib/routes";

type Props = { draft: Choices; explaining: Explaining | null; onClose: () => void };

/** Inside a head opens at the block the part was clicked in. */
function hrefFor(route: string, block: number) {
  const { href } = routeFor(route);
  return route === "head_zoom" ? `${href}/?block=${block}&head=0` : `${href}/`;
}

export function Explain({ draft, explaining, onClose }: Props) {
  const listsQ = useQuery({
    queryKey: ["design", "word-lists"],
    queryFn: ({ signal }) => design.wordLists(signal),
    enabled: explaining !== null,
    staleTime: Infinity,
    retry: false,
  });
  // A word list that is missing or still being read leaves the example without its pieces, as on the Tk page.
  const list = listsQ.data?.lists[draft.V] ?? null;
  const pieces = list ? list.chips.map((chip) => chip.label) : null;
  const sentence = listsQ.data?.sentence ?? null;
  const part = explaining?.part;
  const explainQ = useQuery({
    queryKey: ["design", "explain", draft, part, explaining?.block ?? null, pieces, sentence],
    queryFn: () => design.explain(draft, part!, explaining!.block, pieces, sentence),
    enabled: explaining !== null,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    retry: false,
  });
  const said = explainQ.data;
  return (
    <Card size="sm" data-testid="explain">
      <CardHeader className="flex-row items-center justify-between gap-2">
        <CardTitle>{BUILDER.explain}</CardTitle>
        {explaining && (
          <Button size="xs" variant="ghost" onClick={onClose} data-testid="explain-close">
            {BUILDER.explain_close}
          </Button>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-2" aria-live="polite">
        {!explaining ? (
          <p className="text-xs text-muted-foreground">{BUILDER.explain_hint}</p>
        ) : explainQ.isError ? (
          <p role="alert" className="text-sm text-destructive">
            {BUILDER.explain_failed} {detailOf(explainQ.error)}
          </p>
        ) : !said ? (
          <div role="status" aria-label={BUILDER.explain_loading} aria-busy="true" className="flex flex-col gap-2" data-testid="explain-loading">
            <Skeleton className="h-5 w-1/2" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : (
          <>
            <h3 className="text-base font-medium" data-testid="explain-title">
              {said.title}
            </h3>
            <p className="whitespace-pre-line text-xs">{said.what}</p>
            <p className="text-sm font-medium" title={said.module_text}>
              {said.module}
            </p>
            <p className="text-xs text-muted-foreground">{said.module_text}</p>
            <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{BUILDER.explain_example}</h4>
            <p className="text-xs" data-testid="explain-example">
              {said.example}
            </p>
            {said.links.length > 0 && (
              <div className="flex flex-col gap-0.5">
                <span className="text-xs text-muted-foreground">{BUILDER.explain_see}</span>
                {said.links.map((link) => (
                  <Link key={link.route} href={hrefFor(link.route, link.block)} className="text-sm text-highlight underline-offset-4 hover:underline">
                    {link.label} ›
                  </Link>
                ))}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

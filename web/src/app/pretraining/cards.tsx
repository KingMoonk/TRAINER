"use client";

import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { T, fill } from "./copy";
import { END, HOLD, PALETTE, ROWS, enoughView, perStepLine, selectedLine, type Plan, type Preview, type TextSet } from "./logic";
import { optionBox } from "@/components/option-box";

/** A row of choices (rows per step, held back, how long) as a labelled radio group. */
export function Choice<V extends string>({ label, hint, value, items, onChange }: { label: string; hint?: string; value: V; items: readonly V[]; onChange: (v: V) => void }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-sm font-medium">{label}</span>
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
      <ToggleGroup type="single" variant="outline" aria-label={label} value={value} onValueChange={(next) => next && onChange(next as V)}>
        {items.map((item) => (
          <ToggleGroupItem key={item} value={item}>
            {item}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

/** Which sets of text the run reads. Tiles are keyed by set id, so a count that changes updates the tile in place. */
export function SetsCard({ sets, loading, chosen, onToggle }: { sets: TextSet[] | undefined; loading: boolean; chosen: string[]; onToggle: (set: TextSet) => void }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{T.data}</CardTitle>
        <CardDescription>{T.whatReads}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {loading ? (
          <div className="flex flex-col gap-2" aria-label={T.readingText}>
            <Skeleton className="h-11 w-full" />
            <Skeleton className="h-11 w-full" />
          </div>
        ) : !sets?.length ? (
          <p className="text-sm text-muted-foreground">{T.noText}</p>
        ) : (
          <ul aria-label="Text sets" className="flex flex-col gap-2">
            {sets.map((s) => {
              const on = chosen.includes(s.id);
              return (
                <li key={s.id}>
                  <button
                    type="button"
                    role="checkbox"
                    aria-checked={on}
                    data-testid={`set-${s.id}`}
                    onClick={() => onToggle(s)}
                    className={optionBox(on, "flex w-full items-center gap-3 px-3 py-2")}
                  >
                    <span className={cn("flex size-5 shrink-0 items-center justify-center rounded text-xs", on ? "bg-highlight-strong text-background" : "border")}>{on ? "✓" : ""}</span>
                    <span className="font-medium">{s.name}</span>
                    <span className="text-sm text-muted-foreground">
                      {s.count.toLocaleString("en-US")} {s.count === 1 ? "example" : "examples"}{s.held ? ` · ${T.heldOut}` : ""}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <p className="text-sm text-muted-foreground" data-testid="selected-line">
          {loading ? "" : selectedLine(chosen.length)}
        </p>
        <div className="flex flex-wrap gap-6 text-sm">
          <Link href="/your-data/?kind=text" className="text-highlight hover:underline">
            {T.browse}
          </Link>
        </div>
        <p className="text-sm text-muted-foreground">
          {T.textOnly}{" "}
          <Link href="/post-training" className="text-highlight hover:underline">
            {T.postTrainingR1}
          </Link>
          {T.postTrainingR1End}
        </p>
      </CardContent>
    </Card>
  );
}

/** "Enough text?": the size verdict for the chosen text against this model, and what to expect from the version. */
export function EnoughCard({ plan, estimating }: { plan: Plan | undefined; estimating: boolean }) {
  const view = enoughView(plan, estimating);
  const expectation = plan?.expectation;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{T.enoughText}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {!plan ? (
          <Skeleton className="h-6 w-96" aria-label={T.estimating} />
        ) : estimating ? (
          <>
            <Skeleton className="h-6 w-96" aria-label={T.estimating} />
            <span className="text-xs text-muted-foreground">{T.estimating}</span>
          </>
        ) : (
          <>
            <p className="text-lg font-semibold" data-testid="enough-headline">
              {view.headline}
            </p>
            <p className="text-xs text-muted-foreground">{view.tip}</p>
          </>
        )}
        {expectation && (
          <div data-testid="expectation">
            <p className="text-sm">{expectation.headline}</p>
            <p className="mt-1 whitespace-pre-line text-xs text-muted-foreground">{expectation.details}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RowsGrid({ rows }: { rows: number[][] }) {
  return (
    <div className="flex flex-col gap-1.5 overflow-x-auto" data-testid="rows-grid" role="img" aria-label="How the text is cut into rows">
      {rows.map((row, r) => (
        <div key={r} className="flex items-center gap-0.5">
          <span className="w-12 shrink-0 text-xs text-muted-foreground">row {r + 1}</span>
          {row.map((d, i) => (
            <span key={i} className="size-3.5 shrink-0" style={{ background: d < 0 ? END : PALETTE[d % PALETTE.length] }} />
          ))}
        </div>
      ))}
    </div>
  );
}

/** How the text is fed in: the first rows of a step, rows per step, and how much is held back for testing. */
export function BatchesCard({
  plan,
  preview,
  estimating,
  rows,
  hold,
  onRows,
  onHold,
}: {
  plan: Plan | undefined;
  preview: Preview | null | undefined;
  estimating: boolean;
  rows: (typeof ROWS)[number];
  hold: (typeof HOLD)[number];
  onRows: (v: (typeof ROWS)[number]) => void;
  onHold: (v: (typeof HOLD)[number]) => void;
}) {
  const first = preview?.first_rows ?? [];
  return (
    <Card>
      <CardHeader>
        <CardTitle>{T.batches}</CardTitle>
        <CardDescription>{T.batchesDesc}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="text-sm">{fill(T.oneStep, { rows, context: plan?.context ?? 256 })}</p>
        {estimating ? <Skeleton className="h-24 w-full" aria-label={T.estimating} /> : first.length ? <RowsGrid rows={first} /> : <p className="text-sm text-muted-foreground">{T.noRowsYet}</p>}
        <Choice label={T.rowsPerStep} hint={T.rowsDesc} value={rows} items={ROWS} onChange={onRows} />
        <Choice label={T.heldBack} hint={T.heldBackDesc} value={hold} items={HOLD} onChange={onHold} />
        <p className="text-sm text-muted-foreground" data-testid="per-step">
          {perStepLine(plan, estimating)}
        </p>
      </CardContent>
    </Card>
  );
}

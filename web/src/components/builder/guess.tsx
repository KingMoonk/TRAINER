"use client";

// The Builder's learning loop (builder.py _paint_tray with `asking`, and the guess tray): before a save, a guess at what
// the change will do (skippable); after training, the guess checked against the result. The questions, the split and the
// checking are the server's (/design/questions, /design/split-plan, /design/guess-check); this file only asks and shows.

import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { design, detailOf, type Guess, type Question, type Version } from "@/lib/builder";
import { BUILDER, questionName } from "@/lib/copy/builder";
import { cn } from "@/lib/utils";

const AMBER = "text-warning";

/** One version's questions: a title when several versions are saved at once. */
export type GuessStep = { title: string; questions: Question[] };

type PanelProps = {
  /** null while the questions are still being read. */
  steps: GuessStep[] | null;
  error: string | null;
  picks: Guess[];
  saveLabel: string;
  saving: boolean;
  onPick: (step: number, key: string, value: string) => void;
  onSave: () => void;
  onSkip: () => void;
  onBack: () => void;
};

/** "Before you save · what do you expect?": a row of choices per question, then save with them, save without, or go back. */
export function GuessPanel({ steps, error, picks, saveLabel, saving, onPick, onSave, onSkip, onBack }: PanelProps) {
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => panel.current?.focus(), []);
  return (
    <div
      ref={panel}
      tabIndex={-1}
      role="group"
      aria-labelledby="guess-kicker"
      className="flex flex-col gap-3 rounded-md border px-3 py-2 outline-none"
      data-testid="guess-panel"
    >
      <div className="flex flex-col gap-1">
        <p id="guess-kicker" className={cn("text-sm font-medium", AMBER)}>
          {BUILDER.predict}
        </p>
        <p className="text-xs text-muted-foreground">{BUILDER.predict_text}</p>
      </div>
      {error ? (
        <p role="alert" className="text-sm text-destructive">
          {BUILDER.predict_failed} {error}
        </p>
      ) : !steps ? (
        <div className="flex flex-col gap-2" aria-busy="true" data-testid="guess-loading">
          <Skeleton className="h-8 w-2/3" />
          <Skeleton className="h-8 w-1/2" />
        </div>
      ) : (
        steps.map((step, i) => (
          <fieldset key={i} className="flex flex-col gap-2" data-testid="guess-step">
            {step.title && <legend className="mb-1 text-sm font-medium">{step.title}</legend>}
            {step.questions.map((question) => {
              const name = questionName(question.key);
              return (
                <div key={question.key} className="flex flex-wrap items-center gap-2">
                  <span className="w-48 text-sm" id={`guess-${i}-${question.key}`}>
                    {name}
                  </span>
                  <ToggleGroup
                    type="single"
                    variant="outline"
                    size="sm"
                    value={picks[i]?.[question.key] ?? ""}
                    onValueChange={(value) => onPick(i, question.key, value)}
                    aria-labelledby={`guess-${i}-${question.key}`}
                    data-testid={`guess-${question.key}`}
                  >
                    {question.options.map((option) => (
                      <ToggleGroupItem key={option.value} value={option.value}>
                        {option.label}
                      </ToggleGroupItem>
                    ))}
                  </ToggleGroup>
                </div>
              );
            })}
          </fieldset>
        ))
      )}
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={onSave} disabled={!steps || saving} data-testid="guess-save">
          {saveLabel}
        </Button>
        <Button size="sm" variant="outline" onClick={onSkip} disabled={!steps || saving} data-testid="guess-skip">
          {BUILDER.predict_skip}
        </Button>
        <Button size="sm" variant="ghost" onClick={onBack} disabled={saving} data-testid="guess-back">
          {BUILDER.predict_back}
        </Button>
      </div>
    </div>
  );
}

/** The saved guess of version `version`, checked against what it did next to version `before` (check_text). Nothing when it has none. */
export function GuessCheck({ before, version }: { before: number | null; version: Version }) {
  const guessed = Object.keys(version.prediction ?? {}).some((key) => key !== "question");
  const checkQ = useQuery({
    // What the check reads changes when the version trains: its checkpoint and score.
    queryKey: ["design", "guess-check", before, version.number, version.status, version.best_checkpoint ?? null, version.scorecard ?? null],
    queryFn: ({ signal }) => design.guessCheck(before!, version.number, signal),
    enabled: guessed && before !== null,
    retry: false,
  });
  if (!guessed || before === null) return null;
  if (checkQ.isPending) return <Skeleton className="h-14 w-full" aria-label={BUILDER.guess_checking} data-testid="guess-check-loading" />;
  if (checkQ.isError || !checkQ.data?.tray) {
    return checkQ.isError ? (
      <p role="alert" className="text-sm text-destructive">
        {detailOf(checkQ.error)}
      </p>
    ) : null;
  }
  const [state, kicker, text] = checkQ.data.tray;
  return (
    <div className="flex flex-col gap-1 rounded-md border px-3 py-2" data-testid="guess-check" data-state={state} aria-live="polite">
      <p className={cn("text-sm font-medium", state === "attention" && AMBER, state === "live" && "text-success")}>{kicker}</p>
      <p className="text-xs text-muted-foreground">{text}</p>
    </div>
  );
}

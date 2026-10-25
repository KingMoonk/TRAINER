"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { Hint } from "@/app/runs/hint";
import { SectionTabs } from "@/components/section-tabs";
import { T } from "./copy";

/** The run's name, its shape line (the machines, the coordinator, the settings) and the controls that end it. */
export function RunHeader({ title, shape, controls, freshness }: { title: string; shape: string; controls?: ReactNode; freshness?: string }) {
  return (
    <header className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1.5">
          <Link href="/runs/" className="w-fit text-sm text-highlight underline-offset-4 hover:underline">← {T.back}</Link>
          <h1 className="truncate text-2xl font-semibold tracking-tight" data-testid="run-title">{title}</h1>
          <Hint tip={T.shapeTip} plain>
            <span className="font-mono text-sm text-muted-foreground" data-testid="run-shape">{shape}</span>
          </Hint>
          {freshness && <span className="text-xs text-muted-foreground" data-testid="run-freshness">{freshness}</span>}
        </div>
        {controls}
      </div>
      <SectionTabs />
    </header>
  );
}

/** Where the run is: the round or step, a bar, and what is known of the time left. */
export function ProgressCard({ now, of, percent, status, statusTip }: { now: string; of: string; percent: number; status: string; statusTip: string }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border border-border bg-card px-5 py-4" data-testid="run-progress">
      <span className="text-xl font-semibold tabular-nums">
        {now}
        <span className="font-normal text-muted-foreground">{of}</span>
      </span>
      <div
        role="progressbar"
        aria-label={T.progressLabel}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(percent)}
        className="h-2 min-w-48 flex-1 basis-80 overflow-hidden rounded-full bg-muted"
      >
        <div className="h-full bg-primary" style={{ width: `${percent}%` }} />
      </div>
      {status && (
        <Hint tip={statusTip} className="font-mono text-sm text-muted-foreground">
          <span data-testid="run-status">{status}</span>
        </Hint>
      )}
    </div>
  );
}

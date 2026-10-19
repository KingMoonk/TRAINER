"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { Choice } from "./cards";
import { T, fill } from "./copy";
import { HOURS, sizingView, type Hours, type Plan } from "./logic";

/** The expected validation error as a small line chart: step across, error up. */
function Curve({ points }: { points: [number, number][] }) {
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const at = ([x, y]: [number, number]) => `${(((x - x0) / Math.max(x1 - x0, 1e-9)) * 200).toFixed(1)},${(58 - ((y - y0) / Math.max(y1 - y0, 1e-9)) * 56).toFixed(1)}`;
  return (
    <svg viewBox="0 0 200 60" className="h-16 w-full max-w-md text-highlight" role="img" aria-label={T.curveLabel} data-testid="expected-curve" preserveAspectRatio="none">
      <polyline points={points.map(at).join(" ")} fill="none" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** Is the model the right size for the text and the time: the warning, the time a run takes here, and a design and step count that fit. */
export function SizingCard({
  plan,
  estimating,
  hours,
  onHours,
  chosenSteps,
  onUseSteps,
}: {
  plan: Plan | undefined;
  estimating: boolean;
  hours: Hours;
  onHours: (h: Hours) => void;
  chosenSteps: number | null;
  onUseSteps: (steps: number) => void;
}) {
  const view = sizingView(plan);
  const curve = plan?.sizing?.expected?.curve as [number, number][] | undefined;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{T.sizingTitle}</CardTitle>
        <CardDescription>{T.sizingDesc}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Choice label={T.hoursLabel} hint={T.hoursHint} value={hours} items={Object.keys(HOURS) as Hours[]} onChange={onHours} />
        {!view ? (
          estimating || !plan ? <Skeleton className="h-6 w-96" aria-label={T.estimating} /> : null
        ) : (
          <>
            <p className={cn("text-sm font-medium", view.verdict !== "balanced" && "text-warning")} data-testid="sizing-warning" role={view.verdict === "balanced" ? undefined : "alert"}>
              {view.warning}
            </p>
            <p className="text-sm" data-testid="sizing-time">
              {view.time}
            </p>
            {view.expected && (
              <p className="text-sm text-muted-foreground" data-testid="sizing-expected">
                {view.expected}
              </p>
            )}
            {curve && curve.length > 1 && <Curve points={curve} />}
            {view.suggestion && (
              <div className="flex flex-col gap-2 rounded-md border p-3" data-testid="sizing-suggestion">
                <p className="text-sm">{view.suggestion.text}</p>
                <div className="flex flex-wrap items-center gap-3">
                  {view.suggestion.useSteps && chosenSteps !== view.suggestion.steps && (
                    <Button variant="outline" size="sm" onClick={() => onUseSteps(view.suggestion!.steps)}>
                      {fill(T.useSteps, { steps: view.suggestion.steps.toLocaleString("en-US") })}
                    </Button>
                  )}
                  {view.suggestion.designChange && (
                    <Link href="/builder" className="text-sm text-highlight hover:underline">
                      {T.openDesigner} ›
                    </Link>
                  )}
                </div>
                {chosenSteps !== null && <p className="text-xs text-muted-foreground">{fill(T.usingSteps, { steps: chosenSteps.toLocaleString("en-US") })}</p>}
              </div>
            )}
            {view.gpu && (
              <p className="text-sm text-muted-foreground" data-testid="sizing-gpu">
                {view.gpu}
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

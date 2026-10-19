"use client";

import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { SeriesChart } from "@/components/series-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { runJob, type JobLike } from "../your-data/jobs";
import { T } from "./copy";
import { errorText, lrText, specLr, type RangeResult } from "./logic";

type Outcome = { version: number | string; result: RangeResult | { error: string } };

/**
 * The learning-rate range test as a job: a skeleton while it runs (Stop ends it early), then its result as a chart of error
 * against learning rate and the sentence that reads it. A result belongs to the design it ran on: another version (or, in the Builder, another draft: any string that names the design) hides it.
 */
export function LrCard({ version, spec }: { version: number | string | null; spec: unknown }) {
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const jobId = useRef<string | null>(null);
  const stopped = useRef(false);
  const run = useMutation({
    mutationFn: async () => {
      stopped.current = false;
      return runJob<RangeResult>(async () => {
        const started = (await unwrap(api.POST("/pretraining/lr_range_test", { body: { spec: spec as Record<string, never> } }))) as JobLike;
        jobId.current = started.id;
        return started;
      });
    },
    onSuccess: (result) => setOutcome({ version: version!, result }),
    onError: (error) => {
      if (stopped.current) {
        setOutcome({ version: version!, result: { rates: [], losses: [], smoothed: [], diverged: false, cancelled: true } });
      } else {
        setOutcome({ version: version!, result: { error: errorText(error) } });
        toast.error(errorText(error));
      }
    },
  });
  const stop = () => {
    stopped.current = true;
    if (jobId.current) void api.DELETE("/jobs/{job_id}", { params: { path: { job_id: jobId.current } } });
  };
  const shown = outcome && outcome.version === version ? outcome.result : null;
  const chart = shown && "rates" in shown && shown.rates.length ? shown : null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{T.lrTitle}</CardTitle>
        <CardDescription>{T.lrTip}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          {run.isPending ? (
            <Button variant="outline" onClick={stop}>
              {T.lrStop}
            </Button>
          ) : (
            <Button variant="outline" disabled={version === null} onClick={() => run.mutate()}>
              {T.lrFind}
            </Button>
          )}
          {version === null && <span className="text-sm text-muted-foreground">{T.lrNoDesign}</span>}
        </div>
        {run.isPending ? (
          <div className="flex flex-col gap-2" aria-label={T.lrRunning}>
            <p className="text-sm text-muted-foreground">{T.lrRunning}</p>
            <Skeleton className="h-48 w-full" />
          </div>
        ) : (
          <>
            {chart && (
              <SeriesChart
                xLabel="learning rate (log10)"
                yLabel="error"
                series={[
                  { id: "loss", label: "error", points: chart.losses.map((y, i) => ({ x: Math.log10(chart.rates[i]), y })) },
                  { id: "smoothed", label: "smoothed", points: chart.smoothed.map((y, i) => ({ x: Math.log10(chart.rates[i]), y })) },
                ]}
              />
            )}
            {shown && (
              <p className="text-sm" data-testid="lr-text">
                {lrText(shown, specLr(spec))}
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

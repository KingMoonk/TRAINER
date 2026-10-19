"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, unwrap } from "@/lib/api";
import { cn } from "@/lib/utils";
import { optionBox } from "@/components/option-box";
import { runJob, type JobLike } from "../your-data/jobs";
import { errorText } from "../pretraining/logic";
import { CONTINUE, continueBody, type ContinueKind, type Continued } from "./continue";

/** Continue this version's newest run: more steps on the same data, on other text, or in a bigger design started from these weights. */
export function ContinueCard({ version, step }: { version: number; step: number }) {
  const queryClient = useQueryClient();
  const [kind, setKind] = useState<ContinueKind>("same");
  const [steps, setSteps] = useState(String(Math.max(1000, Math.round(step / 2 / 100) * 100)));
  const [ids, setIds] = useState<string[]>([]);
  const [width, setWidth] = useState("128");
  const [layers, setLayers] = useState("4");
  const [queued, setQueued] = useState<Continued | null>(null);
  const sets = useQuery({ queryKey: ["pretraining", "text_sets"], queryFn: () => unwrap(api.GET("/pretraining/text_sets")), enabled: kind !== "same" });

  const body = continueBody({ kind, steps, ids, width, layers });
  const go = useMutation({
    mutationFn: () =>
      runJob<Continued>(() => unwrap(api.POST("/versions/{number}/continue", { params: { path: { number: version } }, body: body! })) as Promise<JobLike>),
    onSuccess: (found) => {
      setQueued(found);
      toast.success(`Queued: ${found.label}`);
      queryClient.invalidateQueries({ queryKey: ["queue"] });
      queryClient.invalidateQueries({ queryKey: ["versions"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  const toggle = (id: string) => setIds((now) => (now.includes(id) ? now.filter((x) => x !== id) : [...now, id]));

  return (
    <Card data-testid="continue-card">
      <CardHeader>
        <CardTitle>{CONTINUE.title}</CardTitle>
        <CardDescription>{CONTINUE.lead.replace("{step}", step.toLocaleString("en-US"))}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div role="radiogroup" aria-label={CONTINUE.title} className="flex flex-col gap-2">
          {(Object.keys(CONTINUE.kinds) as ContinueKind[]).map((key) => (
            <button key={key} type="button" role="radio" aria-checked={kind === key} onClick={() => setKind(key)} className={optionBox(kind === key, "flex flex-col gap-0.5 px-3 py-2 text-left")}>
              <span className="font-medium">{CONTINUE.kinds[key].title}</span>
              <span className="text-sm text-muted-foreground">{CONTINUE.kinds[key].note}</span>
            </button>
          ))}
        </div>
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium">{kind === "grow" ? CONTINUE.stepsGrow : CONTINUE.stepsAdd}</span>
          <Input inputMode="numeric" className="w-40" value={steps} onChange={(e) => setSteps(e.target.value)} aria-label={CONTINUE.stepsAdd} />
        </label>
        {kind !== "same" && (
          <ul aria-label="Text sets" className="flex flex-col gap-1">
            {(sets.data ?? []).filter((s) => !s.held).map((s) => (
              <li key={s.id}>
                <button type="button" role="checkbox" aria-checked={ids.includes(s.id)} onClick={() => toggle(s.id)} className={optionBox(ids.includes(s.id), "w-full px-3 py-1.5 text-left text-sm")}>
                  {s.name}
                </button>
              </li>
            ))}
          </ul>
        )}
        {kind === "grow" && (
          <div className="flex gap-3">
            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium">{CONTINUE.width}</span>
              <Input inputMode="numeric" className="w-28" value={width} onChange={(e) => setWidth(e.target.value)} aria-label={CONTINUE.width} />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium">{CONTINUE.layers}</span>
              <Input inputMode="numeric" className="w-28" value={layers} onChange={(e) => setLayers(e.target.value)} aria-label={CONTINUE.layers} />
            </label>
          </div>
        )}
        <div>
          <Button disabled={!body || go.isPending} onClick={() => go.mutate()}>
            {go.isPending ? CONTINUE.working : CONTINUE.queue}
          </Button>
        </div>
        {queued && (
          <p className={cn("text-sm")} data-testid="continued">
            {CONTINUE.queued.replace("{label}", queued.label)}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

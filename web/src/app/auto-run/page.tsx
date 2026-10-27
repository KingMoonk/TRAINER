"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { STATUS_TEXT, T, fill } from "./copy";
import { defaultVersion, isFinished, pickRun, runLabel } from "./logic";
import { PlanEditor } from "./plan-editor";
import { RunPanel } from "./run-view";
import { useCreateRun, useDefaults, useRun, useRuns } from "./use-auto-run";

const FIELD = "h-9 w-fit max-w-full rounded-md border border-input bg-background px-2 text-sm text-foreground";
const NEW = "new";

/** A plan for a new run: pick the version, change the plan, and start it (or keep it as a draft). */
function NewRun({ onMade }: { onMade: (id: string) => void }) {
  const versions = useQuery({ queryKey: ["versions"], queryFn: () => unwrap(api.GET("/versions")) });
  const [picked, setPicked] = useState<number | null>(null);
  const version = picked ?? defaultVersion(versions.data);
  const defaults = useDefaults(version);
  const create = useCreateRun(onMade);

  if (versions.isSuccess && !versions.data.length) return <p className="text-sm text-muted-foreground">{T.noVersions}</p>;
  return (
    <div className="flex flex-col gap-4">
      <label className="flex flex-col gap-1 text-sm">
        {T.versionLabel}
        <select className={FIELD} value={version ?? ""} onChange={(e) => setPicked(Number(e.target.value))} disabled={!versions.data?.length}>
          {(versions.data ?? []).map((v) => (
            <option key={v.number} value={v.number}>
              {fill(T.versionN, { n: v.number })} · {v.name}
            </option>
          ))}
        </select>
      </label>
      {defaults.data ? (
        <PlanEditor
          key={version}
          initial={defaults.data}
          pending={create.isPending}
          actions={[
            { label: T.start, primary: true, run: (plan) => create.mutate({ plan, start: true }) },
            { label: T.saveDraft, run: (plan) => create.mutate({ plan, start: false }) },
          ]}
        />
      ) : (
        <Skeleton className="h-40 w-full" aria-label={T.loadingPlan} />
      )}
    </div>
  );
}

function Shown({ id, onAnother }: { id: string; onAnother: () => void }) {
  const known = useRuns().data?.find((r) => r.id === id);
  const going = !!known && !isFinished(known.status);
  const run = useRun(id, going);
  if (!run.data) return <Skeleton className="h-40 w-full" />;
  return (
    <>
      <RunPanel run={run.data} />
      {isFinished(run.data.state.status) && (
        <div>
          <Button variant="outline" onClick={onAnother}>
            {T.another}
          </Button>
        </div>
      )}
    </>
  );
}

export default function AutoRunPage() {
  const runs = useRuns();
  const [chosen, setChosen] = useState<string | null>(null);
  useEffect(() => {
    if (runs.isError) toast.error(T.couldNotLoad, { id: "auto-run-load" });
  }, [runs.isError]);

  // The run the page opens on stays shown when it ends, so its last state can be read; "Start another" or Open changes it.
  const open = pickRun(runs.data, null);
  if (chosen === null && open) setChosen(open);
  const shownId = chosen === NEW ? null : (chosen ?? open);
  const others = useMemo(() => (runs.data ?? []).filter((r) => r.id !== shownId), [runs.data, shownId]);

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={T.title} lead={T.lead} />
      {!runs.data ? (
        <Skeleton className="h-40 w-full" />
      ) : shownId ? (
        <Shown key={shownId} id={shownId} onAnother={() => setChosen(NEW)} />
      ) : (
        <NewRun onMade={setChosen} />
      )}
      {others.length > 0 && (
        <section className="flex flex-col gap-2" aria-label={T.earlier}>
          <h2 className="text-sm font-medium">{T.earlier}</h2>
          <ul className="flex flex-col gap-2">
            {others.map((r) => (
              <li key={r.id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border p-3 text-sm">
                <span>
                  {runLabel(r)} <span className="text-muted-foreground">· {STATUS_TEXT[r.status] ?? r.status}</span>
                </span>
                <Button variant="outline" size="sm" onClick={() => setChosen(r.id)}>
                  {T.open}
                </Button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

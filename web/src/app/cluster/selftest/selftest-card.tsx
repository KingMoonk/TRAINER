"use client";

import { Check, CircleDashed, Loader2, Minus, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { problem } from "@/lib/cluster";
import { SELFTEST as T } from "./copy";
import { headline, ranAt, visible, type SelfTestStep } from "./logic";
import { useSelfTest, useStartSelfTest } from "./use-selftest";

const ICON = { pass: Check, fail: X, skipped: Minus, running: Loader2, pending: CircleDashed } as const;

function StepRow({ step }: { step: SelfTestStep }) {
  const Icon = ICON[step.status];
  return (
    <li className="flex gap-2 text-sm" data-testid={`selftest-step-${step.id}`} data-status={step.status}>
      <Icon
        aria-label={T.states[step.status]}
        role="img"
        className={`mt-0.5 size-4 shrink-0 ${step.status === "fail" ? "text-destructive" : "text-muted-foreground"} ${step.status === "running" ? "animate-spin" : ""}`}
      />
      <div className="min-w-0 flex-1">
        <p className={step.status === "pass" || step.status === "fail" ? "font-medium" : "text-muted-foreground"}>
          {step.title}
          {step.seconds !== null && <span className="ml-2 font-normal text-muted-foreground">{T.seconds.replace("{s}", String(step.seconds))}</span>}
        </p>
        {step.detail && <p className="break-words text-muted-foreground">{step.detail}</p>}
        {step.status === "fail" && step.fix && (
          <p className="break-words text-destructive" data-testid="selftest-fix">
            <span className="font-medium">{T.fix}:</span> {step.fix}
          </p>
        )}
      </div>
    </li>
  );
}

/** The Test the cluster card: pick the machines, run the self-test, watch its steps, keep the last verdict. `peers` are the listed machines. */
export function SelfTestCard({ peers }: { peers: { address: string; hostname: string }[] }) {
  const read = useSelfTest();
  const [note, setNote] = useState("");
  const start = useStartSelfTest((why) => setNote(T.couldNotStart.replace("{why}", why)));
  const [unticked, setUnticked] = useState<string[]>([]);
  const chosen = peers.map((peer) => peer.address).filter((address) => !unticked.includes(address));
  const run = read.data;
  const running = run?.state === "running" || start.isPending;
  const when = (seconds: number) => new Date(seconds * 1000).toLocaleString();
  const last = ranAt(run, when);

  return (
    <Card data-testid="selftest">
      <CardHeader>
        <CardTitle>{T.title}</CardTitle>
        <CardDescription>{T.lead}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {peers.length === 0 ? (
          <p className="text-sm text-muted-foreground">{T.noPeers}</p>
        ) : (
          <fieldset className="flex flex-col gap-1" disabled={running}>
            <legend className="text-sm font-medium">{T.pick}</legend>
            {peers.map((peer) => (
              <label key={peer.address} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={!unticked.includes(peer.address)}
                  onChange={(event) => setUnticked(event.target.checked ? unticked.filter((a) => a !== peer.address) : [...unticked, peer.address])}
                />
                {peer.hostname || peer.address}
                {peer.hostname && peer.hostname !== peer.address && <span className="text-muted-foreground">{peer.address}</span>}
              </label>
            ))}
          </fieldset>
        )}
        <div className="flex flex-wrap items-center gap-3">
          <Button
            disabled={running || chosen.length === 0}
            onClick={() => {
              setNote("");
              start.mutate(chosen);
            }}
          >
            {running ? T.running : T.start}
          </Button>
          <p className="text-sm" data-testid="selftest-headline" role="status">{headline(run)}</p>
        </div>
        {read.isError && <p className="text-sm text-muted-foreground">{T.couldNotLoad.replace("{why}", problem(read.error))}</p>}
        {note && <p role="alert" className="text-sm text-destructive">{note}</p>}
        {last && <p className="text-sm text-muted-foreground" data-testid="selftest-ran">{last}</p>}
        <ol className="flex flex-col gap-2" aria-label={T.title}>
          {visible(run).map((step) => (
            <StepRow key={step.id} step={step} />
          ))}
        </ol>
      </CardContent>
    </Card>
  );
}

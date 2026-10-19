"use client";

import { toast } from "sonner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { JOIN } from "@/lib/copy/join";
import { useAskPeerToPause } from "@/lib/join";
import { fill } from "@/lib/members";
import { busyText, unreachableText, type Readiness } from "./readiness";

export async function copyAddress(paste: string) {
  try {
    await navigator.clipboard.writeText(paste);
    toast.success(fill(JOIN.copied, { address: paste }));
  } catch {
    toast.error(fill(JOIN.couldNotCopy, { address: paste }));
  }
}

function Status({ item }: { item: Readiness }) {
  if (item.state === "ready") return <Badge variant="secondary">{JOIN.ready}</Badge>;
  if (item.state === "busy") return <Badge variant="outline">{JOIN.busy}</Badge>;
  return <Badge variant="destructive">{JOIN.unreachable}</Badge>;
}

/**
 * Are the chosen machines ready? One line each (Ready, Busy with what and for how long, Unreachable and why), and under it an alert for
 * every machine that needs a person: ask a busy one to pause its job, or copy the https:// address of one that cannot be reached.
 */
export function Roster({ items, runLabel }: { items: Readiness[]; runLabel?: string }) {
  const ask = useAskPeerToPause();
  if (items.length === 0) return null;
  return (
    <section aria-label={JOIN.rosterTitle} className="flex flex-col gap-2" data-testid="roster">
      <h3 className="text-sm font-medium">{JOIN.rosterTitle}</h3>
      <p className="text-xs text-muted-foreground">{JOIN.rosterLead}</p>
      <ul className="flex flex-col gap-1">
        {items.map((item) => (
          <li key={item.address} className="flex flex-wrap items-center gap-2 text-sm" data-testid={`roster-${item.address}`} data-state={item.state}>
            <Status item={item} />
            <span>{item.state === "ready" ? fill(JOIN.readyText, { node: item.name }) : item.state === "busy" ? busyText(item) : unreachableText(item)}</span>
          </li>
        ))}
      </ul>
      {items
        .filter((item) => item.state !== "ready")
        .map((item) =>
          item.state === "busy" ? (
            <Alert key={item.address} data-testid={`roster-alert-${item.address}`}>
              <AlertTitle>{item.label ? fill(JOIN.busyAlert, { node: item.name, label: item.label }) : fill(JOIN.busyAlertNoLabel, { node: item.name })}</AlertTitle>
              <AlertDescription>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={ask.isPending}
                  onClick={() => ask.mutate({ address: item.address, node: item.name, label: item.label, runLabel })}
                >
                  {ask.isPending ? JOIN.asking : fill(JOIN.askPause, { node: item.name })}
                </Button>
              </AlertDescription>
            </Alert>
          ) : (
            <Alert key={item.address} variant="destructive" data-testid={`roster-alert-${item.address}`}>
              <AlertTitle>{fill(JOIN.cantReach, { node: item.name })}</AlertTitle>
              <AlertDescription className="flex flex-col gap-2">
                <span>{fill(JOIN.addressHint, { address: item.paste, node: item.name })}</span>
                <span>
                  <Button size="sm" variant="outline" onClick={() => void copyAddress(item.paste)}>
                    {JOIN.copyAddress}
                  </Button>
                </span>
              </AlertDescription>
            </Alert>
          ),
        )}
    </section>
  );
}

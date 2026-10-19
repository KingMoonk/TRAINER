"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { JOIN } from "@/lib/copy/join";
import { useAskPeerToPause } from "@/lib/join";
import { etaText, fill, pasteAddress } from "@/lib/members";
import type { Run } from "./logic";
import { copyAddress } from "./roster";

/**
 * On a run's page: an alert for each machine that cannot join without a person. Waiting behind another job: ask that machine to pause it.
 * Unreachable: copy the https:// address to add (or fix) in the Machines list.
 */
export function MemberAlerts({ run }: { run: Run }) {
  const ask = useAskPeerToPause();
  const blocked = run.members.filter((member) => !member.local && (member.state === "waiting" || member.state === "unreachable"));
  if (blocked.length === 0) return null;
  return (
    <div className="flex flex-col gap-2" data-testid={`member-alerts-${run.id}`}>
      {blocked.map((member) => {
        const blocking = member.blocking_job;
        if (member.state === "unreachable") {
          const paste = pasteAddress(member.address);
          return (
            <Alert key={member.node} variant="destructive" data-testid={`member-alert-${member.node}`}>
              <AlertTitle>{fill(JOIN.cantReach, { node: member.node })}</AlertTitle>
              <AlertDescription className="flex flex-col gap-2">
                <span>{member.reason || fill(JOIN.addressHint, { address: paste, node: member.node })}</span>
                <span>
                  <Button size="sm" variant="outline" onClick={() => void copyAddress(paste)}>
                    {JOIN.copyAddress}
                  </Button>
                </span>
              </AlertDescription>
            </Alert>
          );
        }
        return (
          <Alert key={member.node} data-testid={`member-alert-${member.node}`}>
            <AlertTitle>{blocking ? fill(JOIN.busyAlert, { node: member.node, label: blocking.label }) : fill(JOIN.busyAlertNoLabel, { node: member.node })}</AlertTitle>
            <AlertDescription className="flex flex-col gap-2">
              {blocking && <span>{fill(JOIN.busyText, { node: member.node, label: blocking.label, eta: etaText(blocking.eta_s) })}</span>}
              <span>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={ask.isPending}
                  onClick={() => ask.mutate({ address: member.address, node: member.node, label: blocking?.label ?? "", runLabel: run.label })}
                >
                  {fill(JOIN.askPause, { node: member.node })}
                </Button>
              </span>
            </AlertDescription>
          </Alert>
        );
      })}
    </div>
  );
}

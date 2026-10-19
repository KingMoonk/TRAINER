"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { JOIN } from "@/lib/copy/join";
import { inviteLine, inviteTitle, pauseAndJoinLabel, requestAcceptLabel, requestTitle } from "@/lib/invite-text";
import { useAcceptPause, useDeclinePause, useInvites, useLeave, usePauseAndJoin } from "@/lib/join";
import { stateLabel } from "@/lib/members";

/**
 * On every page of the machine that was added to a run: who added it and where it stands, with no accept step (it joins by itself).
 * [Pause '…' and join now] only while it waits behind a local job; [Leave] always. A pause request from another machine is an alert a
 * person answers; nothing is paused until they do.
 */
export function InviteBanner() {
  const { data } = useInvites();
  const pausing = usePauseAndJoin();
  const leaving = useLeave();
  const accepting = useAcceptPause();
  const declining = useDeclinePause();
  if (!data || (data.invites.length === 0 && data.pause_requests.length === 0)) return null;
  return (
    <section aria-label="Runs this machine was added to" className="flex flex-col gap-2 px-4 pt-3 md:px-6" data-testid="invite-banner">
      {data.pause_requests.map((request) => (
        <Alert key={request.id} data-testid={`pause-request-${request.id}`}>
          <AlertTitle>{requestTitle(request)}</AlertTitle>
          <AlertDescription className="flex flex-wrap items-center gap-2">
            {request.blocking_job && (
              <Button size="sm" disabled={accepting.isPending} onClick={() => accepting.mutate(request)}>
                {requestAcceptLabel(request)}
              </Button>
            )}
            <Button size="sm" variant="outline" disabled={declining.isPending} onClick={() => declining.mutate(request)}>
              {JOIN.notNow}
            </Button>
          </AlertDescription>
        </Alert>
      ))}
      {data.invites.map((invite) => {
        const pauseLabel = pauseAndJoinLabel(invite);
        return (
          <Alert key={invite.job_id} data-testid={`invite-${invite.job_id}`} data-state={invite.state}>
            <AlertTitle className="flex flex-wrap items-center gap-2">
              {inviteTitle(invite)}
              <Badge variant={invite.state === "unreachable" ? "destructive" : "secondary"}>{stateLabel(invite.state)}</Badge>
            </AlertTitle>
            <AlertDescription className="flex flex-col gap-2">
              <span data-testid={`invite-line-${invite.job_id}`}>{inviteLine(invite)}</span>
              <span className="flex flex-wrap items-center gap-2">
                {pauseLabel && (
                  <Button size="sm" disabled={pausing.isPending} onClick={() => pausing.mutate(invite)}>
                    {pauseLabel}
                  </Button>
                )}
                <Button size="sm" variant="outline" disabled={leaving.isPending} onClick={() => leaving.mutate(invite)}>
                  {leaving.isPending ? JOIN.leaving : JOIN.leave}
                </Button>
              </span>
            </AlertDescription>
          </Alert>
        );
      })}
    </section>
  );
}

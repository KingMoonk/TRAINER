// What the invite banner and the Queue page's member row say about a run this machine was added to. Pure, so a test reads every state's words.
import { JOIN } from "@/lib/copy/join";
import type { Invite, PauseRequest } from "@/lib/join";
import { etaText, fill } from "@/lib/members";

/** "Michael's Mac added this machine to 'Version 7'". */
export function inviteTitle(invite: Invite): string {
  return fill(invite.owner ? JOIN.addedBy : JOIN.addedByUnknown, { owner: invite.owner, run: invite.run_label });
}

/** The one status line under the title, by state. */
export function inviteLine(invite: Invite): string {
  const blocking = invite.blocking_job;
  switch (invite.state) {
    case "joining":
      return fill(JOIN.joiningAt, { round: invite.round ?? 0 });
    case "training":
      return fill(JOIN.trainingLine, {
        round: invite.round ?? 0,
        rounds: invite.rounds ?? 0,
        speed: invite.tokens_per_second == null ? "" : fill(JOIN.speed, { n: Math.round(invite.tokens_per_second).toLocaleString("en-US") }),
      });
    case "waiting":
      return blocking ? fill(JOIN.waitingLine, { label: blocking.label, eta: etaText(blocking.eta_s) }) : JOIN.waitingLineNoLabel;
    case "invited":
      return JOIN.invitedLine;
    case "behind":
      return JOIN.behindLine;
    case "unreachable":
      return fill(JOIN.unreachableLine, { why: invite.reason });
    default:
      return invite.reason;
  }
}

/** The button that pauses the local job and joins now: only for a member that waits behind a local job. */
export function pauseAndJoinLabel(invite: Invite): string | null {
  return invite.can_pause_and_join && invite.blocking_job ? fill(JOIN.pauseAndJoin, { label: invite.blocking_job.label }) : null;
}

/** "Michael's Mac asks this machine to pause 'Version 1' so 'Version 7' can use it". */
export function requestTitle(request: PauseRequest): string {
  const forRun = request.run_label ? fill(JOIN.forRun, { run: request.run_label }) : "";
  return request.blocking_job
    ? fill(JOIN.pauseRequest, { from: request.from_node, label: request.blocking_job.label, forRun })
    : fill(JOIN.pauseRequestNoLabel, { from: request.from_node, forRun });
}

export function requestAcceptLabel(request: PauseRequest): string {
  return request.blocking_job ? fill(JOIN.pauseIt, { label: request.blocking_job.label }) : JOIN.pauseItNoLabel;
}

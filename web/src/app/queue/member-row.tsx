"use client";

import { Badge } from "@/components/ui/badge";
import { JOIN } from "@/lib/copy/join";
import type { Invite } from "@/lib/join";
import { fill, stateLabel } from "@/lib/members";

/** How far a member is through the run it joined, as a fraction (0 until the owner has reported a round count). */
export function roundFraction(invite: Invite): number {
  return invite.rounds ? Math.min(1, (invite.round ?? 0) / invite.rounds) : 0;
}

/** "Member of 'Version 7' from Michael's Mac". */
export function memberOfText(invite: Invite): string {
  return fill(JOIN.memberOf, { run: invite.run_label, owner: invite.owner ? fill(JOIN.fromOwner, { owner: invite.owner }) : "" });
}

/** "Round 6 of 50, 1,200 tok/s": the round being trained (the owner reports the rounds finished), or a line saying the owner has not reported yet. */
export function roundText(invite: Invite): string {
  if (!invite.rounds) return fill(JOIN.roundsUnknown, { owner: invite.owner || "the owner" });
  const speed = invite.tokens_per_second == null ? "" : fill(JOIN.speed, { n: Math.round(invite.tokens_per_second).toLocaleString("en-US") });
  return `${fill(JOIN.roundOf, { round: Math.min((invite.round ?? 0) + 1, invite.rounds), rounds: invite.rounds })}${speed}`;
}

/** The Queue page's running member job: the run it belongs to, its state in the shared vocabulary, and the run's round and progress. */
export function MemberRow({ invite }: { invite: Invite }) {
  const percent = Math.round(roundFraction(invite) * 100);
  return (
    <div className="flex flex-col gap-2" data-testid="member-row">
      <p className="flex flex-wrap items-center gap-2 text-base" data-testid="member-of">
        {memberOfText(invite)}
        <Badge variant="secondary">{stateLabel(invite.state)}</Badge>
      </p>
      <p className="text-sm text-muted-foreground" data-testid="member-round">{roundText(invite)}</p>
      <div role="progressbar" aria-label="Run progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} className="h-2 w-full rounded-full bg-muted">
        <div className="h-2 rounded-full bg-highlight-strong transition-all" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

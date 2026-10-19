import { describe, expect, it } from "vitest";
import { inviteLine, inviteTitle, pauseAndJoinLabel, requestAcceptLabel, requestTitle } from "./invite-text";
import type { PauseRequest } from "./join";
import { invite } from "@/test/invites";


describe("the invite banner's words", () => {
  it("names who added this machine to which run", () => {
    expect(inviteTitle(invite())).toBe("Michael's Mac added this machine to 'Version 7'");
    expect(inviteTitle(invite({ owner: "" }))).toBe("A computer added this machine to 'Version 7'");
  });

  it("has one status line per state", () => {
    expect(inviteLine(invite({ state: "joining", round: 4 }))).toBe("Joining at round 4");
    expect(inviteLine(invite({ state: "training", round: 6, rounds: 50, tokens_per_second: 1200 }))).toBe("Training: round 6 of 50, 1,200 tok/s");
    expect(inviteLine(invite({ state: "training", round: 6, rounds: 50 }))).toBe("Training: round 6 of 50");
    expect(inviteLine(invite({ state: "waiting", blocking_job: { label: "Version 1", eta_s: 720 } }))).toBe("Waiting: busy with 'Version 1', about 12 min left, joins when it finishes");
    expect(inviteLine(invite({ state: "waiting", blocking_job: { label: "Version 1", eta_s: null } }))).toBe("Waiting: busy with 'Version 1', joins when it finishes");
    expect(inviteLine(invite({ state: "invited" }))).toBe("Invited: it joins by itself when this machine is free");
    expect(inviteLine(invite({ state: "behind" }))).toBe("Behind: it missed a round");
    expect(inviteLine(invite({ state: "unreachable", reason: "Michael's Mac did not answer." }))).toBe("Unreachable: Michael's Mac did not answer.");
    expect(inviteLine(invite({ state: "left", reason: "You left." }))).toBe("You left.");
  });

  it("offers pause-and-join only while waiting behind a local job", () => {
    const blocking = { label: "Version 1", eta_s: 60 };
    expect(pauseAndJoinLabel(invite({ state: "waiting", can_pause_and_join: true, blocking_job: blocking }))).toBe("Pause 'Version 1' and join now");
    expect(pauseAndJoinLabel(invite({ state: "waiting", can_pause_and_join: false, blocking_job: blocking }))).toBeNull();
    expect(pauseAndJoinLabel(invite({ state: "training", can_pause_and_join: false }))).toBeNull();
  });

  it("words a pause request and its answer", () => {
    const request: PauseRequest = { id: "q1", at: 1, from_node: "Michael's Mac", run_label: "Version 7", blocking_job: { label: "Version 1", eta_s: null } };
    expect(requestTitle(request)).toBe("Michael's Mac asks this machine to pause 'Version 1' so 'Version 7' can use it");
    expect(requestAcceptLabel(request)).toBe("Pause 'Version 1'");
    const bare: PauseRequest = { ...request, run_label: "", blocking_job: null };
    expect(requestTitle(bare)).toBe("Michael's Mac asks this machine to pause what it is training");
    expect(requestAcceptLabel(bare)).toBe("Pause it");
  });
});
